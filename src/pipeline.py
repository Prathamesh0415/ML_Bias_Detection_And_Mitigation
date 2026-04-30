import sys
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_ingestion.loader import CreditDataLoader
from src.detection.proxy_detector import ProxyDetector
from src.detection.nlp_scanner import DemographicScanner
from src.models.train_baseline import train_and_save_baseline
from src.explainability.run_shap import generate_shap_explanations
from src.models.evaluate_fairness import evaluate_aif360_fairness

# --- PHASE 4 IMPORTS ---
from src.models.aif360_baseline import train_aif360_baseline
from src.models.adversarial_network import train_adversarial_network, predict_adversarial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("=====================================================")
    logger.info("    STARTING AUTOFAIR PIPELINE (CREDIT DEFAULT)      ")
    logger.info("=====================================================\n")

    # =================================================================
    # PHASE 1: DATA INGESTION
    # =================================================================
    loader = CreditDataLoader(data_path=config.CREDIT_DATA_PATH, processed_path=config.PROCESSED_DATA_PATH)
    df = loader.load_and_clean()

    # =================================================================
    # PHASE 1.5: DEMOGRAPHIC SCANNING & PROXY DETECTION
    # =================================================================
    logger.info("=== Phase 1.5: Demographic Scanning & Proxy Detection ===")
    scanner = DemographicScanner()
    # Scan columns to find sensitive attributes like 'sex', 'age', 'marriage'
    sensitive_cols = scanner.scan_columns(df.columns.tolist())
    logger.info(f"Detected Sensitive Columns: {sensitive_cols}")

    if sensitive_cols:
        detector = ProxyDetector(random_state=42)
        proxy_results = detector.detect_proxies(df, sensitive_cols)
        
        logger.info("\n--- PROXY DETECTION SUMMARY ---")
        for target, data in proxy_results.items():
            logger.info(f"Target: '{target}' | Prediction {data['metric']}: {data['score']:.4f}")
            logger.info(f"Top 3 Proxies masking as '{target}':")
            for proxy in data['top_proxies']:
                logger.info(f"  -> {proxy['Feature']} (Importance Weight: {proxy['Importance']:.4f})")
    else:
        logger.info("No sensitive columns detected. Skipping Proxy Detection.")

    # =================================================================
    # PHASE 2 & 3: BASELINE & SHAP
    # =================================================================
    logger.info("\n=== Phase 2 & 3: Baseline Training & SHAP ===")
    train_and_save_baseline(df)
    generate_shap_explanations()

    # =================================================================
    # PHASE 4: THE A/B FAIRNESS SHOWDOWN (INTERSECTIONAL)
    # =================================================================
    logger.info("=====================================================")
    logger.info("    PHASE 4: FAIRNESS MITIGATION SHOWDOWN            ")
    logger.info("=====================================================")

    df_clean = df.dropna(subset=['default']).copy()
    X = df_clean.drop(columns=['default']).fillna(df_clean.drop(columns=['default']).median())
    y = df_clean['default'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PREPARE AGE FOR EVALUATION ---
    X_test_eval = X_test.copy()
    X_test_eval['age_binary'] = (X_test_eval['age'] >= 30).astype(int) # 1 = Privileged (>=30), 0 = Unprivileged (<30)

    # ---------------------------------------------------------
    # SHOWDOWN 1: THE BIASED BASELINE
    # ---------------------------------------------------------
    logger.info("\n--- EVALUATING BASELINE MODEL ---")
    model_path = config.MODELS_DIR / "baseline_rf.joblib"
    pipeline_state = joblib.load(model_path)
    baseline_model = pipeline_state['model']
    
    baseline_preds = baseline_model.predict(X_test)
    logger.info(f"Baseline Accuracy: {accuracy_score(y_test, baseline_preds):.4f}")
    
    evaluate_aif360_fairness("Baseline RF (SEX)", y_test, baseline_preds, X_test, protected_col='sex', privileged_val=1, unprivileged_val=2)
    evaluate_aif360_fairness("Baseline RF (AGE)", y_test, baseline_preds, X_test_eval, protected_col='age_binary', privileged_val=1, unprivileged_val=0)

    # ---------------------------------------------------------
    # SHOWDOWN 2: ROUTE A (AIF360 REWEIGHING - INTERSECTIONAL)
    # ---------------------------------------------------------
    logger.info("\n--- EVALUATING ROUTE A (AIF360 INTERSECTIONAL) ---")
    
    def assign_privilege(row):
        if row['sex'] == 1 and row['age'] >= 30:
            return 1 # Privileged (Older Men)
        else:
            return 0 # Unprivileged (All other intersecting groups)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_test_eval = X_test_eval.copy()
    
    X_train['intersectional_protected'] = X_train.apply(assign_privilege, axis=1)
    X_test['intersectional_protected'] = X_test.apply(assign_privilege, axis=1)
    X_test_eval['intersectional_protected'] = X_test_eval.apply(assign_privilege, axis=1)

    aif_model_combined = train_aif360_baseline(
        X_train, 
        y_train, 
        protected_col='intersectional_protected', 
        privileged_val=1, 
        unprivileged_val=0
    )
    
    aif_preds_combined = aif_model_combined.predict(X_test)

    logger.info(f"AIF360 Intersectional Accuracy: {accuracy_score(y_test, aif_preds_combined):.4f}")
    
    evaluate_aif360_fairness("AIF360 Combined Model (Eval on SEX)", y_test, aif_preds_combined, X_test, protected_col='sex', privileged_val=1, unprivileged_val=2)
    evaluate_aif360_fairness("AIF360 Combined Model (Eval on AGE)", y_test, aif_preds_combined, X_test_eval, protected_col='age_binary', privileged_val=1, unprivileged_val=0)
    
    # ---------------------------------------------------------
    # SHOWDOWN 3: ROUTE B (PYTORCH ADVERSARIAL)
    # ---------------------------------------------------------
    logger.info("\n--- EVALUATING ROUTE B (PYTORCH ADVERSARIAL) ---")
    
    protected_train_pt = pd.DataFrame()
    protected_train_pt['sex'] = X_train['sex'] - 1  # 0=Male, 1=Female
    protected_train_pt['age_binary'] = (X_train['age'] < 30).astype(int) # 0=Older, 1=Younger
    
    adv_model = train_adversarial_network(X_train, y_train, protected_train_pt, epochs=30, batch_size=128)
    adv_preds = predict_adversarial(adv_model, X_test)
    
    logger.info(f"PyTorch Adversarial Accuracy: {accuracy_score(y_test, adv_preds):.4f}")
    evaluate_aif360_fairness("PyTorch Adversarial (SEX)", y_test, adv_preds, X_test, protected_col='sex', privileged_val=1, unprivileged_val=2)
    evaluate_aif360_fairness("PyTorch Adversarial (AGE)", y_test, adv_preds, X_test_eval, protected_col='age_binary', privileged_val=1, unprivileged_val=0)

    logger.info("\n=====================================================")
    logger.info("    AUTOFAIR PIPELINE COMPLETED SUCCESSFULLY         ")
    logger.info("=====================================================")

if __name__ == "__main__":
    run_pipeline()