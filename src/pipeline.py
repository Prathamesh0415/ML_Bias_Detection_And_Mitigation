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
    # PHASE 1, 2 & 3: DATA, BASELINE & SHAP
    # =================================================================
    loader = CreditDataLoader(data_path=config.CREDIT_DATA_PATH, processed_path=config.PROCESSED_DATA_PATH)
    df = loader.load_and_clean()

    # Skip scanning to save time in the final run, we know the proxies
    logger.info("=== Phase 2 & 3: Baseline Training & SHAP ===")
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
    # We create a temporary DataFrame for evaluation so we don't mess up the 23 features the ML models expect
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
    
    # Check both Gender and Age Bias
    evaluate_aif360_fairness("Baseline RF (SEX)", y_test, baseline_preds, X_test, protected_col='sex', privileged_val=1, unprivileged_val=2)
    evaluate_aif360_fairness("Baseline RF (AGE)", y_test, baseline_preds, X_test_eval, protected_col='age_binary', privileged_val=1, unprivileged_val=0)

    # ---------------------------------------------------------
    # SHOWDOWN 2: ROUTE A (AIF360 REWEIGHING)
    # ---------------------------------------------------------
    logger.info("\n--- EVALUATING ROUTE A (AIF360) ---")
    # AIF360 handles one primary attribute smoothly. We will let it fix 'sex'.
    aif_model = train_aif360_baseline(X_train, y_train, protected_col='sex', privileged_val=1, unprivileged_val=2)
    aif_preds = aif_model.predict(X_test)
    
    logger.info(f"AIF360 Accuracy: {accuracy_score(y_test, aif_preds):.4f}")
    evaluate_aif360_fairness("AIF360 (SEX)", y_test, aif_preds, X_test, protected_col='sex', privileged_val=1, unprivileged_val=2)
    evaluate_aif360_fairness("AIF360 (AGE)", y_test, aif_preds, X_test_eval, protected_col='age_binary', privileged_val=1, unprivileged_val=0)

    # ---------------------------------------------------------
    # SHOWDOWN 3: ROUTE B (PYTORCH INTERSECTIONAL)
    # ---------------------------------------------------------
    logger.info("\n--- EVALUATING ROUTE B (PYTORCH ADVERSARIAL) ---")
    
    # We pass BOTH Sex and Age to the Adversarial network to cure both simultaneously!
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