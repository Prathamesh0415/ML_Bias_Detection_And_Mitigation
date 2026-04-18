import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_ingestion.loader import TCGALoader
from src.detection.proxy_detector import ProxyDetector
from src.detection.nlp_scanner import ClinicalDemographicScanner

# Import our new Phase 2 and Phase 3 modules
from src.models.train_baseline import train_and_save_baseline
from src.explainability.run_shap import generate_shap_explanations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("=====================================================")
    logger.info("    STARTING AUTOFAIR PIPELINE (END-TO-END)          ")
    logger.info("=====================================================\n")

    # =================================================================
    # PHASE 1: DATA INGESTION & FAIRNESS DETECTION
    # =================================================================
    logger.info("=== Phase 1: Data Ingestion & Proxy Detection ===")
    loader = TCGALoader(
        clinical_path=config.CLINICAL_PATH,
        survival_path=config.SURVIVAL_PATH,
        tpm_path=config.TPM_PATH,
        processed_path=config.PROCESSED_DATA_PATH
    )
    df = loader.load_and_clean()

    clinical_columns = [col for col in df.columns if not col.startswith('ENSG')]
    
    for col in clinical_columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)

    scanner = ClinicalDemographicScanner()
    sensitive_columns = scanner.scan_columns(clinical_columns)

    if not sensitive_columns:
        logger.warning("Scanner found no sensitive columns. Defaulting to standard targets.")
        fallback_targets = ['gender', 'age_at_initial_pathologic_diagnosis', 'menopause_status', 'race']
        sensitive_columns = [col for col in fallback_targets if col in df.columns]
        
    logger.info(f"Sensitive columns moving to proxy detection: {sensitive_columns}")
    
    detector = ProxyDetector(random_state=config.PROXY_RANDOM_STATE)
    proxy_report = detector.detect_proxies(df[clinical_columns], sensitive_columns)
    logger.info("=== Phase 1 Execution Complete ===\n")

    # =================================================================
    # PHASE 2: DIMENSIONALITY REDUCTION & BASELINE TRAINING
    # =================================================================
    # We pass the cleaned DataFrame directly to the training module
    train_and_save_baseline(df)

    # =================================================================
    # PHASE 3: SHAP EXPLAINABILITY
    # =================================================================
    generate_shap_explanations()

    logger.info("=====================================================")
    logger.info("    AUTOFAIR PIPELINE COMPLETED SUCCESSFULLY         ")
    logger.info("=====================================================")

if __name__ == "__main__":
    run_pipeline()