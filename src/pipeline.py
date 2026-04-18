import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_ingestion.loader import TCGALoader
from src.detection.proxy_detector import ProxyDetector
from src.detection.nlp_scanner import ClinicalDemographicScanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase_1():
    logger.info("=== Starting Fairness Detection Pipeline Phase 1 (Genomic Edition) ===")

    loader = TCGALoader(
        clinical_path=config.CLINICAL_PATH,
        survival_path=config.SURVIVAL_PATH,
        tpm_path=config.TPM_PATH,
        processed_path = config.PROCESSED_DATA_PATH
    )
    df = loader.load_and_clean()

    clinical_columns = [col for col in df.columns if not col.startswith('ENSG')]
    
    for col in clinical_columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)

    logger.info(f"Extracted {len(clinical_columns)} clinical/metadata columns for fairness scanning.")

    scanner = ClinicalDemographicScanner()
    sensitive_columns = scanner.scan_columns(clinical_columns)

    if not sensitive_columns:
        logger.warning("Scanner found no sensitive columns. Defaulting to standard demographic targets.")

        fallback_targets = ['gender', 'age_at_initial_pathologic_diagnosis', 'menopause_status', 'race']
        
        sensitive_columns = [col for col in fallback_targets if col in df.columns]
        
    logger.info(f"Sensitive columns moving to proxy detection: {sensitive_columns}")
    

    detector = ProxyDetector(random_state=config.PROXY_RANDOM_STATE)
    proxy_report = detector.detect_proxies(df[clinical_columns], sensitive_columns)
    # -----------------------------------------------
    
    logger.info("\n=== Phase 1 Execution Complete ===")
    logger.info("Ready for Phase 2 (Dimensionality Reduction & Mitigation).")

if __name__ == "__main__":
    run_phase_1()