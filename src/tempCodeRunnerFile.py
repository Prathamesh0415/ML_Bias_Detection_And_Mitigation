import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_ingestion.loader import TCGALoader
from src.detection.nlp_scanner import ZeroShotScanner
from src.detection.proxy_detector import ProxyDetector

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
    
    logger.info(f"Extracted {len(clinical_columns)} clinical/metadata columns for fairness scanning.")

    scanner = ZeroShotScanner(
        model_name=config.ZERO_SHOT_MODEL, 
        candidate_labels=config.SENSITIVE_CANDIDATE_LABELS,
        threshold=config.SENSITIVITY_THRESHOLD
    )
    
    sensitive_columns = scanner.scan_columns(clinical_columns)
    
    if not sensitive_columns:
        logger.info("No sensitive columns detected by NLP scanner.")
    else:
        logger.info(f"Sensitive columns moving to proxy detection: {sensitive_columns}")
        detector = ProxyDetector(random_state=config.PROXY_RANDOM_STATE)
        
        proxy_report = detector.detect_proxies(df[clinical_columns], sensitive_columns)
    
    logger.info("\n=== Phase 1 Execution Complete ===")
    logger.info("Ready for Phase 2 (Dimensionality Reduction & Mitigation).")

if __name__ == "__main__":
    run_phase_1()