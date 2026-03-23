import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_ingestion.loader import DataLoader
from src.detection.nlp_scanner import ZeroShotScanner
from src.detection.proxy_detector import ProxyDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase_1():
    logger.info("=== Starting Fairness Detection Pipeline Phase 1 ===")

    loader = DataLoader(config.RAW_DATA_PATH)
    df = loader.load_and_clean()

    scanner = ZeroShotScanner(
        model_name=config.ZERO_SHOT_MODEL, 
        candidate_labels=config.SENSITIVE_CANDIDATE_LABELS,
        threshold=config.SENSITIVITY_THRESHOLD
    )
    
    columns_to_scan = df.columns.tolist()
    sensitive_columns = scanner.scan_columns(columns_to_scan)
    
    if not sensitive_columns:
        logger.info("No sensitive columns detected by NLP scanner. Exiting pipeline.")
        return
        
    logger.info(f"Sensitive columns moving to proxy detection: {sensitive_columns}")

    detector = ProxyDetector(random_state=config.PROXY_RANDOM_STATE)
    proxy_report = detector.detect_proxies(df, sensitive_columns)
    
    logger.info("\n=== Phase 1 Execution Complete ===")
    logger.info("Summary Report generated. Ready for Phase 2 (Mitigation).")

if __name__ == "__main__":
    run_phase_1()