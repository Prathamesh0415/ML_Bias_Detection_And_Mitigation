import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLINICAL_PATH = RAW_DATA_DIR / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"
SURVIVAL_PATH = RAW_DATA_DIR / "DATASET2.txt"
TPM_PATH = RAW_DATA_DIR / "TCGA-BRCA.star_tpm.tsv" 

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "tcga_brca_merged.parquet"

# --- MODEL & EXPLAINABILITY CONFIG ---
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"

# Ensure these directories exist when config is loaded
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


PROXY_RANDOM_STATE = 42