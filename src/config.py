import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# --- NEW FINANCIAL DATA PATH ---
CREDIT_DATA_PATH = RAW_DATA_DIR / "default of credit card clients.xls" 

# Updated processed cache path
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "credit_card_cleaned.parquet"

# --- MODEL & EXPLAINABILITY CONFIG ---
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"

# Ensure these directories exist when config is loaded
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROXY_RANDOM_STATE = 42