import os
from pathlib import Path

# __file__ -> this contains the string path of the current file
# Path() converts to Path object (to work with it further)
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "adult.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
SENSITIVE_CANDIDATE_LABELS = [
    "sensitive demographic information", 
    "protected class", 
    "personal identity",
    "financial transaction"
]

SENSITIVITY_THRESHOLD = 0.65

PROXY_RANDOM_STATE = 42

