import sys
from pathlib import Path
import logging
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure we can import from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_baseline(df: pd.DataFrame):
    logger.info("=== Starting Baseline Training (Financial Risk) ===")

    # 1. Target Isolation
    target_col = 'default'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
        return

    # Drop rows where target is explicitly missing
    df_clean = df.dropna(subset=[target_col]).copy()
    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col])

    # 2. Smart Imputation (Dataset is entirely numeric now)
    logger.info("Imputing any missing feature values with medians...")
    X = X.fillna(X.median())

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info("Training Baseline Random Forest...")
    
    # We add class_weight='balanced' because financial defaults are usually heavily imbalanced (e.g. 78% No, 22% Yes)
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10, 
        class_weight='balanced', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"Baseline Accuracy on Test Set: {accuracy:.4f}")

    # 4. Save Pipeline State for SHAP
    save_dir = config.BASE_DIR / "models"
    save_dir.mkdir(exist_ok=True)
    
    pipeline_state = {
        'model': model,
        'feature_names': X_train.columns.tolist(),
        'X_sample': X_test.head(100) # Save real data for SHAP to explain
    }
    
    joblib_path = save_dir / "baseline_rf.joblib"
    joblib.dump(pipeline_state, joblib_path)
    logger.info(f"Model and feature map successfully saved to {joblib_path}")

if __name__ == "__main__":
    logger.warning("Please run this via the master pipeline.py")