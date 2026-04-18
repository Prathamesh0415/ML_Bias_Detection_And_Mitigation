import sys
from pathlib import Path
import logging
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FIXED: Function now accepts df from the master pipeline
def train_and_save_baseline(df: pd.DataFrame):
    logger.info("=== Starting Baseline Training & Dimensionality Reduction ===")

    # We predict Overall Survival (os). Drop rows where 'os' is missing.
    df_clean = df.dropna(subset=['os']).copy()
    y = df_clean['os'].astype(int)

    # Separate Genomic (ENSG) and Clinical Data
    gene_cols = [col for col in df_clean.columns if col.startswith('ENSG')]
    
    # --- UPDATED CLINICAL BLOCK ---
    # We are now injecting the exact demographic targets we scanned for in Phase 1
    target_clinical = [
        'age_at_initial_pathologic_diagnosis', 
        'pathologic_stage',
        'gender',
        'menopause_status',
        'race'
    ]
    
    # Failsafe: Only keep columns that actually exist in your specific DataFrame
    existing_cols = [col for col in target_clinical if col in df_clean.columns]
    X_clinical = df_clean[existing_cols].copy()
    
    # Clean the numeric column
    if 'age_at_initial_pathologic_diagnosis' in X_clinical.columns:
        X_clinical['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(X_clinical['age_at_initial_pathologic_diagnosis'], errors='coerce')
    
    # Clean the numeric column
    if 'age_at_initial_pathologic_diagnosis' in X_clinical.columns:
        X_clinical['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(X_clinical['age_at_initial_pathologic_diagnosis'], errors='coerce')
    
    # --- UPDATED SMART IMPUTATION BLOCK ---
    from pandas.api.types import is_numeric_dtype
    
    # Smart Imputation: Median for numbers, "Missing" for text
    for col in X_clinical.columns:
        if is_numeric_dtype(X_clinical[col]):
            X_clinical[col] = X_clinical[col].fillna(X_clinical[col].median())
        else:
            # Force column to string to safely fill NaNs with text
            X_clinical[col] = X_clinical[col].astype(str).fillna("Missing")

    # Convert text columns (like 'Male'/'Female') into binary 1s and 0s
    X_clinical = pd.get_dummies(X_clinical, drop_first=True)
    # --------------------------------------
    # ------------------------------

    X_genes = df_clean[gene_cols]
    logger.info(f"Initial Gene Count: {X_genes.shape[1]}")

    # Phase 2: Dimensionality Reduction on RNA-seq data
    var_filter = VarianceThreshold(threshold=0.1)
    X_genes_filtered = var_filter.fit_transform(X_genes)
    logger.info(f"Gene Count after Variance Filter: {X_genes_filtered.shape[1]}")

    scaler = StandardScaler()
    X_genes_scaled = scaler.fit_transform(X_genes_filtered)
    
    pca = PCA(n_components=128, random_state=42)
    X_genes_pca = pca.fit_transform(X_genes_scaled)
    logger.info(f"Gene Count after PCA Compression: {X_genes_pca.shape[1]}")

    pca_cols = [f"Gene_PC_{i+1}" for i in range(X_genes_pca.shape[1])]
    df_pca = pd.DataFrame(X_genes_pca, columns=pca_cols, index=X_clinical.index)

    X_final = pd.concat([X_clinical, df_pca], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    logger.info("Training Baseline Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"Baseline Accuracy on Test Set: {accuracy:.4f}")

    # SAVE THE PIPELINE USING JOBLIB
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
    # If run as a standalone script, it won't have the df, so this is just for safety.
    logger.warning("Please run this via the master pipeline.py")