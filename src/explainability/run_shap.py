import sys
from pathlib import Path
import logging
import joblib
import shap
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_shap_explanations():
    logger.info("=== Starting SHAP Explainability Engine ===")

    model_path = config.MODELS_DIR / "baseline_rf.joblib"
    
    try:
        pipeline_state = joblib.load(model_path)
        model = pipeline_state['model']
        X_sample = pipeline_state['X_sample'] 
        logger.info("Successfully loaded frozen model from joblib.")
    except FileNotFoundError:
        logger.error(f"Could not find model at {model_path}. Run train_baseline.py first.")
        return

    logger.info("Calculating SHAP values using the Modern API...")
    explainer = shap.TreeExplainer(model)
    
    # Modern SHAP API
    explanation = explainer(X_sample)
    
    # Random Forest returns explanations for both classes (0 = Paid, 1 = Default).
    # We slice to isolate class 1 (Default).
    if len(explanation.shape) == 3:
        default_explanation = explanation[:, :, 1]
    else:
        default_explanation = explanation

    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # --- DYNAMIC FEATURE COUNT ---
    # Gets the exact number of features (23) so SHAP plots everything
    total_features = X_sample.shape[1]
    
    # ---------------------------------------------------------
    # PLOT 1: The Global Importance Bar Chart
    # ---------------------------------------------------------
    logger.info(f"Generating SHAP Bar Chart for all {total_features} features...")
    plt.figure(figsize=(10, 8)) # Increased height slightly so labels don't crowd
    
    shap.plots.bar(default_explanation, max_display=total_features, show=False)
    
    plt.title("All Features Importance (Baseline RF - Default Prediction)")
    plt.savefig(reports_dir / "shap_bar_baseline.png", bbox_inches='tight', dpi=300)
    plt.clf() 
    
    # ---------------------------------------------------------
    # PLOT 2: The Focused Beeswarm Plot
    # ---------------------------------------------------------
    logger.info(f"Generating SHAP Beeswarm Plot for all {total_features} features...")
    plt.figure(figsize=(12, 10)) # Increased height for full feature list
    
    shap.plots.beeswarm(default_explanation, max_display=total_features, show=False)
    
    plt.title("SHAP Beeswarm: Impact on Credit Default Risk")
    plt.savefig(reports_dir / "shap_beeswarm_baseline.png", bbox_inches='tight', dpi=300)
    plt.close() 

    logger.info(f"Success! Clean SHAP plots saved to: {reports_dir}")
    logger.info("=== Phase 3 Execution Complete ===\n")

if __name__ == "__main__":
    generate_shap_explanations()