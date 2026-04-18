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
    
    # --- THE MODERN SHAP API FIX ---
    # Calling the explainer directly creates a strict 'Explanation' object.
    # This prevents the buggy legacy function from making interaction grids.
    explanation = explainer(X_sample)
    
    # Random Forest returns explanations for both survival (1) and death (0).
    # We slice the Explanation object to isolate class 1.
    # Shape is usually (samples, features, classes), so we take [:, :, 1]
    if len(explanation.shape) == 3:
        surv_explanation = explanation[:, :, 1]
    else:
        surv_explanation = explanation

    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # PLOT 1: The Global Importance Bar Chart
    # ---------------------------------------------------------
    logger.info("Generating SHAP Bar Chart...")
    plt.figure(figsize=(10, 6))
    
    # Using shap.plots.bar instead of the legacy summary_plot
    shap.plots.bar(surv_explanation, max_display=15, show=False)
    
    plt.title("Top 15 Most Important Features (Baseline RF)")
    plt.savefig(reports_dir / "shap_bar_baseline.png", bbox_inches='tight', dpi=300)
    plt.clf() 
    
    # ---------------------------------------------------------
    # PLOT 2: The Focused Beeswarm Plot
    # ---------------------------------------------------------
    logger.info("Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(10, 8))
    
    # Using shap.plots.beeswarm strictly forces the dot plot layout
    shap.plots.beeswarm(surv_explanation, max_display=15, show=False)
    
    plt.title("SHAP Beeswarm: Impact on Survival Prediction")
    plt.savefig(reports_dir / "shap_beeswarm_baseline.png", bbox_inches='tight', dpi=300)
    plt.close() 

    logger.info(f"Success! Clean SHAP plots saved to: {reports_dir}")
    logger.info("=== Phase 3 Execution Complete ===\n")

if __name__ == "__main__":
    generate_shap_explanations()