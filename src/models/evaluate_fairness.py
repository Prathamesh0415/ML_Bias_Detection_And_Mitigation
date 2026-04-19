from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_aif360_fairness(model_name, y_true, y_pred, df_features, protected_col="sex", privileged_val=1, unprivileged_val=2):
    logger.info(f"=== FAIRNESS REPORT: {model_name} ===")
    
    df_true = df_features.copy()
    df_true['target'] = y_true.values if hasattr(y_true, 'values') else y_true
    
    df_pred = df_features.copy()
    df_pred['target'] = y_pred

    privileged_groups = [{protected_col: privileged_val}]
    unprivileged_groups = [{protected_col: unprivileged_val}]

    # CRITICAL FIX: Favorable class is 0 (No Default). 1 (Default) is unfavorable.
    dataset_true = StandardDataset(
        df=df_true, label_name='target', favorable_classes=[0],
        protected_attribute_names=[protected_col], privileged_classes=[[privileged_val]]
    )
    
    dataset_pred = StandardDataset(
        df=df_pred, label_name='target', favorable_classes=[0],
        protected_attribute_names=[protected_col], privileged_classes=[[privileged_val]]
    )

    metric = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    results = {
        "Disparate Impact (DI)": round(metric.disparate_impact(), 4),
        "Statistical Parity Difference (SPD)": round(metric.statistical_parity_difference(), 4),
        "Equal Opportunity Difference (EOD)": round(metric.equal_opportunity_difference(), 4)
    }

    for name, value in results.items():
        logger.info(f"{name}: {value}")
        
    return results