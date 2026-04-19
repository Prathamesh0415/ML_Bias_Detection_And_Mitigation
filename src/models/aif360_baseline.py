import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import logging

logger = logging.getLogger(__name__)

def train_aif360_baseline(X_train, y_train, protected_col="sex", privileged_val=1, unprivileged_val=2):
    """
    Trains a Baseline RF using AIF360 Reweighing for Financial Risk.
    """
    logger.info("--- ROUTE A: Training AIF360 Pre-Processing Model ---")
    
    # Recombine X and y temporarily for AIF360 dataset creation
    df_train = X_train.copy()
    # Ensure y_train is safely appended whether it's a Series or Array
    df_train['target'] = y_train.values if hasattr(y_train, 'values') else y_train

    privileged_groups = [{protected_col: privileged_val}]
    unprivileged_groups = [{protected_col: unprivileged_val}]

    # CRITICAL FIX: Explicitly define 0 as Favorable (Paid) and 1 as Unfavorable (Default)
    aif_dataset = BinaryLabelDataset(
        df=df_train,
        label_names=['target'],
        protected_attribute_names=[protected_col],
        favorable_label=0.0,
        unfavorable_label=1.0
    )

    logger.info("Calculating static sample weights via AIF360...")
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf = RW.fit_transform(aif_dataset)

    sample_weights = dataset_transf.instance_weights

    logger.info("Training Random Forest with AIF360 adjusted weights...")
    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model