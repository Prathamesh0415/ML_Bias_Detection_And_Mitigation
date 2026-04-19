import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, balanced_accuracy_score
import logging

logger = logging.getLogger(__name__)

class ProxyDetector:
    def __init__(self, random_state):
        self.random_state = random_state

    def detect_proxies(self, df: pd.DataFrame, sensitive_cols: list) -> dict:
        """ Function to detect proxy variables (features that induce bias when sensitive columns are removed) """ 
        proxy_results = {}
        
        # We operate on a copy to avoid SettingWithCopyWarnings
        df_clean = df.copy()

        for target_col in sensitive_cols:
            logger.info(f"\nAnalyzing potential proxies for sensitive attribute: '{target_col}'")

            # Drop rows where the target variable is NaN. We cannot predict a missing target.
            df_target = df_clean.dropna(subset=[target_col]).copy()
            
            if df_target.empty:
                logger.warning(f"All values for '{target_col}' are missing. Skipping.")
                continue
            
            # Isolate features (X) and target (y)
            X = df_target.drop(columns=sensitive_cols)
            y = df_target[target_col]
            
            # Since the Credit Card dataset is mostly numeric, we just handle remaining missing feature values
            X = X.fillna(X.median(numeric_only=True))

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

            # Determine if the target is categorical (e.g., sex, marriage) or continuous (e.g., age)
            is_categorical = y.dtype == 'object' or len(y.unique()) < 10

            if is_categorical:
                model = RandomForestClassifier(
                    random_state=self.random_state, 
                    class_weight='balanced', 
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = balanced_accuracy_score(y_test, preds)
                metric = "Balanced Accuracy"
            else:
                model = RandomForestRegressor(
                    random_state=self.random_state, 
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                metric = "R2 Score"
                
            logger.info(f"Model trained to predict '{target_col}'. {metric}: {score:.4f}")

            # Extract Feature Importances to find the proxies
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(3)
            
            proxy_results[target_col] = {
                'score': score,
                'metric': metric,
                'top_proxies': feature_imp_df.to_dict('records')
            }
            
            logger.info(f"Top 3 strongest proxy features for '{target_col}':")
            for _, row in feature_imp_df.iterrows():
                logger.info(f"  - {row['Feature']}: {row['Importance']:.4f}")

        return proxy_results