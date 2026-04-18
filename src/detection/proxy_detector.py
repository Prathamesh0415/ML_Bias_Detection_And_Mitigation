import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
import logging

logger = logging.getLogger(__name__)

class ProxyDetector:
    def __init__(self, random_state):
        self.random_state = random_state

    def detect_proxies(self, df: pd.DataFrame, sensitive_cols: list) -> dict:
        """ Function to detect proxy variables (features that induce bias when sensitive columns are removed) """ 
        proxy_results = {}
        
        # 1. Encode nominal text/bool data
        df_encoded = df.copy()
        encoders = {}
        # Ensure we catch both object and bool types
        for col in df_encoded.select_dtypes(include=['object', 'bool']).columns:
            encoders[col] = LabelEncoder()
            # Cast to string first to prevent mixed-type errors
            df_encoded[col] = encoders[col].fit_transform(df_encoded[col].astype(str).fillna("Missing"))

        for target_col in sensitive_cols:
            logger.info(f"\nAnalyzing potential proxies for sensitive attribute: '{target_col}'")

            # --- ADD THIS FIX BLOCK ---
            # Drop rows where the target variable is NaN. We cannot predict a missing target.
            df_target = df_encoded.dropna(subset=[target_col]).copy()
            
            # Failsafe: If a column was entirely empty, skip it so the pipeline doesn't crash
            if df_target.empty:
                logger.warning(f"All values for '{target_col}' are missing. Skipping.")
                continue
            
            # Isolate features and target using the cleaned subset
            X = df_target.drop(columns=sensitive_cols)
            y = df_target[target_col]
            # --------------------------
            
            # 2. Split BEFORE imputation to prevent data leakage
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            
            # 3. Calculate median ONLY on the training set, then apply to both
            train_medians = X_train.median()
            X_train = X_train.fillna(train_medians)
            X_test = X_test.fillna(train_medians)

            is_categorical = df[target_col].dtype == 'object' or len(df[target_col].unique()) < 10

            if is_categorical:
                # 4. Added class_weight='balanced' to penalize ignoring minority classes 
                # 5. Added n_jobs=-1 to parallelize tree building
                model = RandomForestClassifier(
                    random_state=self.random_state, 
                    class_weight='balanced', 
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # 6. Use Balanced Accuracy to reveal true predictive power
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