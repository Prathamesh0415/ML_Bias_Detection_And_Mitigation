import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

class ProxyDetector:
    def __init__(self, random_state):
        self.random_state = random_state

    def detect_proxies(self, df: pd.DataFrame, sensitive_cols: list) -> dict:
        """ Function to detect proxy variables (features that induce bias when sensitive columns are removed) """ 
        proxy_results = {}
        
        # Preprocess categorical data for the tree models
        df_encoded = df.copy()
        encoders = {}
        for col in df_encoded.select_dtypes(include=['object']).columns:
            encoders[col] = LabelEncoder()
            df_encoded[col] = encoders[col].fit_transform(df_encoded[col].fillna("Missing"))

        df_encoded = df_encoded.fillna(df_encoded.median())

        for target_col in sensitive_cols:
            logger.info(f"\nAnalyzing potential proxies for sensitive attribute: '{target_col}'")

            X = df_encoded.drop(columns=sensitive_cols)
            y = df_encoded[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

            if df[target_col].dtype == 'object' or len(df[target_col].unique()) < 10:
                model = RandomForestClassifier(random_state=self.random_state)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                metric = "Accuracy"
            else:
                model = RandomForestRegressor(random_state=self.random_state)
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