import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditDataLoader:
    def __init__(self, data_path, processed_path):
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)

    def load_and_clean(self) -> pd.DataFrame:
        try:
            if self.processed_path.exists():
                logger.info(f"Found cached dataset at {self.processed_path}. Loading fast...")
                return pd.read_parquet(self.processed_path)

            logger.info(f"Loading Credit Card Data from {self.data_path}...")
            
            # The UCI dataset has a double header (X1, X2... on row 0, real names on row 1).
            # header=1 tells Pandas to use the second row as the actual column names.
            if self.data_path.suffix in ['.xls', '.xlsx']:
                df = pd.read_excel(self.data_path, header=1)
            else:
                df = pd.read_csv(self.data_path, header=1)

            # Drop the ID column as it provides no mathematical value and could act as a noisy proxy
            if 'ID' in df.columns:
                df = df.drop(columns=['ID'])

            logger.info("Cleaning and standardizing column names...")
            
            # Rename the target column to something much easier to type
            if 'default payment next month' in df.columns:
                df.rename(columns={'default payment next month': 'default'}, inplace=True)

            # Standardize all column names to lowercase with underscores
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            logger.info(f"Saving processed dataframe to {self.processed_path}...")
            # Ensure the directory exists
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.processed_path)

            logger.info(f"Final Dataset Shape: {df.shape}")
            return df

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}. Ensure the dataset is in the correct folder.")
            raise