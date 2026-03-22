import pandas as pd
import logging

#  logging used for better error logging
# INFO sets the hierarchy in which error logs appear
# logger is the logging class created seprately for this file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean(self) -> pd.DataFrame:
        """Loads and cleans the data"""
        try:
            logger.info(f"Loading the file from {self.file_path}")
            df = pd.read_csv(self.file_path, index_col = 0, na_values=["NA"])

            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("/", "_")
            )

            logger.info(f"File loaded successfully, Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at {self.file_path}. Please check the folder")
            raise
    
