import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCGALoader:
    def __init__(self, clinical_path, survival_path, tpm_path, processed_path):
        self.clinical_path = clinical_path
        self.survival_path = survival_path
        self.tpm_path = tpm_path
        self.processed_path = Path(processed_path)

    def _standardize_barcodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """The TCGA barcodes vary in length, this ensures uniform length for clean merging
            Only the first 12 characters identify the patient the rest are different samples from the
            patient from the same tumor size, thus removing duplicates actually help, as if the 
            duplicate rows are divided between test and train the model might cheat and give direct answers 
            which is not good for the model
        """
        df.index = df.index.str[:15]

        return df[~df.index.duplicated(keep='first')]
    
    def load_and_clean(self) -> pd.DataFrame:
        try:
            if self.processed_path.exists():
                logger.info(f"Found cached dataset at {self.processed_path}. Loading fast...")
                return pd.read_parquet(self.processed_path)

            logger.info("Loading Clinical Phenotype Data...")
            df_clinical = pd.read_csv(self.clinical_path, sep='\t', index_col=0)

            logger.info("Loading Curated Survival Data...")
            df_survival = pd.read_csv(self.survival_path, sep='\t', index_col=0)

            logger.info("Loading STAR-TPM RNA-Seq Data (~60k genes, this will take a moment)...")
            df_tpm = pd.read_csv(self.tpm_path, sep='\t', index_col=0)
            
            # transposing as genes are rows and patients are columns
            df_tpm = df_tpm.T 
            
            gene_columns = df_tpm.columns.tolist()

            logger.info("Standardizing patient barcodes...")
            df_clinical = self._standardize_barcodes(df_clinical)
            df_survival = self._standardize_barcodes(df_survival)
            df_tpm = self._standardize_barcodes(df_tpm)

            logger.info("Performing inner join across all datasets...")
            merged_df = df_clinical.join(df_survival, how='inner', rsuffix='_surv').join(df_tpm, how='inner')

            logger.info("Applying np.log1p transformation to gene expression values...")
            
            #Applying log TRANSFORM
            merged_df[gene_columns] = np.log1p(merged_df[gene_columns].astype(float))

            clinical_cols = [c for c in merged_df.columns if c not in gene_columns]
            rename_dict = {
                c: c.strip().lower().replace(" ", "_").replace("/", "_") 
                for c in clinical_cols
            }
            merged_df.rename(columns=rename_dict, inplace=True)

            logger.info(f"Saving processed dataframe to {self.processed_path}...")
            # Ensure the directory exists
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(self.processed_path)

            logger.info(f"Final Merged Shape: {merged_df.shape}")
            return merged_df

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}. Check if the .part download finished.")
            raise