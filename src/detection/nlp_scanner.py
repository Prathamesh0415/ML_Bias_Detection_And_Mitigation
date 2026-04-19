import re
import logging

logger = logging.getLogger(__name__)

class DemographicScanner:
    def __init__(self):
        # Updated regex patterns to catch standard demographic and financial targets
        self.sensitive_patterns = [
            r'race', r'ethnicity', r'gender', r'sex', 
            r'age', r'birth', r'marri', r'marital', # 'marri' catches 'marriage'
            r'education', r'income', r'insurance', r'dependents', r'family'
        ]
        
    def scan_columns(self, columns: list) -> list:
        logger.info("Scanning metadata for sensitive demographic variables...")
        sensitive_cols = set()
        
        for col in columns:
            col_lower = col.lower()
            for pattern in self.sensitive_patterns:
                # Search for the pattern anywhere in the column name
                if re.search(pattern, col_lower):
                    logger.warning(f"Flagged demographic column: {col}")
                    sensitive_cols.add(col)
                    break # Move to the next column once a match is found
                    
        return list(sensitive_cols)