import re
import logging

logger = logging.getLogger(__name__)

class ClinicalDemographicScanner:
    def __init__(self):
        # A robust list of regex patterns that catch standard demographic terminology
        self.sensitive_patterns = [
            r'race', r'ethnicity', r'gender', r'sex', 
            r'age', r'birth', r'menopause', r'ancestry',
            r'marital', r'education', r'income', r'insurance'
        ]
        
    def scan_columns(self, columns: list) -> list:
        logger.info("Scanning clinical metadata for demographic variables...")
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