from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class ZeroShotScanner:
    def __init__(self, model_name, candidate_labels, threshold):
        logging.info(f"initializing nlp scanner Model: {model_name}")
        self.classifier = pipeline('zero-shot-classification', model=model_name)
        self.candidate_labels = candidate_labels
        self.threshold = threshold

    def scan_columns(self, columns: list) -> list:
        """ Function to identify the columns which might induce bias using ZeroShotClassification """
        flaged_columns = []
        logger.info("Starting zero shot classification to identify columns that might induce bias")

        for col in columns:
            sequence_to_classify = "This columns contains data about persons " + col.replace(" ", "_")
            res = self.classifier(sequence_to_classify, self.candidate_labels)
            top_label = res['labels'][0]
            top_score = res['scores'][0]

            if top_label != "financial transaction" and top_score >= self.threshold:
                logger.warning(f"flagged column: {col} with confidence score: {top_score}")
                flaged_columns.append(col)
            else:
                logger.info(f"Passed: {col}")
        
        return flaged_columns
