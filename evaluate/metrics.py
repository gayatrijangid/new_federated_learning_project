import numpy as np
from typing import List, Dict

class Evaluator:
    """
    Two SEPARATE kinds of metrics are tracked, and must never be confused:

    1. DETECTION metrics (this class's TP/FP/FN/TN):
       did the z-score correctly flag the ACTUAL malicious hospitals?
       -> precision/recall/F1 here are about catching attackers.

    2. CLASSIFICATION metrics (compute_classification_metrics, static):
       how good is the resulting global model at predicting diabetes?
       -> precision/recall/F1 here are about the diabetes outcome.
    """
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update_detection_stats(self, actual_malicious: List[int], flagged_indices: List[int], total_clients: int):
        actual_set = set(actual_malicious)
        flagged_set = set(flagged_indices)

        for i in range(total_clients):
            is_mal = i in actual_set
            is_flag = i in flagged_set
            if is_mal and is_flag:
                self.tp += 1
            elif not is_mal and is_flag:
                self.fp += 1
            elif is_mal and not is_flag:
                self.fn += 1
            else:
                self.tn += 1

    def get_detection_metrics(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        det_acc = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn) if (self.tp + self.fp + self.fn + self.tn) > 0 else 0.0
        return {
            "detection_precision": round(precision * 100, 2),
            "detection_recall": round(recall * 100, 2),
            "detection_f1": round(f1 * 100, 2),
            "detection_accuracy": round(det_acc * 100, 2),
        }

    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return round(float(np.mean(y_true == y_pred) * 100), 2)

    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Standard classification P/R/F1 on the diabetes-outcome task itself
        (positive class = diabetic / Outcome==1). Same TP/FP/FN/TN definitions
        as the detection metrics above, but applied to a completely different
        question -- deliberately named differently so the two are never mixed up."""
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
        }
