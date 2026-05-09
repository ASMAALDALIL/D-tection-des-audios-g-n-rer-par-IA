
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pred_bin = (y_pred > 0.5).astype(int)
    return {
        "acc": accuracy_score(y_true, pred_bin),
        "f1":  f1_score(y_true, pred_bin, zero_division=0),
    }


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)   
