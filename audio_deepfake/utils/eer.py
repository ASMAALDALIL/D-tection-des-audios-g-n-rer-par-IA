
import numpy as np
from sklearn.metrics import roc_curve, f1_score, accuracy_score


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx       = np.nanargmin(np.abs(fpr - fnr))
    eer           = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer * 100, eer_threshold 
def compute_all_metrics(labels, logits):
    scores = 1 / (1 + np.exp(-np.array(logits)))
    preds = (scores > 0.5).astype(int)

    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds) * 100
    eer, thr = compute_eer(labels, scores)

    return {
        "accuracy": acc,
        "f1": f1,
        "eer": eer,
        "threshold": thr
    }
