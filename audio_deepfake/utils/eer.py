
import numpy as np
from sklearn.metrics import roc_curve, f1_score, accuracy_score


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx       = np.nanargmin(np.abs(fpr - fnr))
    eer           = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer * 100, eer_threshold 

