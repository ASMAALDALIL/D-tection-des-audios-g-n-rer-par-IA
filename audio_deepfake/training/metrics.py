
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, roc_auc_score,
    average_precision_score
)

def compute_eer(labels, scores):

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr  
    idx = np.nanargmin(np.abs(fpr - fnr))
    if idx < len(fpr) - 1: 
        far1, far2 = fpr[idx],     fpr[idx + 1]
        frr1, frr2 = fnr[idx],     fnr[idx + 1]
        if (far2 - far1) != (frr1 - frr2):
            t   = (frr1 - far1) / ((far2 - far1) + (frr1 - frr2) + 1e-10)
            eer = far1 + t * (far2 - far1)
        else:
            eer = (fpr[idx] + fnr[idx]) / 2.0
    else:
        eer = (fpr[idx] + fnr[idx]) / 2.0

    threshold = thresholds[idx]
    return float(eer * 100), float(threshold)

def compute_all_metrics(labels, logits, threshold=0.5):
   
    labels  = np.array(labels)
    logits  = np.array(logits)
    scores  = 1 / (1 + np.exp(-logits))     
    preds   = (scores >= threshold).astype(int)
    acc       = accuracy_score(labels, preds)  * 100
    f1        = f1_score(labels, preds, average='binary', zero_division=0) * 100
    precision = precision_score(labels, preds, average='binary', zero_division=0) * 100
    recall    = recall_score(labels, preds, average='binary', zero_division=0)    * 100

   
    auc_roc = roc_auc_score(labels, scores) * 100 if len(np.unique(labels)) > 1 else 0.0

    
    auc_pr  = average_precision_score(labels, scores) * 100 if len(np.unique(labels)) > 1 else 0.0
    eer, eer_threshold = compute_eer(labels, scores)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    far = fp / (fp + tn + 1e-8) * 100
    frr = fn / (fn + tp + 1e-8) * 100

    return {
        'accuracy'      : round(acc,       3),
        'f1'            : round(f1,        3),
        'precision'     : round(precision, 3),
        'recall'        : round(recall,    3),
        'auc_roc'       : round(auc_roc,   3),
        'auc_pr'        : round(auc_pr,    3),
        'eer'           : round(eer,       3),
        'eer_threshold' : round(eer_threshold, 4),
        'far'           : round(far,       3),
        'frr'           : round(frr,       3),
        'tp'            : int(tp),
        'tn'            : int(tn),
        'fp'            : int(fp),
        'fn'            : int(fn),
        'confusion_matrix': cm.tolist()
    }

def aggregate_fold_metrics(fold_metrics_list):
    scalar_keys = [
        'accuracy', 'f1', 'precision', 'recall',
        'auc_roc', 'auc_pr', 'eer', 'far', 'frr'
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [m[key] for m in fold_metrics_list if key in m]
        if values:
            aggregated[f'{key}_mean'] = round(float(np.mean(values)), 3)
            aggregated[f'{key}_std']  = round(float(np.std(values)),  3)
            aggregated[f'{key}_min']  = round(float(np.min(values)),  3)
            aggregated[f'{key}_max']  = round(float(np.max(values)),  3)

    return aggregated


# ─────────────────────────────────────────────
# AFFICHAGE FORMATÉ
# ─────────────────────────────────────────────

def print_metrics(metrics, title="Métriques"):
    """Affiche les métriques de façon lisible dans le terminal."""
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    print(f"  Accuracy   : {metrics.get('accuracy',  0):.2f}%")
    print(f"  F1-Score   : {metrics.get('f1',        0):.2f}%")
    print(f"  Precision  : {metrics.get('precision', 0):.2f}%")
    print(f"  Recall     : {metrics.get('recall',    0):.2f}%")
    print(f"  AUC-ROC    : {metrics.get('auc_roc',   0):.2f}%")
    print(f"  AUC-PR     : {metrics.get('auc_pr',    0):.2f}%")
    print(f"{'─'*50}")
    print(f"  EER        : {metrics.get('eer',  0):.2f}%  (seuil={metrics.get('eer_threshold',0):.4f})")
    print(f"  FAR        : {metrics.get('far',  0):.2f}%")
    print(f"  FRR        : {metrics.get('frr',  0):.2f}%")
    print(f"{'─'*50}")
    cm = metrics.get('confusion_matrix', None)
    if cm:
        print(f"  Matrice de confusion :")
        print(f"           Prédit Real  Prédit Fake")
        print(f"  Vrai Real    {cm[0][0]:>6}       {cm[0][1]:>6}")
        print(f"  Vrai Fake    {cm[1][0]:>6}       {cm[1][1]:>6}")
    print(f"{'─'*50}\n")


def print_aggregated_metrics(agg, title="Résultats Cross-Validation"):
    """Affiche le résumé agrégé sur tous les folds."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    keys = ['accuracy', 'f1', 'precision', 'recall', 'auc_roc', 'eer', 'far', 'frr']
    for key in keys:
        mean = agg.get(f'{key}_mean', 0)
        std  = agg.get(f'{key}_std',  0)
        print(f"  {key:<12} : {mean:.2f}% ± {std:.2f}%")
    print(f"{'='*55}\n")