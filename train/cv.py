# train/cv.py
from sklearn.model_selection import StratifiedKFold
from typing import List, Iterator, Tuple
import numpy as np


def get_folds(
    paths: List[str],
    labels: List[float],
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Yields (fold_idx, train_indices, val_indices).
    Stratifié sur les labels pour garder l'équilibre bonafide/spoof.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(paths, labels)):
        yield i, tr, va
