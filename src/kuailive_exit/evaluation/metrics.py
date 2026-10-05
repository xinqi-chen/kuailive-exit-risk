"""Point metrics for binary risk predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def binary_metrics(y, probability) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {"n": int(len(y)), "positive_count": int(y.sum()), "positive_rate": float(y.mean()),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
            "ap": float(average_precision_score(y, p)) if y.sum() else float("nan")}
