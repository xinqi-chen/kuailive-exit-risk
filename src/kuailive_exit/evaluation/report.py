"""Aggregate experiment metrics with explicit, stable summary definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def landmark_auc_gain(metrics: pd.DataFrame, candidate: str, baseline: str) -> dict:
    """Summarize per-landmark AUC gain for landmarks strictly greater than 30s."""
    required = {"model", "elapsed_seconds", "auc", "n"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"metrics must contain {sorted(required)}")
    pivot = metrics.pivot(index="elapsed_seconds", columns="model", values=["auc", "n"])
    selected = pivot.loc[pivot.index > 30]
    gain = selected[("auc", candidate)] - selected[("auc", baseline)]
    weights = selected[("n", baseline)].astype(float)
    valid = gain.notna() & weights.gt(0)
    return {"candidate": candidate, "baseline": baseline,
            "landmarks": int(valid.sum()),
            "mean_auc_gain": float(gain[valid].mean()) if valid.any() else float("nan"),
            "sample_weighted_mean_auc_gain": float(np.average(gain[valid], weights=weights[valid]))
            if valid.any() else float("nan"),
            "maximum_auc_gain_secondary": float(gain[valid].max()) if valid.any() else float("nan")}
