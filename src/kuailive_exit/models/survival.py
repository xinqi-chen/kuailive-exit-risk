"""Convert piecewise interval hazards into arbitrary-window probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings


def forecast_h30(model, points: pd.DataFrame, edges: list[int], feature_names: list[str],
                 model_name: str, batch_size: int = 25_000, frozen_columns: list[str] | None = None) -> np.ndarray:
    """Predict the next 30 seconds using the hazard at each interval's start."""
    t = points.offset_seconds.to_numpy(dtype=float)
    cursor = t.copy()
    remaining = np.full(len(t), 30.0)
    cumulative = np.zeros(len(t), dtype=float)
    edge = np.asarray(edges, dtype=float)
    while np.any(remaining > 1e-9):
        active = np.flatnonzero(remaining > 1e-9)
        at = cursor[active]
        index = np.clip(np.searchsorted(edge, at, side="right") - 1, 0, len(edge) - 2)
        stop = edge[index + 1].copy()
        beyond = at >= 600
        stop[beyond] = at[beyond] + remaining[active][beyond]
        span = np.maximum(1e-6, np.minimum(remaining[active], stop - at))
        bin_start = edge[index]
        width = edge[index + 1] - edge[index]
        delta = bin_start - t[active]
        for batch_start in range(0, len(active), batch_size):
            sl = slice(batch_start, batch_start + batch_size)
            ids = active[sl]
            d = delta[sl]
            view = points.iloc[ids].copy()
            view["elapsed_watch_seconds"] = bin_start[sl]
            view["room_elapsed_seconds"] = np.maximum(
                0, points.room_elapsed_seconds.iloc[ids].to_numpy() + d)
            if "browse_active_segment_seconds" in feature_names:
                view["browse_active_segment_seconds"] = np.maximum(
                    0, points.browse_active_segment_seconds.iloc[ids].to_numpy() + d)
                age = points.browse_last_start_age_seconds.iloc[ids].to_numpy()
                view["browse_last_start_age_seconds"] = np.where(age > 0, np.maximum(0, age + d), 0)
                local = pd.to_datetime(points.prediction_time_ms.iloc[ids].to_numpy(dtype=np.int64) +
                                       (d * 1000).astype(np.int64), unit="ms", utc=True).tz_convert("Asia/Shanghai")
                view["browse_local_hour"] = local.hour
                view["browse_local_weekday"] = local.weekday
            if "embedding_age_seconds" in feature_names:
                view["embedding_age_seconds"] = np.where(view.has_past_embedding.to_numpy() > 0,
                    np.maximum(0, points.embedding_age_seconds.iloc[ids].to_numpy() + d), 0)
                view["available_age_seconds"] = view["embedding_age_seconds"]
            # A shuffled block must stay shuffled throughout hazard conversion.
            for column in frozen_columns or []:
                view[column] = points[column].iloc[ids].to_numpy()
            x = view[feature_names].to_numpy(dtype=np.float32, copy=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                hazard = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
            cumulative[ids] += -np.log1p(-hazard) * span[sl] / width[sl]
        cursor[active] += span
        remaining[active] -= span
    return -np.expm1(-cumulative)
