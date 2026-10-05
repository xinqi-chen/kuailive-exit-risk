"""Strictly past as-of joins for timestamped room embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_COLUMNS = [f"emb_{index:03d}" for index in range(128)]
CONTENT_META = ["has_past_embedding", "embedding_age_seconds", "available_age_seconds"]


def join_content_bucket(points: pd.DataFrame, files: list[Path]) -> pd.DataFrame:
    """Join each point to the latest unique segment with timestamp strictly < t."""
    import polars as pl

    n = len(points)
    matched_end = np.full(n, -1, dtype=np.int64)
    vectors = np.zeros((n, 128), dtype=np.float32)
    files = [path for path in files if path.exists()]
    if files:
        segments = pl.read_parquet(files).with_columns(pl.col("segment_end_ms").alias("available_ms"))
        duplicates = segments.group_by(["room_id", "available_ms"]).len().filter(pl.col("len") > 1)
        segments = segments.join(duplicates.select(["room_id", "available_ms"]),
                                 on=["room_id", "available_ms"], how="anti")
        left = pl.DataFrame({"position": np.arange(n, dtype=np.int64),
                             "room_id": points.room_id.astype(str).tolist(),
                             "asof_before_ms": (points.prediction_time_ms.astype("int64") - 1).tolist()})
        joined = left.sort("asof_before_ms").join_asof(
            segments.sort("available_ms"), left_on="asof_before_ms", right_on="available_ms",
            by="room_id", strategy="backward", check_sortedness=False)
        matched = joined.filter(pl.col("segment_end_ms").is_not_null())
        if not matched.is_empty():
            positions = matched["position"].to_numpy().astype(np.int64)
            matched_end[positions] = matched["segment_end_ms"].to_numpy().astype(np.int64)
            vectors[positions] = np.asarray(matched["embedding"].to_list(), dtype=np.float32)
    time_ms = points.prediction_time_ms.to_numpy(dtype=np.int64)
    if np.any(matched_end >= time_ms):
        raise AssertionError("content feature is not strictly prior to prediction time")
    points["has_past_embedding"] = (matched_end >= 0).astype(np.int8)
    age = np.where(matched_end >= 0, (time_ms - matched_end) / 1000, 0).astype(np.float32)
    points["embedding_age_seconds"] = age
    points["available_age_seconds"] = age
    embeddings = pd.DataFrame(vectors, columns=EMBEDDING_COLUMNS, index=points.index)
    return pd.concat([points.drop(columns=EMBEDDING_COLUMNS, errors="ignore"), embeddings], axis=1)
