"""Frozen feature group membership (development stage D8, labelled v1.7 in configs)."""

from __future__ import annotations

from .browsing import ACTIONS, ALL
from .content import CONTENT_META, EMBEDDING_COLUMNS
from .events import EVENT_FEATURES
import json
from pathlib import Path

NUMERIC = ["elapsed_watch_seconds", "room_elapsed_seconds", "room_start_missing",
           "is_auto_play", "is_follow_enter", "user_recent20_mean_duration_seconds",
           "user_recent20_median_duration_seconds", "user_short_watch_fraction",
           "user_count_24h", "user_count_7d", "user_last_end_age_seconds",
           "author_count_7d", "author_mean_duration_seconds", "author_last_end_age_seconds",
           "user_broadcast_fraction", "user_survival_30s", "user_survival_60s"]
_CATEGORY_VOCAB_PATH = Path(__file__).resolve().parents[3] / "configs" / "entry_category_vocab.json"
if _CATEGORY_VOCAB_PATH.exists():
    _vocab = json.loads(_CATEGORY_VOCAB_PATH.read_text())
    ENTRY_CATEGORY_FEATURES = [column for columns in _vocab["feature_columns"].values() for column in columns]
else:
    ENTRY_CATEGORY_FEATURES = []
STATE = NUMERIC[:5] + ENTRY_CATEGORY_FEATURES
MODEL_FEATURES = {
    "M1": STATE + NUMERIC[5:],
    "M1_browse": STATE + NUMERIC[5:] + ALL,
    "M_all": STATE + NUMERIC[5:] + ALL + EVENT_FEATURES + CONTENT_META + EMBEDDING_COLUMNS,
}


def enrich_points(points):
    """Add the v1.7 state fields to interval-start point rows."""
    import pandas as pd

    result = points.copy()
    result["elapsed_watch_seconds"] = result["offset_seconds"].astype("float32")
    elapsed = (pd.to_numeric(result.prediction_time_ms, errors="coerce") -
               pd.to_numeric(result.room_start_ms, errors="coerce")) / 1000
    result["room_start_missing"] = elapsed.isna().astype("int8")
    result["room_elapsed_seconds"] = elapsed.clip(lower=0).fillna(0).astype("float32")
    for name in ("is_auto_play", "is_follow_enter"):
        result[name] = pd.to_numeric(result[name], errors="coerce").astype("float32")
    if ENTRY_CATEGORY_FEATURES:
        for field, columns in _vocab["feature_columns"].items():
            kept = _vocab["fields"][field]["kept_categories"]
            sentinel = _vocab["fields"][field]["internal_other_sentinel"]
            values = result[field].astype("string").str.strip()
            values = values.where(values.isin(kept), sentinel).fillna(sentinel)
            for category, column in zip([*kept, sentinel], columns):
                result[column] = values.eq(category).astype("int8")
    return result
