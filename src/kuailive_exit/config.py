"""Configuration loading with all data paths rooted in the user's data directory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def utc_ms(value: str) -> int:
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC boundary must include timezone")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def load_config(path: str | Path | None = None) -> dict:
    """Load protocol settings; KUAILIVE_DATA_DIR points at the raw data directory."""
    base = PACKAGE_ROOT
    data_doc = yaml.safe_load((base / "configs/data.yaml").read_text())
    cohort_doc = yaml.safe_load((base / "configs/cohorts.yaml").read_text())
    protocol_doc = yaml.safe_load((base / "configs/protocol.yaml").read_text())
    raw_dir = Path(os.environ.get("KUAILIVE_DATA_DIR", base / "data/raw")).expanduser().resolve()
    processed_dir = Path(os.environ.get("KUAILIVE_PROCESSED_DIR", base / "data/processed")).expanduser().resolve()
    run_id = os.environ.get("KUAILIVE_RUN_ID", "formal-v1.1.0")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("KUAILIVE_RUN_ID must be a directory name")
    return {
        "data": {"repo_id": data_doc["repo_id"], "revision": data_doc["revision"],
                 "raw_dir": raw_dir, "processed_dir": processed_dir,
                 "run_dir": processed_dir / run_id, "run_id": run_id,
                 "result_dir": Path(os.environ.get("KUAILIVE_RESULTS_DIR", base / "results" / run_id)).expanduser().resolve(),
                 "cleaning_rule_version": "cross_day_stitch_v1"},
        "sampling": {"n_users": cohort_doc["train_users"]},
        "observation": {"verified_start_utc": data_doc["observation_start_utc"],
                        "verified_end_utc_exclusive": data_doc["observation_end_utc_exclusive"],
                        "duration_tolerance_ms": 5000, "overlap_tolerance_ms": 1000},
        "label": {"horizon_seconds": 30, "observation_buffer_seconds": protocol_doc["observation_buffer_seconds"]},
        "landmarks": protocol_doc["landmarks"],
        "split": {"train_end_utc_exclusive": data_doc["train_end_utc_exclusive"],
                  "validation_end_utc_exclusive": data_doc["validation_end_utc_exclusive"],
                  "test_end_utc_exclusive": data_doc["test_end_utc_exclusive"]},
        "cohorts": cohort_doc,
        "protocol": protocol_doc,
    }
