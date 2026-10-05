"""Run a small raw-data-to-H30 pipeline using only this repository's code.

The smoke cohort is selected deterministically from the released user list.
Only core interaction and room metadata are used; embeddings and event-log
tables are deliberately omitted for this 20-user integration check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from kuailive_exit.cleaning.visits import build_visits
from kuailive_exit.config import load_config
from kuailive_exit.data.raw import INTERACTION_COLUMNS, ROOM_COLUMNS, normalize_core
from kuailive_exit.evaluation.metrics import binary_metrics
from kuailive_exit.features.browsing import (
    ALL, ROOM,
    attach_visit_actions,
    local_time_features,
    personal_features,
    r2_room_visits,
    room_features_for_room,
)
from kuailive_exit.features.groups import MODEL_FEATURES, enrich_points
from kuailive_exit.features.history import add_visit_history
from kuailive_exit.labels.periods import h30_from_visits, make_periods
from kuailive_exit.models.lgbm import fit_classifier
from kuailive_exit.models.survival import forecast_h30


EDGES = [0, 5, 10, 15, 20, 25, 35, 50, 75, 120, 225, 345, 600]
EXTRA_INTERACTION_COLUMNS = []
FEATURES = MODEL_FEATURES["M1_browse"]
ROW_KEYS = ["user_id", "visit_id", "prediction_time_ms", "interval_index", "offset_seconds"]
COMPARE_COLUMNS = ROW_KEYS + ["label"] + FEATURES
REPO_ROOT = Path(__file__).resolve().parents[1]


def digest_users(users: list[str]) -> str:
    return hashlib.sha256("\n".join(users).encode()).hexdigest()


def choose_users(raw_dir: Path, candidate_limit: int = 1024) -> list[str]:
    with (raw_dir / "user_id_set.csv").open(newline="") as stream:
        ids = {row["user_id"].strip() for row in csv.DictReader(stream) if row.get("user_id", "").strip()}
    return sorted(ids, key=lambda user: (hashlib.sha256(user.encode()).digest(), user))[:candidate_limit]


def extract_candidate_interactions(raw_dir: Path, candidates: list[str], output: Path) -> dict:
    columns = INTERACTION_COLUMNS + EXTRA_INTERACTION_COLUMNS
    types = {name: pa.string() for name in columns}
    reader = pacsv.open_csv(
        raw_dir / "live_interaction.csv",
        read_options=pacsv.ReadOptions(block_size=16 << 20, use_threads=False),
        convert_options=pacsv.ConvertOptions(
            include_columns=columns, column_types=types, strings_can_be_null=True
        ),
    )
    candidate_array = pa.array(candidates)
    writer = None
    row_number = total = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            mask = pc.is_in(table["user_id"], value_set=candidate_array)
            selected = table.filter(mask)
            if selected.num_rows:
                original = pc.filter(
                    pa.array(np.arange(row_number, row_number + batch.num_rows, dtype=np.int64)), mask
                )
                selected = selected.append_column("raw_row_number", original)
                if writer is None:
                    writer = pq.ParquetWriter(output, selected.schema, compression="zstd")
                writer.write_table(selected)
                total += selected.num_rows
            row_number += batch.num_rows
            if row_number % 2_000_000 < batch.num_rows:
                print(f"scanned_interaction_rows={row_number:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("none of the deterministic smoke candidates occur in live_interaction.csv")
    return {"source_rows_scanned": row_number, "candidate_rows": total}


def eligible_smoke_users(candidate_parquet: Path, candidate_order: list[str]) -> tuple[list[str], list[str]]:
    table = pq.read_table(candidate_parquet, columns=["user_id", "live_play_start_timestamp"])
    frame = table.to_pandas()
    frame["user_id"] = frame.user_id.astype(str)
    start = pd.to_datetime(frame.live_play_start_timestamp, format="mixed", utc=True, errors="coerce")
    frame["start"] = start
    train_end = pd.Timestamp("2026-03-21T16:00:00Z")
    validation_end = pd.Timestamp("2026-03-26T16:00:00Z")
    test_end = pd.Timestamp("2026-03-31T16:00:00Z")
    eligible: list[str] = []
    for user in candidate_order:
        times = frame.loc[frame.user_id.eq(user), "start"]
        in_validation = times.ge(train_end) & times.lt(validation_end)
        in_test = times.ge(validation_end) & times.lt(test_end)
        if times.lt(train_end).any() and in_validation.any() and in_test.any():
            eligible.append(user)
        if len(eligible) >= 20:
            break
    if len(eligible) < 20:
        raise RuntimeError(
            f"only {len(eligible)} candidates have train, validation and test activity; "
            "increase candidate_limit and rerun"
        )
    return eligible[:16], eligible[16:20]


def write_selected_raw(candidate_parquet: Path, users: list[str], processed: Path) -> tuple[list[str], int]:
    source = ds.dataset(candidate_parquet, format="parquet")
    selected = source.to_table(filter=ds.field("user_id").isin(users))
    selected_path = processed / "selected_interactions.parquet"
    pq.write_table(selected, selected_path, compression="zstd")
    rooms = sorted({str(room) for room in selected["live_id"].to_pylist() if room is not None})
    (processed / "extraction_audit.json").write_text(json.dumps({
        "data_revision": "417923d442a793b3e92cc7e079dd4b73fe706c30",
        "users": len(users), "selected_interaction_rows": selected.num_rows,
        "target_rooms": len(rooms),
    }, indent=2))
    return rooms, selected.num_rows


def extract_room_metadata(raw_dir: Path, room_ids: list[str], processed: Path) -> int:
    target = pa.array([int(room) for room in room_ids], type=pa.int64())
    reader = pq.ParquetFile(raw_dir / "live_room_meta.parquet")
    writer = None
    total = selected_rows = 0
    output = processed / "selected_rooms.parquet"
    for batch in reader.iter_batches(batch_size=65536, columns=ROOM_COLUMNS, use_threads=False):
        table = pa.Table.from_batches([batch])
        chosen = table.filter(pc.is_in(table["live_id"], value_set=target))
        if chosen.num_rows:
            if writer is None:
                writer = pq.ParquetWriter(output, chosen.schema, compression="zstd")
            writer.write_table(chosen)
            selected_rows += chosen.num_rows
        total += batch.num_rows
    if writer is not None:
        writer.close()
    if not selected_rows:
        raise RuntimeError("no selected smoke rooms were present in live_room_meta.parquet")
    return selected_rows


def add_features(rows: pd.DataFrame, visits: pd.DataFrame,
                 room_intervals: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    result = enrich_points(rows.reset_index(drop=True))
    result = add_visit_history(result, visits)
    personal = personal_features(result, visits)
    result[personal.columns] = personal
    local = local_time_features(result)
    result[local.columns] = local
    room_pieces = []
    for room, group in result.groupby("room_id", sort=False):
        room_data = room_intervals[room_intervals.room_id.astype(str).eq(str(room))]
        room_values = room_features_for_room(group, room_data)
        room_pieces.append(room_values)
    room_features = pd.concat(room_pieces).sort_index() if room_pieces else pd.DataFrame(index=result.index)
    result[room_features.columns] = room_features
    return result


def compare_formal_rows(generated: pd.DataFrame, formal_root: Path,
                        users: list[str], query_kind: str) -> dict:
    """Exact-value comparison on matching A-cohort rows and shared features."""
    paths = sorted(formal_root.glob("features_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no formal feature shards under {formal_root}")
    columns = ["cohort", "query_kind", *COMPARE_COLUMNS]
    pieces = []
    user_set = set(map(str, users))
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        mask = frame.cohort.eq("A") & frame.query_kind.eq(query_kind) & frame.user_id.astype(str).isin(user_set)
        if mask.any():
            pieces.append(frame.loc[mask, COMPARE_COLUMNS].copy())
    formal = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=COMPARE_COLUMNS)
    left = generated[generated.user_id.astype(str).isin(user_set)][COMPARE_COLUMNS].copy()
    for frame in (left, formal):
        frame["user_id"] = frame.user_id.astype(str)
    key = ["user_id", "visit_id", "prediction_time_ms", "offset_seconds"]
    if left.duplicated(key).any() or formal.duplicated(key).any():
        raise ValueError(f"duplicate {query_kind} comparison row keys")
    merged = left.merge(formal, on=key, how="outer", suffixes=("_new", "_formal"), indicator=True)
    matched = merged[merged._merge.eq("both")]
    mismatch_counts: dict[str, int] = {}
    max_absolute_difference = 0.0
    for name in ["label", *[c for c in FEATURES if c not in ROOM]]:
        a = matched[f"{name}_new"].to_numpy(dtype=np.float32)
        b = matched[f"{name}_formal"].to_numpy(dtype=np.float32)
        both_null = pd.isna(a) & pd.isna(b)
        equal = (a == b) | both_null
        unequal = ~equal
        count = int(unequal.sum())
        if count:
            mismatch_counts[name] = count
            try:
                af, bf = a.astype(float), b.astype(float)
                finite = np.isfinite(af) & np.isfinite(bf)
                if finite.any():
                    max_absolute_difference = max(
                        max_absolute_difference, float(np.max(np.abs(af[finite] - bf[finite])))
                    )
            except (TypeError, ValueError):
                pass
    a_users = set(formal.user_id.astype(str))
    overlap_users = user_set & a_users
    missing_formal = int(merged._merge.eq("left_only").sum())
    missing_generated = int(merged._merge.eq("right_only").sum())
    return {
        "query_kind": query_kind,
        "overlapping_A_users": len(overlap_users),
        "generated_rows": len(left), "formal_A_rows_for_smoke_users": len(formal),
        "matched_rows": len(matched), "missing_formal_rows": missing_formal,
        "missing_generated_rows": missing_generated,
        "numeric_comparison": "exact after common float32 storage conversion",
        "compared_value_columns": ["label", *[c for c in FEATURES if c not in ROOM]],
        "excluded_columns": ROOM,
        "exclusion_reason": "smoke audience covers 20 users; formal audience covers all released sampled users",
        "compared_cells": len(matched) * (len(FEATURES) - len(ROOM) + 1),
        "mismatched_cells": int(sum(mismatch_counts.values())),
        "mismatches_by_column": mismatch_counts,
        "max_absolute_numeric_difference": max_absolute_difference,
        "exact_match": bool(not missing_formal and not missing_generated and not mismatch_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-limit", type=int, default=1024)
    parser.add_argument("--output", type=Path, help="optional output directory under processed data")
    parser.add_argument(
        "--formal-feature-root", type=Path,
        default=None,
        help="formal feature shards used for exact A-cohort row comparison",
    )
    args = parser.parse_args()
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    for name in ("user_id_set.csv", "live_interaction.csv", "live_room_meta.parquet"):
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(f"required raw input is missing: {raw_dir / name}")
    run_id = datetime.now(timezone.utc).strftime("raw-smoke-%Y%m%dT%H%M%S-%fZ")
    root = args.output or (cfg["data"]["processed_dir"] / run_id)
    if not root.is_absolute():
        root = cfg["data"]["processed_dir"] / root
    root.mkdir(parents=True, exist_ok=False)
    processed = root / "normalized"
    processed.mkdir()
    cfg["data"]["processed_dir"] = processed
    start_time = time.time()

    candidates = choose_users(raw_dir, args.candidate_limit)
    candidate_path = root / "candidate_interactions.parquet"
    extraction_scan = extract_candidate_interactions(raw_dir, candidates, candidate_path)
    try:
        train_users, test_users = eligible_smoke_users(candidate_path, candidates)
        users = train_users + test_users
        room_ids, selected_rows = write_selected_raw(candidate_path, users, processed)
    finally:
        candidate_path.unlink(missing_ok=True)
    room_count = extract_room_metadata(raw_dir, room_ids, processed)

    sessions, r01 = normalize_core(cfg)
    (processed / "session_audit.json").write_text(json.dumps(r01, indent=2, default=str))
    visits, r24 = build_visits(sessions, cfg, g_ms=1000, delta_minus_ms=5000, delta_plus_ms=60000)
    raw_sessions = pq.read_table(processed / "selected_interactions.parquet").to_pandas()
    visits = attach_visit_actions(visits, raw_sessions)
    visits.to_parquet(processed / "visits.parquet", index=False)
    (processed / "visit_audit.json").write_text(json.dumps(r24, indent=2, default=str))
    raw_sessions = pq.read_table(processed / "selected_interactions.parquet").to_pandas()
    room_intervals = r2_room_visits(raw_sessions)

    periods = make_periods(visits, EDGES, cfg, target="E2-V")
    periods = periods[periods.label_status.eq("evaluable") & periods.label.notna()].copy()
    points = add_features(periods, visits, room_intervals)
    points.to_parquet(root / "smoke_interval_features.parquet", index=False)
    train_set, test_set = set(train_users), set(test_users)
    train = points[points.user_id.isin(train_set) & points.split.eq("train")]
    validation = points[points.user_id.isin(train_set) & points.split.eq("validation")]
    if train.empty or validation.empty or train.label.nunique() < 2 or validation.label.nunique() < 2:
        raise RuntimeError(
            f"smoke train/validation labels insufficient: train={len(train)} validation={len(validation)}"
        )
    params = {"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20, "feature_fraction": 1.0}
    model = fit_classifier(
        train[FEATURES].to_numpy(dtype=np.float32), train.label.to_numpy(dtype=np.int8),
        validation[FEATURES].to_numpy(dtype=np.float32), validation.label.to_numpy(dtype=np.int8),
        params, n_estimators=100, early_stopping_rounds=15, seed=20261001, threads=2,
    )
    test_visits = visits[visits.user_id.isin(test_set)]
    h30 = h30_from_visits(test_visits, cfg, "E2-V")
    h30 = h30[h30.label_status.eq("evaluable") & h30.label.notna()].copy()
    h30_features = add_features(h30, test_visits, room_intervals)
    if h30_features.empty:
        raise RuntimeError("smoke cohort produced no evaluable test H30 points")
    h30_features.to_parquet(root / "smoke_h30_features.parquet", index=False)
    feature_consistency = {"status": "not_requested"}
    if args.formal_feature_root is not None:
        feature_consistency = {
            "interval": compare_formal_rows(points, args.formal_feature_root, users, "interval"),
            "H30": compare_formal_rows(h30_features, args.formal_feature_root, test_users, "H30"),
        }
    probability = forecast_h30(model, h30_features, EDGES, FEATURES, "M1_browse")
    primary = h30_features.offset_seconds.le(570).to_numpy()
    metrics = binary_metrics(h30_features.label.to_numpy(dtype=np.int8)[primary], probability[primary])
    predictions = h30_features[["visit_id", "offset_seconds", "label"]].copy()
    predictions["prediction"] = probability
    predictions.to_parquet(root / "smoke_predictions.parquet", index=False)
    result = {
        "status": "completed", "data_revision": cfg["data"]["revision"],
        "users": len(users), "train_users": len(train_users), "test_users": len(test_users),
        "cohort_user_ids_sha256": digest_users(users),
        "raw_interaction_rows_scanned": extraction_scan["source_rows_scanned"],
        "candidate_interaction_rows": extraction_scan["candidate_rows"],
        "selected_interaction_rows": selected_rows, "selected_rooms": len(room_ids),
        "selected_room_metadata_rows": room_count, "visits": len(visits),
        "train_person_period_rows": len(train), "validation_person_period_rows": len(validation),
        "test_h30_points": int(primary.sum()), "test_h30_all_points": len(h30_features), "test_h30_metrics": metrics,
        "feature_consistency": feature_consistency,
        "model": "M1_browse", "best_iteration": int(model.best_iteration_),
        "content_embeddings_used": False,
        "elapsed_seconds": time.time() - start_time,
        "raw_inputs": ["user_id_set.csv", "live_interaction.csv", "live_room_meta.parquet"],
        "pipeline_code": "new-repository modules only; no legacy feature shards or model artifacts read",
    }
    (root / "smoke_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
