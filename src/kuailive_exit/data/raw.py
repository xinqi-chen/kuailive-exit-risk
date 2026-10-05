from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .sampling import cohort_hash, selected_users
from .schemas import timestamp_ms


INTERACTION_COLUMNS = [
    "user_id", "author_id", "live_id", "live_play_start_timestamp",
    "live_play_end_timestamp", "p_date", "live_source_category",
    "enter_live_action", "is_auto_play", "is_follow_enter", "is_follow_leave", "play_duration",
    "send_gift_cnt", "like_cnt", "comment_cnt",
]
ROOM_COLUMNS = ["live_id", "author_id", "start_timestamp", "end_timestamp", "live_duration"]


def extract_core(cfg: dict) -> dict:
    raw, processed = cfg["data"]["raw_dir"], cfg["data"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    users = selected_users(raw / "user_id_set.csv", cfg["sampling"]["n_users"])
    (processed / "selected_users.txt").write_text("\n".join(users) + "\n")
    user_values = pa.array(users)
    csv_path = raw / "live_interaction.csv"
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=False),
        convert_options=pacsv.ConvertOptions(
            include_columns=INTERACTION_COLUMNS,
            column_types={c: pa.string() for c in INTERACTION_COLUMNS},
        ),
    )
    writer = None
    seen = retained = 0
    dates = collections.Counter()
    for batch in reader:
        table = pa.Table.from_batches([batch])
        mask = pc.is_in(table["user_id"], value_set=user_values)
        chosen = table.filter(mask)
        if chosen.num_rows:
            indices = pc.filter(pa.array(np.arange(seen, seen + batch.num_rows, dtype=np.int64)), mask)
            chosen = chosen.append_column("raw_row_number", indices)
            if writer is None:
                writer = pq.ParquetWriter(processed / "selected_interactions.parquet.part", chosen.schema, compression="zstd")
            writer.write_table(chosen)
            retained += chosen.num_rows
            dates.update(chosen["p_date"].to_pylist())
        seen += batch.num_rows
    if writer is None:
        raise ValueError("selected users have no interaction records")
    writer.close()
    (processed / "selected_interactions.parquet.part").replace(processed / "selected_interactions.parquet")

    room_ids = set()
    for batch in pq.ParquetFile(processed / "selected_interactions.parquet").iter_batches(columns=["live_id"]):
        room_ids.update(batch.column(0).to_pylist())
    ids_int = pa.array([int(x) for x in room_ids], type=pa.int64())
    room_writer = None
    room_seen = room_retained = 0
    room_file = pq.ParquetFile(raw / "live_room_meta.parquet")
    for batch in room_file.iter_batches(batch_size=65536, columns=ROOM_COLUMNS, use_threads=False):
        table = pa.Table.from_batches([batch])
        chosen = table.filter(pc.is_in(table["live_id"], value_set=ids_int))
        if chosen.num_rows:
            if room_writer is None:
                room_writer = pq.ParquetWriter(processed / "selected_rooms.parquet.part", chosen.schema, compression="zstd")
            room_writer.write_table(chosen)
            room_retained += chosen.num_rows
        room_seen += batch.num_rows
    if room_writer is None:
        raise ValueError("no target rooms found in metadata")
    room_writer.close()
    (processed / "selected_rooms.parquet.part").replace(processed / "selected_rooms.parquet")
    result = {
        "data_revision": cfg["data"]["revision"],
        "cohort_hash": cohort_hash(users), "users": len(users), "interaction_rows_total": seen,
        "interaction_rows_selected": retained, "partition_dates_selected": dict(sorted(dates.items())),
        "target_rooms": len(room_ids), "room_meta_rows_total": room_seen,
        "room_meta_rows_selected": room_retained,
        "interaction_schema": str(reader.schema), "room_schema": str(room_file.schema_arrow),
    }
    (processed / "extraction_audit.json").write_text(json.dumps(result, indent=2))
    return result


def normalize_core(cfg: dict) -> tuple[pd.DataFrame, dict]:
    processed = cfg["data"]["processed_dir"]
    extraction = json.loads((processed / "extraction_audit.json").read_text())
    if extraction.get("data_revision") != cfg["data"]["revision"]:
        raise ValueError("processed interactions were extracted from a different or unrecorded data revision")
    sessions = pq.read_table(processed / "selected_interactions.parquet").to_pandas()
    rooms = pq.read_table(processed / "selected_rooms.parquet").to_pandas()
    sessions = sessions.rename(columns={
        "live_id": "room_id", "live_play_start_timestamp": "session_start_raw",
        "live_play_end_timestamp": "session_end_raw",
    })
    rooms = rooms.rename(columns={
        "live_id": "room_id", "start_timestamp": "room_start_raw",
        "end_timestamp": "room_end_raw",
    })
    for col in ("user_id", "author_id", "room_id"):
        sessions[col] = sessions[col].astype("string")
    for col in ("author_id", "room_id"):
        rooms[col] = rooms[col].astype("string")
    sessions["session_start_ms"] = timestamp_ms(sessions["session_start_raw"], name="session_start")
    sessions["session_end_ms"] = timestamp_ms(sessions["session_end_raw"], name="session_end")
    rooms["room_start_ms"] = timestamp_ms(rooms["room_start_raw"], name="room_start")
    rooms["room_end_ms"] = timestamp_ms(rooms["room_end_raw"], name="room_end")
    sessions["reported_duration_ms"] = pd.to_numeric(sessions["play_duration"], errors="coerce")
    sessions["session_id"] = sessions["raw_row_number"].map(
        lambda x: hashlib.sha256(f"{cfg['data']['revision']}:live_interaction.csv:{x}".encode()).hexdigest()[:24]
    )
    # KuaiLive partitions can truncate an otherwise continuous viewing record
    # at UTC+8 midnight (23:59:59.999 -> 00:00:00.000). Only stitch when the
    # immediately following interaction for that user starts in the same room
    # and author at the next millisecond. We preserve entry state from the first
    # row and exit state from the continuation row.
    ordered = sessions.sort_values(["user_id", "session_start_ms", "raw_row_number"], kind="stable").copy()
    ordered["_next_user"] = ordered["user_id"].shift(-1)
    ordered["_next_room"] = ordered["room_id"].shift(-1)
    ordered["_next_author"] = ordered["author_id"].shift(-1)
    ordered["_next_start"] = ordered["session_start_ms"].shift(-1)
    ordered["_next_row"] = ordered["raw_row_number"].shift(-1)
    local_end = pd.to_datetime(ordered["session_end_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    local_next_start = pd.to_datetime(ordered["_next_start"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    at_partition_end = local_end.dt.strftime("%H:%M:%S.%f").eq("23:59:59.999000")
    at_next_midnight = local_next_start.dt.strftime("%H:%M:%S.%f").eq("00:00:00.000000")
    stitch = (
        at_partition_end & at_next_midnight
        & ordered["_next_user"].eq(ordered["user_id"])
        & ordered["_next_room"].eq(ordered["room_id"])
        & ordered["_next_author"].eq(ordered["author_id"])
        & (ordered["_next_start"] - ordered["session_end_ms"]).eq(1)
    ).fillna(False)
    stitch_first_rows = set(ordered.loc[stitch, "raw_row_number"].astype(int))
    stitch_second_rows = set(ordered.loc[stitch, "_next_row"].astype(int))
    unmatched = ordered.loc[at_partition_end & ~stitch].copy()
    unmatched["next_user_id"] = unmatched["_next_user"]
    unmatched["next_room_id"] = unmatched["_next_room"]
    unmatched["next_author_id"] = unmatched["_next_author"]
    unmatched["next_start_ms"] = unmatched["_next_start"]
    unmatched["gap_ms"] = unmatched["_next_start"] - unmatched["session_end_ms"]
    unmatched["manual_review_reason"] = "no qualifying immediate continuation"
    unmatched.loc[unmatched["_next_user"].ne(unmatched["user_id"]), "manual_review_reason"] = "next user record is absent or belongs to another user"
    unmatched.loc[unmatched["_next_user"].eq(unmatched["user_id"]) & unmatched["_next_room"].ne(unmatched["room_id"]), "manual_review_reason"] = "next user record is in a different room"
    unmatched.loc[unmatched["_next_room"].eq(unmatched["room_id"]) & unmatched["_next_author"].ne(unmatched["author_id"]), "manual_review_reason"] = "room matches but author differs"
    unmatched.loc[unmatched["_next_room"].eq(unmatched["room_id"]) & unmatched["_next_start"].notna() & (unmatched["gap_ms"] != 1), "manual_review_reason"] = "same-room continuation does not start exactly at next midnight millisecond"
    review_cols = ["raw_row_number", "session_id", "user_id", "author_id", "room_id",
                   "session_start_ms", "session_end_ms", "p_date", "next_user_id",
                   "next_author_id", "next_room_id", "next_start_ms", "gap_ms",
                   "manual_review_reason"]
    (processed / "cross_day_unmatched_review.csv").parent.mkdir(parents=True, exist_ok=True)
    unmatched[review_cols].to_csv(processed / "cross_day_unmatched_review.csv", index=False)
    second_for_first = dict(zip(
        ordered.loc[stitch, "raw_row_number"].astype(int),
        ordered.loc[stitch, "_next_row"].astype(int),
    ))
    chains = {}
    for first in second_for_first:
        chain = [first]
        while chain[-1] in second_for_first:
            chain.append(second_for_first[chain[-1]])
        chains[first] = chain
    second_for_first = {first: chain[-1] for first, chain in chains.items()}
    by_row = sessions.set_index("raw_row_number")
    # Carry end-time/exit relation from the latter segment and recompute the
    # combined duration from the stitched session's actual endpoints.
    first_mask = sessions["raw_row_number"].isin(second_for_first.keys())
    continuation_rows = sessions["raw_row_number"].map(second_for_first)
    merged_end = continuation_rows[first_mask].map(by_row["session_end_ms"])
    merged_duration = merged_end.to_numpy(dtype="int64") - sessions.loc[first_mask, "session_start_ms"].to_numpy(dtype="int64")
    sessions.loc[first_mask, "session_end_ms"] = merged_end.to_numpy()
    sessions.loc[first_mask, "reported_duration_ms"] = merged_duration
    sessions.loc[first_mask, "play_duration"] = merged_duration.astype(str)
    if "session_end_raw" in sessions:
        sessions.loc[first_mask, "session_end_raw"] = continuation_rows[first_mask].map(by_row["session_end_raw"])
    if "is_follow_leave" in sessions:
        sessions.loc[first_mask, "is_follow_leave"] = continuation_rows[first_mask].map(by_row["is_follow_leave"])
    source_rows = {int(row): [int(row)] for row in sessions["raw_row_number"]}
    source_rows.update(chains)
    sessions["source_raw_row_numbers"] = sessions["raw_row_number"].map(source_rows)
    if stitch_second_rows:
        sessions = sessions.loc[~sessions["raw_row_number"].isin(stitch_second_rows)].copy()
    room_count = rooms.groupby("room_id", sort=False).size()
    ambiguous_ids = set(room_count[room_count > 1].index)
    rooms = rooms[~rooms["room_id"].isin(ambiguous_ids)]
    sessions = sessions.merge(
        rooms[["room_id", "author_id", "room_start_ms", "room_end_ms"]],
        on="room_id", how="left", suffixes=("", "_room"), validate="many_to_one",
    )
    diff = sessions["session_end_ms"] - sessions["session_start_ms"]
    err = (diff - sessions["reported_duration_ms"]).abs()
    tol = cfg["observation"]["duration_tolerance_ms"]
    sessions["duration_error_ms"] = err
    sessions["online_context_ok"] = (
        sessions["room_start_ms"].notna()
        & (sessions["room_start_ms"] <= sessions["session_start_ms"])
        & (sessions["author_id"] == sessions["author_id_room"])
        & ~sessions["room_id"].isin(ambiguous_ids)
    ).fillna(False)
    sessions["room_end_conflict"] = (
        sessions["room_end_ms"].notna() & (sessions["session_end_ms"] > sessions["room_end_ms"] + tol)
    ).fillna(False)
    sessions["quality_ok"] = (
        sessions["session_start_ms"].notna() & sessions["session_end_ms"].notna()
        & (diff > 0) & sessions["reported_duration_ms"].notna()
        & (err <= tol) & sessions["online_context_ok"]
    ).fillna(False)
    exact_key = ["user_id", "author_id", "room_id", "session_start_ms", "session_end_ms"]
    sessions["exact_duplicate"] = sessions.duplicated(exact_key, keep="first")
    sessions.loc[sessions["exact_duplicate"], "quality_ok"] = False
    ordered = sessions.sort_values(["user_id", "session_start_ms", "session_end_ms"])
    prior_end = ordered.groupby("user_id")["session_end_ms"].shift()
    overlap_ms = prior_end - ordered["session_start_ms"]
    ordered["overlap_previous"] = (overlap_ms > 0).fillna(False)
    ordered["severe_overlap_previous"] = (
        overlap_ms > cfg["observation"]["overlap_tolerance_ms"]
    ).fillna(False)
    all_overlap_ids = set(ordered.loc[ordered["overlap_previous"], "session_id"])
    overlap_ids = set(ordered.loc[ordered["severe_overlap_previous"], "session_id"])
    sessions["overlap_previous"] = sessions["session_id"].isin(all_overlap_ids)
    sessions["severe_overlap_previous"] = sessions["session_id"].isin(overlap_ids)
    sessions.loc[sessions["severe_overlap_previous"], "quality_ok"] = False
    # The previous overlapping record is also uncertain.
    previous_ids = ordered.groupby("user_id")["session_id"].shift()
    ambiguous_previous = set(previous_ids[ordered["severe_overlap_previous"]].dropna())
    sessions.loc[sessions["session_id"].isin(ambiguous_previous), "quality_ok"] = False
    audit = {
        "sessions": len(sessions), "users": int(sessions["user_id"].nunique()),
        "cleaning_rule_version": cfg["data"].get("cleaning_rule_version", "none"),
        "exact_duplicates": int(sessions["exact_duplicate"].sum()),
        "cross_day_stitches": len(stitch_first_rows),
        "cross_day_rows_removed": len(stitch_second_rows),
        "overlap_later_rows": int(sessions["overlap_previous"].sum()),
        "severe_overlap_later_rows": int(ordered["severe_overlap_previous"].sum()),
        "room_id_multiple_meta_rows": len(ambiguous_ids),
        "room_meta_matched": int(sessions["room_start_ms"].notna().sum()),
        "room_end_known": int(sessions["room_end_ms"].notna().sum()),
        "sessions_ending_after_room_end_by_more_than_tolerance": int(sessions["room_end_conflict"].sum()),
        "duration_error_ms_quantiles": {str(q): float(err.quantile(q)) for q in [0.5, 0.9, 0.99]},
        "quality_ok": int(sessions["quality_ok"].sum()),
        "session_start_min_utc": str(pd.to_datetime(sessions["session_start_ms"].min(), unit="ms", utc=True)),
        "session_end_max_utc": str(pd.to_datetime(sessions["session_end_ms"].max(), unit="ms", utc=True)),
    }
    out = sessions.drop(columns=["session_start_raw", "session_end_raw"])
    out.to_parquet(processed / "sessions.parquet", index=False)
    (processed / "session_audit.json").write_text(json.dumps(audit, indent=2))
    return out, audit
