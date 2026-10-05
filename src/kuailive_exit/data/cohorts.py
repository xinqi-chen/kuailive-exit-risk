"""Deterministic SHA-256 cohorts and bounded raw-row extraction."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ..config import load_config
from ..cleaning.visits import build_visits
from ..data.raw import INTERACTION_COLUMNS, ROOM_COLUMNS, normalize_core

from ..config import utc_ms


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _sha_ids(users: list[str], trailing_newline: bool = False) -> str:
    payload = "\n".join(users) + ("\n" if trailing_newline else "")
    return hashlib.sha256(payload.encode()).hexdigest()


def _csv_batches(path: Path, columns: list[str], block_size: int = 32 << 20):
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=False),
        convert_options=pacsv.ConvertOptions(
            include_columns=columns,
            column_types={name: pa.string() for name in columns},
            strings_can_be_null=True,
        ),
    )


def select_cohorts(cfg: dict | None = None) -> dict:
    """Rebuild A/B/C from raw files. User lists are local, ignored outputs."""
    cfg = cfg or load_config()
    raw = cfg["data"]["raw_dir"]
    root = cfg["data"]["run_dir"]
    out = root / "cohorts"
    out.mkdir(parents=True, exist_ok=True)
    with (raw / "user_id_set.csv").open(newline="") as stream:
        universe = {row["user_id"].strip() for row in csv.DictReader(stream)}
    universe.discard("")
    if not universe:
        raise ValueError("user_id_set.csv is empty")
    order = sorted(universe, key=lambda x: (_digest(x), x))
    train = order[:int(cfg["cohorts"]["train_users"])]

    window_start = utc_ms(cfg["cohorts"]["selection_window_start_utc"])
    window_end = utc_ms(cfg["cohorts"]["selection_window_end_utc_exclusive"])
    eligible: set[str] = set()
    scanned = 0
    for batch in _csv_batches(raw / "live_interaction.csv", ["user_id", "live_play_start_timestamp"]):
        table = pa.Table.from_batches([batch])
        starts = pd.to_datetime(table["live_play_start_timestamp"].to_pandas(),
                                format="mixed", utc=True, errors="coerce")
        starts_ms = starts.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000
        ids = table["user_id"].to_pylist()
        ok = (starts_ms >= window_start) & (starts_ms < window_end)
        eligible.update(user for user, keep in zip(ids, ok) if keep and user in universe)
        scanned += batch.num_rows

    test_order = sorted(eligible - set(train), key=lambda x: (_digest(x), x))
    b = test_order[:int(cfg["cohorts"]["B_users"])]
    a = [user for user in order if user not in set(b)][:int(cfg["cohorts"]["A_users"])]
    c = sorted(eligible - set(a) - set(b), key=lambda x: (_digest(x), x))[:int(cfg["cohorts"]["C_users"])]
    cohorts = {"A": a, "B": b, "C": c}
    expected = cfg["cohorts"]["expected_selected_user_ids_sha256"]
    details = {}
    for name, users in cohorts.items():
        if len(users) != int(cfg["cohorts"][f"{name}_users"]):
            raise ValueError(f"{name} cohort has {len(users)} users")
        selected_hash = _sha_ids(users)
        if selected_hash != expected[name]:
            raise ValueError(f"{name} cohort hash mismatch: {selected_hash}")
        (out / f"{name}_users.txt").write_text("\n".join(users) + "\n")
        details[name] = {"users": len(users), "cohort_sha256": _sha_ids(users, True),
                         "selected_user_ids_sha256": selected_hash}
    if not set(train).issubset(a) or set(a) & set(b) or set(a) & set(c) or set(b) & set(c):
        raise ValueError("cohort overlap or original A300 inclusion check failed")
    manifest = {"data_revision": cfg["data"]["revision"], "eligible_universe_users": len(eligible),
                "eligible_scan_rows": scanned, "cohorts": details,
                "selection_window_utc": ["2026-03-26T16:00:00Z", "2026-03-31T16:00:00Z"],
                "selection_window_end_exclusive": True}
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def extract_visits(cfg: dict | None = None, block_users: int = 100) -> dict:
    """Stream raw interactions into 100-user blocks and apply R0-R4."""
    cfg = cfg or load_config()
    raw = cfg["data"]["raw_dir"]
    root = cfg["data"]["run_dir"]
    cohort_dir = root / "cohorts"
    raw_blocks = root / "raw_blocks"
    visits_dir = root / "visits"
    raw_blocks.mkdir(parents=True, exist_ok=True)
    visits_dir.mkdir(parents=True, exist_ok=True)
    users_by_block: dict[tuple[str, int], list[str]] = {}
    user_key: dict[str, tuple[str, int]] = {}
    for cohort in ("A", "B", "C"):
        ids = (cohort_dir / f"{cohort}_users.txt").read_text().splitlines()
        for block, start in enumerate(range(0, len(ids), block_users)):
            values = ids[start:start + block_users]
            users_by_block[(cohort, block)] = values
            user_key.update((u, (cohort, block)) for u in values)
    writers: dict[tuple[str, int], pq.ParquetWriter] = {}
    counts = defaultdict(int)
    room_users: dict[str, set[tuple[str, int]]] = defaultdict(set)
    seen = 0
    try:
        for batch in _csv_batches(raw / "live_interaction.csv", INTERACTION_COLUMNS):
            table = pa.Table.from_batches([batch])
            rownums = np.arange(seen, seen + batch.num_rows, dtype=np.int64)
            ids = table["user_id"].to_pylist()
            destinations: dict[tuple[str, int], list[int]] = defaultdict(list)
            for index, user in enumerate(ids):
                key = user_key.get(user)
                if key is not None:
                    destinations[key].append(index)
            for key, positions in destinations.items():
                part = table.take(pa.array(positions, type=pa.int64()))
                part = part.append_column("raw_row_number", pa.array(rownums[positions]))
                target = raw_blocks / key[0] / f"block_{key[1]:03d}"
                target.mkdir(parents=True, exist_ok=True)
                file = target / "selected_interactions.parquet"
                if key not in writers:
                    writers[key] = pq.ParquetWriter(file, part.schema, compression="zstd")
                writers[key].write_table(part)
                counts[key] += part.num_rows
                for room in part["live_id"].to_pylist():
                    room_users[str(room)].add(key)
            seen += batch.num_rows
    finally:
        for writer in writers.values():
            writer.close()

    room_ids = pa.array([int(x) for x in room_users], type=pa.int64())
    room_writers: dict[tuple[str, int], pq.ParquetWriter] = {}
    for batch in pq.ParquetFile(raw / "live_room_meta.parquet").iter_batches(
            batch_size=65536, columns=ROOM_COLUMNS, use_threads=False):
        table = pa.Table.from_batches([batch])
        chosen = table.filter(pc.is_in(table["live_id"], value_set=room_ids))
        if not chosen.num_rows:
            continue
        destinations = defaultdict(list)
        for i, room in enumerate(chosen["live_id"].to_pylist()):
            for key in room_users.get(str(room), ()):
                destinations[key].append(i)
        for key, positions in destinations.items():
            target = raw_blocks / key[0] / f"block_{key[1]:03d}"
            file = target / "selected_rooms.parquet"
            if key not in room_writers:
                room_writers[key] = pq.ParquetWriter(file, chosen.schema, compression="zstd")
            room_writers[key].write_table(chosen.take(pa.array(positions, type=pa.int64())))
    for writer in room_writers.values():
        writer.close()

    audit = {"data_revision": cfg["data"]["revision"], "blocks": [],
             "cohort_users": {name: sum(len(v) for k, v in users_by_block.items() if k[0] == name)
                              for name in ("A", "B", "C")},
             "raw_interaction_rows_extracted": int(sum(counts.values())),
             "target_rooms": len(room_users)}
    for key in sorted(users_by_block):
        block = raw_blocks / key[0] / f"block_{key[1]:03d}"
        extraction = {"data_revision": cfg["data"]["revision"], "users": len(users_by_block[key]),
                      "selected_interaction_rows": counts[key],
                      "selected_rooms": pq.ParquetFile(block / "selected_rooms.parquet").metadata.num_rows}
        (block / "extraction_audit.json").write_text(json.dumps(extraction, indent=2))
        local_cfg = dict(cfg)
        local_cfg["data"] = dict(cfg["data"], processed_dir=block)
        sessions, r01 = normalize_core(local_cfg)
        visits, r24 = build_visits(sessions, local_cfg, g_ms=cfg["protocol"]["g_seconds"] * 1000,
                                   delta_minus_ms=cfg["protocol"]["delta_minus_seconds"] * 1000,
                                   delta_plus_ms=cfg["protocol"]["delta_plus_seconds"] * 1000)
        from ..features.browsing import attach_visit_actions
        visits = attach_visit_actions(visits, pd.read_parquet(block / "selected_interactions.parquet"))
        (visits_dir / key[0]).mkdir(parents=True, exist_ok=True)
        visits.to_parquet(visits_dir / key[0] / f"block_{key[1]:03d}.parquet", index=False)
        audit["blocks"].append({"cohort": key[0], "block": key[1], "users": len(users_by_block[key]),
                               "source_rows": counts[key], "R0_R1": r01, "R2_R4": r24})
        print(f"raw visits complete: {key[0]}/block_{key[1]:03d}: {len(visits):,}", flush=True)
        del sessions, visits
    (root / "r0_r4_manifest.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit
