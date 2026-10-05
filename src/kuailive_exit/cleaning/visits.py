"""Experimental implementation of SAMPLE_PROCESSING_PROTOCOL_v0.1.

All new tables are written to an isolated v1.5 directory; legacy runs are
never rewritten. Times are epoch milliseconds and split assignment is by visit
start, as in the established pilot protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import utc_ms


HORIZON_MS = 30_000
BUFFER_MS = 5_000
OBSERVATION_END = utc_ms("2026-03-31T16:00:00Z")
DELTA_MINUS_MS = 5_000
DELTA_PLUS_MS = 60_000
ENTRY_CUTOFFS_MS = (10_000, 60_000)


def _raw_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(value)]


def _split(start: int, cfg: dict) -> tuple[str | None, int, int]:
    a, b, c = (utc_ms(cfg["split"][key]) for key in (
        "train_end_utc_exclusive", "validation_end_utc_exclusive", "test_end_utc_exclusive"))
    if start < a:
        return "train", -2**63, a
    if start < b:
        return "validation", a, b
    if start < c:
        return "test", b, c
    return None, b, c


def _exact_dedup(sessions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = ["user_id", "author_id", "room_id", "session_start_ms", "session_end_ms",
           "reported_duration_ms", "live_source_category", "enter_live_action",
           "is_auto_play", "is_follow_enter", "is_follow_leave"]
    key = [c for c in key if c in sessions]
    duplicate_mask = sessions.duplicated(key, keep=False)
    removed = int(sessions.duplicated(key, keep="first").sum())
    out = sessions.loc[~sessions.duplicated(key, keep="first")].copy()
    # Preserve provenance only for actual duplicate keys; do not run a Python
    # list aggregation once per ordinary session in the half-million-row table.
    if duplicate_mask.any() and "source_raw_row_numbers" in sessions:
        duplicate_rows = sessions.loc[duplicate_mask]
        for _, group in duplicate_rows.groupby(key, dropna=False, sort=False):
            first_idx = group.index[0]
            source_ids = sorted({raw for value in group["source_raw_row_numbers"] for raw in _raw_list(value)})
            out.at[first_idx, "source_raw_row_numbers"] = source_ids
    return out, int(removed)


def build_visits(sessions: pd.DataFrame, cfg: dict, *, g_ms: int = 1000,
                 old_overlap_quarantine: bool = False,
                 delta_minus_ms: int = DELTA_MINUS_MS,
                 delta_plus_ms: int = DELTA_PLUS_MS) -> tuple[pd.DataFrame, dict]:
    """Apply R0-R4 and construct visits for one re-entry grace period."""
    if g_ms < 0:
        raise ValueError("g_ms must be nonnegative; overlapping same-room rows still form a union")
    before = len(sessions)
    s, deduped = _exact_dedup(sessions)
    s = s.sort_values(["user_id", "session_start_ms", "raw_row_number"], kind="stable")
    max_date = s["p_date"].astype(str).max()
    midnight = pd.to_datetime(s["session_end_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    midnight_mask = midnight.dt.strftime("%H:%M:%S.%f").eq("23:59:59.999000")
    # R1 was executed during canonical pilot normalization. Count its lineage
    # rather than stitching twice; unmatched midnight rows remain censored.
    processed_dir = cfg.get("data", {}).get("processed_dir")
    source_audit_path = Path(processed_dir) / "session_audit.json" if processed_dir else Path("/__missing__")
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.exists() else {}
    r1_stitches = int(source_audit.get("cross_day_stitches", 0))
    r1_unmatched = int(source_audit.get("cross_day_unmatched", 0))
    if not r1_unmatched:
        review_path = Path(processed_dir) / "cross_day_unmatched_review.csv" if processed_dir else Path("/__missing__")
        r1_unmatched = len(pd.read_csv(review_path)) if review_path.exists() else int(midnight_mask.sum())
    r1_midnight_total = r1_stitches + r1_unmatched
    s["r1_unmatched_midnight"] = midnight_mask
    review_ids: set[int] = set()
    review_path = Path(processed_dir) / "cross_day_unmatched_review.csv" if processed_dir else Path("/__missing__")
    if review_path.exists():
        review_ids = set(pd.read_csv(review_path, usecols=["raw_row_number"])["raw_row_number"].astype(int))
    raw_id = pd.to_numeric(s["raw_row_number"], errors="coerce").fillna(-1).astype(int)
    # The source audit explicitly identifies last-day unmatched records; p_date
    # on neighboring carried rows is not sufficient to infer censoring.
    s["r1_last_partition_end"] = midnight_mask & raw_id.isin(review_ids) & s["p_date"].astype(str).eq(max_date)
    s["r1_censor_time_ms"] = s["session_end_ms"].where(midnight_mask)
    s["end_observation_status"] = np.where(
        s["r1_last_partition_end"], "observation_boundary",
        np.where(midnight_mask, "midnight_censored", "observed_end"))

    s = s.sort_values(["user_id", "room_id", "session_start_ms", "raw_row_number"], kind="stable").copy()
    keys = ["user_id", "room_id"]
    grouped = s.groupby(keys, sort=False, dropna=False)
    previous_max_end = grouped["session_end_ms"].cummax().groupby(
        [s[k] for k in keys], sort=False, dropna=False).shift()
    starts_new_visit = previous_max_end.isna() | ((s["session_start_ms"] - previous_max_end) > g_ms)
    s["_r2_cluster"] = starts_new_visit.groupby([s[k] for k in keys], sort=False, dropna=False).cumsum().astype("int64")
    cluster_keys = keys + ["_r2_cluster"]
    clustered = s.groupby(cluster_keys, sort=False, dropna=False)
    v = clustered.head(1).copy().set_index(cluster_keys, drop=False)
    group_end = clustered["session_end_ms"].max()
    group_size = clustered.size()
    group_sources = clustered["source_raw_row_numbers"].first()
    multi_keys = group_size[group_size > 1].index
    if len(multi_keys):
        source_frame = s.set_index(cluster_keys).loc[multi_keys]
        multi_sources = source_frame.groupby(level=cluster_keys, sort=False)["source_raw_row_numbers"].agg(
            lambda values: [raw for value in values for raw in _raw_list(value)])
        group_sources.loc[multi_sources.index] = multi_sources
    censor_times = pd.to_numeric(s["r1_censor_time_ms"], errors="coerce")
    s["_r1_censor_numeric"] = censor_times
    clustered = s.groupby(cluster_keys, sort=False, dropna=False)
    group_censor = clustered["_r1_censor_numeric"].max()
    v["session_end_ms"] = group_end
    v["reported_duration_ms"] = group_end - v["session_start_ms"]
    v["source_raw_row_numbers"] = group_sources
    v["segment_count"] = group_size
    v["last_segment_end_ms"] = group_end
    v["r1_censor_time_ms"] = group_censor.where(group_censor.eq(group_end))
    v["r1_unmatched_midnight"] = v["r1_censor_time_ms"].notna()
    v["r1_last_partition_end"] = v["r1_unmatched_midnight"] & clustered["r1_last_partition_end"].max()
    v["end_observation_status"] = np.where(
        v["r1_last_partition_end"], "observation_boundary",
        np.where(v["r1_unmatched_midnight"], "midnight_censored", "observed_end"))
    if "is_follow_leave" in s:
        terminal_rows = s.sort_values([*cluster_keys, "session_end_ms", "session_start_ms"], kind="stable").groupby(
            cluster_keys, sort=False, dropna=False).tail(1).set_index(cluster_keys)
        v["is_follow_leave"] = terminal_rows["is_follow_leave"]
    v = v.reset_index(drop=True)
    if v.empty:
        raise ValueError("visit construction returned no rows")
    v["visit_id"] = v["source_raw_row_numbers"].map(
        lambda rows: hashlib.sha256((str(cfg["data"]["revision"]) + ":" + ",".join(map(str, rows))).encode()).hexdigest()[:24])
    v["session_id"] = v["visit_id"]
    v["g_ms"] = int(g_ms)
    # Wait for same-room stitching and the early broadcast-end exclusion band.
    v["history_available_ms"] = v["session_end_ms"] + max(g_ms, delta_minus_ms)
    v["split"] = v["session_start_ms"].map(lambda x: _split(int(x), cfg)[0])
    # Recompute duration validity after R0-R2. A stitched visit is measured by
    # elapsed time from its first entry through its final observed segment.
    v["duration_error_ms"] = 0
    v["quality_ok"] = (v["session_end_ms"] > v["session_start_ms"]) & v["online_context_ok"].fillna(False)
    room_end_numeric = pd.to_numeric(v["room_end_ms"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    v["room_end_conflict"] = (np.isfinite(room_end_numeric) &
                              (v["session_end_ms"].to_numpy(dtype=np.int64) >
                               room_end_numeric + cfg["observation"]["duration_tolerance_ms"]))
    # R3: interval starts that make the earlier cross-room visit ambiguous.
    v = v.reset_index(drop=True)
    overlap_starts: list[list[int]] = [[] for _ in range(len(v))]
    overlap_involved = np.zeros(len(v), dtype=bool)
    starts = v["session_start_ms"].to_numpy(dtype=np.int64)
    ends = v["session_end_ms"].to_numpy(dtype=np.int64)
    rooms = v["room_id"].astype(str).to_numpy()
    for _, idx in v.sort_values(["user_id", "session_start_ms"], kind="stable").groupby("user_id", sort=False).groups.items():
        active: list[int] = []
        for i in list(idx):
            a = starts[i]
            room = rooms[i]
            active = [j for j in active if ends[j] > a + 1000]
            for j in active:
                if rooms[j] != room:
                    overlap_starts[j].append(int(a))
                    overlap_involved[j] = True
                    overlap_involved[i] = True
            active.append(i)
    v["overlap_start_times_ms"] = overlap_starts
    v["overlap_involved"] = overlap_involved
    v["severe_overlap_previous"] = v["overlap_start_times_ms"].map(bool)
    # End time is unknown after another room starts; keep the interval for
    # auditing and prohibit events in that interval from becoming positives.
    uncertain_end = np.full(len(v), -1, dtype=np.int64)
    for i, starts_for_visit in enumerate(overlap_starts):
        if starts_for_visit:
            uncertain_end[i] = min(starts_for_visit)
    v["overlap_uncertain_end_ms"] = uncertain_end
    if old_overlap_quarantine:
        v.loc[v["overlap_involved"], "quality_ok"] = False
    # R4: cause mechanism proxy; no uncertain B-missing room is guessed live.
    end_ms = v["session_end_ms"].astype("int64")
    b_ms = v["room_end_ms"].astype("Int64")
    cause = np.full(len(v), "ambiguous_room_end", dtype=object)
    event_ms = np.full(len(v), -1, dtype=np.int64)
    has_b = b_ms.notna().to_numpy()
    bv = b_ms.fillna(-1).to_numpy(dtype=np.int64)
    av = v["session_start_ms"].to_numpy(dtype=np.int64)
    ev = end_ms.to_numpy(dtype=np.int64)
    before_end = has_b & (av <= bv)
    cause[has_b & (av > bv)] = "entered_after_end"
    user = before_end & (ev < bv - delta_minus_ms)
    broadcast = before_end & (ev >= bv - delta_minus_ms) & (ev <= bv + delta_plus_ms)
    anomaly = before_end & (ev > bv + delta_plus_ms)
    cause[user] = "user"
    cause[broadcast] = "broadcast"
    cause[anomaly] = "record_anomaly"
    event_ms[user] = ev[user]
    event_ms[broadcast] = bv[broadcast]
    # For missing B: room metadata records an end only for broadcasts that ended
    # inside the observation window (audit: 0 known ends >= C; every missing-end
    # room started in the final 7 days). A missing B is therefore a right-censored
    # room end (B >= C), so the broadcast was live at every t < C and any visit
    # ending before C is user-caused. Rooms without a known start stay ambiguous.
    room_continues = (~has_b) & v["room_start_ms"].notna().to_numpy()
    no_b = ~has_b
    user_no_b = no_b & room_continues & (ev < OBSERVATION_END)
    cause[user_no_b] = "user"
    event_ms[user_no_b] = ev[user_no_b]
    v["end_cause"] = cause
    v["event_time_ms"] = pd.array(np.where(event_ms >= 0, event_ms, None), dtype="Int64")
    v["room_live_at_observation_end_confirmed"] = room_continues
    if "r1_unmatched_midnight" in v:
        censored_midnight = v["r1_unmatched_midnight"].fillna(False)
        v.loc[censored_midnight, "end_cause"] = "observation_censored"
        v.loc[censored_midnight, "event_time_ms"] = pd.NA
    if "r1_last_partition_end" in v:
        terminal = v["r1_last_partition_end"].fillna(False).to_numpy(dtype=bool)
        # These 34 source records are explicitly the right-censored observation
        # endpoint; never reinterpret their timestamp as a user exit.
        cause[terminal] = "observation_censored"
        v.loc[terminal, "end_cause"] = "observation_censored"
        v.loc[terminal, "event_time_ms"] = pd.NA
    counts = {"input_rows": before, "after_exact_dedup": len(s), "exact_duplicates_removed": deduped,
              "r1_midnight_rows": r1_midnight_total, "r1_stitches_in_lineage": r1_stitches,
              "r1_unmatched_midnight_rows": r1_unmatched,
              "after_r2_visits": len(v), "r2_segments_absorbed": int(v["segment_count"].sum() - len(v)),
              "g_ms": g_ms, "overlap_visits": int(v["severe_overlap_previous"].sum()),
              "old_overlap_quarantine": old_overlap_quarantine,
              "causes": v["end_cause"].value_counts(dropna=False).to_dict()}
    return v, counts


def build_risk_tables(visits: pd.DataFrame, cfg: dict, *, delta_minus_ms: int = DELTA_MINUS_MS,
                      delta_plus_ms: int = DELTA_PLUS_MS, overlap_policy: str = "window") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct E2-V/E2-A landmarks plus E1 entry classifications."""
    if overlap_policy not in {"window", "session"}:
        raise ValueError("overlap_policy must be window or session")
    rows: list[dict] = []
    entries: list[dict] = []
    first = cfg["landmarks"]["first_offset_seconds"]
    step = cfg["landmarks"]["step_seconds"]
    maximum = cfg["landmarks"]["max_offset_seconds"]
    for v in visits.itertuples(index=False):
        start, end = int(v.session_start_ms), int(v.session_end_ms)
        split, split_start, split_end = _split(start, cfg)
        if split is None:
            continue
        cause = str(v.end_cause)
        b = None if pd.isna(v.room_end_ms) else int(v.room_end_ms)
        obs_censor = not bool(v.end_observation_status == "observed_end")
        if start >= OBSERVATION_END or cause == "entered_after_end":
            continue
        label_entry: int | None = None
        entry_status = cause
        duration = end - start
        if cause == "user":
            label_entry = 0 if duration <= ENTRY_CUTOFFS_MS[0] else 1 if duration <= ENTRY_CUTOFFS_MS[1] else 2
            entry_status = ("exit_0_10s", "exit_10_60s", "stay_beyond_60s")[label_entry]
        elif cause == "broadcast" and b is not None and b - delta_minus_ms > start + ENTRY_CUTOFFS_MS[1]:
            label_entry, entry_status = 2, "stay_beyond_60s"
        elif cause == "broadcast":
            entry_status = "censored_broadcast"
        elif cause == "observation_censored" and duration > ENTRY_CUTOFFS_MS[1]:
            label_entry, entry_status = 2, "stay_beyond_60s"
        elif cause == "observation_censored":
            entry_status = "censored_observation"
        entries.append({"prediction_id": hashlib.sha256(f"{v.visit_id}:entry:v3".encode()).hexdigest()[:24],
                        "visit_id": v.visit_id, "session_id": v.visit_id, "user_id": v.user_id,
                        "author_id": v.author_id, "room_id": v.room_id, "split": split,
                        "prediction_time_ms": start, "session_start_ms": start, "session_end_ms": end,
                        "label": label_entry, "label_status": entry_status, "label_name": entry_status,
                        "end_cause": cause, "live_source_category": v.live_source_category,
                        "enter_live_action": v.enter_live_action, "is_auto_play": v.is_auto_play,
                        "is_follow_enter": v.is_follow_enter, "room_start_ms": v.room_start_ms,
                        "room_end_ms": b, "offset_seconds": 0, "elapsed_watch_seconds": 0.0,
                        "reported_duration_ms": v.reported_duration_ms})
        if not bool(v.quality_ok):
            entries[-1]["label"] = None
            entries[-1]["label_status"] = "invalid_or_ambiguous_session"
            entries[-1]["label_name"] = "invalid_or_ambiguous_session"
        # At-risk rows are created only while both the visit and broadcast are live.
        for offset in range(first, maximum + 1, step):
            t = start + offset * 1000
            if t >= end or t >= split_end or (split_start != -2**63 and t < split_start):
                break
            if b is not None and t >= b:
                break
            overlap_starts = v.overlap_start_times_ms if isinstance(v.overlap_start_times_ms, (list, tuple, np.ndarray)) else []
            base = {"visit_time_id": hashlib.sha256(f"{v.visit_id}:{t}:v3".encode()).hexdigest()[:24],
                    "visit_id": v.visit_id, "session_id": v.visit_id, "user_id": v.user_id,
                    "author_id": v.author_id, "room_id": v.room_id, "split": split,
                    "prediction_time_ms": t, "session_start_ms": start, "session_end_ms": end,
                    "room_end_ms": b, "room_start_ms": v.room_start_ms,
                    "offset_seconds": offset, "elapsed_watch_seconds": float(offset),
                    "end_cause": cause, "label_protocol": "voluntary_exit_v3",
                    "live_source_category": v.live_source_category, "enter_live_action": v.enter_live_action,
                    "is_auto_play": v.is_auto_play, "is_follow_enter": v.is_follow_enter,
                    "visit_duration_ms": int(v.reported_duration_ms), "g_ms": int(v.g_ms) if hasattr(v, "g_ms") else None,
                    "overlap_involved": bool(v.overlap_involved)}
            # All-at-risk outcomes start unavailable; labels are then attached.
            for target in ("E2-V", "E2-A"):
                status, y = "ambiguous_room_end", None
                deadline = max(t + HORIZON_MS + BUFFER_MS,
                               end + getattr(v, "g_ms", 1000))
                if not bool(v.quality_ok):
                    status = "invalid_or_ambiguous_session"
                elif t + HORIZON_MS + BUFFER_MS > split_end:
                    status = "split_boundary_unavailable"
                elif t + HORIZON_MS + BUFFER_MS > OBSERVATION_END:
                    status = "observation_incomplete"
                elif cause == "observation_censored":
                    censor_at = int(v.r1_censor_time_ms) if not pd.isna(v.r1_censor_time_ms) else end
                    if t + HORIZON_MS + BUFFER_MS > censor_at:
                        status = "observation_incomplete"
                    else:
                        status, y = "negative_watch_continue", 0
                elif cause == "record_anomaly":
                    status = "record_anomaly"
                elif cause == "entered_after_end":
                    status = "entered_after_end"
                elif cause == "ambiguous_room_end":
                    status = "ambiguous_room_end"
                elif old_overlap_status(v, t, HORIZON_MS + BUFFER_MS, overlap_policy):
                    status = "ambiguous_overlap"
                elif target == "E2-V" and cause == "broadcast":
                    if b is not None and t + HORIZON_MS < b - delta_minus_ms:
                        status, y = "negative_watch_continue", 0
                    else:
                        status = "censored_broadcast"
                else:
                    event = int(v.event_time_ms) if not pd.isna(v.event_time_ms) else None
                    if target == "E2-A" and cause == "broadcast":
                        event = b
                    if event is None:
                        status = "ambiguous_room_end"
                    elif int(getattr(v, "overlap_uncertain_end_ms", -1)) >= 0 and t + HORIZON_MS + BUFFER_MS >= int(v.overlap_uncertain_end_ms):
                        status = "ambiguous_overlap"
                    elif t < event <= t + HORIZON_MS:
                        status, y = "positive_watch_end", 1
                    elif event > t + HORIZON_MS:
                        status, y = "negative_watch_continue", 0
                    else:
                        status = "not_at_risk"
                rows.append({**base, "prediction_id": hashlib.sha256(f"{v.visit_id}:{t}:{target}:v3".encode()).hexdigest()[:24],
                             "task": target, "label_status": status, "label": y,
                             "label_protocol": "voluntary_exit_v3", "delta_minus_ms": delta_minus_ms,
                             "delta_plus_ms": delta_plus_ms, "overlap_policy": overlap_policy,
                             "ambiguous_overlap_starts_ms": list(overlap_starts)})
    risk = pd.DataFrame(rows)
    entry = pd.DataFrame(entries)
    for frame in (risk, entry):
        if not frame.empty and frame.prediction_id.duplicated().any():
            raise AssertionError("duplicate prediction ID")
    return risk, entry


def old_overlap_status(v: object, t: int, window_ms: int, policy: str) -> bool:
    starts = getattr(v, "overlap_start_times_ms", [])
    if policy == "session" and bool(getattr(v, "overlap_involved", False)):
        return True
    return any(t < int(a) <= t + window_ms for a in starts)


def reclassify_causes(visits: pd.DataFrame, delta_minus_ms: int, delta_plus_ms: int) -> None:
    """Reapply R4 to existing visits for one-way threshold sensitivity."""
    a = visits.session_start_ms.to_numpy(dtype=np.int64)
    e = visits.session_end_ms.to_numpy(dtype=np.int64)
    bseries = pd.to_numeric(visits.room_end_ms, errors="coerce")
    has_b = bseries.notna().to_numpy()
    b = bseries.fillna(-1).to_numpy(dtype=np.int64)
    cause = visits.end_cause.astype(object).to_numpy(copy=True)
    event = pd.to_numeric(visits.event_time_ms, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid = has_b & (a <= b)
    user = valid & (e < b - delta_minus_ms)
    broadcast = valid & (e >= b - delta_minus_ms) & (e <= b + delta_plus_ms)
    anomaly = valid & (e > b + delta_plus_ms)
    cause[user], event[user] = "user", e[user]
    cause[broadcast], event[broadcast] = "broadcast", b[broadcast]
    cause[anomaly], event[anomaly] = "record_anomaly", -1
    cause[has_b & (a > b)] = "entered_after_end"
    censored = visits.r1_unmatched_midnight.fillna(False).to_numpy(dtype=bool)
    cause[censored], event[censored] = "observation_censored", -1
    visits["end_cause"] = cause
    visits["event_time_ms"] = pd.array(np.where(event >= 0, event, None), dtype="Int64")


# Compatibility alias; keep history definitions in one module.
from ..features.history import add_visit_history


def audit_pipeline(visits: pd.DataFrame, risk: pd.DataFrame, entries: pd.DataFrame,
                   cfg: dict, flow: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "voluntary_exit_v3", "cleaning_rule_version": cfg["data"].get("cleaning_rule_version"),
              "flow": flow, "risk_rows": int(len(risk)), "entry_rows": int(len(entries)),
              "cause_counts": visits.end_cause.value_counts(dropna=False).to_dict(),
              "risk_status_counts": {f"{a}|{b}": int(n) for (a, b), n in risk.groupby(["task", "label_status"]).size().items()},
              "entry_status_counts": entries.label_status.value_counts(dropna=False).to_dict()}
    (out_dir / "protocol_audit.json").write_text(json.dumps(result, indent=2, default=str))
    split_flow = []
    for split in ("train", "validation", "test"):
        vs = visits[visits.split.eq(split)] if "split" in visits else visits
        rs = risk[risk.split.eq(split)]
        es = entries[entries.split.eq(split)]
        for stage, n in (("visits_after_R2", len(vs)), ("risk_rows", len(rs)), ("entry_rows", len(es))):
            split_flow.append({"split": split, "stage": stage, "n": int(n)})
        for (task, status), n in rs.groupby(["task", "label_status"]).size().items():
            split_flow.append({"split": split, "stage": f"{task}:{status}", "n": int(n)})
    pd.DataFrame(split_flow).to_csv(out_dir / "pipeline_flow.csv", index=False)
    return result
