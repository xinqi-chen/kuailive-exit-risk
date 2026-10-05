"""Non-uniform person-period and H30 label construction."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from ..config import utc_ms

def choose_cutpoints(visits: pd.DataFrame, cfg: dict) -> tuple[list[int], dict]:
    buffer = cfg["label"]["observation_buffer_seconds"] * 1000
    observation_end = utc_ms(cfg["observation"]["verified_end_utc_exclusive"])
    train_end = utc_ms(cfg["split"]["train_end_utc_exclusive"])
    v = visits[(visits.split == "train") & visits.quality_ok &
               visits.end_cause.isin(["user", "broadcast"])].copy()
    event = pd.to_numeric(v.event_time_ms, errors="coerce")
    durations = (event - v.session_start_ms) / 1000
    eligible = durations.gt(0) & durations.le(600) & (event + buffer < train_end) & \
        (event + buffer <= observation_end)
    if "overlap_uncertain_end_ms" in v:
        eligible &= (v.overlap_uncertain_end_ms.lt(0) | event.lt(v.overlap_uncertain_end_ms))
    durations = durations[eligible].to_numpy(dtype=float)
    if len(durations) < 1200:
        raise ValueError("too few training exits to form roughly twelve intervals")
    raw = np.quantile(durations, np.arange(1, 12) / 12)
    # Rounding is based on train exits alone. Greedy merging enforces the
    # requested minimum widths, and the final 600-second edge is fixed.
    proposed = sorted({int(round(x / (5 if x <= 60 else 15)) * (5 if x <= 60 else 15))
                       for x in raw if 0 < x < 600})
    edges = [0]
    for x in proposed + [600]:
        minimum = 5 if edges[-1] < 60 and x <= 60 else 15
        if x - edges[-1] >= minimum:
            edges.append(x)
    if edges[-1] != 600:
        edges[-1] = 600
    # Extreme concentration at 0-5 s makes twelve equal-exit intervals
    # mathematically impossible under the width rule. Fill remaining slots
    # by bisecting the widest feasible event-bearing bin at its training median.
    while len(edges) - 1 < 12:
        options = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            local = durations[(durations > lo) & (durations <= hi)]
            if len(local) < 200:
                continue
            increment = 5 if hi <= 60 else 15
            minimum = 5 if hi <= 60 else 15
            candidate = int(round(np.median(local) / increment) * increment)
            if candidate - lo >= minimum and hi - candidate >= minimum:
                options.append((len(local), hi - lo, i, candidate))
        if not options:
            break
        _, _, i, candidate = max(options)
        edges.insert(i + 1, candidate)
    # Merge low count late bins deterministically into the preceding interval.
    while True:
        counts = [int(((durations > lo) & (durations <= hi)).sum())
                  for lo, hi in zip(edges[:-1], edges[1:])]
        sparse = next((i for i, n in enumerate(counts) if edges[i] >= 60 and n < 100), None)
        if sparse is None or len(edges) <= 3:
            break
        del edges[sparse if sparse > 0 else 1]
    counts = [int(((durations > lo) & (durations <= hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    return edges, {"train_exit_count": len(durations), "raw_quantiles_seconds": raw.tolist(),
                   "interval_train_exits": counts, "minimum_late_exits": 100,
                   "rounding": "5s through 60s, 15s after 60s"}


def make_periods(visits: pd.DataFrame, edges: list[int], cfg: dict,
                 *, only_split: str | None = None, target: str = "E2-V") -> pd.DataFrame:
    """Person-period rows. E2-V treats broadcast end as a competing event:
    a broadcast-ended visit is a negative while the interval ends before
    B - delta_minus and is censored once the interval reaches the band.
    E2-A counts broadcast end (event time B) as the event."""
    if target not in {"E2-V", "E2-A"}:
        raise ValueError("target must be E2-V or E2-A")
    buffer = cfg["label"]["observation_buffer_seconds"] * 1000
    observation_end = utc_ms(cfg["observation"]["verified_end_utc_exclusive"])
    delta_minus = cfg["protocol"]["delta_minus_seconds"] * 1000
    parts = []
    bounds = {"train": utc_ms(cfg["split"]["train_end_utc_exclusive"]),
              "validation": utc_ms(cfg["split"]["validation_end_utc_exclusive"]),
              "test": utc_ms(cfg["split"]["test_end_utc_exclusive"])}
    v = visits if only_split is None else visits[visits.split.eq(only_split)]
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        t = v.session_start_ms + lo * 1000
        end = v.session_end_ms
        b = pd.to_numeric(v.room_end_ms, errors="coerce")
        at_risk = t.lt(end) & (b.isna() | t.lt(b)) & v.split.isin(bounds)
        x = v.loc[at_risk].copy()
        if x.empty:
            continue
        t = x.session_start_ms + lo * 1000
        u = x.session_start_ms + hi * 1000
        event = pd.to_numeric(x.event_time_ms, errors="coerce")
        split_end = x.split.map(bounds)
        status = np.full(len(x), "evaluable", dtype=object)
        bad = ~x.quality_ok.to_numpy(dtype=bool)
        status[bad] = "invalid_session"
        boundary = (u + buffer >= split_end).to_numpy(dtype=bool)
        status[(status == "evaluable") & boundary] = "split_boundary_unavailable"
        observation = (u + buffer > observation_end).to_numpy(dtype=bool)
        status[(status == "evaluable") & observation] = "observation_incomplete"
        censored = x.end_cause.eq("observation_censored").to_numpy()
        censor_time = pd.to_numeric(x.r1_censor_time_ms, errors="coerce").fillna(x.session_end_ms)
        status[(status == "evaluable") & censored & (u + buffer > censor_time).to_numpy()] = "observation_incomplete"
        causes = x.end_cause.to_numpy()
        status[(status == "evaluable") & np.isin(causes, ["record_anomaly", "ambiguous_room_end",
                                                       "entered_after_end"])] = "uncertain_end"
        overlap = np.fromiter((any(int(ti) < int(a) <= int(ui) + buffer for a in starts)
                               for ti, ui, starts in zip(t, u, x.overlap_start_times_ms)),
                              dtype=bool, count=len(x))
        # The first cross-room overlap censors observation locally. Earlier
        # fully observed negatives remain valid even if the eventual end is unknown.
        uncertain = (x.overlap_uncertain_end_ms.ge(0) &
                     (u + buffer >= x.overlap_uncertain_end_ms)).fillna(False).to_numpy(dtype=bool)
        status[(status == "evaluable") & (overlap | uncertain)] = "ambiguous_overlap"
        if target == "E2-V":
            broadcast = x.end_cause.eq("broadcast").to_numpy()
            b_end = pd.to_numeric(x.room_end_ms, errors="coerce")
            near_band = (u >= b_end - delta_minus).fillna(True).to_numpy(dtype=bool)
            status[(status == "evaluable") & broadcast & near_band] = "censored_broadcast"
            positive = (event <= u).fillna(False).to_numpy(dtype=bool) & ~broadcast
        else:
            positive = (event <= u).fillna(False).to_numpy(dtype=bool)
        x["label"] = np.where(status == "evaluable", positive.astype(float), np.nan)
        x["label_status"] = status
        x["interval_index"] = k
        x["interval_start_seconds"] = lo
        x["interval_end_seconds"] = hi
        x["offset_seconds"] = lo
        x["prediction_time_ms"] = t
        x["session_id"] = x.visit_id
        x["prediction_id"] = [hashlib.sha256(f"{vid}:{lo}:NU".encode()).hexdigest()[:24]
                              for vid in x.visit_id]
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    if out.prediction_id.duplicated().any():
        raise AssertionError("duplicate person-period ID")
    return out


def h30_from_visits(visits: pd.DataFrame, cfg: dict, target: str,
                    *, only_split: str = "test") -> pd.DataFrame:
    landmarks = cfg["landmarks"]
    start, step, maximum = (landmarks[k] for k in
                           ("first_offset_seconds", "step_seconds", "max_offset_seconds"))
    if step != 30:
        raise ValueError("H30 landmarks currently require a 30-second step")
    out = make_periods(visits, list(range(start, maximum + 31, step)), cfg,
                       only_split=only_split, target=target)
    out["interval_index"] = -1
    out["prediction_id"] = [hashlib.sha256(f"{v}:{t}:H30:{target}".encode()).hexdigest()[:24]
                            for v, t in zip(out.visit_id, out.prediction_time_ms)]
    return out
