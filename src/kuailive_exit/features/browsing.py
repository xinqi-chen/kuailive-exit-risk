"""Strictly past browsing-state features for the v1.6 interval experiment."""
from __future__ import annotations

from bisect import bisect_left, insort

import numpy as np
import pandas as pd


WINDOWS_SECONDS = (60, 300, 900)
PERSONAL = [*(f"browse_starts_{w}s" for w in WINDOWS_SECONDS),
            *(f"browse_ends_{w}s" for w in WINDOWS_SECONDS),
            "browse_last1_duration_seconds", "browse_last2_duration_seconds",
            "browse_last3_duration_seconds", "browse_recent5_under5_fraction",
            "browse_last_start_age_seconds", "browse_fast_swipe_streak",
            "browse_active_segment_seconds", "browse_active_segment_visits"]
LOCAL_TIME = ["browse_local_hour", "browse_local_weekday"]
ROOM = ["room_other_present", "room_other_entries_60s", "room_other_exits_60s",
        "room_other_net_entries_60s", "room_sample_covered"]
ACTIONS = ["browse_past_gift_fraction", "browse_past_like_fraction",
           "browse_past_comment_fraction"]
ALL = PERSONAL + LOCAL_TIME + ROOM + ACTIONS


def attach_visit_actions(visits: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-segment counts to each R2 visit; never expose current totals."""
    counts = actions[["raw_row_number", "send_gift_cnt", "like_cnt", "comment_cnt"]].copy()
    for name in ("send_gift_cnt", "like_cnt", "comment_cnt"):
        counts[name] = pd.to_numeric(counts[name], errors="coerce").fillna(0).clip(lower=0)
    counts = counts.drop_duplicates("raw_row_number")
    by_raw = {int(row.raw_row_number): (row.send_gift_cnt > 0, row.like_cnt > 0,
                                         row.comment_cnt > 0)
              for row in counts.itertuples(index=False)}
    flags = np.zeros((len(visits), 3), dtype=np.int8)
    for i, source in enumerate(visits.source_raw_row_numbers):
        if isinstance(source, (list, tuple, np.ndarray)):
            rows = source
        elif pd.isna(source):
            rows = []
        else:
            rows = [source]
        for raw in rows:
            value = by_raw.get(int(raw))
            if value is not None:
                flags[i] |= value
    out = visits.copy()
    out["past_visit_has_gift"] = flags[:, 0]
    out["past_visit_has_like"] = flags[:, 1]
    out["past_visit_has_comment"] = flags[:, 2]
    return out


def local_time_features(points: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(points.prediction_time_ms, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return pd.DataFrame({"browse_local_hour": local.dt.hour.to_numpy(dtype=np.int8),
                         "browse_local_weekday": local.dt.weekday.to_numpy(dtype=np.int8)},
                        index=points.index)


def personal_features(points: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Use only quality R2 visits with known end strictly earlier than query t."""
    original_index = points.index
    points = points.reset_index(drop=True)
    out = {name: np.zeros(len(points), dtype=np.float32) for name in PERSONAL + ACTIONS}
    hist = visits[visits.quality_ok & visits.end_cause.isin(["user", "broadcast"]) &
                  visits.get("overlap_uncertain_end_ms", pd.Series(-1, index=visits.index)).lt(0)].copy()
    hist["_available_ms"] = hist.get("history_available_ms", hist.session_end_ms)
    hist = hist.sort_values(["user_id", "_available_ms", "visit_id"], kind="stable")
    by_user = {u: g for u, g in hist.groupby("user_id", sort=False)}
    for user, rows in points.groupby("user_id", sort=False):
        h = by_user.get(user)
        if h is None:
            out["browse_active_segment_seconds"][rows.index] = (
                rows.prediction_time_ms.to_numpy(dtype=np.int64) -
                rows.session_start_ms.to_numpy(dtype=np.int64)) / 1000
            out["browse_active_segment_visits"][rows.index] = 1
            continue
        ends = h.session_end_ms.to_numpy(dtype=np.int64)
        available = h._available_ms.to_numpy(dtype=np.int64)
        starts = h.session_start_ms.to_numpy(dtype=np.int64)
        duration = h.reported_duration_ms.to_numpy(dtype=np.float64) / 1000
        action = h[["past_visit_has_gift", "past_visit_has_like",
                    "past_visit_has_comment"]].to_numpy(dtype=np.float64)
        action_prefix = np.vstack([np.zeros((1, 3)), np.cumsum(action, axis=0)])
        quick_prefix = np.r_[0, np.cumsum(duration < 5)]
        streak = np.zeros(len(h), dtype=np.int32)
        segment_start = np.zeros(len(h), dtype=np.int64)
        segment_count = np.zeros(len(h), dtype=np.int32)
        for i in range(len(h)):
            streak[i] = (streak[i - 1] if i else 0) + 1 if duration[i] < 5 else 0
            if i and starts[i] - ends[i - 1] <= 600_000:
                segment_start[i] = segment_start[i - 1]
                segment_count[i] = segment_count[i - 1] + 1
            else:
                segment_start[i] = starts[i]
                segment_count[i] = 1
        ordered = rows.sort_values("prediction_time_ms", kind="stable")
        idx = ordered.index.to_numpy(dtype=np.int64)
        t = ordered.prediction_time_ms.to_numpy(dtype=np.int64)
        current_start = ordered.session_start_ms.to_numpy(dtype=np.int64)
        k = np.searchsorted(available, t, side="left")
        nonzero = k > 0
        for w in WINDOWS_SECONDS:
            out[f"browse_ends_{w}s"][idx] = np.maximum(0, k - np.searchsorted(ends, t - w * 1000, side="left"))
        for j in (1, 2, 3):
            valid = k >= j
            out[f"browse_last{j}_duration_seconds"][idx[valid]] = duration[k[valid] - j]
        low = np.maximum(0, k - 5)
        out["browse_recent5_under5_fraction"][idx] = (quick_prefix[k] - quick_prefix[low]) / np.maximum(1, k-low)
        out["browse_fast_swipe_streak"][idx[nonzero]] = streak[k[nonzero] - 1]
        out["browse_last_start_age_seconds"][idx[nonzero]] = (t[nonzero] - starts[k[nonzero]-1]) / 1000
        fractions = action_prefix[k] / np.maximum(1, k)[:, None]
        for j, name in enumerate(ACTIONS):
            out[name][idx] = fractions[:, j]
        seg_start = current_start.copy()
        seg_visits = np.ones(len(t), dtype=np.int32)
        active = nonzero.copy()
        active[nonzero] = current_start[nonzero] - ends[k[nonzero]-1] <= 600_000
        seg_start[active] = np.minimum(current_start[active], segment_start[k[active]-1])
        seg_visits[active] = segment_count[k[active]-1] + 1
        out["browse_active_segment_seconds"][idx] = (t - seg_start) / 1000
        out["browse_active_segment_visits"][idx] = seg_visits
        completed_starts: list[int] = []
        cursor = 0
        for pos, time in zip(idx, t):
            while cursor < len(available) and available[cursor] < time:
                insort(completed_starts, int(starts[cursor]))
                cursor += 1
            for w in WINDOWS_SECONDS:
                out[f"browse_starts_{w}s"][pos] = len(completed_starts) - bisect_left(completed_starts, int(time - w * 1000))
    return pd.DataFrame(out, index=original_index)


def merge_user_room_intervals(sessions: pd.DataFrame) -> pd.DataFrame:
    """Union overlapping raw segments per room and user without future bridging."""
    if sessions.empty:
        return sessions[["room_id", "user_id", "start_ms", "end_ms"]].copy()
    x = sessions[sessions.end_ms.gt(sessions.start_ms)].sort_values(
        ["room_id", "user_id", "start_ms", "end_ms"], kind="stable").copy()
    keys = [x.room_id, x.user_id]
    previous = x.groupby(["room_id", "user_id"], sort=False).end_ms.cummax().groupby(keys, sort=False).shift()
    x["cluster"] = (previous.isna() | (x.start_ms > previous)).groupby(keys, sort=False).cumsum()
    return x.groupby(["room_id", "user_id", "cluster"], sort=False).agg(
        start_ms=("start_ms", "min"), end_ms=("end_ms", "max")).reset_index().drop(columns="cluster")


def room_features_for_room(points: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """At t, count other sampled users with events strictly before t."""
    out = np.zeros((len(points), len(ROOM)), dtype=np.float32)
    if intervals.empty:
        return pd.DataFrame(out, columns=ROOM, index=points.index)
    starts = np.sort(intervals.start_ms.to_numpy(dtype=np.int64))
    ends = np.sort(intervals.end_ms.to_numpy(dtype=np.int64))
    t = points.prediction_time_ms.to_numpy(dtype=np.int64)
    start_before = np.searchsorted(starts, t, side="left")
    end_before = np.searchsorted(ends, t, side="left")
    out[:, 0] = start_before - end_before
    out[:, 1] = start_before - np.searchsorted(starts, t - 60_000, side="left")
    out[:, 2] = end_before - np.searchsorted(ends, t - 60_000, side="left")
    for user, group in points.groupby("user_id", sort=False):
        own = intervals[intervals.user_id.eq(user)]
        if own.empty:
            continue
        loc = points.index.get_indexer(group.index)
        u_t = group.prediction_time_ms.to_numpy(dtype=np.int64)
        own_starts = np.sort(own.start_ms.to_numpy(dtype=np.int64))
        own_ends = np.sort(own.end_ms.to_numpy(dtype=np.int64))
        own_before = np.searchsorted(own_starts, u_t, side="left")
        own_ended = np.searchsorted(own_ends, u_t, side="left")
        out[loc, 0] -= own_before - own_ended
        out[loc, 1] -= own_before - np.searchsorted(own_starts, u_t - 60_000, side="left")
        out[loc, 2] -= own_ended - np.searchsorted(own_ends, u_t - 60_000, side="left")
    out[:, 3] = out[:, 1] - out[:, 2]
    # Coverage must itself be observable at t. A room with only later sampled
    # records cannot be marked covered for an earlier prediction.
    out[:, 4] = np.searchsorted(starts, t, side="right") > 0
    if (out[:, :3] < 0).any():
        raise AssertionError("negative room audience count")
    return pd.DataFrame(out, columns=ROOM, index=points.index)


def r2_room_visits(raw: pd.DataFrame) -> pd.DataFrame:
    # Room observations intentionally retain only identity, time, and action counts.
    keys = ["raw_row_number"]
    raw = raw.drop_duplicates(keys, keep="first").copy()
    start = pd.to_datetime(raw.live_play_start_timestamp, format="mixed", utc=True, errors="coerce").dt.as_unit("ns")
    end = pd.to_datetime(raw.live_play_end_timestamp, format="mixed", utc=True, errors="coerce").dt.as_unit("ns")
    raw["start_ms"] = start.astype("int64") // 1_000_000
    raw["end_ms"] = end.astype("int64") // 1_000_000
    raw = raw[(raw.start_ms > 0) & (raw.end_ms > raw.start_ms)].copy()
    raw["room_id"] = raw.live_id.astype(str)
    # Only union overlapping/contiguous fragments. Bridging a positive gap using
    # a later re-entry would rewrite audience/exits at times inside that gap.
    return merge_user_room_intervals(raw[["room_id", "user_id", "start_ms", "end_ms"]])


# Compatibility alias: there is one event counting implementation.
from .events import asof_action_counts


def add_room_audience(points: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        for name in ROOM:
            points[name] = 0.0
        return points
    intervals = r2_room_visits(raw)
    by_room = {str(room): group for room, group in intervals.groupby("room_id", sort=False)}
    result = np.zeros((len(points), len(ROOM)), dtype=np.float32)
    for room, indices in points.groupby("room_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        p = points.iloc[idx][["user_id", "prediction_time_ms"]].reset_index(drop=True)
        own = by_room.get(str(room), intervals.iloc[0:0])
        if own.empty:
            continue
        t = p.prediction_time_ms.to_numpy(dtype=np.int64)
        starts = np.sort(own.start_ms.to_numpy(dtype=np.int64)); ends = np.sort(own.end_ms.to_numpy(dtype=np.int64))
        starts_before = np.searchsorted(starts, t, side="left")
        ends_before = np.searchsorted(ends, t, side="left")
        counts = np.zeros((len(p), len(ROOM)), dtype=np.float32)
        counts[:,0] = starts_before - ends_before
        counts[:,1] = starts_before - np.searchsorted(starts,t-60_000,side="left")
        counts[:,2] = ends_before - np.searchsorted(ends,t-60_000,side="left")
        counts[:,3] = counts[:,1]-counts[:,2]
        counts[:,4] = starts_before > 0
        # Subtract the query user's own sampled-room intervals using one grouped index.
        user_arrays = {str(user):(np.sort(g.start_ms.to_numpy(dtype=np.int64)),
                                  np.sort(g.end_ms.to_numpy(dtype=np.int64)))
                       for user,g in own.groupby('user_id',sort=False)}
        for user, subindices in p.groupby('user_id',sort=False).groups.items():
            user_values = user_arrays.get(str(user))
            if user_values is None: continue
            local = np.asarray(list(subindices),dtype=np.int64); ut=t[local]
            us,ue=user_values
            sb=np.searchsorted(us,ut,side='left'); eb=np.searchsorted(ue,ut,side='left')
            counts[local,0]-=sb-eb
            counts[local,1]-=sb-np.searchsorted(us,ut-60_000,side='left')
            counts[local,2]-=eb-np.searchsorted(ue,ut-60_000,side='left')
            counts[local,3]=counts[local,1]-counts[local,2]
        result[idx] = counts
    room_frame = pd.DataFrame(result, columns=ROOM, index=points.index)
    for j, name in enumerate(ROOM):
        points[name] = result[:, j]
    return points
