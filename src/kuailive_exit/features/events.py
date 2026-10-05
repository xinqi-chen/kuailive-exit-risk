"""As-of event features for likes and comments."""

from __future__ import annotations

import numpy as np
import pandas as pd


WINDOWS = (60, 300, 900)
EVENT_FEATURES = [f"{who}_{kind}_{window}s" for who in ("own", "room_other")
                  for kind in ("like", "comment") for window in WINDOWS]


def parse_event_time(series: pd.Series, *, milliseconds: bool) -> np.ndarray:
    if milliseconds:
        return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    parsed = pd.to_datetime(series, format="mixed", utc=True, errors="coerce").dt.as_unit("ns")
    result = parsed.astype("int64").to_numpy() // 1_000_000
    result[parsed.isna().to_numpy()] = -1
    return result


def asof_action_counts(points: pd.DataFrame, events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Count each event in [t-window, t), split into the current user and others."""
    if event_type not in {"like", "comment"}:
        raise ValueError("event_type must be like or comment")
    original_index = points.index
    points = points.reset_index(drop=True)
    names = [f"{who}_{event_type}_{window}s" for who in ("own", "room_other") for window in WINDOWS]
    out = np.zeros((len(points), len(names)), dtype=np.float32)
    if "room_id" in points and "room_id" in events:
        by_room = {str(room): group for room, group in events.groupby("room_id", sort=False)}
        for room, indices in points.groupby("room_id", sort=False).groups.items():
            idx = np.asarray(indices, dtype=np.int64)
            room_events = by_room.get(str(room), events.iloc[:0]).drop(columns="room_id")
            out[idx] = asof_action_counts(points.iloc[idx], room_events, event_type).to_numpy()
        return pd.DataFrame(out, columns=names, index=original_index)
    if events.empty:
        return pd.DataFrame(out, columns=names, index=original_index)
    event_times = np.sort(events.event_ms.to_numpy(dtype=np.int64))
    query_times = points.prediction_time_ms.to_numpy(dtype=np.int64)
    totals = {}
    for window in WINDOWS:
        lo = np.searchsorted(event_times, query_times - window * 1000, side="left")
        hi = np.searchsorted(event_times, query_times, side="left")
        totals[window] = hi - lo
    by_user = {str(user): np.sort(group.event_ms.to_numpy(dtype=np.int64))
               for user, group in events.groupby("user_id", sort=False)}
    for user, indices in points.groupby("user_id", sort=False).groups.items():
        own_times = by_user.get(str(user))
        if own_times is None:
            own_times = np.empty(0, dtype=np.int64)
        idx = np.asarray(list(indices), dtype=np.int64)
        times = query_times[idx]
        for j, window in enumerate(WINDOWS):
            own = (np.searchsorted(own_times, times, side="left") -
                   np.searchsorted(own_times, times - window * 1000, side="left"))
            out[idx, j] = own
            out[idx, j + len(WINDOWS)] = totals[window][idx] - own
    return pd.DataFrame(out, columns=names, index=original_index)
