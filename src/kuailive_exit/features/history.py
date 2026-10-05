"""Completed-visit history features (development stage D8 definitions)."""
from __future__ import annotations

import numpy as np
import pandas as pd

def add_visit_history(risk: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """R7: only completed, valid visits strictly before t; fixed-window counts."""
    if risk.empty:
        return risk
    frame = risk.copy()
    context = frame.drop_duplicates(["visit_id", "prediction_time_ms"]).copy()
    for name in ("user_history_count", "user_recent20_mean_duration_seconds",
                 "user_recent20_median_duration_seconds", "user_short_watch_fraction",
                 "user_count_24h", "user_count_7d", "user_last_end_age_seconds",
                 "author_count_7d", "author_mean_duration_seconds", "author_last_end_age_seconds",
                 "user_broadcast_fraction", "user_survival_30s", "user_survival_60s"):
        context[name] = 0.0
    hist = visits[(visits.quality_ok) & visits.end_cause.isin(["user", "broadcast"])].copy()
    hist = hist[hist.get("overlap_uncertain_end_ms", pd.Series(-1, index=hist.index)).lt(0)].copy()
    hist["_available_ms"] = hist.get("history_available_ms", hist.session_end_ms)
    hist = hist.sort_values(["_available_ms", "visit_id"], kind="stable")
    by_user = {u: g for u, g in hist.groupby("user_id", sort=False)}
    by_user_author = {(u, a): g for (u, a), g in hist.groupby(["user_id", "author_id"], sort=False)}
    for user, rows in context.groupby("user_id", sort=False):
        h = by_user.get(user)
        if h is None:
            continue
        ends = h.session_end_ms.to_numpy(dtype=np.int64)
        durations = h.reported_duration_ms.to_numpy(dtype=np.float64) / 1000
        authors = h.author_id.astype(str).to_numpy()
        causes = h.end_cause.to_numpy()
        ts = rows.prediction_time_ms.to_numpy(dtype=np.int64)
        k = np.searchsorted(h._available_ms.to_numpy(dtype=np.int64), ts, side="left")
        context.loc[rows.index, "user_history_count"] = k
        sum_d = np.r_[0.0, np.cumsum(durations)]
        short = np.r_[0, np.cumsum(durations < 30)]
        broad = np.r_[0, np.cumsum(causes == "broadcast")]
        lo20 = np.maximum(0, k - 20)
        den20 = np.maximum(1, k - lo20)
        context.loc[rows.index, "user_recent20_mean_duration_seconds"] = (sum_d[k] - sum_d[lo20]) / den20
        context.loc[rows.index, "user_short_watch_fraction"] = (short[k] - short[lo20]) / den20
        context.loc[rows.index, "user_broadcast_fraction"] = (broad[k] - broad[lo20]) / den20
        context.loc[rows.index, "user_count_24h"] = np.maximum(0, k - np.searchsorted(ends, ts - 86_400_000, side="left"))
        context.loc[rows.index, "user_count_7d"] = np.maximum(0, k - np.searchsorted(ends, ts - 7 * 86_400_000, side="left"))
        valid_k = k > 0
        context.loc[rows.index[valid_k], "user_last_end_age_seconds"] = (ts[valid_k] - ends[k[valid_k] - 1]) / 1000
        # Median is the only non-invertible rolling statistic; compute per context.
        for idx, kk in zip(rows.index, k):
            if kk:
                context.at[idx, "user_recent20_median_duration_seconds"] = float(np.median(durations[max(0, kk - 20):kk]))
        # Author history is a separate completed-visit stream, vectorized by author.
        for author, ar in rows.groupby("author_id", sort=False):
            own_hist = by_user_author.get((user, author))
            if own_hist is None:
                continue
            ae = own_hist.session_end_ms.to_numpy(dtype=np.int64)
            ad = own_hist.reported_duration_ms.to_numpy(dtype=np.float64) / 1000
            at = ar.prediction_time_ms.to_numpy(dtype=np.int64)
            ak = np.searchsorted(own_hist._available_ms.to_numpy(dtype=np.int64), at, side="left")
            context.loc[ar.index, "author_count_7d"] = np.maximum(0, ak - np.searchsorted(ae, at - 7 * 86_400_000, side="left"))
            sum_a = np.r_[0.0, np.cumsum(ad)]
            context.loc[ar.index, "author_mean_duration_seconds"] = np.divide(
                sum_a[ak], ak, out=np.zeros(len(ak), dtype=float), where=ak > 0)
            ok = ak > 0
            context.loc[ar.index[ok], "author_last_end_age_seconds"] = (at[ok] - ae[ak[ok] - 1]) / 1000
    # Exact event-time Kaplan-Meier on completed, causally available visits.
    # Broadcast endings are censored; fractional-second exits are events too.
    for user, rows in context.groupby("user_id", sort=False):
        h = by_user.get(user)
        if h is None or h.empty:
            continue
        event = h.end_cause.eq("user").to_numpy()
        surv_end = h.session_end_ms.to_numpy(dtype=np.int64).copy()
        broadcast = ~event
        surv_end[broadcast] = pd.to_numeric(h.room_end_ms, errors="coerce").to_numpy(dtype=float)[broadcast]
        duration = surv_end - h.session_start_ms.to_numpy(dtype=np.int64)
        ks = np.searchsorted(h._available_ms.to_numpy(dtype=np.int64),
                             rows.prediction_time_ms.to_numpy(dtype=np.int64), side="left")
        surv30, surv60 = prefix_kaplan_meier(duration, event, ks)
        context.loc[rows.index, "user_survival_30s"] = np.where(ks > 0, surv30, 0)
        context.loc[rows.index, "user_survival_60s"] = np.where(ks > 0, surv60, 0)
    feature_names = ["user_history_count", "user_recent20_mean_duration_seconds",
                     "user_recent20_median_duration_seconds", "user_short_watch_fraction",
                     "user_count_24h", "user_count_7d", "user_last_end_age_seconds",
                     "author_count_7d", "author_mean_duration_seconds",
                     "author_last_end_age_seconds", "user_broadcast_fraction", "user_survival_30s",
                     "user_survival_60s"]
    features = context[["visit_id", "prediction_time_ms"] + feature_names]
    frame = frame.drop(columns=[c for c in features if c not in {"visit_id", "prediction_time_ms"} and c in frame], errors="ignore")
    return frame.merge(features, on=["visit_id", "prediction_time_ms"], how="left", validate="many_to_one")



def prefix_kaplan_meier(durations, events, prefix_sizes):
    """Exact S(30), S(60) for history prefixes; bounded event-time chunks."""
    durations = np.asarray(durations, dtype=np.int64)
    events = np.asarray(events, dtype=bool)
    sizes, inverse = np.unique(prefix_sizes, return_inverse=True)
    times = np.unique(durations[events & (durations > 0) & (durations <= 60_000)])
    log30 = np.zeros(len(sizes)); log60 = np.zeros(len(sizes))
    for start in range(0, len(times), 128):
        grid = times[start:start + 128]
        risk = np.vstack([np.zeros((1, len(grid)), dtype=np.int32),
                          np.cumsum(durations[:, None] >= grid, axis=0, dtype=np.int32)])[sizes]
        exits = np.vstack([np.zeros((1, len(grid)), dtype=np.int32),
                           np.cumsum(events[:, None] & (durations[:, None] == grid),
                                     axis=0, dtype=np.int32)])[sizes]
        hazard = np.divide(exits, risk, out=np.zeros_like(exits, dtype=float), where=risk > 0)
        with np.errstate(divide="ignore"):
            terms = np.log1p(-hazard)
        log60 += terms.sum(axis=1)
        log30 += terms[:, grid <= 30_000].sum(axis=1)
    return np.exp(log30[inverse]), np.exp(log60[inverse])
