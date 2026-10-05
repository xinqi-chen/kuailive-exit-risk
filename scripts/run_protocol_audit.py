"""Create the protocol audit outputs from the freshly migrated A visits."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kuailive_exit.cleaning.visits import reclassify_causes
from kuailive_exit.config import load_config, utc_ms


ROOT = Path(__file__).resolve().parents[1]


def risk_sensitivity(visits: pd.DataFrame, cfg: dict) -> list[dict]:
    """Count evaluable H30 points and positives without materializing huge tables."""
    import numpy as np

    offsets = np.arange(cfg["landmarks"]["first_offset_seconds"],
                        cfg["landmarks"]["max_offset_seconds"] + 1,
                        cfg["landmarks"]["step_seconds"], dtype=np.int64)
    starts = visits.session_start_ms.to_numpy(dtype=np.int64)
    ends = visits.session_end_ms.to_numpy(dtype=np.int64)
    b = pd.to_numeric(visits.room_end_ms, errors="coerce").to_numpy(dtype=float)
    ts = starts[:, None] + offsets[None, :] * 1000
    at_risk = (ts < ends[:, None]) & (np.isnan(b[:, None]) | (ts < b[:, None]))
    quality = visits.quality_ok.to_numpy(dtype=bool)[:, None]
    split_end = utc_ms(cfg["split"]["train_end_utc_exclusive"])
    observation_end = utc_ms(cfg["observation"]["verified_end_utc_exclusive"])
    eligible_base = at_risk & quality & ((ts + 35_000) < split_end) & ((ts + 35_000) <= observation_end)
    cause_original = visits.end_cause.to_numpy(dtype=object)
    obs_censored = cause_original == "observation_censored"
    censor_at = pd.to_numeric(visits.r1_censor_time_ms, errors="coerce").fillna(visits.session_end_ms).to_numpy(dtype=np.int64)
    eligible_base &= ~(obs_censored[:, None] & ((ts + 35_000) > censor_at[:, None]))
    overlap_bad = np.zeros_like(at_risk)
    for i, values in enumerate(visits.overlap_start_times_ms):
        if isinstance(values, (list, tuple, np.ndarray)) and len(values):
            a = np.asarray(values, dtype=np.int64)
            overlap_bad[i] = np.any((ts[i, :, None] < a[None, :]) & (a[None, :] <= ts[i, :, None] + 35_000), axis=1)
    uncertain_end = pd.to_numeric(visits.overlap_uncertain_end_ms, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    yield_rows = []
    for dm in (2, 5, 10):
        for dp in (30, 60, 120):
            changed = visits.copy()
            reclassify_causes(changed, dm * 1000, dp * 1000)
            cause = changed.end_cause.to_numpy(dtype=object)
            counts = changed.end_cause.value_counts()
            event = pd.to_numeric(changed.event_time_ms, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            valid_visit = ~np.isin(cause, ["record_anomaly", "entered_after_end", "ambiguous_room_end"])
            uncertain = (uncertain_end[:,None] >= 0) & (ts + 35_000 >= uncertain_end[:,None])
            common = eligible_base & valid_visit[:, None] & ~uncertain & ~overlap_bad
            records = []
            for target in ("E2-V", "E2-A"):
                valid = common.copy()
                positive = (event[:, None] > ts) & (event[:, None] <= ts + 30_000)
                broadcast = cause == "broadcast"
                if target == "E2-V":
                    valid &= (~broadcast[:, None] | (ts + 30_000 < b[:, None] - dm * 1000))
                else:
                    broadcast_positive = broadcast[:, None] & (b[:, None] > ts) & (b[:, None] <= ts + 30_000)
                    positive = np.where(broadcast[:, None], broadcast_positive, positive)
                valid &= obs_censored[:, None] | (event[:, None] >= 0) | (broadcast[:, None] & (target == "E2-V"))
                n = int(valid.sum())
                pos = int((positive & valid).sum())
                records.append({"target": target, "evaluable_h30_points": n,
                                "positive_count": pos, "positive_rate": pos / n if n else None})
            for row in records:
                row.update({"delta_minus_seconds": dm, "delta_plus_seconds": dp})
                row.update({"cause_user": int(counts.get("user", 0)),
                            "cause_broadcast": int(counts.get("broadcast", 0)),
                            "cause_record_anomaly": int(counts.get("record_anomaly", 0))})
                yield_rows.append(row)
    return yield_rows


def main() -> None:
    started = time.time()
    cfg = load_config()
    stage = cfg["data"]["run_dir"]
    files = sorted((stage / "visits" / "A").glob("*.parquet"))
    visits = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    visits = visits.loc[visits.split.eq("train")].copy()
    room_known = pd.to_numeric(visits.room_end_ms, errors="coerce").notna()
    lag = (pd.to_numeric(visits.room_end_ms, errors="coerce") - visits.session_end_ms) / 1000
    valid_lag = lag[room_known & visits.quality_ok]
    lag_bins = [-float("inf"), -120, -60, -30, -10, 0, 10, 30, 60, 120, 300, float("inf")]
    hist = pd.cut(valid_lag, lag_bins, right=False).value_counts(sort=False)
    lag_counts = {str(k): int(v) for k, v in hist.items()}
    gaps = visits.sort_values(["user_id", "room_id", "session_start_ms"])
    gaps["previous_end"] = gaps.groupby(["user_id", "room_id"], sort=False).session_end_ms.shift()
    gap_s = (gaps.session_start_ms - gaps.previous_end) / 1000
    reentries = gap_s[(gap_s >= 0) & gap_s.notna()]
    out = cfg["data"]["result_dir"] / "exp_protocol_audit"
    out.mkdir(parents=True, exist_ok=True)
    sensitivity = risk_sensitivity(visits, cfg)
    summary = {"cohort": "A", "split": "train", "visits": int(len(visits)),
               "cause_counts": visits.end_cause.value_counts().to_dict(),
               "room_end_known": int(room_known.sum()),
               "room_end_missing": int((~room_known).sum()),
               "r0_r4": {"duplicates": int(visits.exact_duplicate.sum()),
                         "cross_day_unmatched": int(visits.r1_unmatched_midnight.sum()),
                         "overlap_involved": int(visits.overlap_involved.sum()),
                         "severe_overlap": int(visits.severe_overlap_previous.sum()),
                         "quality_ok": int(visits.quality_ok.sum())},
               "down_lag_seconds_histogram": lag_counts,
               "same_room_reentry_gap_seconds": {"n": int(len(reentries)),
                   "median": float(reentries.median()) if len(reentries) else None,
                   "p90": float(reentries.quantile(.9)) if len(reentries) else None,
                   "p99": float(reentries.quantile(.99)) if len(reentries) else None},
               "delta_sensitivity": sensitivity,
               "count_feature_note": "The v1.7 predictor feature list includes fixed 24h/7d counts; cumulative user_history_count is not a model feature.",
               "elapsed_seconds": time.time() - started}
    pd.DataFrame(sensitivity).to_csv(out / "delta_sensitivity.csv", index=False)
    pd.DataFrame([{"metric": "down_lag_seconds", "bin": k, "count": v}
                  for k, v in lag_counts.items()]).to_csv(out / "metrics_long.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    input_files = [stage / "r0_r4_manifest.json", *files]
    digests = {}
    for path in input_files:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        digests[path.name] = h.hexdigest()
    (out / "run_manifest.json").write_text(json.dumps({
        "experiment": "exp_protocol_audit", "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if (ROOT / ".git").exists() else None,
        "config": {"cohort": "A", "split": "train", "delta_minus_seconds": [2, 5, 10],
                   "delta_plus_seconds": [30, 60, 120]},
        "input_sha256": digests, "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(lag_counts)), list(lag_counts.values()))
    plt.xticks(range(len(lag_counts)), list(lag_counts.keys()), rotation=40, ha="right")
    plt.ylabel("Quality visits")
    plt.title("Difference between recorded room end and viewer visit end")
    plt.tight_layout()
    plt.savefig(out / "down_lag.png", dpi=160)


if __name__ == "__main__":
    main()
