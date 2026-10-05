"""Run the frozen corrected-model ablation; C is a separate locked phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib.pyplot as plt

from kuailive_exit.features.browsing import ACTIONS, LOCAL_TIME, PERSONAL, ROOM
from kuailive_exit.features.content import CONTENT_META, EMBEDDING_COLUMNS
from kuailive_exit.features.events import EVENT_FEATURES
from kuailive_exit.features.groups import ENTRY_CATEGORY_FEATURES, MODEL_FEATURES, NUMERIC, STATE
from kuailive_exit.models.lgbm import fit_classifier
from kuailive_exit.models.survival import forecast_h30
from kuailive_exit.evaluation.metrics import binary_metrics
from kuailive_exit.evaluation.report import landmark_auc_gain
from kuailive_exit.config import load_config


ROOT = Path(__file__).resolve().parents[1]
EXP = load_config()["data"]["run_dir"]
RESULT = load_config()["data"]["result_dir"] / "exp_final_ablation"
EDGES = json.loads((EXP / "feature_manifest.json").read_text())["edges_seconds"] if (EXP / "feature_manifest.json").exists() else []
from kuailive_exit.provenance import source_manifest, runtime_manifest
from kuailive_exit.evaluation.uncertainty import paired_user_bootstrap
FEATURES = {
    "M0": list(STATE),
    "M1": list(STATE) + NUMERIC[5:],
    "M1_browse_personal": list(STATE) + NUMERIC[5:] + PERSONAL,
    "M1_browse_time": list(STATE) + NUMERIC[5:] + PERSONAL + LOCAL_TIME,
    "M1_browse_room": list(STATE) + NUMERIC[5:] + PERSONAL + LOCAL_TIME + ROOM,
    "M1_browse_actions": list(STATE) + NUMERIC[5:] + PERSONAL + LOCAL_TIME + ROOM + ACTIONS,
    "M1_events": list(STATE) + NUMERIC[5:] + PERSONAL + LOCAL_TIME + ROOM + ACTIONS + EVENT_FEATURES,
    "M_all": list(MODEL_FEATURES["M_all"]),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reference_dir() -> Path:
    return EXP / "model_features"


def feature_columns() -> list[str]:
    return list(dict.fromkeys(MODEL_FEATURES["M_all"] + ["cohort", "split", "query_kind", "user_id", "visit_id", "room_id", "label",
        "interval_index", "offset_seconds", "prediction_time_ms", "room_elapsed_seconds",
        "browse_active_segment_seconds", "browse_last_start_age_seconds", "browse_local_hour",
        "browse_local_weekday", "embedding_age_seconds", "available_age_seconds"]))


def read_rows(cohort: str, split: str, kind: str) -> pd.DataFrame:
    import pyarrow.dataset as ds
    dataset = ds.dataset(reference_dir(), format="parquet")
    frame = dataset.to_table(columns=feature_columns(), filter=(ds.field("cohort") == cohort) &
                             (ds.field("split") == split) & (ds.field("query_kind") == kind)).to_pandas()
    if frame.empty:
        raise ValueError(f"No {cohort}/{split}/{kind} feature rows")
    return frame


def load_arrays() -> dict:
    """Materialize split arrays once, retaining user IDs for the learning curve."""
    out = EXP / "training_arrays"
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(["cohort", "split", "query_kind", "label", "user_id"] + MODEL_FEATURES["M_all"]))
    paths = sorted(reference_dir().glob("features_*.parquet"))
    result = {}
    user_order = (EXP / "cohorts/A_users.txt").read_text().splitlines()
    user_index = {u:i for i,u in enumerate(user_order)}
    for split in ("train", "validation"):
        prefix = "train" if split == "train" else "validation"
        count = 0
        for p in paths:
            for batch in pq.ParquetFile(p).iter_batches(batch_size=50000, columns=["cohort", "split", "query_kind"]):
                f = batch.to_pandas()
                count += int((f.cohort.eq("A") & f.split.eq(split) & f.query_kind.eq("interval")).sum())
        xpath, ypath, upath = (out / f"{prefix}_{suffix}.npy" for suffix in ("X", "y", "users"))
        x = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float32,
            shape=(count, len(MODEL_FEATURES["M_all"])))
        y = np.lib.format.open_memmap(ypath, mode="w+", dtype=np.int8, shape=(count,))
        users = np.lib.format.open_memmap(upath, mode="w+", dtype=np.int32, shape=(count,))
        cursor = 0
        for p in paths:
            for batch in pq.ParquetFile(p).iter_batches(batch_size=50000, columns=cols):
                f = batch.to_pandas()
                mask = f.cohort.eq("A") & f.split.eq(split) & f.query_kind.eq("interval")
                if not mask.any():
                    continue
                s = f.loc[mask]
                n = len(s)
                x[cursor:cursor+n] = s[MODEL_FEATURES["M_all"]].to_numpy(dtype=np.float32)
                y[cursor:cursor+n] = s.label.to_numpy(dtype=np.int8)
                users[cursor:cursor+n] = s.user_id.astype(str).map(user_index).to_numpy(dtype=np.int32)
                cursor += n
        if cursor != count:
            raise RuntimeError(f"{split}: streamed {cursor} rows, counted {count}")
        x.flush(); y.flush(); users.flush()
        result[prefix] = (xpath, ypath, upath, count)
    return result


def model_fit(xtr, ytr, xva, yva, names, params, rounds, patience, seed, threads):
    idx = [MODEL_FEATURES["M_all"].index(c) for c in names]
    return fit_classifier(xtr[:, idx], ytr, xva[:, idx], yva,
        {k: params[k] for k in ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction")},
        n_estimators=rounds, early_stopping_rounds=patience, seed=seed, threads=threads)


def metric_records(model_name: str, model, names: list[str], cohort: str, split: str,
                  placebo_columns: list[str] | None = None) -> list[dict]:
    points = read_rows(cohort, split, "H30")
    if placebo_columns:
        ix = np.random.default_rng(20261001 + sum(ord(c) for c in cohort + split + "H30")).permutation(len(points))
        points.loc[:, placebo_columns] = points[placebo_columns].to_numpy()[ix]
    prob = forecast_h30(model, points, EDGES, names, model_name, frozen_columns=placebo_columns)
    primary = points.offset_seconds.le(570).to_numpy()
    pred_dir = EXP / "predictions"; pred_dir.mkdir(exist_ok=True)
    prediction = points.loc[primary, ["user_id", "visit_id", "prediction_time_ms", "offset_seconds", "label"]].copy()
    prediction["probability"] = prob[primary]
    prediction.to_parquet(pred_dir / f"{cohort}_{model_name}.parquet", index=False)
    h30 = binary_metrics(points.label.to_numpy(dtype=np.int8)[primary], prob[primary])
    result = [{"model": model_name, "cohort": cohort, "split": split, "task": "H30", "max_landmark_seconds": 570,
               "elapsed_seconds": "all", **h30},
              {"model": model_name, "cohort": cohort, "split": split, "task": "H30_all",
               "max_landmark_seconds": 600, "elapsed_seconds": "all", **binary_metrics(points.label, prob)}]
    for landmark, group in points.groupby("offset_seconds", sort=True):
        if int(landmark) > 570:
            continue
        take = group.index.to_numpy()
        result.append({"model": model_name, "cohort": cohort, "split": split, "task": "H30_landmark",
                       "elapsed_seconds": int(landmark), **binary_metrics(group.label, prob[points.index.get_indexer(take)])})
    interval = read_rows(cohort, split, "interval")
    if placebo_columns:
        ix = np.random.default_rng(20261001 + sum(ord(c) for c in cohort + split + "interval")).permutation(len(interval))
        interval.loc[:, placebo_columns] = interval[placebo_columns].to_numpy()[ix]
    chunks = []
    for start in range(0, len(interval), 25000):
        chunks.append(model.predict_proba(interval.iloc[start:start+25000][names].to_numpy(dtype=np.float32))[:, 1])
    p = np.concatenate(chunks)
    result.append({"model": model_name, "cohort": cohort, "split": split, "task": "interval", "interval_index": "all", **binary_metrics(interval.label, p)})
    interval_prob = pd.Series(p, index=interval.index)
    for index, group in interval.groupby("interval_index", sort=True):
        take = group.index.to_numpy()
        result.append({"model": model_name, "cohort": cohort, "split": split, "task": "interval_bin",
                       "interval_index": int(index), **binary_metrics(group.label, interval_prob.loc[take].to_numpy())})
    return result


def fit_all() -> dict:
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    repaired = config["repaired"]
    arrays = load_arrays()
    xtr = np.load(arrays["train"][0], mmap_mode="r"); ytr = np.load(arrays["train"][1], mmap_mode="r")
    xva = np.load(arrays["validation"][0], mmap_mode="r"); yva = np.load(arrays["validation"][1], mmap_mode="r")
    models_dir = EXP / "models"; models_dir.mkdir(parents=True, exist_ok=True)
    training: dict = {"arrays": {k: {"rows": v[3]} for k, v in arrays.items()}, "models": {}, "tuning_candidates": []}
    fixed_params = {k: repaired["fixed"][k] for k in
                    ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction")}
    for name, cols in FEATURES.items():
        start = time.time()
        model = model_fit(xtr, ytr, xva, yva, cols, fixed_params,
            repaired["fixed"]["n_estimators"], repaired["fixed"]["early_stopping_rounds"], config["seed"], config["threads"])
        p = models_dir / f"{name}_fixed.joblib"; joblib.dump(model, p)
        training["models"][f"{name}_fixed"] = {"best_iteration": int(model.best_iteration_), "sha256": sha256(p),
            "seconds": time.time()-start, "params": fixed_params}
        iteration = training["models"][f"{name}_fixed"]["best_iteration"]
        print(f"trained {name}_fixed: iteration {iteration}", flush=True)
        del model
    for name in repaired["tuned_models"]:
        cols = FEATURES["M_all"]
        params = repaired["tuned_grid"]["params"]
        start = time.time()
        model = model_fit(xtr, ytr, xva, yva, cols, params,
            repaired["tuned_grid"]["n_estimators"], repaired["tuned_grid"]["early_stopping_rounds"], config["seed"], config["threads"])
        prob = np.concatenate([model.predict_proba(xva[j:j+25000, [MODEL_FEATURES["M_all"].index(c) for c in cols]])[:, 1]
                               for j in range(0, len(xva), 25000)])
        loss = binary_metrics(yva, prob)["log_loss"]
        row = {"model": name, "candidate": repaired["tuned_grid"]["selected_candidate"],
               "params": params, "validation_log_loss": loss,
               "best_iteration": int(model.best_iteration_), "seconds": time.time()-start,
               "candidate_selection_source": repaired["tuned_grid"]["params_source"]}
        training["tuning_candidates"].append(row)
        path = models_dir / f"{name}.joblib"; joblib.dump(model, path)
        training["models"][name] = {**row, "sha256": sha256(path)}
        print(f"trained {name}: iteration {model.best_iteration_}", flush=True)
        del model
    placebo_blocks = [("M1_browse_personal", PERSONAL), ("M1_browse_time", LOCAL_TIME),
                      ("M1_browse_room", ROOM), ("M1_browse_actions", ACTIONS),
                      ("M1_events", EVENT_FEATURES),
                      ("M_all", CONTENT_META + EMBEDDING_COLUMNS)]
    previous = list(STATE)
    for model_key, block in placebo_blocks:
        cols = FEATURES[model_key]
        ix = [MODEL_FEATURES["M_all"].index(c) for c in cols]
        xtrp = np.asarray(xtr[:, ix]).copy(); xvap = np.asarray(xva[:, ix]).copy()
        block_ix = [cols.index(c) for c in block]
        xtr_perm = np.random.default_rng(config["seed"] + 101).permutation(len(xtrp))
        xva_perm = np.random.default_rng(config["seed"] + 202).permutation(len(xvap))
        xtrp[:, block_ix] = xtrp[xtr_perm][:, block_ix]
        xvap[:, block_ix] = xvap[xva_perm][:, block_ix]
        model = fit_classifier(xtrp, ytr, xvap, yva, fixed_params,
            n_estimators=repaired["fixed"]["n_estimators"],
            early_stopping_rounds=repaired["fixed"]["early_stopping_rounds"],
            seed=config["seed"], threads=config["threads"])
        path = models_dir / f"{model_key}_placebo.joblib"; joblib.dump(model, path)
        training["models"][f"{model_key}_placebo"] = {"best_iteration": int(model.best_iteration_),
            "sha256": sha256(path), "permuted_block": block}
        print(f"trained {model_key}_placebo: iteration {model.best_iteration_}", flush=True)
        del model, xtrp, xvap
    return training


def collect_metrics(phase: str, training: dict) -> pd.DataFrame:
    rows = []
    model_defs = {f"{name}_fixed": (name, cols) for name, cols in FEATURES.items()}
    model_defs["M_all_tuned"] = ("M_all", FEATURES["M_all"])
    placebo_blocks = {"M1_browse_personal": PERSONAL, "M1_browse_time": LOCAL_TIME,
                      "M1_browse_room": ROOM, "M1_browse_actions": ACTIONS,
                      "M1_events": EVENT_FEATURES, "M_all": CONTENT_META + EMBEDDING_COLUMNS}
    for key in placebo_blocks:
        model_defs[f"{key}_placebo"] = (key, FEATURES[key])
    cohorts = ["B"] if phase == "train-b" else ["C"]
    for key, (forecast_name, cols) in model_defs.items():
        model_path = EXP / "models" / (f"{key}.joblib")
        model = joblib.load(model_path)
        for cohort in cohorts:
            base_key = key[:-8] if key.endswith("_placebo") else key
            rows.extend(metric_records(key, model, cols, cohort, "test",
                                       placebo_blocks.get(base_key) if key.endswith("_placebo") else None))
        del model
    return pd.DataFrame(rows)


def train_b() -> None:
    if (EXP / "training_manifest.json").exists() or (RESULT / "pre_c_freeze_manifest.json").exists():
        raise FileExistsError("Training is immutable; choose a fresh run ID for another fit")
    if not EDGES:
        raise FileNotFoundError("Build raw features first: python scripts/build_features.py")
    started = time.time(); run = fit_all()
    (EXP / "training_manifest.json").write_text(json.dumps(run, indent=2))
    metrics = collect_metrics("train-b", run)
    RESULT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULT / "metrics_long.csv", index=False)
    summary = {"metrics": metrics.to_dict(orient="records"),
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / ((1024**3) if __import__("sys").platform == "darwin" else (1024**2)),
               "elapsed_seconds": time.time()-started,
               "history_count_features": "fixed 24h/7d counts; cumulative user_history_count is excluded",
               "runtime": runtime_manifest(),
               "auc_gain_summary": {}}
    by_landmark = metrics[(metrics.task == "H30_landmark") & (metrics.cohort == "B")]
    for model in ("M1_browse_actions_fixed", "M_all_fixed", "M_all_tuned"):
        summary["auc_gain_summary"][model] = landmark_auc_gain(by_landmark, model, "M1_fixed")
    (RESULT / "B_summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = metrics[(metrics.task == "H30") & (metrics.cohort == "B")]
    ax.bar(subset.model, subset.log_loss); ax.set_ylabel("H30 log loss")
    ax.tick_params(axis="x", labelrotation=50); fig.tight_layout()
    fig.savefig(RESULT / "B_h30_logloss.png", dpi=160)
    input_hashes = {p.name: sha256(p) for p in sorted(reference_dir().glob("features_*.parquet"))}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if (ROOT / ".git").exists() else None
    freeze = {"status": "B_complete_current_run_C_not_evaluated", "git_commit": git_commit,
              "source": source_manifest(ROOT), "run_id": EXP.name,
              "models": run["models"], "feature_manifest_sha256": sha256(EXP / "feature_manifest.json"),
              "C_history": "held-out test cohort; not used for training, early stopping or selection", "edges_seconds": EDGES,
              "features": FEATURES, "config": yaml.safe_load((ROOT / "configs/models.yaml").read_text())["repaired"],
              "input_feature_shards_sha256": input_hashes, "training_manifest_sha256": sha256(EXP / "training_manifest.json"),
              "B_metrics_sha256": sha256(RESULT / "metrics_long.csv"), "C_evaluation_count": 0,
              "started_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (RESULT / "pre_c_freeze_manifest.json").write_text(json.dumps(freeze, indent=2))
    (RESULT / "run_manifest.json").write_text(json.dumps({
        "experiment": "exp_final_ablation", "git_commit": git_commit,
        "config": freeze["config"], "feature_groups": FEATURES,
        "input_feature_shards_sha256": input_hashes,
        "training_manifest_sha256": freeze["training_manifest_sha256"],
        "B_metrics_sha256": freeze["B_metrics_sha256"],
        "C_evaluation_count": 0, "elapsed_seconds": summary["elapsed_seconds"],
        "peak_rss_gib": summary["peak_rss_gib"]}, indent=2))
    bootstrap_results("B")
    print(json.dumps({"status":"B_completed", "freeze_manifest":str(RESULT / "pre_c_freeze_manifest.json")}, indent=2))


def eval_c() -> None:
    freeze_path = RESULT / "pre_c_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    marker = RESULT / "C_evaluation_started.json"
    if marker.exists() or (RESULT / "C_metrics_long.csv").exists():
        raise RuntimeError("C evaluation is one-shot and has already started")
    if source_manifest(ROOT)["sha256"] != freeze["source"]["sha256"]:
        raise RuntimeError("Executable source/config changed after B freeze")
    if sha256(EXP / "feature_manifest.json") != freeze["feature_manifest_sha256"]:
        raise RuntimeError("Feature manifest changed after B freeze")
    if sha256(EXP / "training_manifest.json") != freeze["training_manifest_sha256"]:
        raise RuntimeError("Training manifest changed after B freeze")
    for name, detail in freeze["models"].items():
        if sha256(EXP / "models" / f"{name}.joblib") != detail["sha256"]:
            raise RuntimeError(f"Model changed after freeze: {name}")
    for name, digest in freeze["input_feature_shards_sha256"].items():
        if sha256(reference_dir() / name) != digest:
            raise RuntimeError(f"corrected input feature shard changed: {name}")
    if sha256(RESULT / "metrics_long.csv") != freeze["B_metrics_sha256"]:
        raise RuntimeError("B metrics changed after freeze")
    marker.write_text(json.dumps({"started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                  "freeze_manifest_sha256": sha256(freeze_path)}, indent=2))
    dummy = {"models": {}}
    metrics = collect_metrics("eval-c", dummy)
    metrics.to_csv(RESULT / "C_metrics_long.csv", index=False)
    by_landmark = metrics[(metrics.task == "H30_landmark") & (metrics.cohort == "C")]
    c_summary = {"metrics": metrics.to_dict(orient="records"), "C_evaluation_count": 1,
                 "prior_C_uses_minimum": 3,
                 "evaluation_scope": "single frozen evaluation on held-out test cohort C",
                 "auc_gain_summary": {model: landmark_auc_gain(by_landmark, model, "M1_fixed")
                     for model in ("M1_browse_actions_fixed", "M_all_fixed", "M_all_tuned")}}
    (RESULT / "C_summary.json").write_text(json.dumps(c_summary, indent=2))
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = metrics[(metrics.task == "H30") & (metrics.cohort == "C")]
    ax.bar(subset.model, subset.log_loss); ax.set_ylabel("H30 log loss (t ≤ 570 s)")
    ax.tick_params(axis="x", labelrotation=50); fig.tight_layout()
    fig.savefig(RESULT / "C_h30_logloss.png", dpi=160)
    marker.write_text(json.dumps({"completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                  "freeze_manifest_sha256": sha256(freeze_path),
                                  "C_metrics_sha256": sha256(RESULT / "C_metrics_long.csv")}, indent=2))
    run_manifest = json.loads((RESULT / "run_manifest.json").read_text())
    run_manifest.update({"C_metrics_sha256": sha256(RESULT / "C_metrics_long.csv"),
                         "C_summary_sha256": sha256(RESULT / "C_summary.json"),
                         "C_evaluation_count": 1})
    (RESULT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    bootstrap_results("C")
    print("C post-audit evaluation completed", flush=True)



def bootstrap_results(cohort):
    baseline = pd.read_parquet(EXP / "predictions" / f"{cohort}_M1_fixed.parquet")
    keys = ["user_id", "visit_id", "prediction_time_ms", "offset_seconds"]
    records = []
    for path in sorted((EXP / "predictions").glob(f"{cohort}_*.parquet")):
        name = path.stem[len(cohort)+1:]
        candidate = pd.read_parquet(path)
        if len(candidate) != len(baseline) or not candidate[keys].equals(baseline[keys]) or not candidate.label.equals(baseline.label):
            raise AssertionError(f"Unpaired evaluation rows: {name}")
        records.append({"model":name,"baseline":"M1_fixed","cohort":cohort,
            **paired_user_bootstrap(baseline.user_id, baseline.label, candidate.probability, baseline.probability)})
    fixed_order = [f"{name}_fixed" for name in FEATURES] + ["M_all_tuned"]
    for before, after in zip(fixed_order[:-1], fixed_order[1:]):
        b = pd.read_parquet(EXP / "predictions" / f"{cohort}_{before}.parquet")
        c = pd.read_parquet(EXP / "predictions" / f"{cohort}_{after}.parquet")
        if not b[keys].equals(c[keys]) or not b.label.equals(c.label):
            raise AssertionError("Non-matching nested ablation predictions")
        records.append({"model":after,"baseline":before,"cohort":cohort,
            **paired_user_bootstrap(b.user_id,b.label,c.probability,b.probability)})
    (RESULT / f"{cohort}_user_bootstrap.json").write_text(json.dumps(records, indent=2))
    print(f"{cohort} paired user bootstrap complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train-b", "eval-c"], required=True)
    args = parser.parse_args()
    train_b() if args.phase == "train-b" else eval_c()


if __name__ == "__main__":
    main()
