"""Evaluate fixed-parameter learning curves on the untouched B cohort."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_final_ablation import (EDGES, EXP, FEATURES, RESULT as ABLATION_RESULT,
                                metric_records, model_fit, sha256)
from kuailive_exit.features.groups import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
from kuailive_exit.config import load_config
RESULT = load_config()["data"]["result_dir"] / "exp_learning_curve"


def main() -> None:
    started = time.time()
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    arrays_dir = EXP / "training_arrays"
    x = np.load(arrays_dir / "train_X.npy", mmap_mode="r")
    y = np.load(arrays_dir / "train_y.npy", mmap_mode="r")
    users = np.load(arrays_dir / "train_users.npy", mmap_mode="r")
    ordered = (EXP / "cohorts/A_users.txt").read_text().splitlines()
    fixed = {k: config["repaired"]["fixed"][k] for k in
             ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction")}
    records = []
    model_dir = EXP / "learning_curve_models"; model_dir.mkdir(parents=True, exist_ok=True)
    for n_users in yaml.safe_load((ROOT / "configs/exp_learning_curve.yaml").read_text())["training_user_counts"]:
        mask = users < n_users
        for name in ("M1", "M1_browse_actions"):
            key = "M1_browse" if name == "M1_browse_actions" else name
            cols = FEATURES[name]
            start = time.time()
            if n_users == len(ordered):
                model = joblib.load(EXP / "models" / f"{name}_fixed.joblib")
            else:
                model = model_fit(x[mask], y[mask], np.load(arrays_dir / "validation_X.npy", mmap_mode="r"),
                    np.load(arrays_dir / "validation_y.npy", mmap_mode="r"), cols, fixed,
                    config["repaired"]["fixed"]["n_estimators"], config["repaired"]["fixed"]["early_stopping_rounds"],
                    config["seed"], config["threads"])
            path = model_dir / f"{key}_n{n_users}.joblib"; joblib.dump(model, path)
            new_records = metric_records(f"{key}_n{n_users}", model, cols, "B", "test")
            for r in new_records:
                r["training_users"] = n_users
                r["best_iteration"] = int(model.best_iteration_)
                r["training_seconds"] = time.time() - start
            records.extend(new_records)
            print(f"learning curve completed: {key}, {n_users} users, best iteration {model.best_iteration_}", flush=True)
            del model
    metrics = pd.DataFrame(records)
    RESULT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULT / "metrics_long.csv", index=False)
    summary = {"peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / ((1024**3) if __import__("sys").platform == "darwin" else (1024**2)),
               "elapsed_seconds": time.time() - started,
               "models": metrics[metrics.task.eq("H30")].to_dict(orient="records"),
               "fixed_parameters": fixed, "train_user_order_sha256": sha256(EXP / "cohorts/A_users.txt"),
               "input_feature_manifest_sha256": sha256(EXP / "feature_manifest.json")}
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    h = metrics[metrics.task.eq("H30")]
    h = h.assign(family=h.model.str.replace(r"_n\d+$", "", regex=True))
    for model in h.family.unique():
        s = h[h.family.eq(model)].sort_values("training_users")
        ax.plot(s.training_users, s.log_loss, marker="o", label=model)
    ax.set_xscale("log"); ax.set_xlabel("A training users"); ax.set_ylabel("B H30 log loss")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULT / "learning_curve.png", dpi=160)
    (RESULT / "run_manifest.json").write_text(json.dumps({"git_commit": __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if (ROOT / ".git").exists() else None,
        "code": "scripts/run_learning_curve.py", "input_feature_manifest_sha256": sha256(EXP / "feature_manifest.json"),
        "elapsed_seconds": summary["elapsed_seconds"], "peak_rss_gib": summary["peak_rss_gib"]}, indent=2))


if __name__ == "__main__":
    main()
