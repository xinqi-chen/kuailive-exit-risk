"""Draw the README figure from the committed interval metrics.

Usage: python docs/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/formal-v1.1.0"
OUT = Path(__file__).resolve().parent
EDGES = json.loads((RES / "feature_manifest.json").read_text())["edges_seconds"]
STEPS = [  # (candidate, baseline, label, color)
    ("M1_browse_personal_fixed", "M1_fixed", "browsing rhythm", "#2a78d6"),
    ("M1_browse_room_fixed", "M1_browse_time_fixed", "co-viewers", "#eb6834"),
    ("M1_events_fixed", "M1_browse_actions_fixed", "recent likes/comments", "#1baf7a"),
    ("M_all_fixed", "M1_events_fixed", "content embedding", "#eda100"),
]
INK2, GRID = "#52514e", "#e4e3df"


def bins(path):
    m = pd.read_csv(path)
    b = m[m.task.eq("interval_bin")].copy()
    b["interval_index"] = b.interval_index.astype(int)
    return b.pivot(index="interval_index", columns="model", values="log_loss")


ll = bins(RES / "exp_final_ablation/C_metrics_long.csv")
labels = [f"{EDGES[i]}–{EDGES[i + 1]}" for i in ll.index]
x = range(len(labels))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=170, gridspec_kw={"width_ratios": [1, 1.35]})
hist = 1000 * (ll["M1_fixed"] - ll["M0_fixed"])
a1.bar(x, hist, width=0.7, color="#2a78d6")
a1.set_title("Adding viewing history (M0 → M1)", fontsize=9.5, loc="left")
for i, (cand, base, lab, col) in enumerate(STEPS):
    d = 1000 * (ll[cand] - ll[base])
    a2.plot(x, d, color=col, lw=2, marker="o", ms=4, label=lab)
a2.axhline(0, color="#8a8984", lw=1)
a2.set_title("Each later feature block vs the model before it", fontsize=9.5, loc="left")
a2.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
for ax in (a1, a2):
    ax.set_xticks(list(x), labels, rotation=60, fontsize=7.5)
    ax.set_xlabel("interval (seconds since entering)", fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
a1.set_ylabel("Change in log loss (×10⁻³; lower = better)", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "interval_gains.png")
print("wrote", OUT / "interval_gains.png")
