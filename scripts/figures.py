#!/usr/bin/env python3
"""README figure: how well standing dead trees are found, by canopy position.

    python scripts/figures.py   -> figures/visibility.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
b = json.load(open(ROOT / "results" / "visibility_bootstrap.json"))
v = json.load(open(ROOT / "results" / "visibility.json"))
L = ["sunlit", "partly shaded", "shaded"]
labels = ["Full sun", "Partly shaded", "Shaded"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
for k, (name, col, lab) in enumerate((("lidar_all", "#6f7f99", "LiDAR only"),
                                      ("lidar_plus_colour", "#2a7f62", "LiDAR + camera colour"))):
    y = np.arange(3) + (k - .5) * .26
    auc = [b[name][l]["roc_auc"] for l in L]
    lo = [b[name][l]["roc_auc"] - b[name][l]["ci95"][0] for l in L]
    hi = [b[name][l]["ci95"][1] - b[name][l]["roc_auc"] for l in L]
    ax.errorbar(auc, y, xerr=[lo, hi], fmt="o", color=col, ms=8, capsize=3, lw=1.6, label=lab)
    for a, h, yy in zip(auc, hi, y):
        ax.text(a + h + .008, yy, f"{a:.2f}", va="center", fontsize=9.5, color=col)
ax.axvline(.5, color="0.6", ls="--", lw=1); ax.text(.505, -.45, "chance", color="0.45", fontsize=9)
ax.set_yticks(range(3), labels, fontsize=11); ax.set_ylim(2.5, -.6)
ax.set_xlim(.45, 1); ax.set_xlabel("ROC AUC on unseen plots (bars: 95% plot bootstrap)")
ax.set_title("How well standing dead trees are found", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

dead = v["dead_by_light_by_site"]
left = np.zeros(3)
for site, col in (("SOAP", "#c8a27a"), ("TEAK", "#8a6a4a")):
    n = np.array([dead.get(site, {}).get(l, 0) for l in L])
    ax2.barh(range(3), n, left=left, color=col, height=.55, label=site); left += n
tot = left.sum()
for i, n in enumerate(left):
    ax2.text(n + 2, i, f"{int(n)}  ({n / tot:.0%})", va="center", fontsize=10)
ax2.set_yticks(range(3), labels, fontsize=11); ax2.invert_yaxis()
ax2.set_xlim(0, left.max() * 1.45); ax2.set_xlabel("standing dead trees in the field survey")
ax2.set_title("Where the dead trees are", loc="left", fontsize=12)
ax2.legend(frameon=False, loc="lower right", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "visibility.png", dpi=150)
print("saved figures/visibility.png")
