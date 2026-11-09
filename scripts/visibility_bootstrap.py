#!/usr/bin/env python3
"""Plot-level bootstrap for the visibility result (resamples whole plots, 1000 times).

    python scripts/visibility_bootstrap.py   -> results/visibility_bootstrap.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from visibility import load, STRUCT  # noqa: E402
from deadwood.features import ALL_FEATURES, COLOUR_FEATURES  # noqa: E402
from deadwood.model import grouped_cv_predict  # noqa: E402

LIGHTS = ("sunlit", "partly shaded", "shaded")
df = load()
rng = np.random.default_rng(0)
out = {}
for name, cols in (("lidar_all", ALL_FEATURES), ("lidar_plus_colour", ALL_FEATURES + COLOUR_FEATURES)):
    p = np.full(len(df), np.nan)
    for site, d in df.groupby("site"):
        p[d.index] = grouped_cv_predict(d[cols], d.is_dead.to_numpy(), d.group.to_numpy())
    df["p"] = p
    point = {l: roc_auc_score(df[df.light == l].is_dead, df[df.light == l].p) for l in LIGHTS}
    groups = df.group.unique()
    by = df.set_index("group")
    samples = {l: [] for l in LIGHTS}; diffs = []
    for _ in range(1000):
        b = by.loc[rng.choice(groups, len(groups), replace=True)]
        a = {}
        for l in LIGHTS:
            s = b[b.light == l]
            a[l] = roc_auc_score(s.is_dead, s.p) if s.is_dead.nunique() == 2 else np.nan
            samples[l].append(a[l])
        diffs.append(a["sunlit"] - a["shaded"])
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]
    out[name] = {l: {"roc_auc": round(float(point[l]), 3), "ci95": ci(samples[l])} for l in LIGHTS}
    d = np.array(diffs)
    out[name]["sunlit_minus_shaded"] = {"ci95": ci(d), "share_positive": round(float(np.nanmean(d > 0)), 3)}
    print(name, json.dumps(out[name]))
json.dump(out, open(ROOT / "results" / "visibility_bootstrap.json", "w"), indent=2)
