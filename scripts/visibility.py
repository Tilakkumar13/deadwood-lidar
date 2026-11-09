#!/usr/bin/env python3
"""Which standing dead trees can airborne LiDAR see?

    python scripts/visibility.py

Pools every site and year, normalises the acquisition-dependent features (point
density, intensity) by each flight's median, and scores out-of-fold predictions
(grouped by plot within each site) separately for trees whose crown reaches the
light and trees under other crowns, using the canopy position NEON's field crews
record for every tree. Writes results/visibility.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deadwood.features import ALL_FEATURES, COLOUR_FEATURES, FEATURE_GROUPS  # noqa: E402
from deadwood.model import grouped_cv_predict  # noqa: E402

SUNLIT = {"Full sun", "Open grown"}
SHADED = {"Mostly shaded", "Full shade"}
PER_FLIGHT = FEATURE_GROUPS["density"] + FEATURE_GROUPS["intensity"]
STRUCT = [c for g in ("height", "returns", "shape", "density") for c in FEATURE_GROUPS[g]]


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "features_*.parquet"))):
        tag = Path(f).stem.replace("features_", "")
        d = pd.read_parquet(f).dropna(subset=["h_max"])
        st = pd.read_csv(ROOT / "data" / f"stems_{tag}.csv")[["individualID", "height", "canopyPosition"]]
        d = d.merge(st, on="individualID", how="left").assign(flight=tag)
        for c in PER_FLIGHT:                       # remove flight-to-flight differences
            d[c] = d[c] / d[c].median()
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["light"] = np.where(df.canopyPosition.isin(SUNLIT), "sunlit",
                           np.where(df.canopyPosition.isin(SHADED), "shaded",
                                    np.where(df.canopyPosition == "Partially shaded", "partly shaded", "not recorded")))
    df["group"] = df.site + ":" + df.plotID     # same plot in different years stays in one fold
    return df


def score(y, p):
    return {"n": int(len(y)), "dead": int(y.sum()),
            "roc_auc": round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None,
            "average_precision": round(float(average_precision_score(y, p)), 3) if y.sum() else None,
            "base_rate": round(float(y.mean()), 3)}


def main():
    df = load()
    out = {"trees": int(len(df)), "dead": int(df.is_dead.sum()), "flights": sorted(df.flight.unique()), "feature_sets": {}}
    for name, cols in (("lidar_structure", STRUCT), ("lidar_all", ALL_FEATURES), ("lidar_plus_colour", ALL_FEATURES + COLOUR_FEATURES)):
        p = np.full(len(df), np.nan)
        for site, d in df.groupby("site"):          # train and test within a site, folds by plot
            p[d.index] = grouped_cv_predict(d[cols], d.is_dead.to_numpy(), d.group.to_numpy())
        res = {"overall": score(df.is_dead.to_numpy(), p)}
        for site in sorted(df.site.unique()):
            m = (df.site == site).to_numpy()
            res[site] = score(df.is_dead.to_numpy()[m], p[m])
        for light in ("sunlit", "partly shaded", "shaded"):
            m = (df.light == light).to_numpy()
            res[light] = score(df.is_dead.to_numpy()[m], p[m])
        out["feature_sets"][name] = res
        print(name, json.dumps(res))
    dead = df[df.is_dead == 1]
    out["dead_by_light"] = dead.light.value_counts().to_dict()
    out["dead_by_light_by_site"] = {s: d.light.value_counts().to_dict() for s, d in dead.groupby("site")}
    print("standing dead trees by canopy position:", out["dead_by_light_by_site"])
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "visibility.json", "w"), indent=2)


if __name__ == "__main__":
    main()
