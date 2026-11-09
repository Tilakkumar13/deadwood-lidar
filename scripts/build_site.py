#!/usr/bin/env python3
"""Build the feature table for one NEON site and one survey/flight pairing.

    python scripts/build_site.py SOAP --veg 2023-09 2023-10 --lidar 2023-06 2023-07

Steps:
  1. Field survey (DP1.10098.001): trees with a live or standing-dead status.
  2. Stem coordinates from distance/azimuth to surveyed plot points (/locations).
  3. LiDAR (DP1.30003.001), one 1 km tile at a time: download, keep only the
     points within 15 m of a surveyed stem, delete the full tile. A NEON tile is
     hundreds of MB; the crops are a few MB.
  4. Features per stem (deadwood.dataset.build_table).

Writes data/stems_<SITE>_<YEAR>.csv, data/crops/<SITE>_<YEAR>/*.laz and
data/features_<SITE>_<YEAR>.parquet. Needs a NEON API token (~/.neon_token).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deadwood.dataset import build_table  # noqa: E402
from deadwood.neon import (  # noqa: E402
    LIDAR_POINT_CLOUD, VEG_STRUCTURE, download, label_stems, list_files, named_location, stem_xy,
)

TREE_FORMS = {"single bole tree", "multi-bole tree", "small tree"}
KEEP_RADIUS = 15.0   # metres of point cloud kept around each stem


def field_stems(site: str, months: list[str], min_height: float) -> pd.DataFrame:
    vdir = ROOT / "data" / "veg" / site
    ai, mt = [], None
    for m in months:
        fs = [f for f in list_files(VEG_STRUCTURE, site, m) if f.name.endswith(".csv")]
        for p in download([f for f in fs if "vst_apparentindividual" in f.name], vdir):
            ai.append(pd.read_csv(p))
        mt_files = download([f for f in fs if "vst_mappingandtagging" in f.name], vdir)
        if mt_files:
            mt = pd.read_csv(mt_files[-1])     # cumulative table: the latest copy has every tagged stem
    ai = pd.concat(ai).sort_values("date").drop_duplicates("individualID", keep="last")
    ai = ai[ai.growthForm.isin(TREE_FORMS)]
    lab = label_stems(ai)
    lab = lab[lab.height.fillna(0) >= min_height]

    mt = mt.dropna(subset=["pointID", "stemDistance", "stemAzimuth"]).drop_duplicates("individualID", keep="last")
    mt = mt[mt.individualID.isin(lab.individualID)]

    # surveyed point coordinates, cached per plot
    cache = vdir / "points.json"
    pts = json.loads(cache.read_text()) if cache.exists() else {}
    for loc in mt.namedLocation.unique():
        if loc in pts:
            continue
        d = named_location(loc)
        pts[loc] = {}
        for child in d.get("locationChildren", []):
            c = named_location(child)
            pts[loc][child.rsplit(".", 1)[-1]] = [c["locationUtmEasting"], c["locationUtmNorthing"]]
        cache.write_text(json.dumps(pts))
    rows = []
    for loc, g in mt.groupby("namedLocation"):
        coords = {float(k): tuple(v) for k, v in pts.get(loc, {}).items() if re.fullmatch(r"\d+(\.\d+)?", k)}
        g = g.assign(pointID=g.pointID.astype(float))
        rows.append(stem_xy(g, coords))
    xy = pd.concat(rows)
    out = lab.merge(xy, on="individualID")[["individualID", "plotID", "is_dead", "plantStatus", "height",
                                            "stemDiameter", "canopyPosition", "easting", "northing"]]
    return out


def crop_tiles(site: str, months: list[str], stems: pd.DataFrame, crop_dir: Path) -> list[Path]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "data" / "tmp"
    files = []
    for m in months:
        files += [f for f in list_files(LIDAR_POINT_CLOUD, site, m) if f.name.endswith(".laz")]
    by_tile = {}
    for f in files:
        mm = re.search(r"_(\d{6})_(\d{7})_", f.name)
        if mm:
            by_tile.setdefault((int(mm.group(1)), int(mm.group(2))), f)
    need = set()
    for e, n in zip(stems.easting, stems.northing):
        for de in (-KEEP_RADIUS, 0, KEEP_RADIUS):
            for dn in (-KEEP_RADIUS, 0, KEEP_RADIUS):
                need.add((int((e + de) // 1000 * 1000), int((n + dn) // 1000 * 1000)))
    need &= set(by_tile)
    print(f"{len(need)} LiDAR tiles cover the plots ({sum(by_tile[k].size for k in need) / 1e6:.0f} MB to stream)", flush=True)
    xy = stems[["easting", "northing"]].to_numpy()
    crops = []
    for k in sorted(need):
        out = crop_dir / f"{site}_{k[0]}_{k[1]}.laz"
        if out.exists():
            crops.append(out); continue
        f = by_tile[k]
        (p,) = download([f], tmp)
        las = laspy.read(str(p))
        x, y = np.asarray(las.x), np.asarray(las.y)
        near = np.zeros(len(x), bool)
        for e, n in xy:
            if k[0] - KEEP_RADIUS <= e < k[0] + 1000 + KEEP_RADIUS and k[1] - KEEP_RADIUS <= n < k[1] + 1000 + KEEP_RADIUS:
                box = (np.abs(x - e) <= KEEP_RADIUS) & (np.abs(y - n) <= KEEP_RADIUS)
                near |= box
        if near.any():
            sub = laspy.LasData(las.header)
            sub.points = las.points[near].copy()
            sub.write(str(out))
            crops.append(out)
        print(f"  {f.name}: {near.sum():,} of {len(x):,} points kept", flush=True)
        del las
        p.unlink()                           # the full tile is not kept
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--veg", nargs="+", required=True, help="survey months, YYYY-MM")
    ap.add_argument("--lidar", nargs="+", required=True, help="flight months, YYYY-MM")
    ap.add_argument("--min-height", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=3.0)
    a = ap.parse_args()
    tag = f"{a.site}_{a.lidar[0][:4]}"
    stems = field_stems(a.site, a.veg, a.min_height)
    stems.to_csv(ROOT / "data" / f"stems_{tag}.csv", index=False)
    print(f"{tag}: {len(stems)} mapped trees, {int(stems.is_dead.sum())} standing dead, {stems.plotID.nunique()} plots", flush=True)
    crops = crop_tiles(a.site, a.lidar, stems, ROOT / "data" / "crops" / tag)
    feats = build_table(stems, crops, radius_m=a.radius)
    feats["site"], feats["year"] = a.site, int(a.lidar[0][:4])
    feats = feats.drop_duplicates("individualID")
    feats.to_parquet(ROOT / "data" / f"features_{tag}.parquet")
    print(f"features: {len(feats)} trees, {int(feats.is_dead.sum())} dead -> data/features_{tag}.parquet")


if __name__ == "__main__":
    main()
