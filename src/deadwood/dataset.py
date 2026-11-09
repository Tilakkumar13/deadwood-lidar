"""Join NEON stem labels to LiDAR neighbourhoods and build a feature table."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from .features import stem_features

GROUND_CLASS = 2  # ASPRS standard classification for ground


def load_tile(path: Path) -> dict[str, np.ndarray]:
    """Read a LAS/LAZ tile into plain arrays."""
    las = laspy.read(str(path))
    return {
        "xyz": np.column_stack([las.x, las.y, las.z]).astype(float),
        "intensity": np.asarray(las.intensity, dtype=float),
        "return_number": np.asarray(las.return_number, dtype=int),
        "number_of_returns": np.asarray(las.number_of_returns, dtype=int),
        "classification": np.asarray(las.classification, dtype=int),
        "rgb": (np.column_stack([las.red, las.green, las.blue]).astype(float)
                if "red" in las.point_format.dimension_names else None),
    }


def local_ground_z(tile: dict, cx: float, cy: float, radius_m: float) -> float:
    """Ground elevation near a stem, from classified ground returns.

    Uses a wider search than the stem radius, because a dense crown can leave
    no ground returns directly beneath it. Falls back to the local minimum if
    the tile carries no ground classification.
    """
    xyz, cls = tile["xyz"], tile["classification"]
    search = max(radius_m * 3, 10.0)
    d2 = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2
    near = d2 <= search**2
    if not np.any(near):
        return float("nan")
    ground = near & (cls == GROUND_CLASS)
    if np.any(ground):
        return float(np.median(xyz[ground, 2]))
    return float(np.percentile(xyz[near, 2], 2))  # fallback: low percentile


def extract_stem(tile: dict, cx: float, cy: float, radius_m: float) -> dict | None:
    """Features for one stem, or None if the neighbourhood is empty."""
    xyz = tile["xyz"]
    d2 = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2
    sel = d2 <= radius_m**2
    if not np.any(sel):
        return None
    gz = local_ground_z(tile, cx, cy, radius_m)
    if not np.isfinite(gz):
        return None
    return stem_features(
        xyz=xyz[sel],
        heights=xyz[sel, 2] - gz,
        intensity=tile["intensity"][sel],
        return_number=tile["return_number"][sel],
        number_of_returns=tile["number_of_returns"][sel],
        radius_m=radius_m,
        rgb=tile["rgb"][sel] if tile.get("rgb") is not None else None,
    )


def build_table(
    stems: pd.DataFrame, tile_paths: list[Path], radius_m: float = 3.0
) -> pd.DataFrame:
    """Feature table for stems that fall inside the supplied tiles.

    stems needs: individualID, easting, northing, is_dead, plotID.
    Tiles are loaded one at a time - a NEON LAZ tile is large, and holding
    several at once will exhaust memory on a laptop.
    """
    required = {"individualID", "easting", "northing", "is_dead", "plotID"}
    missing = required - set(stems.columns)
    if missing:
        raise ValueError(f"stems missing columns: {sorted(missing)}")

    rows = []
    for path in tile_paths:
        tile = load_tile(path)
        xmin, ymin = tile["xyz"][:, 0].min(), tile["xyz"][:, 1].min()
        xmax, ymax = tile["xyz"][:, 0].max(), tile["xyz"][:, 1].max()
        inside = stems[
            stems.easting.between(xmin, xmax) & stems.northing.between(ymin, ymax)
        ]
        for rec in inside.itertuples(index=False):
            f = extract_stem(tile, rec.easting, rec.northing, radius_m)
            if f is None:
                continue
            f.update(
                individualID=rec.individualID,
                plotID=rec.plotID,
                is_dead=int(rec.is_dead),
                tile=path.name,
            )
            rows.append(f)
        del tile  # release before the next tile
    return pd.DataFrame(rows)
