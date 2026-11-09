"""End-to-end check on a synthetic LAZ tile with known live/dead stems.

Writes a real LAS file, runs the same extraction the NEON pipeline uses, and
confirms a model trained on it separates the classes. If this fails, the break
is in the chain rather than in any one function.
"""

import numpy as np
import pandas as pd
import laspy
import pytest

from deadwood.dataset import build_table, extract_stem, load_tile, local_ground_z
from deadwood.features import ALL_FEATURES
from deadwood.model import evaluate, grouped_cv_predict

RNG = np.random.default_rng(7)
GROUND_Z = 100.0


def _tree(cx, cy, dead: bool, n=500):
    """Points for one tree. Dead = narrow, single-return, dim."""
    z = RNG.uniform(1.5, 20.0, n)
    if dead:
        r = RNG.uniform(0, 0.3, n)
        nr = np.ones(n, dtype=int)
        inten = RNG.normal(45, 8, n)
    else:
        frac = (z - 1.5) / 18.5
        r = 3.0 * np.sin(np.pi * frac).clip(0.2) * np.sqrt(RNG.uniform(0, 1, n))
        nr = RNG.integers(1, 5, n)
        inten = RNG.normal(115, 15, n)
    th = RNG.uniform(0, 2 * np.pi, n)
    xyz = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), GROUND_Z + z])
    rn = np.array([RNG.integers(1, k + 1) for k in nr])
    return xyz, inten.clip(0), rn, nr


def _ground(cx, cy, n=400, span=60):
    xy = RNG.uniform(-span, span, (n, 2)) + np.array([cx, cy])
    xyz = np.column_stack([xy, np.full(n, GROUND_Z)])
    return xyz, RNG.normal(60, 5, n).clip(0), np.ones(n, int), np.ones(n, int)


@pytest.fixture(scope="module")
def synthetic_tile(tmp_path_factory):
    """A LAS tile holding 30 live and 30 dead stems on a 20 m grid."""
    parts, stems = [], []
    # Plot is set by column (j) and class by row (i), so every plot holds both
    # classes. If plot predicted class, GroupKFold would test on single-class
    # folds and the score would be meaningless - which is exactly the confound
    # grouped CV exists to expose.
    coords = [(500000 + 20 * i, 4100000 + 20 * j, i, j) for i in range(10) for j in range(6)]
    for k, (cx, cy, i, j) in enumerate(coords):
        dead = i % 2 == 0
        parts.append(_tree(cx, cy, dead))
        stems.append(
            {
                "individualID": f"T{k}",
                "easting": cx,
                "northing": cy,
                "is_dead": int(dead),
                "plotID": f"P{j}",       # 6 plots, each with 5 dead and 5 live
            }
        )
    parts.append(_ground(500090, 4100050, n=6000, span=200))

    xyz = np.vstack([p[0] for p in parts])
    inten = np.concatenate([p[1] for p in parts])
    rn = np.concatenate([p[2] for p in parts])
    nr = np.concatenate([p[3] for p in parts])
    cls = np.concatenate(
        [np.ones(len(p[0]), int) * 5 for p in parts[:-1]]
        + [np.full(len(parts[-1][0]), 2)]          # ground class
    )

    hdr = laspy.LasHeader(point_format=1, version="1.2")
    hdr.offsets, hdr.scales = xyz.min(axis=0), [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = inten.astype(np.uint16)
    las.return_number = rn.astype(np.uint8)
    las.number_of_returns = nr.astype(np.uint8)
    las.classification = cls.astype(np.uint8)

    path = tmp_path_factory.mktemp("las") / "synthetic.las"
    las.write(str(path))
    return path, pd.DataFrame(stems)


def test_tile_round_trips(synthetic_tile):
    path, _ = synthetic_tile
    tile = load_tile(path)
    assert tile["xyz"].shape[0] > 10000
    assert set(np.unique(tile["classification"])) >= {2, 5}


def test_ground_elevation_is_recovered(synthetic_tile):
    path, _ = synthetic_tile
    tile = load_tile(path)
    assert local_ground_z(tile, 500000, 4100000, 3.0) == pytest.approx(GROUND_Z, abs=0.5)


def test_extract_stem_returns_full_feature_vector(synthetic_tile):
    path, _ = synthetic_tile
    f = extract_stem(load_tile(path), 500000, 4100000, radius_m=3.0)
    assert f is not None and set(ALL_FEATURES) <= set(f)


def test_extract_stem_outside_tile_is_none(synthetic_tile):
    path, _ = synthetic_tile
    assert extract_stem(load_tile(path), 9_000_000, 9_000_000, 3.0) is None


def test_build_table_finds_every_stem(synthetic_tile):
    path, stems = synthetic_tile
    df = build_table(stems, [path], radius_m=3.0)
    assert len(df) == len(stems)
    assert df.is_dead.sum() == stems.is_dead.sum()


def test_pipeline_separates_live_from_dead(synthetic_tile):
    """The payoff test: features -> grouped CV -> usable separation."""
    path, stems = synthetic_tile
    df = build_table(stems, [path], radius_m=3.0)
    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df.is_dead.to_numpy()
    proba = grouped_cv_predict(X, y, df.plotID.to_numpy(), n_splits=6)
    m = evaluate(y, proba)
    assert m["roc_auc"] > 0.9, f"expected clean separation, got {m['roc_auc']:.3f}"


def test_build_table_rejects_missing_columns(synthetic_tile):
    path, stems = synthetic_tile
    with pytest.raises(ValueError, match="missing columns"):
        build_table(stems.drop(columns=["is_dead"]), [path])
