"""Per-stem features for separating standing dead trees from live ones in ALS.

The feature set is deliberately physics-led rather than exhaustive. Three
signals do most of the work for live/dead separation in discrete-return ALS:

1. Return structure. Foliage splits a pulse into several returns; a bare snag
   mostly produces single returns. The single-return fraction is usually the
   single strongest discriminator, and it needs no radiometric calibration.

2. Geometry. A live crown fills a volume, so its point neighbourhood is roughly
   spherical. A snag is a bare stem, so its neighbourhood is close to linear.
   PCA eigenvalues of the neighbourhood capture this directly.

3. Intensity. Dead wood reflects less in the NIR than live foliage. Useful, but
   intensity is sensor- and flight-dependent, so it is kept separate and should
   be normalised per acquisition before combining sites.

Keeping these as named groups matters: if the model turns out to lean entirely
on intensity, the result will not transfer to another sensor, and you want to
be able to see that.
"""

from __future__ import annotations

import numpy as np

# Height above ground below which points are treated as understory/ground
# clutter rather than part of the stem.
MIN_HEIGHT_M = 1.3  # breast height, the usual forestry convention


def _percentiles(values: np.ndarray, qs=(25, 50, 75, 95)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    pct = np.percentile(values, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, pct)}


def height_features(heights: np.ndarray) -> dict[str, float]:
    """Vertical distribution of returns above ground."""
    if heights.size == 0:
        return {"h_max": float("nan"), "h_mean": float("nan"), "h_std": float("nan"),
                **{f"h_{k}": float("nan") for k in ("p25", "p50", "p75", "p95")}}
    out = {
        "h_max": float(heights.max()),
        "h_mean": float(heights.mean()),
        "h_std": float(heights.std()),
    }
    out.update({f"h_{k}": v for k, v in _percentiles(heights).items()})
    return out


def return_features(
    return_number: np.ndarray, number_of_returns: np.ndarray
) -> dict[str, float]:
    """Return-structure features.

    single_return_frac is the headline feature: a pulse that hits only bare
    wood tends to come back once, while foliage splits it. Expect snags high
    and live crowns low.
    """
    n = return_number.size
    if n == 0:
        return {"single_return_frac": float("nan"), "first_return_frac": float("nan"),
                "mean_returns_per_pulse": float("nan")}
    single = np.count_nonzero(number_of_returns == 1)
    first = np.count_nonzero(return_number == 1)
    return {
        "single_return_frac": float(single / n),
        "first_return_frac": float(first / n),
        "mean_returns_per_pulse": float(number_of_returns.mean()),
    }


def shape_features(xyz: np.ndarray) -> dict[str, float]:
    """PCA shape descriptors of the point neighbourhood.

    linearity  -> high for a bare stem (points strung along one axis)
    sphericity -> high for a live crown (points fill a volume)

    Standard eigenvalue features; see Weinmann et al. on 3D point cloud
    geometric descriptors.
    """
    nan = {"linearity": float("nan"), "planarity": float("nan"),
           "sphericity": float("nan"), "verticality": float("nan")}
    if xyz.shape[0] < 3:
        return nan

    centred = xyz - xyz.mean(axis=0)
    # Covariance needs at least 3 non-degenerate points; guard against collapse.
    cov = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return nan
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total = eigvals.sum()
    if total <= 0:
        return nan
    e1, e2, e3 = (eigvals / total)

    # Verticality: how closely the dominant axis aligns with Z.
    verticality = abs(float(eigvecs[2, 0]))

    return {
        "linearity": float((e1 - e2) / e1) if e1 > 0 else float("nan"),
        "planarity": float((e2 - e3) / e1) if e1 > 0 else float("nan"),
        "sphericity": float(e3 / e1) if e1 > 0 else float("nan"),
        "verticality": verticality,
    }


def intensity_features(intensity: np.ndarray) -> dict[str, float]:
    """Radiometric features. Sensor-dependent: normalise per acquisition."""
    if intensity.size == 0:
        return {"i_mean": float("nan"), "i_std": float("nan"),
                **{f"i_{k}": float("nan") for k in ("p25", "p50", "p75", "p95")}}
    out = {"i_mean": float(intensity.mean()), "i_std": float(intensity.std())}
    out.update({f"i_{k}": v for k, v in _percentiles(intensity).items()})
    return out


def colour_features(rgb: np.ndarray | None, heights: np.ndarray) -> dict[str, float]:
    """Crown colour from the camera NEON flies with the LiDAR ("colorized" point clouds).

    Not a LiDAR signal: it comes from co-acquired RGB imagery, so it is kept out of
    FEATURE_GROUPS and compared separately. Uses the upper half of the crown, where a
    live tree shows foliage and a snag shows grey or brown wood.
    """
    keys = ("c_gcc", "c_rcc", "c_exg", "c_bright", "c_gcc_p75")
    if rgb is None or len(rgb) == 0 or heights.size == 0:
        return {k: float("nan") for k in keys}
    top = heights >= 0.5 * heights.max()
    c = rgb[top].astype(float)
    tot = c.sum(1) + 1e-9
    r, g, b = c[:, 0] / tot, c[:, 1] / tot, c[:, 2] / tot
    return {"c_gcc": float(g.mean()), "c_rcc": float(r.mean()), "c_exg": float((2 * g - r - b).mean()),
            "c_bright": float(tot.mean() / 3 / 65535), "c_gcc_p75": float(np.percentile(g, 75))}


def density_features(heights: np.ndarray, radius_m: float) -> dict[str, float]:
    """Point count and vertical occupancy within the stem neighbourhood."""
    if heights.size == 0:
        return {"n_points": 0.0, "point_density": 0.0, "occupied_bin_frac": float("nan")}
    area = np.pi * radius_m**2
    # Fraction of 1 m height bins between ground and top that contain any point.
    top = heights.max()
    if top <= MIN_HEIGHT_M:
        occupied = float("nan")
    else:
        bins = np.arange(MIN_HEIGHT_M, top + 1.0, 1.0)
        counts, _ = np.histogram(heights, bins=bins)
        occupied = float(np.count_nonzero(counts) / max(len(counts), 1))
    return {
        "n_points": float(heights.size),
        "point_density": float(heights.size / area),
        "occupied_bin_frac": occupied,
    }


def stem_features(
    xyz: np.ndarray,
    heights: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    radius_m: float,
    rgb: np.ndarray | None = None,
) -> dict[str, float]:
    """Full feature vector for one stem neighbourhood.

    Args:
        xyz: (N, 3) point coordinates.
        heights: (N,) height above ground, metres.
        intensity: (N,) raw intensity.
        return_number / number_of_returns: (N,) LAS return fields.
        radius_m: neighbourhood radius used to extract the points.

    Points below MIN_HEIGHT_M are dropped before anything is computed, so
    ground and low understory do not contaminate the stem description.
    """
    mask = heights >= MIN_HEIGHT_M
    xyz, heights = xyz[mask], heights[mask]
    intensity = intensity[mask]
    return_number, number_of_returns = return_number[mask], number_of_returns[mask]
    rgb = rgb[mask] if rgb is not None else None

    feats: dict[str, float] = {}
    feats.update(height_features(heights))
    feats.update(return_features(return_number, number_of_returns))
    feats.update(shape_features(xyz))
    feats.update(intensity_features(intensity))
    feats.update(density_features(heights, radius_m))
    feats.update(colour_features(rgb, heights))
    return feats


FEATURE_GROUPS = {
    "height": ["h_max", "h_mean", "h_std", "h_p25", "h_p50", "h_p75", "h_p95"],
    "returns": ["single_return_frac", "first_return_frac", "mean_returns_per_pulse"],
    "shape": ["linearity", "planarity", "sphericity", "verticality"],
    "intensity": ["i_mean", "i_std", "i_p25", "i_p50", "i_p75", "i_p95"],
    "density": ["n_points", "point_density", "occupied_bin_frac"],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Camera colour, not LiDAR: compared separately, never part of ALL_FEATURES.
COLOUR_FEATURES = ["c_gcc", "c_rcc", "c_exg", "c_bright", "c_gcc_p75"]
