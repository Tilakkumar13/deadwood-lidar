"""Tests on synthetic stems.

The point of these is not coverage, it is to check that the features behave
the way the physics says they should. A synthetic bare stem and a synthetic
live crown should separate on return structure and shape. If they do not, the
feature set is wrong and no amount of model tuning will rescue it.
"""

import numpy as np
import pytest

from deadwood.features import (
    ALL_FEATURES,
    MIN_HEIGHT_M,
    COLOUR_FEATURES,
    colour_features,
    density_features,
    height_features,
    intensity_features,
    return_features,
    shape_features,
    stem_features,
)

RNG = np.random.default_rng(0)


def make_snag(n=400, height=18.0, radius=0.25):
    """A bare vertical stem: points strung along Z, almost all single returns."""
    z = RNG.uniform(MIN_HEIGHT_M, height, n)
    theta = RNG.uniform(0, 2 * np.pi, n)
    r = RNG.uniform(0, radius, n)
    xyz = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    intensity = RNG.normal(40, 8, n).clip(0)
    number_of_returns = np.ones(n, dtype=int)          # bare wood: one return
    return_number = np.ones(n, dtype=int)
    return xyz, z, intensity, return_number, number_of_returns


def make_live_tree(n=400, height=18.0, crown_radius=3.5):
    """A live crown: points filling a volume, many multi-return pulses."""
    z = RNG.uniform(MIN_HEIGHT_M, height, n)
    # Crown widens with height then tapers - rough conifer shape.
    frac = (z - MIN_HEIGHT_M) / (height - MIN_HEIGHT_M)
    local_r = crown_radius * np.sin(np.pi * frac).clip(0.15)
    theta = RNG.uniform(0, 2 * np.pi, n)
    r = local_r * np.sqrt(RNG.uniform(0, 1, n))
    xyz = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    intensity = RNG.normal(110, 15, n).clip(0)         # foliage: brighter in NIR
    number_of_returns = RNG.integers(1, 5, n)          # foliage splits pulses
    return_number = np.array([RNG.integers(1, k + 1) for k in number_of_returns])
    return xyz, z, intensity, return_number, number_of_returns


def feats(maker, radius=5.0):
    xyz, h, i, rn, nr = maker()
    return stem_features(xyz, h, i, rn, nr, radius_m=radius)


# --- the discrimination tests: these are the ones that matter ----------------

def test_snag_has_higher_single_return_fraction():
    assert feats(make_snag)["single_return_frac"] > feats(make_live_tree)["single_return_frac"]


def test_snag_is_more_linear_than_live_crown():
    assert feats(make_snag)["linearity"] > feats(make_live_tree)["linearity"]


def test_live_crown_is_more_spherical_than_snag():
    assert feats(make_live_tree)["sphericity"] > feats(make_snag)["sphericity"]


def test_snag_dominant_axis_is_vertical():
    assert feats(make_snag)["verticality"] > 0.9


def test_live_crown_is_brighter():
    assert feats(make_live_tree)["i_mean"] > feats(make_snag)["i_mean"]


# --- correctness of the individual feature functions ------------------------

def test_all_declared_features_are_produced():
    produced = set(feats(make_snag))
    expected = set(ALL_FEATURES) | set(COLOUR_FEATURES)   # colour is NaN without RGB, but always present
    assert expected == produced, expected ^ produced


def test_colour_separates_green_crown_from_grey_snag():
    h = np.linspace(2, 20, 200)
    green = np.tile([8000, 20000, 7000], (200, 1))
    grey = np.tile([15000, 15000, 15000], (200, 1))
    assert colour_features(green, h)["c_gcc"] > colour_features(grey, h)["c_gcc"]
    assert np.isnan(colour_features(None, h)["c_gcc"])


def test_points_below_breast_height_are_dropped():
    xyz = np.array([[0, 0, 0.1], [0, 0, 0.5], [0, 0, 10.0], [0, 0, 12.0]], float)
    h = xyz[:, 2].copy()
    n = len(h)
    f = stem_features(xyz, h, np.full(n, 50.0), np.ones(n, int), np.ones(n, int), 5.0)
    assert f["n_points"] == 2, "ground and understory points should be excluded"


def test_empty_neighbourhood_returns_nan_not_crash():
    empty = np.empty((0, 3))
    f = stem_features(empty, np.array([]), np.array([]),
                      np.array([], int), np.array([], int), 5.0)
    assert f["n_points"] == 0
    assert np.isnan(f["h_max"]) and np.isnan(f["linearity"])


def test_single_point_shape_is_nan_not_an_exception():
    s = shape_features(np.array([[0.0, 0.0, 5.0]]))
    assert all(np.isnan(v) for v in s.values())


def test_return_features_on_known_counts():
    nr = np.array([1, 1, 3, 2])
    rn = np.array([1, 1, 2, 1])
    r = return_features(rn, nr)
    assert r["single_return_frac"] == pytest.approx(0.5)
    assert r["first_return_frac"] == pytest.approx(0.75)
    assert r["mean_returns_per_pulse"] == pytest.approx(1.75)


def test_point_density_uses_circle_area():
    h = np.full(100, 10.0)
    d = density_features(h, radius_m=10.0)
    assert d["point_density"] == pytest.approx(100 / (np.pi * 100), rel=1e-6)


def test_height_percentiles_are_ordered():
    h = RNG.uniform(2, 25, 500)
    f = height_features(h)
    assert f["h_p25"] <= f["h_p50"] <= f["h_p75"] <= f["h_p95"] <= f["h_max"]


def test_intensity_features_on_constant_input():
    f = intensity_features(np.full(50, 77.0))
    assert f["i_mean"] == pytest.approx(77.0) and f["i_std"] == pytest.approx(0.0)
