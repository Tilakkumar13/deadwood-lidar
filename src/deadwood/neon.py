"""Client for the NEON data API, and label derivation for live/dead stems.

NEON products used:
  DP1.10098.001  Woody plant vegetation structure (tabular; carries plantStatus)
  DP1.30003.001  Discrete-return LiDAR point cloud (LAZ tiles)
  DP3.30015.001  Ecosystem structure / canopy height model (optional)

IMPORTANT — verify before trusting:
NEON revises table schemas between releases. The column names below are the
ones in current documentation, but confirm them against the portal for the
release you download. Every place that depends on a NEON column name is
marked with NEON-SCHEMA so you can find them quickly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

API = "https://data.neonscience.org/api/v0"

VEG_STRUCTURE = "DP1.10098.001"
LIDAR_POINT_CLOUD = "DP1.30003.001"
CANOPY_HEIGHT_MODEL = "DP3.30015.001"

# NEON-SCHEMA: plantStatus values that count as a standing dead stem.
# Fallen/downed material is excluded - it is a different detection problem and
# is not what an ALS snag classifier should be asked to find.
DEAD_STATUSES = {
    "Standing dead",
    "Dead, broken bole",
}
LIVE_STATUSES = {
    "Live",
    "Live, disease damaged",
    "Live, insect damaged",
    "Live, physically damaged",
    "Live, other damage",
}
# Everything else (e.g. "Downed", "Removed", blank) is dropped rather than
# guessed at. Silent mislabelling is worse than a smaller training set.


class NeonError(RuntimeError):
    pass


def _headers() -> dict:
    """NEON requires an API token for data downloads since June 2026.

    Read from the NEON_TOKEN environment variable, or from ~/.neon_token.
    Get one from the Data Portal: My Account -> API Token.
    """
    import os
    tok = os.environ.get("NEON_TOKEN")
    f = Path.home() / ".neon_token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    return {"X-API-Token": tok} if tok else {}


@dataclass
class NeonFile:
    name: str
    url: str
    size: int


def _get(path: str, **params) -> dict:
    try:
        r = requests.get(f"{API}{path}", params=params, headers=_headers(), timeout=60)
    except requests.RequestException as exc:
        raise NeonError(f"could not reach NEON API: {exc}") from exc
    if r.status_code in (401, 403) and "/data/" in path and not _headers():
        raise NeonError("NEON needs an API token for data downloads: put it in ~/.neon_token or NEON_TOKEN")
    if r.status_code == 404:
        raise NeonError(f"not found: {path} (check product code, site and month)")
    if r.status_code >= 400:
        raise NeonError(f"HTTP {r.status_code} from {path}: {r.text[:300]}")
    return r.json()


def available_months(product: str, site: str) -> list[str]:
    """Months with data for a product at a site, as 'YYYY-MM'."""
    data = _get(f"/products/{product}")["data"]
    for s in data.get("siteCodes", []):
        if s.get("siteCode") == site:
            return sorted(s.get("availableMonths", []))
    raise NeonError(f"site {site} has no data for {product}")


def list_files(product: str, site: str, month: str) -> list[NeonFile]:
    """Files published for one product/site/month."""
    data = _get(f"/data/{product}/{site}/{month}")["data"]
    return [
        NeonFile(name=f["name"], url=f["url"], size=int(f.get("size", 0)))
        for f in data.get("files", [])
    ]


def download(files: list[NeonFile], out_dir: Path, contains: str | None = None) -> list[Path]:
    """Download files, optionally filtering by substring. Skips existing files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        if contains and contains not in f.name:
            continue
        dest = out_dir / f.name
        if dest.exists() and dest.stat().st_size == f.size:
            saved.append(dest)
            continue
        with requests.get(f.url, stream=True, headers=_headers(), timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        saved.append(dest)
    return saved


def label_stems(apparent: pd.DataFrame) -> pd.DataFrame:
    """Map plantStatus to a binary is_dead label, dropping ambiguous rows.

    NEON-SCHEMA: expects columns 'individualID' and 'plantStatus'.
    """
    for col in ("individualID", "plantStatus"):
        if col not in apparent.columns:
            raise NeonError(
                f"expected column {col!r} in vst_apparentindividual; "
                f"found {list(apparent.columns)[:12]}... - check the NEON schema"
            )
    status = apparent["plantStatus"].astype("string").str.strip()
    keep = status.isin(DEAD_STATUSES | LIVE_STATUSES)
    out = apparent.loc[keep].copy()
    out["is_dead"] = status.loc[keep].isin(DEAD_STATUSES).astype(int)
    return out


def stem_xy(
    mapping: pd.DataFrame, point_coords: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Derive stem easting/northing from NEON's distance/azimuth mapping.

    NEON does not publish a stem coordinate directly. Each stem is recorded as
    a bearing and distance from a surveyed pointID, so the coordinate has to be
    reconstructed. point_coords maps pointID -> (easting, northing), which you
    get from the /locations endpoint for the plot's named locations.

    NEON-SCHEMA: expects 'individualID', 'pointID', 'stemDistance', 'stemAzimuth'.
    Azimuth is degrees clockwise from grid north.
    """
    required = {"individualID", "pointID", "stemDistance", "stemAzimuth"}
    missing = required - set(mapping.columns)
    if missing:
        raise NeonError(f"vst_mappingandtagging missing columns: {sorted(missing)}")

    rows = []
    for rec in mapping.itertuples(index=False):
        origin = point_coords.get(getattr(rec, "pointID"))
        dist, azim = getattr(rec, "stemDistance"), getattr(rec, "stemAzimuth")
        if origin is None or pd.isna(dist) or pd.isna(azim):
            continue  # unmappable stem; dropped rather than approximated
        theta = math.radians(float(azim))
        rows.append(
            {
                "individualID": getattr(rec, "individualID"),
                # Azimuth is clockwise from north, so north is the cosine term.
                "easting": origin[0] + float(dist) * math.sin(theta),
                "northing": origin[1] + float(dist) * math.cos(theta),
            }
        )
    return pd.DataFrame(rows)


def named_location(name: str) -> dict:
    """Fetch one NEON named location (used to resolve pointID coordinates)."""
    return _get(f"/locations/{name}")["data"]
