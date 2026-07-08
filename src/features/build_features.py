"""Static geospatial features: distances (metres, EPSG:27700) and POI density.

Every feature here is computed purely from a transaction's location and fixed
external geometry, so none of them can leak the target: they are safe to compute
once, globally, before cross-validation. (The one target-derived feature - the
spatial lag - lives in spatial_lag.py and is fold-safe by construction.)

Distances use British National Grid eastings/northings already present in the
processed data; reference points and external layers are reprojected to EPSG:27700.

Run (from the repository root):
    python -m src.features.build_features
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from ..data.config import CRS_BNG, CRS_WGS84
from .config import (
    CBD_KEY,
    EMPLOYMENT_CENTRES_WGS84,
    POI_RADII_M,
    POIS_FILE,
    PROCESSED_PARQUET,
    STATIONS_FILE,
    STATIC_GEO_FEATURES,
    THAMES_FILE,
)

LOGGER = logging.getLogger(__name__)
_WGS84_TO_BNG = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)


# --------------------------------------------------------------------------- #
# Loaders (external geometry -> EPSG:27700)
# --------------------------------------------------------------------------- #
def _ensure_bng(gdf):
    """Return the layer in EPSG:27700, assuming BNG if the CRS is absent."""
    if gdf.crs is None:
        LOGGER.warning("Layer has no CRS; assuming EPSG:%d. Verify the source.", CRS_BNG)
        return gdf.set_crs(CRS_BNG)
    return gdf.to_crs(CRS_BNG)


def load_thames(path: Path = THAMES_FILE):
    import geopandas as gpd

    if not path.exists():
        raise FileNotFoundError(
            f"Thames geometry not found at {path}. Download OS Open Rivers (OGL) or an "
            "OSM waterway extract and place the Thames line/network there."
        )
    gdf = _ensure_bng(gpd.read_file(path))
    name_col = next((c for c in gdf.columns
                     if c.lower() in ("name", "name1", "watercourse", "river")), None)
    if name_col is not None:
        mask = gdf[name_col].astype(str).str.contains("Thames", case=False, na=False)
        if mask.any():
            gdf = gdf[mask]
            LOGGER.info("Filtered Thames layer to %d named feature(s).", len(gdf))
    return gdf.reset_index(drop=True)


def load_points(path: Path, label: str):
    """Load a point layer (stations or POIs); use centroids for non-point geoms."""
    import geopandas as gpd

    if not path.exists():
        raise FileNotFoundError(f"{label} layer not found at {path}.")
    gdf = _ensure_bng(gpd.read_file(path))
    if not (gdf.geometry.geom_type == "Point").all():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.centroid  # safe: already projected to metres
    return gdf.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
def _xy(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["easting"].to_numpy(float), df["northing"].to_numpy(float)])


def add_distance_to_centres(df: pd.DataFrame, centres=EMPLOYMENT_CENTRES_WGS84) -> pd.DataFrame:
    """Euclidean distance (m) from each sale to each named employment centre."""
    e = df["easting"].to_numpy(float)
    n = df["northing"].to_numpy(float)
    for key, (lon, lat) in centres.items():
        cx, cy = _WGS84_TO_BNG.transform(lon, lat)
        df[f"dist_{key}_m"] = np.hypot(e - cx, n - cy)
    df["dist_cbd_m"] = df[f"dist_{CBD_KEY}_m"]  # explicit CBD alias (Charing Cross)
    return df


def add_distance_to_thames(df: pd.DataFrame, thames_gdf) -> pd.DataFrame:
    """Nearest distance (m) from each sale to the Thames, via a spatial index."""
    import geopandas as gpd

    pts = gpd.GeoDataFrame(
        {"_row": np.arange(len(df))},
        geometry=gpd.points_from_xy(df["easting"].to_numpy(float),
                                    df["northing"].to_numpy(float)),
        crs=CRS_BNG,
    )
    thames = thames_gdf[["geometry"]].to_crs(CRS_BNG)
    joined = gpd.sjoin_nearest(pts, thames, how="left", distance_col="dist_thames_m")
    # sjoin_nearest can emit duplicate rows on exact ties; keep one per point.
    joined = joined.drop_duplicates(subset="_row", keep="first").sort_values("_row")
    df["dist_thames_m"] = joined["dist_thames_m"].to_numpy()
    return df


def add_distance_to_nearest_station(df: pd.DataFrame, stations_gdf) -> pd.DataFrame:
    """Distance (m) to the nearest station using a KD-tree 1-NN query."""
    st = stations_gdf.to_crs(CRS_BNG)
    station_xy = np.column_stack([st.geometry.x.to_numpy(), st.geometry.y.to_numpy()])
    tree = cKDTree(station_xy)
    dist, _ = tree.query(_xy(df), k=1)
    df["dist_station_m"] = dist
    return df


def add_poi_density(df: pd.DataFrame, pois_gdf, radii=POI_RADII_M) -> pd.DataFrame:
    """Count of POIs within each radius (m), via KD-tree ball counts."""
    pois = pois_gdf.to_crs(CRS_BNG)
    poi_xy = np.column_stack([pois.geometry.x.to_numpy(), pois.geometry.y.to_numpy()])
    tree = cKDTree(poi_xy)
    txn_xy = _xy(df)
    for r in radii:
        df[f"poi_count_{int(r)}m"] = tree.query_ball_point(txn_xy, r, return_length=True)
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame,
    thames_gdf,
    stations_gdf,
    pois_gdf,
    *,
    centres=EMPLOYMENT_CENTRES_WGS84,
    radii=POI_RADII_M,
) -> pd.DataFrame:
    """Attach all static geospatial features and return a new DataFrame."""
    out = df.copy()
    out = add_distance_to_centres(out, centres)
    out = add_distance_to_thames(out, thames_gdf)
    out = add_distance_to_nearest_station(out, stations_gdf)
    out = add_poi_density(out, pois_gdf, radii)
    LOGGER.info("Added static geo features: %s", ", ".join(STATIC_GEO_FEATURES))
    return out


def run_build(processed_path: Path = PROCESSED_PARQUET, *, write: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(processed_path)
    feats = build_features(df, load_thames(), load_points(STATIONS_FILE, "Stations"),
                           load_points(POIS_FILE, "POIs"))
    if write:
        dest = processed_path.with_name("london_ppd_2021_2024_features.parquet")
        feats.to_parquet(dest, index=False)
        LOGGER.info("Wrote %s (%s rows)", dest, f"{len(feats):,}")
    return feats


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    run_build()


if __name__ == "__main__":
    main()
