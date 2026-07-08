"""Configuration for the feature-engineering stage.

All distance features are computed in EPSG:27700 (British National Grid, metres),
consistent with the ingestion and EDA stages. Reference points are stored in
WGS84 (human-verifiable) and reprojected to EPSG:27700 in code.
"""
from __future__ import annotations

from ..data.config import CRS_BNG, CRS_WGS84, EXTERNAL_DIR, PROCESSED_DIR, PROCESSED_FILENAME

# --- Inputs ------------------------------------------------------------------
PROCESSED_PARQUET = PROCESSED_DIR / PROCESSED_FILENAME

# External geometry, manually downloaded and placed under data/external:
#   - Thames: OS Open Rivers (OGL) watercourses, or an OSM waterway extract.
#   - Stations: TfL / NaPTAN rail-tube-DLR-Overground-Elizabeth line points.
#   - POIs: an OpenStreetMap points-of-interest extract for Greater London.
THAMES_FILE = EXTERNAL_DIR / "thames.geojson"
STATIONS_FILE = EXTERNAL_DIR / "london_stations.geojson"
POIS_FILE = EXTERNAL_DIR / "london_pois.geojson"

# --- Reference points (WGS84 lon, lat) --------------------------------------
# The CBD is Charing Cross: the conventional centre of London and the historical
# datum from which road distances to London are measured. Bank and Canary Wharf
# are added as secondary employment centres (London is polycentric).
CBD_KEY = "charing_cross"
CBD_NAME = "Charing Cross"
EMPLOYMENT_CENTRES_WGS84 = {
    "charing_cross": (-0.1277, 51.5073),  # CBD (Charles I statue, top of Whitehall)
    "bank": (-0.0886, 51.5134),           # City of London financial core
    "canary_wharf": (-0.0235, 51.5054),   # Docklands financial core
}

# --- POI density -------------------------------------------------------------
POI_RADII_M = (500, 1000)   # walkable-neighbourhood radii, in metres

# --- Spatial lag -------------------------------------------------------------
AREA_COL = "lsoa"                       # areal unit for the spatial lag
SPATIAL_LAG_TARGET = "log_price"        # target whose neighbourhood mean we lag
SPATIAL_LAG_FEATURE = "spatial_lag_log_price"
ADJACENCY_METHOD = "queen"              # "queen" (primary) or "knn"
ADJACENCY_KNN_K = 8

# --- Output feature names (for reference / selection downstream) -------------
DISTANCE_FEATURES = ["dist_cbd_m", "dist_bank_m", "dist_canary_wharf_m",
                     "dist_thames_m", "dist_station_m"]
DENSITY_FEATURES = [f"poi_count_{int(r)}m" for r in POI_RADII_M]
STATIC_GEO_FEATURES = DISTANCE_FEATURES + DENSITY_FEATURES
