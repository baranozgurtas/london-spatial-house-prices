"""Configuration for the geospatial EDA / spatial-autocorrelation stage."""
from __future__ import annotations

from ..data.config import (
    CRS_BNG,
    CRS_WGS84,
    EXTERNAL_DIR,
    PROCESSED_DIR,
    PROCESSED_FILENAME,
    REPO_ROOT,
)

# --- Inputs ------------------------------------------------------------------
PROCESSED_PARQUET = PROCESSED_DIR / PROCESSED_FILENAME

# LSOA 2021 boundaries: manually download from the ONS Open Geography Portal
# (Generalised Clipped, BGC, is recommended for mapping) and place the GeoJSON
# (or shapefile) here. The loader auto-detects the LSOA code column.
LSOA_BOUNDARY_FILE = EXTERNAL_DIR / "LSOA_2021_EW_BGC.geojson"
LSOA_CODE_CANDIDATES = ("LSOA21CD", "lsoa21cd", "LSOA21cd", "geo_code", "code")

# --- Aggregation -------------------------------------------------------------
MIN_SALES_PER_LSOA = 5           # drop thin LSOAs whose area median would be noisy
VALUE_COL = "median_log_price"   # areal value used for Moran's I / LISA
CHOROPLETH_VALUE_COL = "median_price"  # interpretable value for the display map

# --- Spatial weights + inference --------------------------------------------
WEIGHTS_METHOD = "queen"         # primary weights; "knn" is reported as robustness
KNN_K = 8
PERMUTATIONS = 999
LISA_ALPHA = 0.05
RANDOM_SEED = 42

# --- Outputs -----------------------------------------------------------------
REPORTS_DIR = REPO_ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "moran_metrics.json"

# Standard LISA cluster colours.
LISA_COLORS = {
    "HH": "#d7191c",   # high value, high-value neighbours (hot spot)
    "LL": "#2c7bb6",   # low value, low-value neighbours (cold spot)
    "LH": "#abd9e9",   # low value surrounded by high (spatial outlier)
    "HL": "#fdae61",   # high value surrounded by low (spatial outlier)
    "ns": "#e8e8e8",   # not significant
}
