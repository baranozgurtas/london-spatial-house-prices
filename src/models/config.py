"""Configuration for the modelling / CV-contrast stage."""
from __future__ import annotations

from ..data.config import PROCESSED_DIR, REPO_ROOT
from ..features.config import AREA_COL

# --- Data --------------------------------------------------------------------
FEATURES_PARQUET = PROCESSED_DIR / "london_ppd_2021_2024_features.parquet"
TARGET = "log_price"
PRICE_COL = "price"

# --- Feature groups ----------------------------------------------------------
NUMERIC_FEATURES = [
    "dist_cbd_m", "dist_bank_m", "dist_canary_wharf_m",
    "dist_thames_m", "dist_station_m",
    "poi_count_500m", "poi_count_1000m",
]
CATEGORICAL_FEATURES = ["property_type", "old_new", "duration", "year_quarter"]
COORD_FEATURES = ["easting", "northing"]

# --- Cross-validation --------------------------------------------------------
CV_K = 5
RANDOM_SEED = 42
N_SPATIAL_BLOCKS = 5
BUFFER_M = 1000.0          # dead-zone radius for buffered spatial CV
RESIDUAL_MORAN_KNN = 8

# --- Models ------------------------------------------------------------------
KNN_K = 10                 # spatial-kNN neighbours
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.001
LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)

DEFAULT_MODELS = [
    "global_mean", "global_median", "spatial_knn",
    "ridge_no_spatial", "ridge_spatial", "lasso_spatial", "lightgbm",
]
SCHEMES = ["random", "spatial", "spatial_buffered"]

# --- Outputs -----------------------------------------------------------------
REPORTS_DIR = REPO_ROOT / "reports"
TIDY_RESULTS_FILE = REPORTS_DIR / "cv_results_tidy.csv"
SUMMARY_RESULTS_FILE = REPORTS_DIR / "cv_results_summary.csv"
RESIDUAL_MORAN_FILE = REPORTS_DIR / "residual_moran.json"

AREA_COLUMN = AREA_COL
