"""Configuration for interpretability, uncertainty and residual mapping."""
from __future__ import annotations

from ..models.config import (  # noqa: F401  (re-exported for convenience)
    AREA_COLUMN,
    BUFFER_M,
    CATEGORICAL_FEATURES,
    CV_K,
    LGBM_PARAMS,
    N_SPATIAL_BLOCKS,
    NUMERIC_FEATURES,
    PRICE_COL,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET,
)

FIGURE_DIR = REPORTS_DIR / "figures"

# SHAP
HEADLINE_FEATURE = "dist_cbd_m"   # dependence plot feature (distance to Charing Cross)
SHAP_SAMPLE = 2000                # rows explained (sampled from the training blocks)

# Honest spatial hold-out
HOLDOUT_BLOCK = 0                 # which KMeans block is held out as the unseen region

# Conformalised quantile regression
QUANTILE_LO = 0.05
QUANTILE_HI = 0.95
NOMINAL_COVERAGE = 0.90           # target interval coverage
CALIB_FRACTION = 0.30             # share of the training blocks used for conformal calibration

# Output files
SHAP_IMPORTANCE_FILE = REPORTS_DIR / "shap_importance.json"
UNCERTAINTY_FILE = REPORTS_DIR / "uncertainty.json"
RESIDUAL_MAP_METRICS_FILE = REPORTS_DIR / "residual_map_metrics.json"
