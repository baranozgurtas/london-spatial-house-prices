"""Prediction intervals via conformalized quantile regression (CQR).

Method (Romano, Patterson & Candes, 2019):
  1. Fit LightGBM quantile models for the 5th and 95th percentiles on a
     train-proper subset.
  2. On a held-back calibration subset, compute conformity scores
     E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)) and take Q, the conformal
     quantile at the target coverage.
  3. The 90% prediction interval is [q_lo(x) - Q, q_hi(x) + Q].

INTERPRETATION: a 90% prediction interval - under EXCHANGEABILITY, about 90% of
actual sale prices fall inside it. To show what that assumption is worth here we
evaluate CQR two ways, mirroring the CV contrast:
  * random test split - exchangeability holds, so coverage should sit near 90%
    (CQR corrects base-model miscalibration by construction);
  * spatially held-out region - exchangeability is violated by spatial
    distribution shift, so coverage can fall below nominal.
Reporting the gap honestly is the point; we do not assume the guarantee transfers
to an unseen region.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CALIB_FRACTION,
    HOLDOUT_BLOCK,
    LGBM_PARAMS,
    NOMINAL_COVERAGE,
    PRICE_COL,
    QUANTILE_HI,
    QUANTILE_LO,
    RANDOM_SEED,
    TARGET,
    UNCERTAINTY_FILE,
)
from ._common import build_lgbm_frame, honest_holdout, spatial_block_labels

LOGGER = logging.getLogger(__name__)


def _quantile_model(alpha):
    from lightgbm import LGBMRegressor

    params = dict(LGBM_PARAMS)
    params.update(objective="quantile", alpha=alpha)
    return LGBMRegressor(**params)


def _coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def _cqr_once(train_df: pd.DataFrame, test_df: pd.DataFrame, adjacency) -> dict:
    """Fit quantiles + conformal calibration on train_df; evaluate on test_df."""
    proper_df, calib_df = train_test_split(
        train_df, test_size=CALIB_FRACTION, random_state=RANDOM_SEED
    )
    x_proper, x_calib, _, cats = build_lgbm_frame(proper_df, calib_df, adjacency)
    _, x_test, _, _ = build_lgbm_frame(proper_df, test_df, adjacency)
    y_proper = proper_df[TARGET].to_numpy(float)
    y_calib = calib_df[TARGET].to_numpy(float)
    y_test = test_df[TARGET].to_numpy(float)

    model_lo = _quantile_model(QUANTILE_LO).fit(x_proper, y_proper, categorical_feature=cats)
    model_hi = _quantile_model(QUANTILE_HI).fit(x_proper, y_proper, categorical_feature=cats)

    qlo_cal, qhi_cal = model_lo.predict(x_calib), model_hi.predict(x_calib)
    qlo_te, qhi_te = model_lo.predict(x_test), model_hi.predict(x_test)

    alpha = 1.0 - NOMINAL_COVERAGE
    scores = np.maximum(qlo_cal - y_calib, y_calib - qhi_cal)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = float(np.quantile(scores, level, method="higher"))

    lo_te, hi_te = qlo_te - Q, qhi_te + Q
    return {
        "coverage_raw_quantile": _coverage(y_test, qlo_te, qhi_te),
        "coverage_cqr": _coverage(y_test, lo_te, hi_te),
        "median_width_log": float(np.median(hi_te - lo_te)),
        "median_width_gbp": float(np.median(np.exp(hi_te) - np.exp(lo_te))),
        "conformal_Q_log": Q,
        "n_test": int(len(y_test)),
    }


def run_uncertainty(df: pd.DataFrame, adjacency, *, write: bool = True) -> dict:
    df = df.reset_index(drop=True)

    # 1) Spatially held-out region (exchangeability violated - the honest test).
    blocks = spatial_block_labels(df)
    train_sp, test_sp = honest_holdout(df, blocks, HOLDOUT_BLOCK)
    spatial = _cqr_once(train_sp, test_sp, adjacency)

    # 2) Random test split of matching size (exchangeability holds - the control).
    frac = max(0.1, min(0.5, len(test_sp) / len(df)))
    train_rd, test_rd = train_test_split(df, test_size=frac, random_state=RANDOM_SEED)
    random_ = _cqr_once(train_rd, test_rd, adjacency)

    result = {
        "nominal_coverage": NOMINAL_COVERAGE,
        "random_split": random_,
        "spatial_holdout": spatial,
    }
    LOGGER.info("CQR 90%% prediction intervals (coverage should be ~0.90 under exchangeability):")
    LOGGER.info("  random test split   -> CQR coverage %.3f (assumptions hold)",
                random_["coverage_cqr"])
    LOGGER.info("  spatial held-out    -> CQR coverage %.3f (spatial shift; honest)",
                spatial["coverage_cqr"])
    LOGGER.info("  median interval width (spatial): %.3f log (~ GBP %s)",
                spatial["median_width_log"], f"{spatial['median_width_gbp']:,.0f}")

    if write:
        UNCERTAINTY_FILE.parent.mkdir(parents=True, exist_ok=True)
        UNCERTAINTY_FILE.write_text(json.dumps(result, indent=2))
    return result
