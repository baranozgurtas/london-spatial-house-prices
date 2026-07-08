"""Model fitting, metrics, and the random-vs-spatial CV contrast (the money table).

Every data-derived component - the spatial-lag transformer, the scaler, the
one-hot encoder - is fit INSIDE each fold on training rows only, for both CV
schemes. Static geo features (distances, POI density) are location-deterministic
and were computed once upstream; they carry no target information.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features.spatial_lag import SpatialLagTransformer
from .config import (
    AREA_COLUMN,
    BUFFER_M,
    CATEGORICAL_FEATURES,
    COORD_FEATURES,
    CV_K,
    DEFAULT_MODELS,
    KNN_K,
    LASSO_ALPHA,
    LGBM_PARAMS,
    N_SPATIAL_BLOCKS,
    NUMERIC_FEATURES,
    PRICE_COL,
    RANDOM_SEED,
    RESIDUAL_MORAN_FILE,
    RESIDUAL_MORAN_KNN,
    REPORTS_DIR,
    SCHEMES,
    SUMMARY_RESULTS_FILE,
    TARGET,
    TIDY_RESULTS_FILE,
)
from .cv import build_folds, lsoa_centroids

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-fold feature assembly (fit on train only)
# --------------------------------------------------------------------------- #
def _fit_spatial_lag(train_df, val_df, adjacency):
    tr = SpatialLagTransformer(adjacency, area_col=AREA_COLUMN)
    tr.fit(train_df[[AREA_COLUMN]], train_df[TARGET].to_numpy(float))
    return tr.transform(train_df[[AREA_COLUMN]]).ravel(), tr.transform(val_df[[AREA_COLUMN]]).ravel()


def _linear_design(train_df, val_df, use_spatial_lag, adjacency):
    """Scaled numerics (+ optional spatial lag) plus one-hot categoricals.

    Scaler and encoder are fit on the training fold only.
    """
    num_tr = train_df[NUMERIC_FEATURES].to_numpy(float)
    num_va = val_df[NUMERIC_FEATURES].to_numpy(float)
    if use_spatial_lag:
        lag_tr, lag_va = _fit_spatial_lag(train_df, val_df, adjacency)
        num_tr = np.column_stack([num_tr, lag_tr])
        num_va = np.column_stack([num_va, lag_va])
    scaler = StandardScaler().fit(num_tr)
    num_tr, num_va = scaler.transform(num_tr), scaler.transform(num_va)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train_df[CATEGORICAL_FEATURES])
    cat_tr = ohe.transform(train_df[CATEGORICAL_FEATURES])
    cat_va = ohe.transform(val_df[CATEGORICAL_FEATURES])
    return np.column_stack([num_tr, cat_tr]), np.column_stack([num_va, cat_va])


def _lgbm_frames(train_df, val_df, adjacency):
    import lightgbm  # noqa: F401  (import guarded here so the dep is optional)

    lag_tr, lag_va = _fit_spatial_lag(train_df, val_df, adjacency)
    x_tr = train_df[NUMERIC_FEATURES].copy()
    x_va = val_df[NUMERIC_FEATURES].copy()
    x_tr["spatial_lag"] = lag_tr
    x_va["spatial_lag"] = lag_va
    for c in CATEGORICAL_FEATURES:
        x_tr[c] = train_df[c].astype("category")
        # Align validation codes to the training categories; unseen -> NaN.
        x_va[c] = pd.Categorical(val_df[c], categories=x_tr[c].cat.categories)
    return x_tr, x_va


# --------------------------------------------------------------------------- #
# Model registry: each returns (val_pred_log, train_pred_log)
# --------------------------------------------------------------------------- #
def fit_predict(model_name, train_df, val_df, adjacency):
    y_tr = train_df[TARGET].to_numpy(float)

    if model_name == "global_mean":
        c = float(np.mean(y_tr))
        return np.full(len(val_df), c), np.full(len(train_df), c)

    if model_name == "global_median":
        c = float(np.median(y_tr))
        return np.full(len(val_df), c), np.full(len(train_df), c)

    if model_name == "spatial_knn":
        knn = KNeighborsRegressor(n_neighbors=KNN_K, weights="distance")
        knn.fit(train_df[COORD_FEATURES].to_numpy(float), y_tr)
        return (knn.predict(val_df[COORD_FEATURES].to_numpy(float)),
                knn.predict(train_df[COORD_FEATURES].to_numpy(float)))

    if model_name in ("ridge_no_spatial", "ridge_spatial", "lasso_spatial"):
        use_lag = model_name != "ridge_no_spatial"
        x_tr, x_va = _linear_design(train_df, val_df, use_lag, adjacency)
        est = (Lasso(alpha=LASSO_ALPHA, max_iter=10000)
               if model_name.startswith("lasso") else Ridge(alpha=1.0))
        est.fit(x_tr, y_tr)
        return est.predict(x_va), est.predict(x_tr)

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        x_tr, x_va = _lgbm_frames(train_df, val_df, adjacency)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(x_tr, y_tr, categorical_feature=CATEGORICAL_FEATURES)
        return model.predict(x_va), model.predict(x_tr)

    raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true_log, y_pred_log, price_true, train_true_log, train_pred_log):
    resid = y_true_log - y_pred_log
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
    price_pred = np.exp(y_pred_log)
    medape = float(np.median(np.abs(price_pred - price_true) / price_true) * 100)

    # Duan smearing: debias exp(prediction) using the training residuals.
    smearing = float(np.mean(np.exp(train_true_log - train_pred_log)))
    mae_price = float(np.mean(np.abs(price_pred * smearing - price_true)))

    return {
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "mae_log": float(np.mean(np.abs(resid))),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "medape_pct": medape,
        "mae_price_gbp": mae_price,
        "duan_smearing": smearing,
    }


# --------------------------------------------------------------------------- #
# Residual spatial autocorrelation
# --------------------------------------------------------------------------- #
def residual_moran(lsoa_residual: pd.Series, centroids: pd.DataFrame, k: int = RESIDUAL_MORAN_KNN):
    import geopandas as gpd
    from esda.moran import Moran
    from libpysal.weights import KNN

    common = lsoa_residual.index.intersection(centroids.index)
    c = centroids.loc[common]
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(c["easting"], c["northing"]),
        crs=27700, index=common,
    )
    w = KNN.from_dataframe(pts, k=min(k, len(common) - 1))
    w.transform = "r"
    np.random.seed(RANDOM_SEED)
    m = Moran(lsoa_residual.loc[common].to_numpy(float), w)
    return {"I": float(m.I), "p_sim": float(m.p_sim), "z_sim": float(m.z_sim),
            "n_lsoa": int(len(common))}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(df: pd.DataFrame, adjacency, *, models=None, schemes=None,
                   area_col=AREA_COLUMN, write: bool = True):
    """Evaluate every model under every CV scheme; return tidy results + summaries."""
    models = models or DEFAULT_MODELS
    schemes = schemes or SCHEMES
    df = df.reset_index(drop=True)
    n = len(df)
    price = df[PRICE_COL].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    rows = []
    oof = {(m, s): np.full(n, np.nan) for m in models for s in schemes}

    for scheme in schemes:
        folds = build_folds(df, scheme, area_col=area_col, k=CV_K, seed=RANDOM_SEED,
                            n_blocks=N_SPATIAL_BLOCKS, buffer_m=BUFFER_M)
        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
            for model in models:
                val_pred, train_pred = fit_predict(model, train_df, val_df, adjacency)
                oof[(model, scheme)][va_idx] = val_pred
                met = compute_metrics(y[va_idx], val_pred, price[va_idx],
                                      y[tr_idx], train_pred)
                for metric, value in met.items():
                    rows.append({"model": model, "scheme": scheme,
                                 "fold": fold_id, "metric": metric, "value": value})
            LOGGER.info("%-16s fold %d/%d done (train=%d, val=%d)",
                        scheme, fold_id + 1, len(folds), len(tr_idx), len(va_idx))

    tidy = pd.DataFrame(rows)
    summary = summarize(tidy)

    # Residual Moran's I for the main model under random vs spatial CV.
    centroids = lsoa_centroids(df, area_col)
    resid_moran = {}
    for scheme in ("random", "spatial"):
        if ("lightgbm", scheme) in oof and not np.isnan(oof[("lightgbm", scheme)]).any():
            resid = pd.Series(y - oof[("lightgbm", scheme)], index=df[area_col])
            lsoa_resid = resid.groupby(level=0).mean()
            resid_moran[scheme] = residual_moran(lsoa_resid, centroids)

    if write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(TIDY_RESULTS_FILE, index=False)
        summary.to_csv(SUMMARY_RESULTS_FILE, index=False)
        RESIDUAL_MORAN_FILE.write_text(json.dumps(resid_moran, indent=2))
        LOGGER.info("Wrote %s, %s, %s", TIDY_RESULTS_FILE, SUMMARY_RESULTS_FILE,
                    RESIDUAL_MORAN_FILE)

    _log_money_table(summary)
    _log_residual_moran(resid_moran)
    return tidy, summary, resid_moran, oof


def summarize(tidy: pd.DataFrame) -> pd.DataFrame:
    agg = (tidy.groupby(["model", "scheme", "metric"])["value"]
           .agg(["mean", "std"]).reset_index())
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1)
    return agg


def money_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = summary[summary["metric"] == metric]
    return sub.pivot(index="model", columns="scheme", values="cell")


def _log_money_table(summary: pd.DataFrame) -> None:
    for metric in ("rmse_log", "r2", "medape_pct"):
        LOGGER.info("\n==== %s (mean +/- sd across folds) ====\n%s",
                    metric, money_table(summary, metric).to_string())


def _log_residual_moran(resid_moran: dict) -> None:
    if not resid_moran:
        return
    LOGGER.info("==== residual spatial autocorrelation (LightGBM OOF) ====")
    for scheme, m in resid_moran.items():
        verdict = ("remains" if (m["p_sim"] < 0.05 and m["I"] > 0)
                   else "not significant")
        LOGGER.info("%-16s Moran's I=%.3f  p=%.4f  -> residual autocorrelation %s",
                    scheme, m["I"], m["p_sim"], verdict)


def main() -> None:
    from .config import FEATURES_PARQUET
    from ..eda.spatial_autocorrelation import load_lsoa_boundaries, detect_code_col
    from ..features.spatial_lag import build_area_adjacency

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    df = pd.read_parquet(FEATURES_PARQUET)
    boundaries = load_lsoa_boundaries()
    adjacency = build_area_adjacency(boundaries, detect_code_col(boundaries))
    run_evaluation(df, adjacency)


if __name__ == "__main__":
    main()
