"""Shared helpers: the honest spatial hold-out and the LightGBM design frame.

All three interpret modules (SHAP, uncertainty, residual maps) fit on the spatial
TRAINING blocks with a fold-safe spatial lag, so none of them reintroduces the
leakage the CV contrast exposed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.spatial_lag import SpatialLagTransformer
from ..models.cv import kmeans_blocks, lsoa_centroids
from .config import (
    AREA_COLUMN,
    CATEGORICAL_FEATURES,
    N_SPATIAL_BLOCKS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET,
)


def spatial_block_labels(df: pd.DataFrame, area_col: str = AREA_COLUMN,
                         n_blocks: int = N_SPATIAL_BLOCKS, seed: int = RANDOM_SEED) -> pd.Series:
    return kmeans_blocks(lsoa_centroids(df, area_col), n_blocks, seed)


def honest_holdout(df: pd.DataFrame, block_of_lsoa: pd.Series, holdout_block: int,
                   area_col: str = AREA_COLUMN):
    """Hold out one spatial block as the unseen test region."""
    row_block = df[area_col].map(block_of_lsoa)
    train = df[row_block != holdout_block].copy()
    test = df[row_block == holdout_block].copy()
    return train, test


def build_lgbm_frame(train_df: pd.DataFrame, other_df: pd.DataFrame, adjacency,
                     *, with_lag: bool = True):
    """Numeric + fold-safe spatial lag + integer-coded categoricals.

    The spatial lag is fit on train_df only. Categoricals are integer-coded using
    the training categories (unseen -> -1), which keeps SHAP robust to category
    dtypes. Returns (X_train, X_other, feature_names, categorical_names).
    """
    x_tr = train_df[NUMERIC_FEATURES].copy()
    x_ot = other_df[NUMERIC_FEATURES].copy()
    if with_lag:
        lag = SpatialLagTransformer(adjacency, area_col=AREA_COLUMN)
        lag.fit(train_df[[AREA_COLUMN]], train_df[TARGET].to_numpy(float))
        x_tr["spatial_lag"] = lag.transform(train_df[[AREA_COLUMN]]).ravel()
        x_ot["spatial_lag"] = lag.transform(other_df[[AREA_COLUMN]]).ravel()
    for c in CATEGORICAL_FEATURES:
        categories = train_df[c].astype("category").cat.categories
        x_tr[c] = pd.Categorical(train_df[c], categories=categories).codes
        x_ot[c] = pd.Categorical(other_df[c], categories=categories).codes
    return x_tr, x_ot, list(x_tr.columns), list(CATEGORICAL_FEATURES)
