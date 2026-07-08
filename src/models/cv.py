"""Cross-validation fold construction.

Two schemes:
  * random k-fold - shuffle transactions, ignore geography (the naive baseline).
  * spatial-block CV - KMeans on LSOA centroids gives k spatially compact blocks;
    each block is held out in turn. Optionally a buffer (dead-zone) removes
    TRAINING rows within a distance of the held-out block's points, so adjacency
    leakage across the fold boundary is eliminated.

Every scheme partitions all rows into validation exactly once, so out-of-fold
predictions cover the whole dataset.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from .config import COORD_FEATURES

LOGGER = logging.getLogger(__name__)


def lsoa_centroids(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Approximate LSOA centroids from transaction coordinates (EPSG:27700)."""
    return df.groupby(area_col)[COORD_FEATURES].mean()


def kmeans_blocks(centroids: pd.DataFrame, n_blocks: int, seed: int) -> pd.Series:
    """Assign each LSOA to a spatially compact block via KMeans on centroids."""
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    labels = km.fit_predict(centroids[COORD_FEATURES].to_numpy(float))
    return pd.Series(labels, index=centroids.index, name="block")


def random_folds(n_rows: int, k: int, seed: int):
    """Standard shuffled k-fold over row positions."""
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n_rows)))


def spatial_folds(df: pd.DataFrame, block_of_lsoa: pd.Series, area_col: str,
                  buffer_m: float | None = None):
    """Hold out one spatial block at a time; optionally buffer the training set."""
    coords = df[COORD_FEATURES].to_numpy(float)
    row_block = df[area_col].map(block_of_lsoa).to_numpy()
    folds = []
    for b in sorted(pd.unique(block_of_lsoa)):
        val_idx = np.where(row_block == b)[0]
        train_idx = np.where(row_block != b)[0]
        if buffer_m:
            # Drop training rows within buffer_m of any held-out (validation) point.
            tree = cKDTree(coords[val_idx])
            nearest, _ = tree.query(coords[train_idx], k=1)
            kept = train_idx[nearest >= buffer_m]
            LOGGER.debug("Block %s: buffer dropped %d training rows.",
                         b, len(train_idx) - len(kept))
            train_idx = kept
        folds.append((train_idx, val_idx))
    return folds


def build_folds(df: pd.DataFrame, scheme: str, *, area_col: str, k: int, seed: int,
                n_blocks: int, buffer_m: float):
    """Return a list of (train_idx, val_idx) for the requested scheme."""
    if scheme == "random":
        return random_folds(len(df), k, seed)
    centroids = lsoa_centroids(df, area_col)
    blocks = kmeans_blocks(centroids, n_blocks, seed)
    if scheme == "spatial":
        return spatial_folds(df, blocks, area_col, buffer_m=None)
    if scheme == "spatial_buffered":
        return spatial_folds(df, blocks, area_col, buffer_m=buffer_m)
    raise ValueError(f"Unknown scheme: {scheme}")
