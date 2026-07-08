"""Fold-safe spatial lag of the target.

WHY THIS IS THE MOST LEAKAGE-PRONE FEATURE, AND HOW WE MAKE IT SAFE
------------------------------------------------------------------
The spatial lag of the target is, for each area, the row-standardised weighted
mean of its NEIGHBOURING areas' target. It is powerful precisely because house
prices are spatially autocorrelated - which is also exactly why it leaks if built
naively. Three structural safeguards, none relying on discipline at call sites:

  1. It is a scikit-learn transformer. Inside a CV loop it is ``fit`` on the
     TRAINING fold and ``transform``-ed on the validation fold, so only training
     targets ever enter it. Never fit it once globally.
  2. A row's OWN AREA target is excluded by construction: the lag for area i is
     built only from neighbouring areas j != i (contiguity/KNN weights have no
     self-loops). So even a row's own-area training target does not enter its lag.
  3. If none of an area's neighbours appear in the training fold (common under
     spatial/block CV, where a held-out region's neighbours are also held out),
     the lag falls back to the global training mean - i.e. we honestly admit we
     have no neighbourhood information for an unseen region.

Consequence worth noting: under RANDOM CV a validation area's neighbours sit in
the training fold, so this feature is informative and inflates the score; under
SPATIAL CV those neighbours are held out, so it degrades to the global mean. That
asymmetry is the leakage the project is designed to expose.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (
    ADJACENCY_KNN_K,
    ADJACENCY_METHOD,
    AREA_COL,
    SPATIAL_LAG_FEATURE,
)

LOGGER = logging.getLogger(__name__)


def build_area_adjacency(gdf, code_col: str, method: str = ADJACENCY_METHOD,
                         k: int = ADJACENCY_KNN_K) -> dict[str, dict[str, float]]:
    """Row-standardised area adjacency as {area: {neighbour: weight}}.

    Reuses the EDA weights builder (Queen contiguity with island handling, or
    KNN), keyed by LSOA code. Weights are row-standardised, so each area's
    neighbour weights sum to 1 and never include the area itself.
    """
    from ..eda.spatial_autocorrelation import build_spatial_weights

    g = gdf.drop_duplicates(subset=[code_col]).set_index(code_col)
    w = build_spatial_weights(g, method=method, k=k)
    return {
        area: {nb: float(wt) for nb, wt in zip(w.neighbors[area], w.weights[area])}
        for area in w.id_order
    }


class SpatialLagTransformer(BaseEstimator, TransformerMixin):
    """Neighbourhood mean of the target, fit on training rows only.

    Parameters
    ----------
    adjacency : dict[str, dict[str, float]]
        Row-standardised {area: {neighbour: weight}} mapping (see
        ``build_area_adjacency``). Built once from geometry; it is fixed
        structure, not data, so sharing it across folds does not leak.
    area_col : str
        Column in X holding each row's area code.
    agg : {"mean", "median"}
        How to aggregate the target within each training area before lagging.
    feature_name : str
        Output feature name.
    """

    def __init__(self, adjacency, area_col: str = AREA_COL, agg: str = "mean",
                 feature_name: str = SPATIAL_LAG_FEATURE):
        self.adjacency = adjacency
        self.area_col = area_col
        self.agg = agg
        self.feature_name = feature_name

    # -- helpers ----------------------------------------------------------- #
    def _areas(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            col = self.area_col if self.area_col in X.columns else X.columns[0]
            return np.asarray(X[col])
        if hasattr(X, "name"):            # a Series
            return np.asarray(X)
        return np.asarray(X).ravel()

    # -- sklearn API ------------------------------------------------------- #
    def fit(self, X, y):
        if y is None:
            raise ValueError("SpatialLagTransformer requires y (the target) at fit time.")
        areas = self._areas(X)
        y = np.asarray(y, dtype=float)
        if len(areas) != len(y):
            raise ValueError("X and y length mismatch.")

        # Per-area target aggregate FROM TRAINING ROWS ONLY.
        s = pd.Series(y, index=pd.Index(areas, name="area"))
        grp = s.groupby(level=0)
        train_area_value = (grp.median() if self.agg == "median" else grp.mean()).to_dict()
        self.global_value_ = float(np.mean(y))

        # Lag per area = weighted mean of NEIGHBOURS' training values only,
        # renormalised over whichever neighbours are present in training. The
        # area's own value is never used (no self in adjacency).
        lag_map: dict[str, float] = {}
        for area, neighbours in self.adjacency.items():
            num = wsum = 0.0
            for nb, w in neighbours.items():
                v = train_area_value.get(nb)
                if v is not None:
                    num += w * v
                    wsum += w
            lag_map[area] = (num / wsum) if wsum > 0 else np.nan
        self.lag_map_ = lag_map
        self.n_features_out_ = 1
        return self

    def transform(self, X) -> np.ndarray:
        areas = self._areas(X)
        vals = np.array([self.lag_map_.get(a, np.nan) for a in areas], dtype=float)
        # Fallback for areas with no training-neighbour information.
        vals = np.where(np.isnan(vals), self.global_value_, vals)
        return vals.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([self.feature_name], dtype=object)
