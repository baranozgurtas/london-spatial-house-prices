"""Smoke test for the modelling / CV-contrast harness on synthetic data.

We build a spatially autocorrelated price surface over a grid of LSOAs and assert
that the harness REVEALS the leakage signature:
  * spatial-kNN (the leakage-prone baseline) scores much better under random CV
    than under spatial CV,
  * buffered spatial CV is no more optimistic than unbuffered,
  * the money table and residual Moran's I are produced.

No network, no real data. Runnable directly or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.spatial_lag import build_area_adjacency  # noqa: E402
from src.models.evaluate import money_table, run_evaluation  # noqa: E402

RNG = np.random.default_rng(7)
GRID = 6            # 6 x 6 = 36 LSOAs
CELL = 1000.0
SALES = 40


def _synthetic():
    import geopandas as gpd
    from shapely.geometry import box

    cells, rows = [], []
    for r in range(GRID):
        for c in range(GRID):
            code = f"E01{r:02d}{c:02d}"
            x0, y0 = c * CELL, r * CELL
            cells.append({"lsoa": code, "geometry": box(x0, y0, x0 + CELL, y0 + CELL)})
            for _ in range(SALES):
                ex = x0 + RNG.uniform(50, CELL - 50)
                ny = y0 + RNG.uniform(50, CELL - 50)
                # Smooth spatial surface in log-price + property effects + noise.
                base = 12.0 + 0.00035 * ex + 0.00035 * ny
                ptype = RNG.choice(["D", "S", "T", "F"])
                bump = {"D": 0.25, "S": 0.12, "T": 0.0, "F": -0.1}[ptype]
                logp = base + bump + RNG.normal(0, 0.06)
                rows.append({
                    "lsoa": code, "easting": ex, "northing": ny,
                    "log_price": logp, "price": float(np.exp(logp)),
                    "property_type": ptype,
                    "old_new": RNG.choice(["Y", "N"]),
                    "duration": RNG.choice(["F", "L"]),
                    "year_quarter": RNG.choice(["2021Q1", "2022Q3", "2023Q2", "2024Q4"]),
                    "dist_cbd_m": float(np.hypot(ex, ny)),
                    "dist_bank_m": float(np.hypot(ex - 3000, ny - 3000)),
                    "dist_canary_wharf_m": float(np.hypot(ex - 5000, ny - 1000)),
                    "dist_thames_m": float(abs(ny - 3000)),
                    "dist_station_m": float(RNG.uniform(50, 1200)),
                    "poi_count_500m": int(RNG.integers(0, 40)),
                    "poi_count_1000m": int(RNG.integers(0, 120)),
                })
    gdf = gpd.GeoDataFrame(cells, geometry="geometry", crs=27700)
    return pd.DataFrame(rows), gdf


def test_cv_contrast_reveals_leakage():
    df, gdf = _synthetic()
    adjacency = build_area_adjacency(gdf, "lsoa", method="queen")
    tidy, summary, resid_moran, oof = run_evaluation(df, adjacency, write=False)

    # All models under all three schemes are present.
    assert set(tidy["scheme"]) == {"random", "spatial", "spatial_buffered"}
    assert set(tidy["model"]) >= {"spatial_knn", "lightgbm", "ridge_spatial"}

    def mean_rmse(model, scheme):
        s = tidy[(tidy.model == model) & (tidy.scheme == scheme)
                 & (tidy.metric == "rmse_log")]
        return s["value"].mean()

    # The leakage signature: spatial-kNN looks far better under random CV.
    assert mean_rmse("spatial_knn", "random") < mean_rmse("spatial_knn", "spatial")

    # Buffered spatial CV is at least as pessimistic as unbuffered.
    assert (mean_rmse("spatial_knn", "spatial_buffered")
            >= mean_rmse("spatial_knn", "spatial") - 1e-9)

    # Money table builds and residual Moran is produced for LightGBM.
    mt = money_table(summary, "rmse_log")
    assert "random" in mt.columns and "spatial" in mt.columns
    assert "random" in resid_moran and "spatial" in resid_moran
    assert set(resid_moran["random"]) == {"I", "p_sim", "z_sim", "n_lsoa"}


if __name__ == "__main__":
    test_cv_contrast_reveals_leakage()
    print("OK: modelling CV-contrast smoke test passed")
