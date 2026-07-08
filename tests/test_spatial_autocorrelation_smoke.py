"""Smoke test for the spatial-autocorrelation EDA on a synthetic grid.

We build an N x N grid of LSOA-like polygons carrying a deliberately
autocorrelated price surface (a smooth spatial gradient plus mild noise) and a
matching set of transactions. The pipeline should then report a strongly
positive, significant global Moran's I and find HH / LL LISA clusters, and all
figures + metrics should be written. No network and no real data.

Runnable directly (``python tests/test_spatial_autocorrelation_smoke.py``) or
under pytest.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eda.config import METRICS_FILE, VALUE_COL  # noqa: E402
from src.eda.spatial_autocorrelation import (  # noqa: E402
    aggregate_to_lsoa,
    build_spatial_weights,
    classify_lisa,
    global_moran,
    local_moran,
    run_eda,
)

GRID = 8            # 8 x 8 = 64 LSOA-like cells
CELL = 1000.0       # metres
SALES_PER_CELL = 12
RNG = np.random.default_rng(0)


def _grid_boundaries():
    from shapely.geometry import box
    import geopandas as gpd

    records = []
    for row in range(GRID):
        for col in range(GRID):
            code = f"E01{row:02d}{col:02d}"
            x0, y0 = col * CELL, row * CELL
            records.append({"LSOA21CD": code, "row": row, "col": col,
                            "geometry": box(x0, y0, x0 + CELL, y0 + CELL)})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=27700)


def _grid_transactions(boundaries) -> pd.DataFrame:
    """Prices follow a smooth diagonal gradient => strong positive autocorrelation."""
    rows = []
    for _, cell in boundaries.iterrows():
        # base log-price rises with row+col; small noise keeps areas distinct.
        base = 12.0 + 0.18 * (cell["row"] + cell["col"])
        for _ in range(SALES_PER_CELL):
            log_price = base + RNG.normal(0, 0.05)
            rows.append({"lsoa": cell["LSOA21CD"],
                         "log_price": log_price,
                         "price": float(np.exp(log_price))})
    return pd.DataFrame(rows)


def test_aggregate_drops_thin_lsoas():
    df = pd.DataFrame({
        "lsoa": ["A"] * 6 + ["B"] * 2,           # B has only 2 sales
        "price": [3e5] * 8,
        "log_price": [np.log(3e5)] * 8,
    })
    agg = aggregate_to_lsoa(df, min_sales=5)
    assert set(agg["lsoa"]) == {"A"}
    assert int(agg.loc[agg["lsoa"] == "A", "n_sales"].iloc[0]) == 6


def test_global_moran_strongly_positive():
    boundaries = _grid_boundaries()
    df = _grid_transactions(boundaries)
    agg = aggregate_to_lsoa(df)
    from src.eda.spatial_autocorrelation import attach_geometry
    gdf = attach_geometry(agg, boundaries, "LSOA21CD")
    w = build_spatial_weights(gdf, "queen")
    moran = global_moran(gdf[VALUE_COL].to_numpy(float), w)
    assert moran.I > 0.5, moran.I           # gradient => high autocorrelation
    assert moran.p_sim < 0.05, moran.p_sim


def test_lisa_finds_hot_and_cold_spots():
    boundaries = _grid_boundaries()
    df = _grid_transactions(boundaries)
    agg = aggregate_to_lsoa(df)
    from src.eda.spatial_autocorrelation import attach_geometry
    gdf = attach_geometry(agg, boundaries, "LSOA21CD")
    w = build_spatial_weights(gdf, "queen")
    lisa = local_moran(gdf[VALUE_COL].to_numpy(float), w)
    labels = classify_lisa(lisa)
    assert (labels == "HH").any()           # top-right corner
    assert (labels == "LL").any()           # bottom-left corner


def test_run_eda_writes_outputs():
    boundaries = _grid_boundaries()
    df = _grid_transactions(boundaries)
    with tempfile.TemporaryDirectory() as d:
        figdir = Path(d) / "figures"
        gdf, results = run_eda(processed_df=df, boundaries_gdf=boundaries,
                               write=True, figure_dir=figdir)
        assert results["moran_queen"].I > 0.5
        for name in ("choropleth_median_price.png", "moran_scatter.png",
                     "lisa_clusters.png", "choropleth_interactive.html"):
            assert (figdir / name).exists(), name
        assert METRICS_FILE.exists()
        payload = json.loads(METRICS_FILE.read_text())
        assert payload["global_moran_queen"]["I"] > 0.5
        assert "lisa_class" in gdf.columns


if __name__ == "__main__":
    test_aggregate_drops_thin_lsoas()
    test_global_moran_strongly_positive()
    test_lisa_finds_hot_and_cold_spots()
    test_run_eda_writes_outputs()
    print("OK: all spatial-EDA smoke tests passed")
