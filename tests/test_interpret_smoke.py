"""Smoke test for the interpret stage on synthetic data.

Validates that SHAP runs on the honestly-fit model and returns finite importances
(including distance-to-CBD), that CQR produces sane coverages and positive-width
intervals, and that the residual pipeline computes Moran's I, assigns LISA
classes, and writes every figure. No network, no real data.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.spatial_lag import build_area_adjacency  # noqa: E402
from src.interpret.config import HEADLINE_FEATURE  # noqa: E402
from src.interpret.explain import run_shap  # noqa: E402
from src.interpret.residual_maps import run_residual_maps  # noqa: E402
from src.interpret.uncertainty import run_uncertainty  # noqa: E402
from tests.test_modeling_smoke import _synthetic  # noqa: E402


def test_shap_runs_on_honest_model():
    df, gdf = _synthetic()
    adjacency = build_area_adjacency(gdf, "lsoa", method="queen")
    with tempfile.TemporaryDirectory() as d:
        importance, model = run_shap(df, adjacency, write=True, figure_dir=Path(d))
        assert HEADLINE_FEATURE in importance.index
        assert np.isfinite(importance.to_numpy()).all()
        assert (importance >= 0).all()
        assert (Path(d) / "shap_importance.png").exists()
        assert (Path(d) / f"shap_dependence_{HEADLINE_FEATURE}.png").exists()


def test_cqr_intervals_are_sane():
    df, gdf = _synthetic()
    adjacency = build_area_adjacency(gdf, "lsoa", method="queen")
    res = run_uncertainty(df, adjacency, write=False)
    for split in ("random_split", "spatial_holdout"):
        assert 0.0 <= res[split]["coverage_cqr"] <= 1.0
        assert res[split]["median_width_log"] > 0
        assert res[split]["median_width_gbp"] > 0
    # Under exchangeability CQR should not undercover more than the spatial region.
    assert res["random_split"]["coverage_cqr"] >= res["spatial_holdout"]["coverage_cqr"] - 1e-9


def test_residual_maps_pipeline():
    df, gdf = _synthetic()
    adjacency = build_area_adjacency(gdf, "lsoa", method="queen")
    with tempfile.TemporaryDirectory() as d:
        metrics = run_residual_maps(df, adjacency, gdf, code_col="lsoa",
                                    write=True, figure_dir=Path(d))
        assert "residual_moran_I" in metrics and np.isfinite(metrics["residual_moran_I"])
        assert (Path(d) / "residual_choropleth.png").exists()
        assert (Path(d) / "residual_lisa.png").exists()


if __name__ == "__main__":
    test_shap_runs_on_honest_model()
    test_cqr_intervals_are_sane()
    test_residual_maps_pipeline()
    print("OK: interpret smoke tests passed")
