"""Spatial residual mapping: the visual capstone of the leakage thesis.

Computes LightGBM out-of-fold residuals under the HONEST spatial-CV protocol,
aggregates them to LSOA, and renders:
  * a diverging residual choropleth centered at zero, and
  * a LISA of residuals locating clusters of systematic over/under-prediction.

Residual = actual - predicted (log-price): positive => the model UNDER-predicts
(actual higher than predicted); negative => it OVER-predicts. If the residual
Moran's I is positive and significant, spatial structure remains and these maps
show where.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..eda.config import LISA_COLORS  # noqa: E402
from ..eda.spatial_autocorrelation import (  # noqa: E402
    build_spatial_weights,
    classify_lisa,
    detect_code_col,
    global_moran,
    local_moran,
)
from ..models.cv import build_folds  # noqa: E402
from ..models.evaluate import fit_predict  # noqa: E402
from .config import (  # noqa: E402
    AREA_COLUMN,
    BUFFER_M,
    CV_K,
    FIGURE_DIR,
    N_SPATIAL_BLOCKS,
    RANDOM_SEED,
    RESIDUAL_MAP_METRICS_FILE,
    TARGET,
)

LOGGER = logging.getLogger(__name__)


def lightgbm_oof_residuals(df: pd.DataFrame, adjacency) -> pd.Series:
    """LightGBM out-of-fold residuals under spatial-block CV (actual - predicted)."""
    df = df.reset_index(drop=True)
    y = df[TARGET].to_numpy(float)
    oof = np.full(len(df), np.nan)
    folds = build_folds(df, "spatial", area_col=AREA_COLUMN, k=CV_K, seed=RANDOM_SEED,
                        n_blocks=N_SPATIAL_BLOCKS, buffer_m=BUFFER_M)
    for tr_idx, va_idx in folds:
        val_pred, _ = fit_predict("lightgbm", df.iloc[tr_idx], df.iloc[va_idx], adjacency)
        oof[va_idx] = val_pred
    return pd.Series(y - oof, index=df[AREA_COLUMN], name="residual")


def run_residual_maps(df: pd.DataFrame, adjacency, boundaries, *, code_col: Optional[str] = None,
                      write: bool = True, figure_dir: Path = FIGURE_DIR) -> dict:
    import geopandas as gpd

    code_col = code_col or detect_code_col(boundaries)
    resid = lightgbm_oof_residuals(df, adjacency)
    lsoa_resid = resid.groupby(level=0).mean()

    merged = boundaries.merge(lsoa_resid.rename("residual"),
                              left_on=code_col, right_index=True, how="inner")
    gdf = gpd.GeoDataFrame(merged, geometry=boundaries.geometry.name, crs=boundaries.crs)
    gdf = gdf.to_crs(27700)

    w = build_spatial_weights(gdf, "queen")
    m_global = global_moran(gdf["residual"].to_numpy(float), w)
    lisa = local_moran(gdf["residual"].to_numpy(float), w)
    gdf["lisa_class"] = classify_lisa(lisa)

    remains = bool(m_global.p_sim < 0.05 and m_global.I > 0)
    metrics = {
        "residual_moran_I": float(m_global.I),
        "residual_moran_p": float(m_global.p_sim),
        "residual_autocorrelation_remains": remains,
        "n_lsoa": int(len(gdf)),
        "lisa_counts": gdf["lisa_class"].value_counts().to_dict(),
    }
    LOGGER.info("Residual Moran's I=%.3f p=%.4f -> autocorrelation %s",
                m_global.I, m_global.p_sim, "remains" if remains else "not significant")

    if write:
        figure_dir.mkdir(parents=True, exist_ok=True)
        _plot_residual_choropleth(gdf, figure_dir / "residual_choropleth.png")
        _plot_residual_lisa(gdf, figure_dir / "residual_lisa.png")
        RESIDUAL_MAP_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESIDUAL_MAP_METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    return metrics


def _plot_residual_choropleth(gdf, out_png: Path) -> None:
    vmax = float(np.nanmax(np.abs(gdf["residual"].to_numpy(float)))) or 1e-6
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="residual", cmap="RdBu_r",
             norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
             linewidth=0.05, edgecolor="white", legend=True,
             legend_kwds={"label": "OOF residual (log-price): + under / - over", "shrink": 0.6},
             ax=ax)
    ax.set_axis_off()
    ax.set_title("LightGBM spatial-CV residuals over London LSOAs", fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_lisa(gdf, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    handles = []
    for cls, color in LISA_COLORS.items():
        subset = gdf[gdf["lisa_class"] == cls]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=0.05, edgecolor="white")
            handles.append(Patch(facecolor=color, edgecolor="white",
                                 label=f"{cls} ({len(subset)})"))
    ax.set_axis_off()
    ax.set_title("LISA of residuals - clusters of systematic over/under-prediction",
                 fontsize=13)
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=9, title="Residual cluster")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
