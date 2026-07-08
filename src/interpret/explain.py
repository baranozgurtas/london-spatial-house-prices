"""SHAP interpretability for the honestly-fit LightGBM.

The model is fit on the spatial TRAINING blocks with a fold-safe spatial lag (not
a leaky global fit). SHAP then attributes the model's log-price predictions to
its inputs.

IMPORTANT: SHAP explains THE MODEL, not causation. A large SHAP magnitude for
distance-to-CBD means the model's predictions move strongly with that feature -
it does not establish the causal price effect of proximity, which is confounded
by everything correlated with location.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import (  # noqa: E402
    FIGURE_DIR,
    HEADLINE_FEATURE,
    HOLDOUT_BLOCK,
    LGBM_PARAMS,
    RANDOM_SEED,
    SHAP_IMPORTANCE_FILE,
    SHAP_SAMPLE,
    TARGET,
)
from ._common import build_lgbm_frame, honest_holdout, spatial_block_labels  # noqa: E402

LOGGER = logging.getLogger(__name__)


def run_shap(df: pd.DataFrame, adjacency, *, write: bool = True, figure_dir: Path = FIGURE_DIR):
    import shap
    from lightgbm import LGBMRegressor

    blocks = spatial_block_labels(df)
    train_df, _ = honest_holdout(df, blocks, HOLDOUT_BLOCK)
    x_tr, _, feat_names, cats = build_lgbm_frame(train_df, train_df, adjacency)
    model = LGBMRegressor(**LGBM_PARAMS).fit(
        x_tr, train_df[TARGET].to_numpy(float), categorical_feature=cats
    )

    sample = x_tr.sample(min(SHAP_SAMPLE, len(x_tr)), random_state=RANDOM_SEED)
    shap_values = shap.TreeExplainer(model).shap_values(sample)

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_names)
    importance = importance.sort_values(ascending=False)
    LOGGER.info("SHAP global importance (mean |SHAP|, log-price units):\n%s",
                importance.to_string())

    if write:
        figure_dir.mkdir(parents=True, exist_ok=True)
        _plot_importance(importance, figure_dir / "shap_importance.png")
        _plot_dependence(sample, shap_values, feat_names, HEADLINE_FEATURE,
                         figure_dir / f"shap_dependence_{HEADLINE_FEATURE}.png")
        SHAP_IMPORTANCE_FILE.write_text(json.dumps(importance.to_dict(), indent=2))
        LOGGER.info("SHAP note: explains the model, not causation.")

    return importance, model


def _plot_importance(importance: pd.Series, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.iloc[::-1].plot.barh(ax=ax, color="#2c7bb6")
    ax.set_xlabel("Mean |SHAP| (log-price units)")
    ax.set_title("LightGBM SHAP global importance (honest spatial fit)")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_dependence(sample, shap_values, feat_names, feature, out_png: Path) -> None:
    idx = feat_names.index(feature)
    x = sample[feature].to_numpy(float)
    y = shap_values[:, idx]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=8, alpha=0.4, color="#333333")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature} (log-price)")
    ax.set_title(f"SHAP dependence: {feature}\n(model behaviour, not causation)")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
