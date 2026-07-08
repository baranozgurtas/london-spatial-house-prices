"""Geospatial EDA: LSOA aggregation, choropleths, global Moran's I and LISA.

This stage produces the spatial-autocorrelation evidence that motivates the
random-CV-vs-spatial-CV contrast. If the LSOA price surface is strongly and
significantly autocorrelated, a naive random split is expected to leak
neighbourhood signal - which is exactly what the modelling stage will show.

Pipeline:
    1. Load the processed transactions (Parquet) and LSOA 2021 boundaries.
    2. Aggregate transactions to LSOA (median log-price / median price, counts),
       dropping thin LSOAs below a minimum sale count.
    3. Attach geometry and build row-standardised spatial weights
       (Queen contiguity primary; KNN as a robustness check).
    4. Compute global Moran's I (permutation inference) under both weights.
    5. Compute local Moran's I (LISA), classify HH/LL/HL/LH clusters.
    6. Save a choropleth (static + interactive), a Moran scatterplot, a LISA
       cluster map, and a small metrics JSON; log the headline evidence.

Run (from the repository root):
    python -m src.eda.spatial_autocorrelation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import (  # noqa: E402
    CHOROPLETH_VALUE_COL,
    CRS_BNG,
    CRS_WGS84,
    FIGURE_DIR,
    KNN_K,
    LISA_ALPHA,
    LISA_COLORS,
    LSOA_BOUNDARY_FILE,
    LSOA_CODE_CANDIDATES,
    METRICS_FILE,
    MIN_SALES_PER_LSOA,
    PERMUTATIONS,
    PROCESSED_PARQUET,
    RANDOM_SEED,
    VALUE_COL,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_processed(path: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed transactions not found at {path}. Run the ingestion stage "
            "first: python -m src.data.ingest"
        )
    return pd.read_parquet(path)


def load_lsoa_boundaries(path: Path = LSOA_BOUNDARY_FILE):
    import geopandas as gpd

    if not path.exists():
        raise FileNotFoundError(
            f"LSOA boundaries not found at {path}. Download the LSOA (December 2021) "
            "EW boundaries (Generalised Clipped, BGC) from the ONS Open Geography "
            "Portal (geoportal.statistics.gov.uk) and place the GeoJSON there."
        )
    gdf = gpd.read_file(path)
    # ONS boundaries are British National Grid. Force EPSG:27700 for correct
    # planar contiguity/distance; set it if the file omits a CRS.
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_BNG)
    else:
        gdf = gdf.to_crs(CRS_BNG)
    return gdf


def detect_code_col(gdf) -> str:
    for cand in LSOA_CODE_CANDIDATES:
        if cand in gdf.columns:
            return cand
    for col in gdf.columns:
        upper = col.upper()
        if upper.startswith("LSOA") and upper.endswith("CD"):
            return col
    raise KeyError(
        f"Could not find an LSOA code column in {list(gdf.columns)}. "
        "Add the correct name to LSOA_CODE_CANDIDATES in eda/config.py."
    )


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def aggregate_to_lsoa(df: pd.DataFrame, min_sales: int = MIN_SALES_PER_LSOA) -> pd.DataFrame:
    """Collapse transactions to one row per LSOA, dropping thin areas."""
    grouped = df.groupby("lsoa").agg(
        n_sales=("price", "size"),
        median_price=("price", "median"),
        median_log_price=("log_price", "median"),
        mean_log_price=("log_price", "mean"),
    )
    kept = grouped[grouped["n_sales"] >= min_sales].copy()
    LOGGER.info(
        "Aggregated to %d LSOAs; dropped %d below %d sales.",
        len(kept), len(grouped) - len(kept), min_sales,
    )
    return kept.reset_index()


def attach_geometry(agg: pd.DataFrame, boundaries, code_col: str):
    """Inner-join aggregated LSOA values onto their polygons (London subset)."""
    import geopandas as gpd

    merged = boundaries.merge(agg, left_on=code_col, right_on="lsoa", how="inner")
    gdf = gpd.GeoDataFrame(merged, geometry=boundaries.geometry.name, crs=boundaries.crs)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    return gdf.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Spatial weights
# --------------------------------------------------------------------------- #
def build_spatial_weights(gdf, method: str = "queen", k: int = KNN_K):
    """Row-standardised weights. Queen contiguity attaches islands via KNN(1)."""
    from libpysal.weights import KNN, Queen

    def _queen():
        try:
            return Queen.from_dataframe(gdf, use_index=True)
        except TypeError:
            return Queen.from_dataframe(gdf)

    def _knn(kk):
        try:
            return KNN.from_dataframe(gdf, k=kk)
        except TypeError:
            return KNN.from_dataframe(gdf, k=kk, use_index=True)

    if method == "knn":
        w = _knn(k)
    else:
        w = _queen()
        if w.islands:
            LOGGER.warning("Queen weights have %d island(s); attaching via KNN(1).",
                           len(w.islands))
            try:
                try:
                    from libpysal.weights import attach_islands
                except ImportError:
                    from libpysal.weights.util import attach_islands
                w = attach_islands(w, _knn(1))
            except Exception as err:  # noqa: BLE001
                LOGGER.warning("attach_islands failed (%s); falling back to KNN(%d).",
                               err, k)
                w = _knn(k)
    w.transform = "r"
    return w


# --------------------------------------------------------------------------- #
# Autocorrelation statistics
# --------------------------------------------------------------------------- #
def global_moran(values: np.ndarray, w, permutations: int = PERMUTATIONS):
    from esda.moran import Moran

    np.random.seed(RANDOM_SEED)
    return Moran(values, w, permutations=permutations)


def local_moran(values: np.ndarray, w, permutations: int = PERMUTATIONS):
    from esda.moran import Moran_Local

    np.random.seed(RANDOM_SEED)
    # Pin the current conditional-randomisation semantics explicitly: a future
    # esda release changes the default to 'two-sided'. 'directed' keeps the
    # cluster counts stable across esda versions.
    return Moran_Local(values, w, permutations=permutations, alternative="directed")


def classify_lisa(lisa, alpha: float = LISA_ALPHA) -> np.ndarray:
    """Map LISA quadrants to HH/LH/LL/HL, or 'ns' when not significant.

    esda quadrant codes: 1=HH, 2=LH, 3=LL, 4=HL.
    """
    names = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    significant = lisa.p_sim < alpha
    return np.array(
        [names[q] if sig else "ns" for q, sig in zip(lisa.q, significant)]
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_choropleth(gdf, value_col: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column=value_col, scheme="Quantiles", k=7, cmap="plasma",
        linewidth=0.05, edgecolor="white", legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 8, "title": value_col},
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title("Greater London LSOA house prices, 2021-2024", fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_moran_scatter(gdf, value_col: str, w, moran, out_png: Path) -> None:
    from libpysal.weights import lag_spatial

    y = gdf[value_col].to_numpy(dtype=float)
    z = (y - y.mean()) / y.std()
    lag = lag_spatial(w, z)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z, lag, s=8, alpha=0.4, color="#333333")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.axvline(0, color="grey", linewidth=0.7)
    xs = np.array([z.min(), z.max()])
    ax.plot(xs, moran.I * xs, color="#d7191c", linewidth=1.6,
            label=f"slope = Moran's I = {moran.I:.3f}")
    ax.set_xlabel("Standardised median log-price (z)")
    ax.set_ylabel("Spatial lag of z")
    ax.set_title("Moran scatterplot")
    ax.legend(fontsize=9)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lisa(gdf, out_png: Path) -> None:
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 10))
    handles = []
    for cls, color in LISA_COLORS.items():
        subset = gdf[gdf["lisa_class"] == cls]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=0.05, edgecolor="white")
            handles.append(Patch(facecolor=color, edgecolor="white",
                                 label=f"{cls} ({len(subset)})"))
    ax.set_axis_off()
    ax.set_title("LISA clusters - median log-price (p < %.2f)" % LISA_ALPHA, fontsize=13)
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=9, title="Cluster")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_interactive(gdf, value_col: str, out_html: Path) -> None:
    """Folium interactive choropleth (reprojected to WGS84 for web display)."""
    cols = [value_col, "n_sales", "lsoa", "lisa_class", "geometry"]
    web = gdf[[c for c in cols if c in gdf.columns]].to_crs(CRS_WGS84)
    m = web.explore(
        column=value_col, scheme="Quantiles", k=7, cmap="plasma",
        tiles="CartoDB positron", legend=True,
        style_kwds={"weight": 0.2, "fillOpacity": 0.75},
        name="LSOA median price",
    )
    m.save(str(out_html))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _moran_dict(moran) -> dict:
    return {
        "I": float(moran.I),
        "expected_I": float(moran.EI),
        "z_sim": float(moran.z_sim),
        "p_sim": float(moran.p_sim),
        "permutations": int(moran.permutations),
    }


def save_metrics(mi_queen, mi_knn, labels: np.ndarray, n_lsoa: int, path: Path) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    payload = {
        "n_lsoa": int(n_lsoa),
        "value_col": VALUE_COL,
        "global_moran_queen": _moran_dict(mi_queen),
        "global_moran_knn": _moran_dict(mi_knn),
        "lisa_cluster_counts": {str(k): int(v) for k, v in zip(unique, counts)},
        "lisa_alpha": LISA_ALPHA,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _log_headline(mi_queen, mi_knn, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    counts_by = dict(zip(unique, counts))
    LOGGER.info("================ autocorrelation evidence ================")
    LOGGER.info("Global Moran's I (Queen): I=%.3f  z=%.1f  p=%.4f",
                mi_queen.I, mi_queen.z_sim, mi_queen.p_sim)
    LOGGER.info("Global Moran's I (KNN%2d): I=%.3f  z=%.1f  p=%.4f",
                KNN_K, mi_knn.I, mi_knn.z_sim, mi_knn.p_sim)
    LOGGER.info("Significant LISA clusters: HH=%d  LL=%d  HL=%d  LH=%d  (ns=%d)",
                counts_by.get("HH", 0), counts_by.get("LL", 0),
                counts_by.get("HL", 0), counts_by.get("LH", 0),
                counts_by.get("ns", 0))
    LOGGER.info("=> Strong positive autocorrelation implies random CV will leak "
                "neighbourhood signal; spatial CV is required for an honest estimate.")
    LOGGER.info("=========================================================")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(
    processed_df: Optional[pd.DataFrame] = None,
    boundaries_gdf=None,
    *,
    write: bool = True,
    figure_dir: Path = FIGURE_DIR,
):
    """Run the full spatial-EDA pipeline. Returns (gdf, results_dict)."""
    df = processed_df if processed_df is not None else load_processed()
    boundaries = boundaries_gdf if boundaries_gdf is not None else load_lsoa_boundaries()
    code_col = detect_code_col(boundaries)

    agg = aggregate_to_lsoa(df)
    gdf = attach_geometry(agg, boundaries, code_col)
    LOGGER.info("LSOAs with geometry and sufficient sales: %d", len(gdf))
    if len(gdf) < 3:
        raise ValueError("Too few LSOAs after aggregation/join to compute Moran's I.")

    values = gdf[VALUE_COL].to_numpy(dtype=float)
    w_queen = build_spatial_weights(gdf, "queen")
    w_knn = build_spatial_weights(gdf, "knn", k=KNN_K)

    mi_queen = global_moran(values, w_queen)
    mi_knn = global_moran(values, w_knn)
    lisa = local_moran(values, w_queen)
    labels = classify_lisa(lisa)
    gdf = gdf.assign(lisa_class=labels)

    if write:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_choropleth(gdf, CHOROPLETH_VALUE_COL, figure_dir / "choropleth_median_price.png")
        plot_moran_scatter(gdf, VALUE_COL, w_queen, mi_queen, figure_dir / "moran_scatter.png")
        plot_lisa(gdf, figure_dir / "lisa_clusters.png")
        make_interactive(gdf, CHOROPLETH_VALUE_COL, figure_dir / "choropleth_interactive.html")
        save_metrics(mi_queen, mi_knn, labels, len(gdf), METRICS_FILE)
        LOGGER.info("Figures written to %s; metrics to %s", figure_dir, METRICS_FILE)

    _log_headline(mi_queen, mi_knn, labels)
    return gdf, {"moran_queen": mi_queen, "moran_knn": mi_knn,
                 "lisa": lisa, "labels": labels}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_eda()


if __name__ == "__main__":
    main()
