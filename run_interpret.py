import logging
import pandas as pd

from src.models.config import FEATURES_PARQUET
from src.eda.spatial_autocorrelation import load_lsoa_boundaries, detect_code_col
from src.features.spatial_lag import build_area_adjacency
from src.interpret.explain import run_shap
from src.interpret.uncertainty import run_uncertainty
from src.interpret.residual_maps import run_residual_maps

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

df = pd.read_parquet(FEATURES_PARQUET)
boundaries = load_lsoa_boundaries()
code_col = detect_code_col(boundaries)
adjacency = build_area_adjacency(boundaries, code_col)

print("\n===== SHAP =====")
run_shap(df, adjacency)

print("\n===== UNCERTAINTY =====")
res = run_uncertainty(df, adjacency)
print(res)

print("\n===== RESIDUAL MAPS =====")
run_residual_maps(df, adjacency, boundaries, code_col=code_col)

print("\nAll interpret outputs written to reports/figures/")
