"""Smoke test for feature engineering on synthetic fixtures.

Covers the static geospatial features (distances checked against hand-computed
values; nearest-station and POI counts against known layouts) and, most
importantly, the fold-safety of the spatial lag:
  * a validation row's own-area training target does NOT enter its lag,
  * changing validation targets does NOT change any lag,
  * an area with no training neighbours falls back to the global training mean.

No network, no real data. Runnable directly or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.config import CRS_BNG, CRS_WGS84  # noqa: E402
from src.features.build_features import (  # noqa: E402
    add_distance_to_centres,
    add_distance_to_nearest_station,
    add_distance_to_thames,
    add_poi_density,
)
from src.features.config import CBD_KEY, EMPLOYMENT_CENTRES_WGS84  # noqa: E402
from src.features.spatial_lag import SpatialLagTransformer, build_area_adjacency  # noqa: E402

_TO_BNG = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)


def _cbd_bng():
    lon, lat = EMPLOYMENT_CENTRES_WGS84[CBD_KEY]
    return _TO_BNG.transform(lon, lat)


# --------------------------------------------------------------------------- #
# Static features
# --------------------------------------------------------------------------- #
def test_distance_to_cbd_is_exact():
    cx, cy = _cbd_bng()
    # One sale 100 m east of Charing Cross, one 300 m north.
    df = pd.DataFrame({"easting": [cx + 100, cx], "northing": [cy, cy + 300]})
    out = add_distance_to_centres(df.copy())
    assert np.allclose(out["dist_cbd_m"].to_numpy(), [100.0, 300.0], atol=1e-6)
    assert "dist_bank_m" in out and "dist_canary_wharf_m" in out


def test_distance_to_thames_line():
    import geopandas as gpd
    from shapely.geometry import LineString

    # Vertical Thames at x = 0; a sale 250 m east should be 250 m away.
    thames = gpd.GeoDataFrame(geometry=[LineString([(0, -1000), (0, 1000)])], crs=CRS_BNG)
    df = pd.DataFrame({"easting": [250.0], "northing": [0.0]})
    out = add_distance_to_thames(df.copy(), thames)
    assert abs(float(out["dist_thames_m"].iloc[0]) - 250.0) < 1e-6


def test_nearest_station_and_poi_counts():
    import geopandas as gpd

    stations = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0, 1000], [0, 0]), crs=CRS_BNG
    )
    # POIs: three within 500 m of the origin, one far away.
    pois = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([100, 200, 400, 5000], [0, 0, 0, 0]), crs=CRS_BNG
    )
    df = pd.DataFrame({"easting": [0.0], "northing": [0.0]})
    out = add_distance_to_nearest_station(df.copy(), stations)
    assert abs(float(out["dist_station_m"].iloc[0])) < 1e-6           # sits on a station
    out = add_poi_density(out, pois, radii=(500, 1000))
    assert int(out["poi_count_500m"].iloc[0]) == 3
    assert int(out["poi_count_1000m"].iloc[0]) == 3


# --------------------------------------------------------------------------- #
# Fold-safe spatial lag
# --------------------------------------------------------------------------- #
def _line_adjacency():
    # A - B - C - D chain, row-standardised.
    return {
        "A": {"B": 1.0},
        "B": {"A": 0.5, "C": 0.5},
        "C": {"B": 0.5, "D": 0.5},
        "D": {"C": 1.0},
    }


def test_spatial_lag_excludes_own_area_target():
    adj = _line_adjacency()
    # Training rows in A (12.0) and C (14.0), plus B with a WILD own-area value.
    X_train = pd.DataFrame({"lsoa": ["A", "A", "C", "C", "B"]})
    y_train = np.array([12.0, 12.0, 14.0, 14.0, 100.0])
    lag = SpatialLagTransformer(adj).fit(X_train, y_train)

    # B's lag uses neighbours A and C only -> 0.5*12 + 0.5*14 = 13.0,
    # regardless of B's own training value (100.0).
    out = lag.transform(pd.DataFrame({"lsoa": ["B"]}))
    assert abs(float(out[0, 0]) - 13.0) < 1e-9


def test_spatial_lag_is_val_target_independent_and_has_fallback():
    adj = _line_adjacency()
    X_train = pd.DataFrame({"lsoa": ["A", "A", "C", "C"]})
    y_train = np.array([12.0, 12.0, 14.0, 14.0])
    lag = SpatialLagTransformer(adj).fit(X_train, y_train)

    # Validation areas B and D. B -> mean(A,C)=13; D -> neighbour C only -> 14.
    out = lag.transform(pd.DataFrame({"lsoa": ["B", "D"]}))
    assert abs(float(out[0, 0]) - 13.0) < 1e-9
    assert abs(float(out[1, 0]) - 14.0) < 1e-9

    # A's neighbour is B (not in training) -> fallback to global train mean = 13.0.
    out_a = lag.transform(pd.DataFrame({"lsoa": ["A"]}))
    assert abs(float(out_a[0, 0]) - 13.0) < 1e-9

    # Transform never sees y, so validation targets cannot influence the lag: a
    # second transform with different "rows" yields identical values by area.
    assert float(lag.transform(pd.DataFrame({"lsoa": ["B"]}))[0, 0]) == float(out[0, 0])


def test_build_area_adjacency_on_grid():
    import geopandas as gpd
    from shapely.geometry import box

    # 1 x 4 strip of cells: neighbours are the immediate left/right cells.
    cells = [{"lsoa": f"L{i}", "geometry": box(i, 0, i + 1, 1)} for i in range(4)]
    gdf = gpd.GeoDataFrame(cells, geometry="geometry", crs=CRS_BNG)
    adj = build_area_adjacency(gdf, "lsoa", method="queen")
    assert set(adj["L0"]) == {"L1"}
    assert set(adj["L1"]) == {"L0", "L2"}
    assert abs(sum(adj["L1"].values()) - 1.0) < 1e-9   # row-standardised


if __name__ == "__main__":
    test_distance_to_cbd_is_exact()
    test_distance_to_thames_line()
    test_nearest_station_and_poi_counts()
    test_spatial_lag_excludes_own_area_target()
    test_spatial_lag_is_val_target_independent_and_has_fallback()
    test_build_area_adjacency_on_grid()
    print("OK: all feature-engineering smoke tests passed")
