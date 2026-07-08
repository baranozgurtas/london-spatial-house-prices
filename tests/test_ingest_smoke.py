"""Smoke test for the ingestion pipeline on synthetic fixtures.

No network and no real data: we fabricate a handful of rows in the exact PPD
16-column layout plus a matching ONSPD table, and assert that every section-4
rule fires as intended. Runnable directly (``python tests/test_ingest_smoke.py``)
or under pytest.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.config import PPD_COLUMNS  # noqa: E402
from src.data.ingest import (  # noqa: E402
    load_ppd,
    normalize_postcode,
    run_ingest,
)


def _ppd_fixture() -> pd.DataFrame:
    """Rows exercising each cleaning rule. Comment marks the expected fate."""
    rows = [
        # transaction_id, price, date, postcode, type, old_new, duration,
        # paon, saon, street, locality, town, district, county, cat, status
        ("T1", "850000", "2022-06-15 00:00", "SW1A 1AA", "D", "N", "F",
         "10", "", "DOWNING STREET", "", "LONDON", "WESTMINSTER", "GREATER LONDON", "A", "A"),  # KEEP
        ("T2", "500000", "2022-01-10 00:00", "SW1A 1AA", "T", "N", "L",
         "12", "", "DOWNING STREET", "", "LONDON", "WESTMINSTER", "GREATER LONDON", "B", "A"),  # drop: Category B
        ("T3", "400000", "2022-04-01 00:00", "E1 6AN", "O", "N", "F",
         "1", "", "BRICK LANE", "", "LONDON", "TOWER HAMLETS", "GREATER LONDON", "A", "A"),      # drop: type O
        ("T4", "1000", "2022-05-01 00:00", "SE1 7PB", "F", "N", "L",
         "5", "3", "BELVEDERE ROAD", "", "LONDON", "LAMBETH", "GREATER LONDON", "A", "A"),       # drop: below floor
        ("T5", "300000", "2022-07-01 00:00", "M1 1AE", "T", "N", "F",
         "2", "", "PICCADILLY", "", "MANCHESTER", "MANCHESTER", "GREATER MANCHESTER", "A", "A"), # drop: not London
        ("T1", "999999", "2022-06-15 00:00", "SW1A 1AA", "D", "N", "F",
         "10", "", "DOWNING STREET", "", "LONDON", "WESTMINSTER", "GREATER LONDON", "A", "A"),   # drop: dup id
        ("T7", "500000", "2022-08-01 00:00", "", "S", "N", "F",
         "9", "", "NOWHERE ROAD", "", "LONDON", "WESTMINSTER", "GREATER LONDON", "A", "A"),      # drop: no postcode
        ("T8", "700000", "2020-05-01 00:00", "SW1A 1AA", "D", "N", "F",
         "11", "", "DOWNING STREET", "", "LONDON", "WESTMINSTER", "GREATER LONDON", "A", "A"),   # drop: out of window
        ("T9", "450000", "2023-02-02 00:00", "ZZ1 1ZZ", "F", "N", "L",
         "1", "", "PHANTOM STREET", "", "LONDON", "NOWHERE", "GREATER LONDON", "A", "A"),        # drop: no ONSPD match
        ("T10", "600000", "2023-03-20 00:00", "E1 6AN", "F", "Y", "L",
         "1", "4", "BRICK LANE", "", "LONDON", "TOWER HAMLETS", "GREATER LONDON", "A", "A"),     # KEEP
    ]
    return pd.DataFrame(rows, columns=PPD_COLUMNS)


def _onspd_fixture() -> pd.DataFrame:
    # Plausible BNG eastings/northings (metres). ZZ1 1ZZ intentionally absent.
    return pd.DataFrame(
        {
            "postcode": ["SW1A 1AA", "E1 6AN", "SE1 7PB", "M1 1AE"],
            "easting": [530047, 533900, 531000, 384500],
            "northing": [179951, 181500, 179000, 398000],
            "lsoa": ["E01004736", "E01004300", "E01003100", "E01005200"],
            "msoa": ["E02000977", "E02000900", "E02000600", "E02001000"],
            "lad": ["E09000033", "E09000030", "E09000022", "E08000003"],  # last = Manchester
            "region": ["E12000007", "E12000007", "E12000007", "E12000002"],
        }
    )


def test_normalize_postcode():
    got = normalize_postcode(pd.Series(["sw1a1aa", "SW1A  1AA", "E16AN", "M11AE", ""]))
    assert list(got[:4]) == ["SW1A 1AA", "SW1A 1AA", "E1 6AN", "M1 1AE"]
    assert pd.isna(got.iloc[4])


def test_load_ppd_headerless(tmp_path):
    raw = (
        '"{ABC}","850000","2022-06-15 00:00","SW1A 1AA","D","N","F",'
        '"10","","DOWNING STREET","","LONDON","WESTMINSTER","GREATER LONDON","A","A"\n'
    )
    p = tmp_path / "pp-2022.csv"
    p.write_text(raw)
    df = load_ppd([p])
    assert list(df.columns) == PPD_COLUMNS
    assert df.loc[0, "price"] == "850000"
    assert df.loc[0, "postcode"] == "SW1A 1AA"


def test_pipeline_end_to_end():
    out = run_ingest(ppd_df=_ppd_fixture(), onspd_df=_onspd_fixture(), write=False)

    # Only T1 and T10 should survive every rule.
    assert set(out["transaction_id"]) == {"T1", "T10"}, out["transaction_id"].tolist()
    assert len(out) == 2

    # Derived fields.
    import math

    t1 = out.set_index("transaction_id").loc["T1"]
    assert t1["year_quarter"] == "2022Q2"
    assert abs(float(t1["log_price"]) - math.log(850000)) < 1e-6
    assert t1["property_key"].startswith("10|")

    # Reprojection lands the London points in a sane WGS84 box.
    assert out["lat"].between(51.2, 51.8).all()
    assert out["lon"].between(-0.6, 0.3).all()

    # The high-price flag must never drop rows, and the lower-priced sale (T10)
    # sits below the p99.9 threshold so is not flagged. (On tiny N the quantile
    # interpolates just under the max, so the top row may be flagged - expected.)
    assert out["is_high_price_outlier"].dtype == bool
    assert len(out) == 2
    assert bool(out.set_index("transaction_id").loc["T10", "is_high_price_outlier"]) is False


if __name__ == "__main__":
    test_normalize_postcode()
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_load_ppd_headerless(Path(d))
    test_pipeline_end_to_end()
    print("OK: all smoke tests passed")
