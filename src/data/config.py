"""Configuration constants for the London spatial house-price ingestion pipeline.

All paths resolve relative to the repository root, so the pipeline behaves
identically regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --- Repository layout -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # auto-downloaded PPD yearly files
EXTERNAL_DIR = DATA_DIR / "external"  # manually placed ONSPD / boundary files
PROCESSED_DIR = DATA_DIR / "processed"

# --- Price Paid Data (HM Land Registry) --------------------------------------
# Stable S3 host. Yearly files are pp-YYYY.csv and are published WITHOUT headers.
# The yearly / complete files already have additions, changes and deletions
# applied (unlike the monthly delta files), so record-status A/C/D markers do not
# need to be replayed.
PPD_BASE_URL = (
    "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
)
PPD_YEARS = (2021, 2022, 2023, 2024)

# Transfer-date window. Yearly files are grouped by *registration* date, which
# lags completion by ~2 weeks to 2 months, so a pp-YYYY file can hold transfers
# from the previous December. We cut on transfer date for a clean cross-section.
DATE_MIN = "2021-01-01"
DATE_MAX = "2024-12-31"

# The 16 PPD columns, in file order (no header row is present in the source).
PPD_COLUMNS = [
    "transaction_id",   # unique per published sale; regenerated on amendment
    "price",            # sale price on the transfer deed (GBP)
    "date_of_transfer",
    "postcode",
    "property_type",    # D=Detached S=Semi T=Terraced F=Flat/Maisonette O=Other
    "old_new",          # Y=newly built  N=established
    "duration",         # F=Freehold  L=Leasehold  U=Unknown
    "paon",             # Primary Addressable Object Name (house number / name)
    "saon",             # Secondary Addressable Object Name (flat / unit)
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",     # A=standard market sale  B=additional (repossession/BTL/...)
    "record_status",    # A/C/D in monthly files; already applied in the yearly files
]

# --- Cleaning thresholds (spec section 4) ------------------------------------
KEEP_PROPERTY_TYPES = frozenset({"D", "S", "T", "F"})  # drop "O" (Other) as ambiguous
KEEP_PPD_CATEGORY = "A"          # arm's-length market sales only
PRICE_FLOOR_GBP = 20_000         # below this: data errors / nominal transfers -> drop
HIGH_PRICE_QUANTILE = 0.999      # flag (not drop) extreme highs for downstream review

# --- ONSPD (postcode -> BNG coords + 2021 statistical geographies) -----------
# ONS occasionally renames columns between releases. These defaults match recent
# (2024/2025) ONSPD layouts; verify against your release's user guide. The loader
# raises a clear error listing available columns if a required field is missing.
# Keys are the canonical names used downstream; values are the raw ONSPD headers.
ONSPD_COLUMNS = {
    "postcode": "pcds",        # postcode with a single space, e.g. "SW1A 2AA"
    "easting": "east1m",     # British National Grid easting  (metres, EPSG:27700)
    "northing": "north1m",    # British National Grid northing (metres, EPSG:27700)
    "lsoa": "lsoa21cd",          # 2021 Lower layer Super Output Area code
    "msoa": "msoa21cd",          # 2021 Middle layer Super Output Area code
    "lad": "lad25cd",           # Local Authority District (GSS) code
    "region": "rgn25cd",           # Region (GSS) code (optional)
    "grid_quality": "gridind",  # grid-reference quality indicator (optional)
}
ONSPD_FILENAME = "ONSPD.csv"                     # expected under EXTERNAL_DIR
ONSPD_REQUIRED = ("postcode", "easting", "northing", "lsoa", "msoa", "lad")

# --- Greater London selector -------------------------------------------------
# The 33 Greater London authorities (City of London + 32 boroughs) have GSS codes
# E09000001..E09000033; "E09" is a unique, release-stable prefix for exactly them.
LONDON_LAD_PREFIX = "E09"
LONDON_REGION_CODE = "E12000007"   # documented fallback selector

# --- Coordinate reference systems --------------------------------------------
CRS_BNG = 27700     # OSGB36 / British National Grid (metres) -> distance features
CRS_WGS84 = 4326    # lon/lat -> web maps only

# --- Output ------------------------------------------------------------------
PROCESSED_FILENAME = "london_ppd_2021_2024.parquet"

# Final column schema (ordered) written to the processed Parquet.
FINAL_COLUMNS = [
    "transaction_id", "price", "log_price", "is_high_price_outlier",
    "date_of_transfer", "year", "quarter", "year_quarter",
    "property_type", "old_new", "duration",
    "postcode", "pc_norm", "property_key",
    "easting", "northing", "lon", "lat",
    "lsoa", "msoa", "lad", "region",
    "paon", "saon", "street", "locality", "town_city", "district", "county",
]

# --- Attribution (required on any published output) --------------------------
ATTRIBUTION = (
    "Contains HM Land Registry data (c) Crown copyright and database right {year}. "
    "Contains OS data (c) Crown copyright and database right {year}. "
    "Contains Royal Mail data (c) Royal Mail copyright and database right {year}. "
    "Contains National Statistics data (c) Crown copyright and database right {year}. "
    "Licensed under the Open Government Licence v3.0."
)
