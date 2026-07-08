"""Data ingestion and cleaning for the London spatial house-price project.

Single-pass pipeline:
    1. Download the HM Land Registry Price Paid Data yearly files (2021-2024).
    2. Parse the headerless CSVs and concatenate them.
    3. Join to the ONS Postcode Directory (ONSPD) for British National Grid
       coordinates and 2021 LSOA / MSOA / LAD codes.
    4. Filter to Greater London.
    5. Clean per section 4 of the spec (Category A only; drop "Other" property
       type; de-duplicate; price sanity; repeat-sale key; log price; year-quarter).
    6. Reproject BNG -> WGS84 lon/lat for map display only.
    7. Write a processed Parquet file and log a stage-by-stage retention funnel.

Coordinates used for distance features stay in EPSG:27700 (metres). lon/lat are
added purely for visualisation.

Run (from the repository root):
    python -m src.data.ingest              # download + full pipeline
    python -m src.data.ingest --no-write   # dry run, no Parquet output
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from .config import (
    ATTRIBUTION,
    CRS_BNG,
    CRS_WGS84,
    DATE_MAX,
    DATE_MIN,
    EXTERNAL_DIR,
    FINAL_COLUMNS,
    HIGH_PRICE_QUANTILE,
    KEEP_PPD_CATEGORY,
    KEEP_PROPERTY_TYPES,
    LONDON_LAD_PREFIX,
    ONSPD_COLUMNS,
    ONSPD_FILENAME,
    ONSPD_REQUIRED,
    PPD_BASE_URL,
    PPD_COLUMNS,
    PPD_YEARS,
    PRICE_FLOOR_GBP,
    PROCESSED_DIR,
    PROCESSED_FILENAME,
    RAW_DIR,
)

LOGGER = logging.getLogger(__name__)

# One reusable transformer (British National Grid metres -> WGS84 lon/lat).
_BNG_TO_WGS84 = Transformer.from_crs(CRS_BNG, CRS_WGS84, always_xy=True)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest: Path, *, retries: int = 3, timeout: int = 120) -> None:
    """Stream a URL to disk with retries and an atomic rename."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size == 0:
                raise IOError("downloaded file is empty")
            tmp.replace(dest)
            return
        except Exception as err:  # noqa: BLE001 - retried and re-raised below
            last_err = err
            LOGGER.warning("Download failed: %s", err)
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_err}")


def download_ppd(years=PPD_YEARS, raw_dir: Path = RAW_DIR, *, force: bool = False) -> list[Path]:
    """Download the PPD yearly files, caching by presence and non-zero size."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for year in years:
        dest = raw_dir / f"pp-{year}.csv"
        if dest.exists() and dest.stat().st_size > 0 and not force:
            LOGGER.info("Cached PPD %d -> %s", year, dest)
        else:
            _download_file(f"{PPD_BASE_URL}/pp-{year}.csv", dest)
        paths.append(dest)
    return paths


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_ppd(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate the headerless PPD yearly CSVs as strings."""
    frames = []
    for p in paths:
        frames.append(
            pd.read_csv(
                p,
                header=None,
                names=PPD_COLUMNS,
                usecols=range(len(PPD_COLUMNS)),
                dtype=str,
                quotechar='"',
                keep_default_na=False,   # do not turn address text like "NA" into NaN
                na_values=[""],          # only genuinely empty fields become NaN
            )
        )
    return pd.concat(frames, ignore_index=True)


def _prepare_onspd(onspd: pd.DataFrame) -> pd.DataFrame:
    """Normalise postcode + coords and enforce one row per postcode."""
    onspd = onspd.copy()
    onspd["pc_norm"] = normalize_postcode(onspd["postcode"])
    onspd["easting"] = pd.to_numeric(onspd["easting"], errors="coerce")
    onspd["northing"] = pd.to_numeric(onspd["northing"], errors="coerce")
    onspd = onspd.dropna(subset=["pc_norm"]).drop_duplicates("pc_norm", keep="last")
    return onspd


def load_onspd(path: Path = EXTERNAL_DIR / ONSPD_FILENAME, columns=ONSPD_COLUMNS) -> pd.DataFrame:
    """Load the ONSPD, resilient to column-name drift between releases."""
    if not path.exists():
        raise FileNotFoundError(
            f"ONSPD not found at {path}. Download the ONS Postcode Directory from the "
            "ONS Open Geography Portal (geoportal.statistics.gov.uk), unzip, and place "
            f"the multi-CSV or the single combined CSV there as '{ONSPD_FILENAME}'."
        )
    available = pd.read_csv(path, nrows=0).columns
    wanted = {canon: raw for canon, raw in columns.items() if raw in available}
    missing = [c for c in ONSPD_REQUIRED if c not in wanted]
    if missing:
        raise KeyError(
            f"ONSPD is missing required columns {missing}. Update ONSPD_COLUMNS in "
            f"config.py to match your release. First available columns: "
            f"{list(available)[:25]}"
        )
    onspd = pd.read_csv(path, usecols=list(wanted.values()), dtype=str, low_memory=False)
    onspd = onspd.rename(columns={raw: canon for canon, raw in wanted.items()})
    return _prepare_onspd(onspd)


# --------------------------------------------------------------------------- #
# Transform helpers
# --------------------------------------------------------------------------- #
def normalize_postcode(values: pd.Series) -> pd.Series:
    """Canonicalise UK postcodes to the ONSPD 'pcds' form (single space).

    Uppercases, strips all whitespace, then reinserts one space before the
    3-character inward code. Values that are not 5-7 characters once compacted
    are treated as malformed and returned as NA.
    """
    compact = (
        values.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    )
    spaced = compact.str.slice(0, -3).str.cat(compact.str.slice(-3), sep=" ")
    valid = compact.str.len().between(5, 7)
    return spaced.where(valid, other=pd.NA)


def _property_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-property key so repeat sales can share a CV fold downstream."""
    cols = ["paon", "saon", "street", "pc_norm"]
    norm = [df[c].fillna("").astype(str).str.upper().str.strip() for c in cols]
    return norm[0].str.cat(norm[1:], sep="|")


def _bng_to_wgs84(easting: np.ndarray, northing: np.ndarray):
    """Vectorised EPSG:27700 -> EPSG:4326 transform. Returns (lon, lat)."""
    lon, lat = _BNG_TO_WGS84.transform(easting, northing)
    return lon, lat


def geocode(ppd: pd.DataFrame, onspd: pd.DataFrame) -> pd.DataFrame:
    """Attach ONSPD coordinates and area codes via the normalised postcode."""
    ppd = ppd.copy()
    ppd["pc_norm"] = normalize_postcode(ppd["postcode"])
    attach = [c for c in ("easting", "northing", "lsoa", "msoa", "lad", "region")
              if c in onspd.columns]
    return ppd.merge(onspd[["pc_norm", *attach]], on="pc_norm", how="left")


def filter_greater_london(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose LAD code sits in Greater London (GSS prefix E09)."""
    mask = df["lad"].fillna("").str.startswith(LONDON_LAD_PREFIX)
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _log_funnel(funnel: list[tuple[str, int]]) -> None:
    if not funnel:
        return
    start = funnel[0][1] or 1
    LOGGER.info("---------------- ingestion funnel ----------------")
    for label, n in funnel:
        LOGGER.info("%-34s %11s  (%5.1f%% of start)", label, f"{n:,}", 100 * n / start)
    LOGGER.info("--------------------------------------------------")


def run_ingest(
    years=PPD_YEARS,
    *,
    force_download: bool = False,
    write: bool = True,
    ppd_df: Optional[pd.DataFrame] = None,
    onspd_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Execute the full ingestion + cleaning pipeline.

    ``ppd_df`` / ``onspd_df`` allow in-memory injection (used by tests) and, when
    provided, bypass the download / file-load steps.
    """
    funnel: list[tuple[str, int]] = []

    def track(df: pd.DataFrame, label: str) -> pd.DataFrame:
        funnel.append((label, len(df)))
        LOGGER.info("%-34s rows=%s", label, f"{len(df):,}")
        return df

    # 1. Acquire raw PPD -----------------------------------------------------
    if ppd_df is None:
        ppd = load_ppd(download_ppd(years, force=force_download))
    else:
        ppd = ppd_df.copy()
    ppd = track(ppd, "loaded PPD 2021-2024")

    # 2. Type coercion + transfer-date window --------------------------------
    ppd["price"] = pd.to_numeric(ppd["price"], errors="coerce")
    ppd["date_of_transfer"] = pd.to_datetime(ppd["date_of_transfer"], errors="coerce")
    in_window = ppd["date_of_transfer"].between(pd.Timestamp(DATE_MIN), pd.Timestamp(DATE_MAX))
    ppd = track(ppd[in_window], "within 2021-2024 window")

    # 3. Drop rows we can never geocode or price -----------------------------
    ppd = track(ppd.dropna(subset=["postcode"]), "non-null postcode")
    ppd = track(ppd.dropna(subset=["price"]), "non-null price")

    # 4. Geocode via ONSPD ---------------------------------------------------
    onspd = _prepare_onspd(onspd_df) if onspd_df is not None else load_onspd()
    merged = geocode(ppd, onspd)
    coords_ok = (
        merged["easting"].notna() & merged["northing"].notna()
        & (merged["easting"] != 0) & (merged["northing"] != 0)
    )
    matched = coords_ok & merged["lad"].notna()
    LOGGER.info("ONSPD match rate: %.2f%%", 100 * (matched.mean() if len(merged) else 0.0))
    merged = track(merged[matched], "geocoded (ONSPD matched)")

    # 5. Greater London ------------------------------------------------------
    lon = track(filter_greater_london(merged), "Greater London (LAD E09*)")

    # 6. Cleaning per section 4 ---------------------------------------------
    lon = track(lon[lon["ppd_category"] == KEEP_PPD_CATEGORY], "Category A only")
    lon = track(lon[lon["property_type"].isin(KEEP_PROPERTY_TYPES)], "property type in D/S/T/F")
    lon = track(lon.drop_duplicates("transaction_id", keep="first"), "unique transaction_id")
    lon = track(lon[lon["price"] >= PRICE_FLOOR_GBP], f"price >= {PRICE_FLOOR_GBP:,} GBP")

    # 7. Derived fields ------------------------------------------------------
    lon = lon.copy()
    lon["log_price"] = np.log(lon["price"].astype(float))
    threshold = lon["price"].quantile(HIGH_PRICE_QUANTILE) if len(lon) else np.inf
    lon["is_high_price_outlier"] = lon["price"] > threshold          # flag, do not drop
    LOGGER.info("High-price flag threshold (p%.1f): GBP %s",
                100 * HIGH_PRICE_QUANTILE, f"{threshold:,.0f}")
    dt = lon["date_of_transfer"].dt
    lon["year"] = dt.year
    lon["quarter"] = dt.quarter
    lon["year_quarter"] = lon["year"].astype(str) + "Q" + lon["quarter"].astype(str)
    lon["property_key"] = _property_key(lon)
    if "region" not in lon.columns:
        lon["region"] = pd.NA

    # 8. Reproject BNG -> WGS84 (display only) -------------------------------
    lon["lon"], lon["lat"] = _bng_to_wgs84(
        lon["easting"].to_numpy(dtype=float), lon["northing"].to_numpy(dtype=float)
    )

    # 9. Final schema --------------------------------------------------------
    out = track(lon[FINAL_COLUMNS].reset_index(drop=True), "final processed")

    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / PROCESSED_FILENAME
        out.to_parquet(dest, index=False)
        LOGGER.info("Wrote %s (%s rows)", dest, f"{len(out):,}")

    _log_funnel(funnel)
    LOGGER.info("Attribution to display on any published output:\n%s",
                ATTRIBUTION.format(year=pd.Timestamp.now().year))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest + clean London PPD (2021-2024).")
    parser.add_argument("--years", type=int, nargs="+", default=list(PPD_YEARS))
    parser.add_argument("--force-download", action="store_true",
                        help="re-download PPD files even if cached")
    parser.add_argument("--no-write", action="store_true",
                        help="run the pipeline but do not write the Parquet output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_ingest(tuple(args.years), force_download=args.force_download, write=not args.no_write)


if __name__ == "__main__":
    main()
