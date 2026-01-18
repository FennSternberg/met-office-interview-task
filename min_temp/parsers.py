from __future__ import annotations

from pathlib import Path
from typing import Set, Union

import pandas as pd

from .predictors import KTable

PathLike = Union[str, Path]

def parse_k_table(path: PathLike) -> KTable:
    """Parse a K-table CSV into a ``KTable`` instance.
    Expects the CSV to have the following format:
    - The first row is the cloud-cover band maxima (as column headers).
    - The first column is the wind-speed band maxima (as the index).

    All non-numeric cell values are coerced to NaN (unknown/undefined K).
    """

    path = Path(path)
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError("K table CSV is empty")

    try:
        df.index = pd.to_numeric(df.index, errors="raise").astype(float)
        df.columns = pd.to_numeric(df.columns, errors="raise").astype(float)
    except Exception as e:
        raise ValueError("K table edges (index/columns) must be numeric") from e

    df = df.apply(pd.to_numeric, errors="coerce")
    return KTable(df)

def _normalize_heading(name: str) -> str:
    return " ".join(name.strip().lower().split())

def parse_initial_data(
    path: PathLike,
    *,
    midday_temp_header: str = "Midday Temperature (deg C)",
    midday_dew_point_header: str = "Midday Dew Point (deg C)",
    wind_speed_header: str = "Wind (Kn)",
    cloud_cover_header: str = "Cloud (Oktas)",
) -> pd.DataFrame:
    """Parse observation rows like initial_data.csv into canonical columns.

    Reads the CSV and converts essential input columns into canonical names:
    - midday_temp_c
    - midday_dew_point_c
    - wind_kn
    - cloud_oktas

    Optional header override arguments allow the input CSV to use different
    column names for these essential fields.
    """

    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df.columns) == 0:
        raise ValueError(f"CSV has no columns: {path}")

    header_to_canonical = {
        _normalize_heading(midday_temp_header): "midday_temp_c",
        _normalize_heading(midday_dew_point_header): "midday_dew_point_c",
        _normalize_heading(wind_speed_header): "wind_kn",
        _normalize_heading(cloud_cover_header): "cloud_oktas",
    }

    rename = {}
    for col in df.columns:
        canonical = header_to_canonical.get(_normalize_heading(str(col)))
        if canonical:
            rename[col] = canonical

    df = df.rename(columns=rename)

    required: Set[str] = {
        "midday_temp_c",
        "midday_dew_point_c",
        "wind_kn",
        "cloud_oktas",
    }

    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} in {path}")

    # Ensure the essential columns are numeric, leave unessential ones as-is
    for col in required:
        raw = df[col].astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().any():
            bad_rows = numeric[numeric.isna()].index.tolist()
            raise ValueError(f"Non-numeric values found in required column '{col}' at rows {bad_rows}")
        df[col] = numeric.astype(float)

    return df

__all__ = ["parse_k_table", "parse_initial_data"]
