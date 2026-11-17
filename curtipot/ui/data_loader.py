"""Data loading utilities for experimental titration uploads."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd


def parse_numeric_series(series: pd.Series) -> np.ndarray:
    """Parse pandas series into a float array, converting comma decimals to dots."""
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float)


def load_experimental_dataframe(content: bytes) -> pd.DataFrame:
    """Load experimental titration data from raw CSV/TXT/TSV bytes with automatic delimiter sniffing."""
    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(content))


def extract_matched_volume_ph(
    df: pd.DataFrame, v_col: str, ph_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned volume and pH numpy arrays, dropping missing or non-numeric rows."""
    v_raw = parse_numeric_series(df[v_col])
    ph_raw = parse_numeric_series(df[ph_col])
    valid_mask = ~(np.isnan(v_raw) | np.isnan(ph_raw))
    return v_raw[valid_mask], ph_raw[valid_mask]
