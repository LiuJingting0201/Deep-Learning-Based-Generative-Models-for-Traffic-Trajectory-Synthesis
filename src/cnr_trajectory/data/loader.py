"""Data loading helpers for trajectory CSV files.

The original file has an `.xls` suffix, but the notebook reads it with
`pandas.read_csv(..., sep=",")`. This module preserves that behavior.
"""

from pathlib import Path

import pandas as pd


def resolve_path(path: str | Path) -> Path:
    """Return a normalized Path without reading or modifying data."""
    return Path(path).expanduser().resolve()


def load_trajectory_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw trajectory table using the original notebook CSV reader."""
    return pd.read_csv(path, sep=",", low_memory=False, nrows=nrows)
