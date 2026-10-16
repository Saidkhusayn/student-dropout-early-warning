"""Load and validate the UCI 'Predict Students' Dropout and Academic Success' data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from dropout import config


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | str = config.RAW_DATA_PATH, rename: bool = True) -> pd.DataFrame:
    """Load the raw CSV.

    The file is ';'-separated and starts with a UTF-8 byte-order mark.
    Column names are stripped of stray whitespace and, if ``rename`` is True,
    mapped to the snake_case names in ``config.COLUMN_RENAMES``.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if rename:
        df = df.rename(columns=config.COLUMN_RENAMES)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the data does not look like the expected UCI file."""
    problems = []
    if df.shape != (config.EXPECTED_N_ROWS, config.EXPECTED_N_COLS):
        problems.append(
            f"shape is {df.shape}, expected "
            f"({config.EXPECTED_N_ROWS}, {config.EXPECTED_N_COLS})"
        )
    expected_cols = list(config.COLUMN_RENAMES.values())
    if list(df.columns) != expected_cols:
        missing = set(expected_cols) - set(df.columns)
        extra = set(df.columns) - set(expected_cols)
        problems.append(f"column mismatch; missing={missing}, extra={extra}")
    if df.isna().any().any():
        problems.append(f"{int(df.isna().sum().sum())} missing values")
    if config.TARGET in df and set(df[config.TARGET].unique()) != set(config.TARGET_CLASSES):
        problems.append(f"unexpected target classes {sorted(df[config.TARGET].unique())}")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))


def load_data(path: Path | str = config.RAW_DATA_PATH, check: bool = True) -> pd.DataFrame:
    """Load the raw data with clean column names and run validation checks."""
    df = load_raw(path)
    if check:
        validate(df)
    return df
