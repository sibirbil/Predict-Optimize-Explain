"""
Date manipulation utilities for YYYYMM format used throughout the codebase.
"""
import pandas as pd
from datetime import datetime


def shift_yyyymm(yyyymm: int, k: int) -> int:
    """
    Shift a YYYYMM date by k months.

    Args:
        yyyymm: Date in YYYYMM format (e.g., 202401 for Jan 2024)
        k: Number of months to shift (positive = future, negative = past)

    Returns:
        Shifted date in YYYYMM format

    Example:
        >>> shift_yyyymm(202401, 1)
        202402
        >>> shift_yyyymm(202401, -1)
        202312
    """
    y, m = divmod(yyyymm, 100)
    dt = datetime(y, m, 1)
    shifted = dt + pd.DateOffset(months=k)
    return shifted.year * 100 + shifted.month


def yyyymm_to_dt(yyyymm: int) -> pd.Timestamp:
    """
    Convert YYYYMM integer to pandas Timestamp.

    Args:
        yyyymm: Date in YYYYMM format (e.g., 202401)

    Returns:
        pandas Timestamp for the first day of that month

    Example:
        >>> yyyymm_to_dt(202401)
        Timestamp('2024-01-01 00:00:00')
    """
    y, m = divmod(yyyymm, 100)
    return pd.Timestamp(y, m, 1)
