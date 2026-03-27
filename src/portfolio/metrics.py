"""
Portfolio performance metrics, diagnostics, and visualization.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def perf_stats_excess(r) -> Dict[str, float]:
    """
    Compute performance statistics for excess returns.

    Matches PTOColab.py perf_stats_excess exactly.

    Args:
        r: Array or Series of monthly excess returns

    Returns:
        Dictionary with performance metrics:
        - n_months: Number of months
        - mean_m, vol_m: Monthly mean and volatility
        - mean_a, vol_a: Annualized mean and volatility
        - sharpe_a: Annualized Sharpe ratio
        - cum_simple: Cumulative simple return
        - max_drawdown: Maximum drawdown
        - hit_rate: Fraction of positive months (decimal, e.g. 0.636)
        - worst_month, best_month: Extreme monthly returns
    """
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]

    if len(r) == 0:
        return {}

    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_a = float(mean_m * 12.0)
    vol_a = float(vol_m * np.sqrt(12.0))
    sharpe_a = float(mean_a / vol_a) if vol_a > 0 else np.nan

    wealth = np.cumprod(1.0 + r)
    cum_simple = float(wealth[-1] - 1.0)

    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    max_dd = float(dd.min())

    return {
        "n_months": int(len(r)),
        "mean_m": mean_m,
        "vol_m": vol_m,
        "mean_a": mean_a,
        "vol_a": vol_a,
        "sharpe_a": sharpe_a,
        "cum_simple": cum_simple,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
    }


def perf_stats(r: np.ndarray) -> Dict[str, float]:
    """
    Compute performance statistics for returns (E2E version).

    Matches PAOColab.py perf_stats exactly.

    Args:
        r: Array of monthly returns

    Returns:
        Dictionary with:
        - mean_m, vol_m: Monthly mean and volatility
        - mean_a, vol_a: Annualized mean and volatility
        - sharpe_a: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown
        - n_months: Number of observations
    """
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]

    if len(r) == 0:
        return {
            "mean_a": 0.0,
            "vol_a": 1.0,
            "sharpe_a": 0.0,
            "n_months": 0,
            "max_drawdown": 0.0,
            "mean_m": 0.0,
            "vol_m": 0.0,
        }

    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    mean_a = mean_m * 12.0
    vol_a = vol_m * math.sqrt(12.0)
    sharpe_a = mean_a / (vol_a + 1e-12)

    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    max_dd = float(dd.min())

    return {
        "mean_m": mean_m,
        "vol_m": vol_m,
        "mean_a": float(mean_a),
        "vol_a": float(vol_a),
        "sharpe_a": float(sharpe_a),
        "max_drawdown": max_dd,
        "n_months": int(len(r)),
    }


def rolling_sharpe(series, window=36):
    """
    Compute rolling annualized Sharpe ratio.

    Matches PTOColab.py rolling_sharpe exactly.

    Args:
        series: Series or array of monthly returns
        window: Rolling window size in months (default: 36 = 3 years)

    Returns:
        Series of rolling Sharpe ratios
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=1)
    return (m * 12.0) / (s * np.sqrt(12.0))


def weights_diagnostics(w: np.ndarray) -> Dict[str, float]:
    """
    Compute portfolio weight concentration metrics.

    Matches PAOColab.py weights_diagnostics exactly.

    Args:
        w: Portfolio weights (N,)

    Returns:
        Dictionary with:
        - hhi: Herfindahl-Hirschman Index (sum of squared weights)
        - n_eff: Effective number of assets (1/hhi)
        - max_w: Maximum weight
        - top10: Sum of top 10 weights
        - active: Number of active positions (w > 1e-6)
    """
    w = np.asarray(w, float)

    hhi = float(np.sum(w ** 2))
    n_eff = float(1.0 / hhi) if hhi > 0 else np.nan
    max_w = float(np.max(w))
    top10 = float(np.sort(w)[-10:].sum()) if w.size >= 10 else float(np.sum(w))
    active = int(np.sum(w > 1e-6))

    return {
        "hhi": hhi,
        "n_eff": n_eff,
        "max_w": max_w,
        "top10": top10,
        "active": active,
    }


def plot_wealth_paths(wealth_df: pd.DataFrame, title: str):
    """
    Plot cumulative wealth paths for multiple strategies.

    Matches PTOColab.py plot_wealth_paths exactly.

    Args:
        wealth_df: DataFrame with 'date' column and one column per strategy
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    for col in wealth_df.columns:
        if col in ["yyyymm", "date"]:
            continue
        plt.plot(wealth_df["date"], wealth_df[col].values, label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Wealth (start = 1)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpes(ret_df: pd.DataFrame, title: str, window=36):
    """
    Plot rolling Sharpe ratios for multiple strategies.

    Matches PTOColab.py plot_rolling_sharpes exactly.

    Args:
        ret_df: DataFrame with 'date' column and one column per strategy of returns
        title: Plot title
        window: Rolling window in months
    """
    plt.figure(figsize=(12, 5))
    for col in ret_df.columns:
        if col in ["yyyymm", "date"]:
            continue
        rs = rolling_sharpe(ret_df[col].astype(float), window)
        plt.plot(ret_df["date"], rs.values, label=col)
    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Rolling Sharpe (annualized), window={window}m")
    plt.legend()
    plt.tight_layout()
    plt.show()
