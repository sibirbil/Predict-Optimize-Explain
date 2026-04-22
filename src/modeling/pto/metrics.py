from __future__ import annotations

import math

import numpy as np
import pandas as pd


def weights_diagnostics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    hhi = float(np.sum(weights**2))
    effective_n = float(1.0 / hhi) if hhi > 0 else 0.0
    max_weight = float(np.max(weights)) if len(weights) else 0.0
    active_positions = int(np.sum(weights > 1e-6))
    return {
        "herfindahl": hhi,
        "effective_n": effective_n,
        "max_weight": max_weight,
        "active_positions": active_positions,
    }


def compute_turnover(
    previous_weights: pd.Series | None,
    previous_returns: pd.Series | None,
    current_weights: pd.Series,
) -> float:
    if previous_weights is None or previous_returns is None:
        return 1.0

    gross = previous_weights * (1.0 + previous_returns.reindex(previous_weights.index).fillna(0.0))
    gross_sum = float(gross.sum())
    if gross_sum > 0:
        post_return_weights = gross / gross_sum
    else:
        post_return_weights = previous_weights * 0.0

    aligned_index = post_return_weights.index.union(current_weights.index)
    previous_aligned = post_return_weights.reindex(aligned_index).fillna(0.0)
    current_aligned = current_weights.reindex(aligned_index).fillna(0.0)
    return float(0.5 * np.abs(current_aligned - previous_aligned).sum())


def summarize_backtest(monthly_frame: pd.DataFrame) -> dict[str, float]:
    if monthly_frame.empty:
        return {}

    total_returns = monthly_frame["portfolio_return"].to_numpy(dtype=float)
    excess_returns = monthly_frame["portfolio_excess_return"].to_numpy(dtype=float)
    volatility_monthly = float(np.std(excess_returns, ddof=1)) if len(excess_returns) > 1 else 0.0
    annualized_volatility = float(volatility_monthly * math.sqrt(12.0))
    annualized_return = float(np.mean(total_returns) * 12.0)
    annualized_excess_return = float(np.mean(excess_returns) * 12.0)
    sharpe = float(annualized_excess_return / annualized_volatility) if annualized_volatility > 0 else 0.0

    wealth = np.cumprod(1.0 + total_returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0

    return {
        "n_months": float(len(monthly_frame)),
        "annualized_return": annualized_return,
        "annualized_excess_return": annualized_excess_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "average_turnover": float(monthly_frame["turnover"].mean()),
        "average_herfindahl": float(monthly_frame["herfindahl"].mean()),
        "average_effective_n": float(monthly_frame["effective_n"].mean()),
        "average_max_weight": float(monthly_frame["max_weight"].mean()),
        "average_active_positions": float(monthly_frame["active_positions"].mean()),
    }
