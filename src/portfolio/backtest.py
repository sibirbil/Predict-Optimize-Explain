"""
Portfolio backtesting engines for PTO strategies.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from ..utils.dates import shift_yyyymm
    from ..optimization.risk import ewma_cov_full_history_matrix
    from ..optimization.solvers import solve_mvo_long_only, solve_robust_longonly
    from .metrics import weights_diagnostics
except ImportError:
    # Fallback for direct imports
    from utils.dates import shift_yyyymm
    from optimization.risk import ewma_cov_full_history_matrix
    from optimization.solvers import solve_mvo_long_only, solve_robust_longonly
    from portfolio.metrics import weights_diagnostics


def run_pto_backtest(
    df_results: pd.DataFrame,
    df_ret_all: pd.DataFrame,
    id_col: str,
    topk: int = 200,
    preselect_factor: int = 3,
    lookback: int = 60,
    lam: float = 0.94,
    shrink: float = 0.10,
    ridge: float = 1e-6,
    lambda_: float = 5.0,
    kappa: float = 0.0,
    omega_mode: str = "diagSigma",
    min_assets: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Predict-Then-Optimize backtest with monthly rebalancing.

    Universe selection pipeline (per month):
    1. Pre-select top (topk × preselect_factor) by FNN prediction
    2. Filter to assets with complete lookback-month history
    3. Select final topk assets

    Portfolio optimization:
    - Compute EWMA covariance from lookback window
    - Solve MVO (kappa=0) or Robust MVO (kappa>0)
    - Fallback to equal-weight if solver fails

    Args:
        df_results: Test set with [yyyymm, id_col, pred_return, actual_return]
        df_ret_all: Full history with [yyyymm, id_col, ret]
        id_col: Asset identifier column
        topk: Final portfolio size (default: 200)
        preselect_factor: Pre-selection multiplier (default: 3)
        lookback: Historical window for covariance (default: 60 months)
        lam: EWMA decay factor (default: 0.94)
        shrink: Covariance shrinkage (default: 0.10)
        ridge: Ridge regularization (default: 1e-6)
        lambda_: Risk aversion (default: 5.0)
        kappa: Robustness penalty (default: 0.0)
        omega_mode: Uncertainty set mode ("diagSigma" or "identity")
        min_assets: Minimum portfolio size (default: 2)

    Returns:
        Tuple of (perf, weights):
        - perf: Monthly performance DataFrame
        - weights: Detailed holdings DataFrame
    """
    perf_list = []
    weights_list = []

    # Get unique test months
    test_months = sorted(df_results["yyyymm"].unique())

    for t in test_months:
        # Cross-section at month t
        cross_t = df_results[df_results["yyyymm"] == t].copy()

        # Pre-select top (topk × preselect_factor) by prediction
        preK = topk * preselect_factor
        cross_t = cross_t.nlargest(preK, "pred_return", keep="first")

        if len(cross_t) < min_assets:
            continue

        # Define lookback window [t-lookback, t-1]
        start_m = shift_yyyymm(int(t), -lookback)
        end_m = shift_yyyymm(int(t), -1)

        # Get historical returns for pre-selected assets
        hist = df_ret_all[
            (df_ret_all["yyyymm"] >= start_m) &
            (df_ret_all["yyyymm"] <= end_m) &
            (df_ret_all[id_col].isin(cross_t[id_col]))
        ].copy()

        # Pivot to matrix form (time × assets)
        hist_wide = hist.pivot(index="yyyymm", columns=id_col, values="ret")

        # Filter to assets with complete history (no NaN in lookback)
        full_history_assets = hist_wide.columns[hist_wide.notna().all(axis=0)].tolist()

        # Intersect with cross-section and select final topk
        cross_t = cross_t[cross_t[id_col].isin(full_history_assets)]
        cross_t = cross_t.nlargest(topk, "pred_return", keep="first")

        if len(cross_t) < min_assets:
            continue

        final_ids = cross_t[id_col].tolist()
        K = len(final_ids)

        # Extract returns matrix for selected assets
        R = hist_wide[final_ids].values  # (lookback × K)

        if R.shape[0] < lookback or R.shape[1] < min_assets:
            continue

        # Compute EWMA covariance
        Sigma = ewma_cov_full_history_matrix(
            R, lam=lam, shrink=shrink, ridge=ridge, psd_proj=True
        )

        if Sigma is None:
            # Fallback: equal-weight
            w = np.ones(K) / K
            fallback = True
        else:
            # Get expected returns (predictions)
            mu = cross_t.set_index(id_col).loc[final_ids, "pred_return"].values

            # Solve optimization
            if kappa == 0.0:
                w = solve_mvo_long_only(mu, Sigma, lambda_=lambda_)
            else:
                w = solve_robust_longonly(mu, Sigma, lambda_=lambda_, kappa=kappa, omega_mode=omega_mode)

            if w is None:
                # Solver failed: equal-weight fallback
                w = np.ones(K) / K
                fallback = True
            else:
                fallback = False

        # Get actual returns
        r_actual = cross_t.set_index(id_col).loc[final_ids, "actual_return"].values

        # Compute portfolio return
        port_ret = float(np.dot(w, r_actual))

        # Weight diagnostics
        diag = weights_diagnostics(w)

        # Store performance
        perf_list.append({
            "yyyymm": t,
            "n_assets": K,
            "port_ret": port_ret,
            "fallback": fallback,
            "HHI": diag["hhi"],
            "N_eff": diag["n_eff"],
            "max_w": diag["max_w"],
            "active": diag["active"],
        })

        # Store weights
        for i, asset_id in enumerate(final_ids):
            weights_list.append({
                "yyyymm": t,
                id_col: asset_id,
                "w": w[i],
            })

    perf = pd.DataFrame(perf_list)
    weights = pd.DataFrame(weights_list)

    return perf, weights


def run_equal_weight_topk(
    df_results: pd.DataFrame,
    df_ret_all: pd.DataFrame,
    id_col: str,
    topk: int = 200,
    preselect_factor: int = 3,
    lookback: int = 60,
    min_assets: int = 2
) -> pd.DataFrame:
    """
    Run equal-weight baseline strategy with same universe selection as PTO.

    Uses identical asset selection logic but allocates 1/N weights.

    Args:
        df_results: Test set with [yyyymm, id_col, pred_return, actual_return]
        df_ret_all: Full history with [yyyymm, id_col, ret]
        id_col: Asset identifier column
        topk: Final portfolio size (default: 200)
        preselect_factor: Pre-selection multiplier (default: 3)
        lookback: Historical window requirement (default: 60 months)
        min_assets: Minimum portfolio size (default: 2)

    Returns:
        DataFrame with monthly performance
    """
    perf_list = []
    test_months = sorted(df_results["yyyymm"].unique())

    for t in test_months:
        # Cross-section at month t
        cross_t = df_results[df_results["yyyymm"] == t].copy()

        # Pre-select
        preK = topk * preselect_factor
        cross_t = cross_t.nlargest(preK, "pred_return", keep="first")

        if len(cross_t) < min_assets:
            continue

        # Lookback window
        start_m = shift_yyyymm(int(t), -lookback)
        end_m = shift_yyyymm(int(t), -1)

        hist = df_ret_all[
            (df_ret_all["yyyymm"] >= start_m) &
            (df_ret_all["yyyymm"] <= end_m) &
            (df_ret_all[id_col].isin(cross_t[id_col]))
        ].copy()

        hist_wide = hist.pivot(index="yyyymm", columns=id_col, values="ret")
        full_history_assets = hist_wide.columns[hist_wide.notna().all(axis=0)].tolist()

        cross_t = cross_t[cross_t[id_col].isin(full_history_assets)]
        cross_t = cross_t.nlargest(topk, "pred_return", keep="first")

        if len(cross_t) < min_assets:
            continue

        K = len(cross_t)
        w = np.ones(K) / K  # Equal weights

        # Get actual returns
        r_actual = cross_t["actual_return"].values

        # Portfolio return
        port_ret = float(np.dot(w, r_actual))

        perf_list.append({
            "yyyymm": t,
            "n_assets": K,
            "port_ret": port_ret,
        })

    return pd.DataFrame(perf_list)
