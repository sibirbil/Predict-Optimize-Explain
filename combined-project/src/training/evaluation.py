"""
E2E Portfolio Evaluation Module.

Implements model evaluation and backtesting functions.
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

try:
    from ..utils.io import save_json
    from ..models.pao_portfolio import PAOPortfolioModel
    from ..portfolio.metrics import perf_stats, weights_diagnostics
except ImportError:
    from utils.io import save_json
    from models.pao_portfolio import PAOPortfolioModel
    from portfolio.metrics import perf_stats, weights_diagnostics


@torch.no_grad()
def eval_dataset(
    model: PAOPortfolioModel,
    ds,  # MonthCacheDataset
    device: str
) -> Dict[str, Any]:
    """
    Evaluate E2E portfolio model on a dataset.

    Computes:
    - Return statistics (mean, Sharpe, volatility, etc.)
    - Utility statistics (where utility = return - lambda/2 * risk)
    - Weight diagnostics (HHI, effective N, etc.)
    - Information coefficient (IC) between predictions and realized returns

    Args:
        model: PAOPortfolioModel instance
        ds: MonthCacheDataset with cached month-level data
        device: Device to run evaluation on ('cpu' or 'cuda')

    Returns:
        Dictionary with:
        - return stats: mean_m, vol_m, mean_a, vol_a, sharpe_a, max_drawdown
        - utility stats: util_m, util_a
        - risk: risk_m (average monthly risk)
        - weight diagnostics: max_w_med, n_eff_med, top10_med, ic_med, mu_std_med
        - raw arrays: returns, utilities, risks
    """
    model.eval()

    rets, utils, risks = [], [], []
    maxw, neff, top10, ic_list, mu_std_list = [], [], [], [], []

    for i in range(len(ds)):
        s = ds[i]
        X = s["X"].to(device)
        y = s["y"].to(device)
        U = s["Sigma_factor"].to(device)
        sigma_vol = s["sigma_vol"].to(device)

        w, mu_raw = model(X, U, sigma_vol)

        # Realized return
        r = float((w * y).sum().item())

        # Realized risk = w'Σw = ||U w||^2
        Uw = torch.mv(U, w)
        risk = float(torch.sum(Uw * Uw).item())

        # Realized utility
        util = r - (model.lambda_ / 2.0) * risk

        rets.append(r)
        risks.append(risk)
        utils.append(util)

        w_np = w.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        mu_np = mu_raw.detach().cpu().numpy()

        wd = weights_diagnostics(w_np)
        maxw.append(wd["max_w"])
        neff.append(wd["n_eff"])
        top10.append(wd["top10"])

        mu_std_list.append(float(np.std(mu_np, ddof=1)) if mu_np.size > 1 else 0.0)

        if (np.std(mu_np, ddof=1) > 1e-12) and (np.std(y_np, ddof=1) > 1e-12):
            ic_list.append(float(np.corrcoef(mu_np, y_np)[0, 1]))
        else:
            ic_list.append(np.nan)

    ret_stats = perf_stats(np.array(rets, dtype=float))
    util_stats = perf_stats(np.array(utils, dtype=float))
    risk_mean_m = float(np.mean(risks)) if len(risks) else float("nan")

    out = {
        **ret_stats,
        "util_m": float(util_stats["mean_m"]),
        "util_a": float(util_stats["mean_a"]),
        "risk_m": risk_mean_m,

        "max_w_med": float(np.nanmedian(maxw)),
        "n_eff_med": float(np.nanmedian(neff)),
        "top10_med": float(np.nanmedian(top10)),
        "ic_med": float(np.nanmedian(ic_list)),
        "mu_std_med": float(np.nanmedian(mu_std_list)),

        "returns": np.array(rets, dtype=float),
        "utilities": np.array(utils, dtype=float),
        "risks": np.array(risks, dtype=float),
    }
    return out


@torch.no_grad()
def backtest_and_save(
    run_dir: Path,
    model: PAOPortfolioModel,
    test_ds,  # MonthCacheDataset
    device: str
) -> Dict[str, Any]:
    """
    Backtest E2E portfolio model on test set and save results.

    Computes performance metrics, diagnostics, and saves:
    - test_timeseries.csv: Per-month performance
    - test_summary.json: Aggregate statistics

    Args:
        run_dir: Directory to save results
        model: PAOPortfolioModel instance
        test_ds: Test dataset (MonthCacheDataset)
        device: Device to run evaluation on

    Returns:
        Dictionary with:
        - Model performance: test_sharpe, test_mean_a, test_vol_a, test_max_dd
        - Utility metrics: test_util_a, test_util_vol_a, test_util_sharpe_a
        - Risk metrics: test_risk_m
        - Weight diagnostics: test_max_w_med, test_n_eff_med, test_ic_med
        - Equal-weight baseline: equal_sharpe, equal_mean_a, equal_vol_a, equal_max_dd
        - Number of months: n_months

    Side effects:
        - Saves test_timeseries.csv with monthly performance
        - Saves test_summary.json with aggregate statistics
    """
    model.eval()
    rows = []
    model_rets, ew_rets, model_utils = [], [], []

    for i in range(len(test_ds)):
        s = test_ds[i]
        t = int(s["yyyymm"])
        X = s["X"].to(device)
        y = s["y"].to(device)
        U = s["Sigma_factor"].to(device)
        sigma_vol = s["sigma_vol"].to(device)

        w, mu_raw = model(X, U, sigma_vol)

        r_model = float((w * y).sum().item())
        r_ew = float(y.mean().item())

        Uw = torch.mv(U, w)
        risk = float(torch.sum(Uw * Uw).item())
        util = r_model - (model.lambda_ / 2.0) * risk

        w_np = w.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        mu_np = mu_raw.detach().cpu().numpy()

        wd = weights_diagnostics(w_np)
        ic = (
            float(np.corrcoef(mu_np, y_np)[0, 1])
            if (np.std(mu_np, ddof=1) > 1e-12 and np.std(y_np, ddof=1) > 1e-12)
            else np.nan
        )

        model_rets.append(r_model)
        ew_rets.append(r_ew)
        model_utils.append(util)

        rows.append({
            "yyyymm": t,
            "ret_model": r_model,
            "ret_equal": r_ew,
            "utility_model": util,
            "risk_model": risk,
            "max_w": wd["max_w"],
            "n_eff": wd["n_eff"],
            "top10": wd["top10"],
            "active": wd["active"],
            "ic": ic,
            "mu_std": float(np.std(mu_np, ddof=1)) if mu_np.size > 1 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("yyyymm").reset_index(drop=True)
    df["wealth_model"] = np.cumprod(1.0 + df["ret_model"].values)
    df["wealth_equal"] = np.cumprod(1.0 + df["ret_equal"].values)
    df.to_csv(run_dir / "test_timeseries.csv", index=False)

    st_model = perf_stats(df["ret_model"].values)
    st_ew = perf_stats(df["ret_equal"].values)
    st_util = perf_stats(df["utility_model"].values)

    out = {
        "test_sharpe": st_model["sharpe_a"],
        "test_mean_a": st_model["mean_a"],
        "test_vol_a": st_model["vol_a"],
        "test_max_dd": st_model["max_drawdown"],

        "test_util_a": st_util["mean_a"],  # "annualized utility" (mean*12)
        "test_util_vol_a": st_util["vol_a"],
        "test_util_sharpe_a": st_util["sharpe_a"],

        "test_risk_m": float(df["risk_model"].mean()),
        "test_max_w_med": float(df["max_w"].median()),
        "test_n_eff_med": float(df["n_eff"].median()),
        "test_ic_med": float(np.nanmedian(df["ic"].values)),
        "n_months": int(len(df)),

        "equal_sharpe": st_ew["sharpe_a"],
        "equal_mean_a": st_ew["mean_a"],
        "equal_vol_a": st_ew["vol_a"],
        "equal_max_dd": st_ew["max_drawdown"],
    }

    save_json(run_dir / "test_summary.json", out)
    return out
