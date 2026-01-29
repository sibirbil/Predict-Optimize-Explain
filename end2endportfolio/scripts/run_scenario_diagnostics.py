#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:20:17 2025

@author: batuhanatas
"""

# run_scenario_diagnostics.py
# Usage:
#   python scripts/run_scenario_diagnostics.py --in artifacts/scenarios/generated_scenarios_full_2131_assets.pt --alpha 0.95 --topk 50
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "diagnostics" / "scenario_diagnostics_out"


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        try:
            return torch.tensor(x)
        except Exception:
            return None
    return None


def _skew_excess(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1) if x.size > 1 else 0.0
    if s == 0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 3))


def _kurt_excess(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1) if x.size > 1 else 0.0
    if s == 0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 4) - 3.0)


def _var_cvar(arr: np.ndarray, alpha=0.95) -> Tuple[float, float]:
    q = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= q]
    cvar = tail.mean() if tail.size else q
    return float(q), float(cvar)


def _drawdown_like(series: np.ndarray) -> Dict[str, float]:
    # If returns are > -1, compute multiplicative DD; else additive DD on cumsum.
    if series.size == 0:
        return {"max_drawdown": float("nan"), "terminal_value": float("nan")}
    if np.all(series > -1.0):
        cum = np.cumprod(1.0 + series)
    else:
        cum = np.cumsum(series)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return {
        "max_drawdown": float(dd.min()),
        "terminal_value": float(cum[-1]),
    }


def load_scenarios(path: str, prefer_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    meta: Dict[str, Any] = {}

    scenario = None
    if isinstance(obj, dict):
        candidate_keys = [
            "trajectory",
            "scenarios",
            "returns",
            "data",
            "X",
            "matrix",
            "scenario_matrix",
            "R",
            "final_state",
        ]
        if prefer_key:
            candidate_keys = [prefer_key] + [k for k in candidate_keys if k != prefer_key]
        for key in candidate_keys:
            if key in obj:
                t = _to_tensor(obj[key])
                if t is not None:
                    scenario = t
                    meta["source_key"] = key
                    break
        # stash light metadata
        for k, v in obj.items():
            if k == meta.get("source_key"):
                continue
            if isinstance(v, (str, int, float)):
                meta[k] = v
            elif isinstance(v, (list, tuple)):
                meta[k] = f"{type(v).__name__}[{len(v)}]"
            elif isinstance(v, torch.Tensor):
                meta[k] = f"Tensor{tuple(v.shape)}"
            elif isinstance(v, np.ndarray):
                meta[k] = f"ndarray{v.shape}"
            elif isinstance(v, dict):
                meta[k] = f"dict({len(v)})"
            else:
                meta[k] = type(v).__name__
    elif isinstance(obj, (torch.Tensor, np.ndarray, list, tuple)):
        scenario = _to_tensor(obj)
    else:
        meta["object_type"] = type(obj).__name__

    if scenario is None:
        raise RuntimeError(
            f"Could not locate a scenario matrix inside {path}. "
            f"Top-level object type: {type(obj).__name__}. Metadata: {json.dumps(meta)[:500]}..."
        )

    scenario = scenario.detach().cpu().float()
    if scenario.ndim != 2:
        # Flatten best-effort: keep first dim as assets
        scenario = scenario.reshape(scenario.shape[0], -1)

    A, B = scenario.shape
    # Heuristic: rows are assets if rows>cols or if a known asset count matches (e.g., 2131)
    rows_are_assets = (A >= B) or (A in (2131, 500, 1000, 2000))
    if not rows_are_assets:
        scenario = scenario.T
        A, B = scenario.shape
        meta["transposed"] = True
    else:
        meta["transposed"] = False

    meta["shape_assets_by_scenarios"] = f"{A}x{B}"
    return scenario, meta


def compute_diagnostics(
    scenario: torch.Tensor,
    alpha: float = 0.95,
    topk: int = 50,
) -> Dict[str, Any]:
    A, T = scenario.shape
    X = scenario.numpy()
    # Basic finite check
    finite_mask = np.isfinite(X)
    n_total = X.size
    n_finite = int(finite_mask.sum())
    # Per-asset stats
    means = X.mean(axis=1)
    stds = X.std(axis=1, ddof=1) if T > 1 else np.zeros(A, dtype=float)
    skews = np.apply_along_axis(_skew_excess, 1, X)
    kurts = np.apply_along_axis(_kurt_excess, 1, X)
    vars_95 = np.empty(A, dtype=float)
    cvars_95 = np.empty(A, dtype=float)
    for i in range(A):
        v, c = _var_cvar(X[i, :], alpha)
        vars_95[i] = v
        cvars_95[i] = c
    df_assets = pd.DataFrame(
        {
            "asset_id": np.arange(A),
            "mean": means,
            "std": stds,
            "skew": skews,
            "excess_kurtosis": kurts,
            "VaR_alpha": vars_95,
            "CVaR_alpha": cvars_95,
        }
    ).set_index("asset_id")

    # Global distribution
    all_vals = X.reshape(-1)
    global_stats = {
        "assets": int(A),
        "scenarios_per_asset": int(T),
        "finite_ratio": float(n_finite / n_total),
        "global_mean": float(np.mean(all_vals)),
        "global_std": float(np.std(all_vals, ddof=1) if all_vals.size > 1 else 0.0),
        "global_skew": float(_skew_excess(all_vals)),
        "global_excess_kurtosis": float(_kurt_excess(all_vals)),
        "min": float(np.min(all_vals)),
        "p01": float(np.quantile(all_vals, 0.01)),
        "p05": float(np.quantile(all_vals, 0.05)),
        "p50": float(np.quantile(all_vals, 0.50)),
        "p95": float(np.quantile(all_vals, 0.95)),
        "p99": float(np.quantile(all_vals, 0.99)),
        "max": float(np.max(all_vals)),
    }
    df_global = pd.DataFrame([global_stats])

    # Covariance / correlation
    Xc = X - means[:, None]
    denom = max(T - 1, 1)
    cov = (Xc @ Xc.T) / denom
    cov = 0.5 * (cov + cov.T)
    # Eigen-decomp (top-k)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0, None)
        eigvals_sorted = np.sort(eigvals)[::-1]
        k = min(topk, eigvals_sorted.size)
        top_eigs = eigvals_sorted[:k]
        tot_var = eigvals_sorted.sum() + 1e-12
        explained = top_eigs / tot_var
        cum_explained = np.cumsum(explained)
        p = eigvals_sorted / tot_var
        p = p[p > 0]
        H = -np.sum(p * np.log(p + 1e-12))
        effective_rank = float(np.exp(H))
    except Exception:
        top_eigs, explained, cum_explained, effective_rank = [], [], [], float("nan")

    # Average pairwise correlation
    try:
        stds_safe = np.where(stds == 0, 1.0, stds)
        Xn = Xc / stds_safe[:, None]
        corr = (Xn @ Xn.T) / denom
        np.fill_diagonal(corr, 1.0)
        m = corr.shape[0]
        off = corr[~np.eye(m, dtype=bool)]
        avg_pair_corr = float(off.mean())
        # Also capture correlation percentiles for flavor
        corr_p05 = float(np.quantile(off, 0.05))
        corr_p50 = float(np.quantile(off, 0.50))
        corr_p95 = float(np.quantile(off, 0.95))
    except Exception:
        avg_pair_corr, corr_p05, corr_p50, corr_p95 = float("nan"), float("nan"), float("nan"), float("nan")

    df_cov = pd.DataFrame(
        {
            "top_eigenvalue": [float(top_eigs[0]) if len(top_eigs) else float("nan")],
            "sum_top_k_eigs": [float(np.sum(top_eigs)) if len(top_eigs) else float("nan")],
            "explained_by_top_k": [float(np.sum(explained)) if len(top_eigs) else float("nan")],
            "effective_rank": [effective_rank],
            "avg_pairwise_corr": [avg_pair_corr],
            "corr_p05": [corr_p05],
            "corr_p50": [corr_p50],
            "corr_p95": [corr_p95],
        }
    )

    # Equal-weight portfolio diagnostics (distribution over scenarios)
    w_eq = np.full(A, 1.0 / A, dtype=np.float64)
    ew_series = w_eq @ X
    ew_stats = {
        "mean": float(np.mean(ew_series)),
        "std": float(np.std(ew_series, ddof=1) if ew_series.size > 1 else 0.0),
        "VaR_alpha": float(np.quantile(ew_series, 1 - alpha)),
        "CVaR_alpha": float(ew_series[ew_series <= np.quantile(ew_series, 1 - alpha)].mean())
        if ew_series.size
        else float("nan"),
    }
    ew_stats.update(_drawdown_like(ew_series))
    df_ew = pd.DataFrame([ew_stats])

    return {
        "df_assets": df_assets,
        "df_global": df_global,
        "df_cov": df_cov,
        "df_ew": df_ew,
        "all_vals": all_vals,
        "stds": stds,
        "top_eigs": np.asarray(top_eigs) if len(top_eigs) else np.array([]),
        "cum_explained": np.asarray(cum_explained) if len(top_eigs) else np.array([]),
    }


def save_outputs(
    out_dir: Path,
    meta: Dict[str, Any],
    alpha: float,
    topk: int,
    results: Dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSVs
    results["df_assets"].to_csv(out_dir / "per_asset_diagnostics.csv")
    results["df_global"].to_csv(out_dir / "global_diagnostics.csv", index=False)
    results["df_cov"].to_csv(out_dir / "cov_corr_diagnostics.csv", index=False)
    results["df_ew"].to_csv(out_dir / "equal_weight_portfolio_diagnostics.csv", index=False)

    # Plots (Matplotlib; no seaborn; single-plot figures)
    all_vals = results["all_vals"]
    stds = results["stds"]
    plt.figure()
    plt.hist(all_vals, bins=100)
    plt.title("Distribution of all scenario values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_all_values.png", dpi=150)
    plt.close()

    plt.figure()
    plt.hist(stds, bins=100)
    plt.title("Distribution of per-asset volatility (std)")
    plt.xlabel("Std")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_asset_vols.png", dpi=150)
    plt.close()

    if results["top_eigs"].size:
        plt.figure()
        plt.plot(np.arange(1, results["top_eigs"].size + 1), results["top_eigs"])
        plt.title("Scree plot (top eigenvalues of covariance)")
        plt.xlabel("Eigenvalue rank")
        plt.ylabel("Eigenvalue")
        plt.tight_layout()
        plt.savefig(out_dir / "scree_top_eigenvalues.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(np.arange(1, results["cum_explained"].size + 1), results["cum_explained"])
        plt.title("Cumulative explained variance (top eigenvalues)")
        plt.xlabel("Top-k")
        plt.ylabel("Cumulative fraction")
        plt.tight_layout()
        plt.savefig(out_dir / "cumulative_explained_variance.png", dpi=150)
        plt.close()

    # Compact text/markdown report
    g = results["df_global"].iloc[0].to_dict()
    c = results["df_cov"].iloc[0].to_dict()
    e = results["df_ew"].iloc[0].to_dict()

    lines = []
    lines.append("# Scenario Diagnostics Summary\n")
    lines.append(f"- Shape (assets × scenarios): {meta.get('shape_assets_by_scenarios')}")
    lines.append(f"- Finite ratio: {g['finite_ratio']:.6f}")
    lines.append(
        f"- Global mean {g['global_mean']:.6f}, std {g['global_std']:.6f}, skew {g['global_skew']:.6f}, ex.kurt {g['global_excess_kurtosis']:.6f}"
    )
    if not math.isnan(c.get("top_eigenvalue", float("nan"))):
        lines.append(
            f"- Top eigenvalue: {c['top_eigenvalue']:.6f}, effective rank ≈ {c['effective_rank']:.2f}, "
            f"explained by top-k (k={topk}): {c['explained_by_top_k']:.4f}"
        )
    if not math.isnan(c.get("avg_pairwise_corr", float("nan"))):
        lines.append(
            f"- Avg pairwise corr: {c['avg_pairwise_corr']:.6f} "
            f"(p05={c['corr_p05']:.4f}, med={c['corr_p50']:.4f}, p95={c['corr_p95']:.4f})"
        )
    lines.append(
        f"- Equal-weight: mean {e['mean']:.6f}, std {e['std']:.6f}, VaRα {e['VaR_alpha']:.6f}, CVaRα {e['CVaR_alpha']:.6f}, "
        f"max DD-like {e['max_drawdown']:.6f}"
    )
    lines.append("\n## Notes\n- VaR/CVaR computed per asset (left tail), α as provided.\n"
                 "- Drawdown-like metric assumes scenarios may be sequential; interpret with care if not time-ordered.\n"
                 "- Eigen-spectrum/effective rank summarize cross-sectional dependence.\n")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Diagnostics for scenario matrix saved in a .pt file.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to .pt file")
    ap.add_argument(
        "--out",
        dest="out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default reports/diagnostics/scenario_diagnostics_out)",
    )
    ap.add_argument("--alpha", type=float, default=0.95, help="VaR/CVaR confidence level (default 0.95)")
    ap.add_argument("--topk", type=int, default=50, help="How many top eigenvalues to summarize (default 50)")
    ap.add_argument(
        "--load-key",
        dest="load_key",
        default=None,
        help="Optional key to pull from the saved object (e.g., 'trajectory', 'final_state').",
    )
    args = ap.parse_args()

    in_path = args.in_path
    out_dir = Path(args.out_dir)

    scenario, meta = load_scenarios(in_path, prefer_key=args.load_key)
    results = compute_diagnostics(scenario, alpha=args.alpha, topk=args.topk)
    save_outputs(out_dir, meta, args.alpha, args.topk, results)

    print("Diagnostics written to:", out_dir)
    print(" - report.md")
    print(" - per_asset_diagnostics.csv")
    print(" - global_diagnostics.csv")
    print(" - cov_corr_diagnostics.csv")
    print(" - equal_weight_portfolio_diagnostics.csv")
    print(" - plots: *.png")


if __name__ == "__main__":
    main()
