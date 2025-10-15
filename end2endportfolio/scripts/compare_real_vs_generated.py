#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:26:49 2025

@author: batuhanatas
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "diagnostics" / "quick_checks"

from src.data_utils import load_panel, clean_missing_xy  # noqa: E402

FILL_VALUE = -99.99

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------- loading helpers ---------------------- #
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


def _load_matrix(
    path: str,
    prefer_key: Optional[str] = None,
    returns_only: bool = False,
    feature_index: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a 2D matrix from .pt/.npz/.csv and return assets × scenarios."""
    p = Path(path)
    meta: Dict[str, Any] = {"source": str(p)}
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p, header=None)
        arr = df.values.astype(np.float32)
        t = torch.from_numpy(arr)
        scenario = t
        meta["format"] = "csv"
    elif suffix == ".npz":
        panel = load_panel(str(p))
        X_list, y_list, _, _ = clean_missing_xy(panel.X, panel.y, panel.asset_ids, panel.tickers)

        if returns_only:
            min_assets = min(len(y) for y in y_list)
            if min_assets == 0:
                raise RuntimeError("No valid returns after cleaning real panel")
            matrix = np.stack([y[:min_assets] for y in y_list], axis=1)
            scenario = torch.from_numpy(matrix.astype(np.float32))
            meta["pt_key"] = "returns"
        else:
            if feature_index is not None:
                feats = [x[:, feature_index] for x in X_list if x.shape[1] > feature_index]
                if not feats:
                    raise IndexError("feature_index out of range for cleaned real panel")
                min_assets = min(len(v) for v in feats)
                matrix = np.stack([v[:min_assets] for v in feats], axis=1)
                scenario = torch.from_numpy(matrix.astype(np.float32))
                meta["pt_key"] = f"feature_{feature_index}"
            else:
                min_assets = min(x.shape[0] for x in X_list)
                feature_dim = X_list[0].shape[1]
                matrix = np.concatenate([x[:min_assets].astype(np.float32) for x in X_list], axis=1)
                scenario = torch.from_numpy(matrix)
                meta["pt_key"] = "data"

        meta["format"] = "npz"
    else:
        obj = torch.load(p, map_location="cpu")
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
                        meta["pt_key"] = key
                        break
        elif isinstance(obj, (torch.Tensor, np.ndarray, list, tuple)):
            scenario = _to_tensor(obj)
        else:
            raise RuntimeError(f"Unknown object in {path}: {type(obj).__name__}")

        if scenario is None:
            raise RuntimeError(f"Could not find a matrix inside {path}")

        if returns_only and scenario.ndim == 3:
            # assume shape (steps, assets, features)
            if scenario.shape[-1] < 1:
                raise RuntimeError("trajectory tensor missing return column")
            scenario = scenario[..., 0]
        elif feature_index is not None and scenario.ndim == 3:
            idx = feature_index + 1 if meta.get("pt_key") == "trajectory" else feature_index
            if idx >= scenario.shape[-1]:
                raise RuntimeError("feature_index out of range for generated tensor")
            scenario = scenario[..., idx]

    scenario = scenario.detach().cpu().float()
    if scenario.ndim == 3:
        raise RuntimeError("Tensor is still 3D after extraction; specify --returns-only or --feature-index.")
    if scenario.ndim != 2:
        scenario = scenario.reshape(scenario.shape[0], -1)

    A, B = scenario.shape
    # heuristic: rows=assets if rows>=cols OR rows looks like big cross-section
    if not (A >= B or A in (2131, 1000, 500, 2000)):
        scenario = scenario.T
        A, B = scenario.shape
        meta["transposed"] = True
    else:
        meta["transposed"] = False

    meta["shape_assets_by_scenarios"] = f"{A}x{B}"
    return scenario, meta


def _maybe_load_weights(path: str, A: int) -> np.ndarray:
    if path is None:
        return None
    p = Path(path)
    if p.suffix.lower() == ".csv":
        w = pd.read_csv(p, header=None).values.squeeze()
    elif p.suffix.lower() == ".npy":
        w = np.load(p)
    else:
        # try pt
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "w" in obj:
            w = obj["w"]
        else:
            w = obj
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.shape[0] != A:
        raise ValueError(f"Weights length {w.shape[0]} != assets {A}")
    return w


# ---------------------- simple stats ---------------------- #
def _per_asset_stats(X: np.ndarray, alpha: float = 0.95) -> pd.DataFrame:
    A, T = X.shape
    means = X.mean(axis=1)
    stds  = X.std(axis=1, ddof=1) if T > 1 else np.zeros(A)
    # light tails (no SciPy): VaR/CVaR left tail
    q = np.quantile(X, 1 - alpha, axis=1)
    cvar = np.array([row[row <= qi].mean() if (row <= qi).any() else qi
                     for row, qi in zip(X, q)], dtype=float)
    return pd.DataFrame({
        "mean": means, "std": stds, f"VaR_{alpha}": q, f"CVaR_{alpha}": cvar
    })


def _avg_corr(X: np.ndarray) -> float:
    A, T = X.shape
    if T < 2:
        return np.nan
    Xc = X - X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, ddof=1, keepdims=True)
    stds[stds == 0] = 1.0
    Z = Xc / stds
    C = (Z @ Z.T) / (T - 1)
    m = C.shape[0]
    off = C[~np.eye(m, dtype=bool)]
    return float(off.mean())


def _ew_series(X: np.ndarray) -> np.ndarray:
    A, T = X.shape
    w = np.full(A, 1.0 / A)
    return w @ X  # (T,)


def _gmv_weights(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """ Global minimum variance weights (no-short) relaxed to unconstrained with sum-to-1 only """
    A, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = (Xc @ Xc.T) / max(T - 1, 1)
    cov = 0.5 * (cov + cov.T) + ridge * np.eye(A)
    ones = np.ones(A)
    # solve cov w = λ 1  s.t. 1' w = 1
    inv = np.linalg.pinv(cov)
    w = inv @ ones
    w = w / (ones @ w)
    return w  # may have small negatives; clip if you want long-only


# ---------------------- plotting (clean & minimal) ---------------------- #
def _hist_overlay(vals_real, vals_gen, title, outpath):
    plt.figure()
    plt.hist(vals_real, bins=100, alpha=0.5, label="real")
    plt.hist(vals_gen,  bins=100, alpha=0.5, label="generated")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("freq")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _scatter_compare(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.6)
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    plt.plot([lo, hi], [lo, hi])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Quick side-by-side diagnostics for generated vs real scenarios")
    ap.add_argument("--gen",  required=True, help=".pt or .csv with generated scenarios")
    ap.add_argument("--real", required=True, help=".pt or .csv with real scenarios")
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help="output folder (default reports/diagnostics/quick_checks)",
    )
    ap.add_argument("--alpha", type=float, default=0.95, help="VaR/CVaR alpha (default 0.95)")
    ap.add_argument("--weights", default=None, help="optional portfolio weights (.csv/.npy/.pt)")
    ap.add_argument("--clip_assets", type=int, default=None, help="use only first N assets (optional)")
    ap.add_argument("--gen-key", default=None, help="optional key inside generated .pt (e.g. trajectory)")
    ap.add_argument("--real-key", default=None, help="optional key inside real .pt/.npz")
    ap.add_argument("--returns-only", action="store_true", help="compare only the return dimension")
    ap.add_argument("--feature-index", type=int, default=None, help="compare a specific feature index")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.returns_only and args.feature_index is not None:
        raise ValueError("Specify either --returns-only or --feature-index, not both.")

    G, meta_g = _load_matrix(
        args.gen,
        prefer_key=args.gen_key,
        returns_only=args.returns_only,
        feature_index=args.feature_index,
    )
    R, meta_r = _load_matrix(
        args.real,
        prefer_key=args.real_key,
        returns_only=args.returns_only,
        feature_index=args.feature_index,
    )

    # Align asset count (use intersection by clipping if requested)
    A_g, T_g = G.shape
    A_r, T_r = R.shape
    if args.clip_assets is not None:
        A_use = min(args.clip_assets, A_g, A_r)
    else:
        A_use = min(A_g, A_r)
    G = G[:A_use, :]
    R = R[:A_use, :]
    A = A_use

    # Simple global distributions (all entries)
    all_g = G.numpy().reshape(-1)
    all_r = R.numpy().reshape(-1)

    _hist_overlay(all_r, all_g, "All entries: real vs generated", outdir / "hist_all_entries.png")

    # Per-asset stats & comparisons
    df_g = _per_asset_stats(G.numpy(), alpha=args.alpha)
    df_r = _per_asset_stats(R.numpy(), alpha=args.alpha)

    df_g.to_csv(outdir / "per_asset_generated.csv")
    df_r.to_csv(outdir / "per_asset_real.csv")

    # mean/std scatter
    _scatter_compare(
        df_r["mean"].values, df_g["mean"].values,
        "Per-asset mean: generated vs real",
        "real mean", "generated mean",
        outdir / "scatter_mean_gen_vs_real.png",
    )
    _scatter_compare(
        df_r["std"].values, df_g["std"].values,
        "Per-asset std: generated vs real",
        "real std", "generated std",
        outdir / "scatter_std_gen_vs_real.png",
    )

    # Correlation structure (average off-diagonal)
    avg_corr_r = _avg_corr(R.numpy())
    avg_corr_g = _avg_corr(G.numpy())

    with open(outdir / "quick_summary.json", "w") as f:
        json.dump({
            "real_shape_assets_by_scenarios": meta_r["shape_assets_by_scenarios"],
            "gen_shape_assets_by_scenarios": meta_g["shape_assets_by_scenarios"],
            "avg_pairwise_corr_real": avg_corr_r,
            "avg_pairwise_corr_generated": avg_corr_g,
        }, f, indent=2)

    # Overlay equal-weight portfolio return distributions
    ew_r = _ew_series(R.numpy())
    ew_g = _ew_series(G.numpy())
    _hist_overlay(ew_r, ew_g, "Equal-weight portfolio returns: real vs generated",
                  outdir / "hist_EW_portfolio_returns.png")

    # If user supplies weights, evaluate portfolio in both worlds
    w_user = _maybe_load_weights(args.weights, A) if args.weights else None

    # Also compute a simple GMV (unconstrained) on *real* and test it on both
    try:
        w_gmv_real = _gmv_weights(R.numpy())
        w_gmv_real = np.clip(w_gmv_real, 0, None)
        if w_gmv_real.sum() > 0:
            w_gmv_real /= w_gmv_real.sum()
    except Exception:
        w_gmv_real = None

    def _portfolio_series(X, w):
        return w @ X  # (T,)

    port_sets = []
    port_sets.append(("EW", np.full(A, 1.0 / A)))
    if w_user is not None:
        port_sets.append(("User", w_user))
    if w_gmv_real is not None:
        port_sets.append(("GMV_from_real", w_gmv_real))

    # simple dist plots for each portfolio across real & generated
    for name, w in port_sets:
        pr = _portfolio_series(R.numpy(), w)
        pg = _portfolio_series(G.numpy(), w)
        _hist_overlay(pr, pg, f"{name} portfolio returns: real vs generated",
                      outdir / f"hist_{name}_portfolio_returns.png")

    # Save the weights we actually evaluated
    weights_out = {name: w.tolist() for name, w in port_sets}
    with open(outdir / "weights_used.json", "w") as f:
        json.dump(weights_out, f, indent=2)

    print("Wrote quick diagnostics to:", outdir)
    print(" - hist_all_entries.png")
    print(" - scatter_mean_gen_vs_real.png, scatter_std_gen_vs_real.png")
    print(" - hist_EW_portfolio_returns.png (+ per-portfolio hist_*.png)")
    print(" - per_asset_generated.csv, per_asset_real.csv")
    print(" - quick_summary.json, weights_used.json")


if __name__ == "__main__":
    main()
