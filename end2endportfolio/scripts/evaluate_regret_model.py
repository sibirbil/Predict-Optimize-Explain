#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained regret model on train/valid/test panels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Keep NumPy/OpenMP usage predictable inside sandboxes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_CREATE_SHARED", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (  # noqa: E402
    FILL_VALUE,
    build_return_history,
    clean_missing_xy,
    covariance_from_history,
    load_panel,
    scale_features,
)
from src.covariance_utils import (  # noqa: E402
    build_ticker_return_panel,
    covariance_from_ticker_panel,
)
from src.model import FNN  # noqa: E402
from src.regret_loss import regret_loss  # noqa: E402
from src.portfolio_layer import build_portfolio_layer, prepare_layer_inputs  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURE_MASK_PATH = ARTIFACTS_DIR / "feature_mask.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained regret model.")
    parser.add_argument("--model", default="trained_model.pth", help="Model filename under artifacts/models/")
    parser.add_argument("--train", default="panel_train.npz", help="Training panel filename under data/")
    parser.add_argument("--valid", default="panel_valid.npz", help="Validation panel filename under data/")
    parser.add_argument("--test", default="panel_test.npz", help="Test panel filename under data/")
    parser.add_argument(
        "--risk",
        type=float,
        default=1.0,
        help="Risk aversion parameter (lambda) to use inside regret loss (default 1.0).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write evaluation summary as JSON (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--forward-panel",
        choices=["train", "valid", "test"],
        default=None,
        help="Optional split to run one-period-ahead evaluation (use features at t-1 to trade returns at t).",
    )
    parser.add_argument(
        "--forward-start",
        default=None,
        help="Start date (e.g. 2024-07 or 2024-07-31) for forward evaluation; defaults to earliest available.",
    )
    parser.add_argument(
        "--forward-count",
        type=int,
        default=None,
        help="Number of timestamps to include in forward evaluation (default uses all until panel end).",
    )
    parser.add_argument(
        "--weights-out",
        default=None,
        help="Optional path (JSON) to dump per-asset portfolio weights across evaluations.",
    )
    return parser.parse_args()


def load_feature_mask(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature mask not found at {path}. Run scripts/train_regret_model.py before evaluation."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    kept = payload.get("kept_features")
    if not kept:
        raise ValueError(f"Feature mask file {path} missing 'kept_features'.")
    return [str(name) for name in kept]


def apply_feature_mask(
    X_list: List[np.ndarray],
    feature_names: np.ndarray,
    keep_names: List[str],
) -> List[np.ndarray]:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    indices = []
    for name in keep_names:
        if name not in name_to_idx:
            raise RuntimeError(f"Feature '{name}' from mask not found in panel columns.")
        indices.append(name_to_idx[name])
    indices = np.asarray(indices, dtype=int)
    return [x[:, indices] for x in X_list]


def _geometric_growth(returns: np.ndarray) -> float:
    """
    Convert a series of simple returns into cumulative growth (final value - 1).
    Caps large drawdowns at -100% to avoid log issues.
    """
    if returns.size == 0:
        return 0.0
    safe = np.clip(returns, -0.999999, None)
    log_growth = np.log1p(safe)
    return float(np.expm1(np.sum(log_growth)))


def _align_assets(
    X_prev: np.ndarray,
    y_curr: np.ndarray,
    ids_prev: Optional[np.ndarray],
    ids_curr: Optional[np.ndarray],
    tickers_prev: Optional[np.ndarray],
    tickers_curr: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """
    Align assets across consecutive timestamps using identifiers.
    Returns feature matrix for previous timestamp, returns for current timestamp,
    and aligned asset/ticker lists (or None if unavailable).
    """
    if ids_prev is None or ids_curr is None:
        n = min(len(X_prev), len(y_curr))
        tickers_aligned = None
        if tickers_prev is not None and tickers_curr is not None:
            tickers_aligned = [str(t) for t in tickers_curr[:n]]
        return X_prev[:n], y_curr[:n], None, tickers_aligned

    prev_map = {str(asset_id): idx for idx, asset_id in enumerate(ids_prev)}
    idx_prev: List[int] = []
    idx_curr: List[int] = []
    aligned_ids: List[str] = []
    for curr_idx, asset_id in enumerate(ids_curr):
        key = str(asset_id)
        prev_idx = prev_map.get(key)
        if prev_idx is not None:
            idx_prev.append(prev_idx)
            idx_curr.append(curr_idx)
            aligned_ids.append(key)

    if not idx_prev:
        return (
            np.zeros((0, X_prev.shape[1]), dtype=X_prev.dtype),
            np.zeros((0,), dtype=y_curr.dtype),
            [],
            [] if tickers_curr is not None else None,
        )

    X_aligned = X_prev[idx_prev]
    y_aligned = y_curr[idx_curr]
    tickers_aligned: Optional[List[str]] = None
    if tickers_curr is not None:
        tickers_aligned = [str(tickers_curr[i]) for i in idx_curr]
    return X_aligned, y_aligned, aligned_ids, tickers_aligned


def _resolve_start_index(
    dates: Optional[np.ndarray],
    start_token: Optional[str],
) -> int:
    """
    Resolve the index of the forward evaluation start date.
    Requires at least one prior timestamp (index >= 1).
    """
    if dates is None or dates.size == 0:
        return 1
    if start_token is None:
        return 1

    # Try exact match, falling back to month resolution
    try:
        target = np.datetime64(start_token, "D")
    except ValueError:
        target = np.datetime64(f"{start_token}-01", "D")

    matches = np.flatnonzero(dates == target)
    if matches.size == 0:
        # fallback: match by month when exact day not present
        try:
            dates_month = dates.astype("datetime64[M]")
            target_month = target.astype("datetime64[M]")
            matches = np.flatnonzero(dates_month == target_month)
        except Exception:
            matches = np.array([], dtype=int)
    if matches.size == 0:
        raise ValueError(f"Start date {start_token} not found in panel.")
    idx = int(matches[0])
    if idx == 0:
        raise ValueError("Forward evaluation start date must have a preceding timestamp (idx >= 1).")
    return idx


def evaluate_split(
    name: str,
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    lambda_: float,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    asset_ids_list: Optional[List[np.ndarray]],
    tickers_list: Optional[List[np.ndarray]],
    date_seq: Optional[Sequence],
    return_history,
    ticker_panel: Optional[pd.DataFrame],
    weights_recorder: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    snapshot_regrets: List[float] = []
    snapshot_mse: List[float] = []
    total_assets = 0
    ew_returns: List[float] = []
    model_returns: List[float] = []
    timeline: List[str] = []

    for idx, (X_np, y_np) in enumerate(zip(X_list, y_list)):
        if X_np.size == 0 or y_np.size == 0:
            continue
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        sigma_override = None
        ids_snapshot = None
        tickers_snapshot = None
        if asset_ids_list is not None and idx < len(asset_ids_list):
            ids_snapshot = asset_ids_list[idx]
            if ids_snapshot is not None and len(ids_snapshot) == len(X_np):
                cov_np = covariance_from_history(ids_snapshot, return_history)
                sigma_override = torch.from_numpy(cov_np.astype(np.float64))
            elif ids_snapshot is not None and len(ids_snapshot) != len(X_np):
                ids_snapshot = None
        if tickers_list is not None and idx < len(tickers_list):
            tickers_snapshot = tickers_list[idx]
            if tickers_snapshot is not None and len(tickers_snapshot) != len(X_np):
                tickers_snapshot = None

        raw_date = date_seq[idx] if date_seq is not None and idx < len(date_seq) else None
        date_ts = pd.to_datetime(raw_date) if raw_date is not None else None

        if ticker_panel is not None and tickers_snapshot is not None and len(tickers_snapshot):
            panel_slice = ticker_panel
            if date_ts is not None and not pd.isna(date_ts) and isinstance(ticker_panel.index, pd.DatetimeIndex):
                panel_slice = ticker_panel.loc[:date_ts]
            cov_np = covariance_from_ticker_panel(tickers_snapshot, panel_slice)
            if cov_np is not None:
                sigma_override = torch.from_numpy(cov_np.astype(np.float64))

        with torch.no_grad():
            regretsq = regret_loss(
                model,
                lambda_,
                params,
                X_tensor,
                y_tensor,
                portfolio_layer=None,
                Sigma_override=sigma_override,
            ).item()
            preds = model(X_tensor).squeeze(-1)
            mse = torch.mean((preds - y_tensor) ** 2).item()

        snapshot_regrets.append(regretsq)
        snapshot_mse.append(mse)
        total_assets += len(X_np)

        # Equal-weight portfolio return
        ew_returns.append(float(np.mean(y_np)))

        # Model-optimised portfolio return
        preds = model(X_tensor).detach()
        preds_cpu, L_cpu = prepare_layer_inputs(
            preds,
            X_batch=X_tensor if sigma_override is None else None,
            Sigma_override=sigma_override,
        )
        y_cpu = y_tensor.to(dtype=torch.double, device="cpu")
        layer = build_portfolio_layer(len(y_np), lambda_)
        with torch.no_grad():
            (weights,) = layer(preds_cpu, L_cpu)
            model_port_ret = torch.dot(y_cpu, weights).item()
        model_returns.append(float(model_port_ret))

        if date_ts is not None:
            date_str = date_ts.strftime("%Y-%m-%d")
        elif raw_date is not None:
            date_str = str(raw_date)
        else:
            date_str = str(idx)

        if weights_recorder is not None:
            weights_np = weights.detach().cpu().numpy()
            preds_arr = preds_cpu.detach().cpu().numpy()
            y_arr = y_cpu.detach().cpu().numpy()
            ids_arr = ids_snapshot if ids_snapshot is not None else None
            tickers_arr = tickers_snapshot if tickers_snapshot is not None else None
            for asset_idx, weight_val in enumerate(weights_np):
                record = {
                    "split": name,
                    "date": date_str,
                    "asset_index": int(asset_idx),
                    "weight": float(weight_val),
                    "prediction": float(preds_arr[asset_idx]),
                    "return": float(y_arr[asset_idx]),
                }
                if ids_arr is not None and len(ids_arr) > asset_idx:
                    record["asset_id"] = str(ids_arr[asset_idx])
                if tickers_arr is not None and len(tickers_arr) > asset_idx:
                    record["ticker"] = str(tickers_arr[asset_idx])
                weights_recorder.append(record)

        timeline.append(date_str)

    if not snapshot_regrets:
        return {"split": name, "snapshots": 0}

    regrets_np = np.array(snapshot_regrets, dtype=np.float64)
    mse_np = np.array(snapshot_mse, dtype=np.float64)
    ew_np = np.array(ew_returns, dtype=np.float64)
    model_np = np.array(model_returns, dtype=np.float64)
    diff_np = model_np - ew_np

    summary = {
        "split": name,
        "snapshots": int(len(regrets_np)),
        "assets_total": int(total_assets),
        "avg_regret": float(regrets_np.mean()),
        "std_regret": float(regrets_np.std(ddof=1)) if len(regrets_np) > 1 else 0.0,
        "median_regret": float(np.median(regrets_np)),
        "p90_regret": float(np.quantile(regrets_np, 0.90)),
        "max_regret": float(regrets_np.max()),
        "avg_mse": float(mse_np.mean()),
        "std_mse": float(mse_np.std(ddof=1)) if len(mse_np) > 1 else 0.0,
        "ew_avg_return": float(ew_np.mean()),
        "ew_std_return": float(ew_np.std(ddof=1) if len(ew_np) > 1 else 0.0),
        "ew_total_return": _geometric_growth(ew_np),
        "model_avg_return": float(model_np.mean()),
        "model_std_return": float(model_np.std(ddof=1) if len(model_np) > 1 else 0.0),
        "model_total_return": _geometric_growth(model_np),
        "model_vs_ew_avg_diff": float(diff_np.mean()),
        "model_vs_ew_win_rate": float(np.mean(diff_np >= 0.0)),
    }

    if timeline:
        summary["timeline"] = {
            "dates": timeline,
            "ew_returns": ew_np.tolist(),
            "model_returns": model_np.tolist(),
        }

    return summary


def evaluate_forward_split(
    name: str,
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    lambda_: float,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    asset_ids_list: Optional[List[np.ndarray]],
    tickers_list: Optional[List[np.ndarray]],
    dates: Optional[np.ndarray],
    return_history,
    ticker_panel: Optional[pd.DataFrame],
    start_token: Optional[str],
    count: Optional[int],
    weights_recorder: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    if len(X_list) < 2 or len(y_list) < 2:
        return {"split": name, "snapshots": 0}

    start_idx = _resolve_start_index(dates, start_token)
    end_idx = len(X_list) - 1
    if count is not None and count > 0:
        end_idx = min(end_idx, start_idx + count - 1)
    if start_idx > end_idx:
        return {"split": name, "snapshots": 0}

    ew_returns: List[float] = []
    model_returns: List[float] = []
    timeline_dates: List[str] = []
    total_assets = 0

    for t in range(start_idx, end_idx + 1):
        X_prev = X_list[t - 1]
        y_curr = y_list[t]
        date_ts = pd.to_datetime(dates[t]) if dates is not None and t < len(dates) else None
        ids_prev = asset_ids_list[t - 1] if asset_ids_list is not None and t - 1 < len(asset_ids_list) else None
        ids_curr = asset_ids_list[t] if asset_ids_list is not None and t < len(asset_ids_list) else None
        ticks_prev = tickers_list[t - 1] if tickers_list is not None and t - 1 < len(tickers_list) else None
        ticks_curr = tickers_list[t] if tickers_list is not None and t < len(tickers_list) else None

        X_aligned, y_aligned, aligned_ids, aligned_tickers = _align_assets(
            X_prev,
            y_curr,
            ids_prev,
            ids_curr,
            ticks_prev,
            ticks_curr,
        )
        if y_aligned.size == 0:
            continue

        X_tensor = torch.tensor(X_aligned, dtype=torch.float32)
        y_tensor = torch.tensor(y_aligned, dtype=torch.float32)

        sigma_override = None
        if ticker_panel is not None and aligned_tickers is not None and len(aligned_tickers):
            panel_slice = ticker_panel
            if date_ts is not None and not pd.isna(date_ts) and isinstance(ticker_panel.index, pd.DatetimeIndex):
                panel_slice = ticker_panel.loc[:date_ts]
            cov_np = covariance_from_ticker_panel(aligned_tickers, panel_slice)
            if cov_np is not None:
                sigma_override = torch.from_numpy(cov_np.astype(np.float64))
        if sigma_override is None and aligned_ids:
            cov_np = covariance_from_history(aligned_ids, return_history)
            sigma_override = torch.from_numpy(cov_np.astype(np.float64))

        preds = model(X_tensor).detach()
        preds_cpu, L_cpu = prepare_layer_inputs(
            preds,
            X_batch=X_tensor if sigma_override is None else None,
            Sigma_override=sigma_override,
        )
        y_cpu = y_tensor.to(dtype=torch.double, device="cpu")
        layer = build_portfolio_layer(len(y_aligned), lambda_)
        with torch.no_grad():
            (weights,) = layer(preds_cpu, L_cpu)
            model_return = torch.dot(y_cpu, weights).item()

        ew_return = float(np.mean(y_aligned))
        ew_returns.append(ew_return)
        model_returns.append(float(model_return))
        total_assets += len(y_aligned)

        if date_ts is not None:
            date_str = date_ts.strftime("%Y-%m-%d")
        else:
            date_str = str(t)
        timeline_dates.append(date_str)

        if weights_recorder is not None:
            weights_np = weights.detach().cpu().numpy()
            preds_arr = preds_cpu.detach().cpu().numpy()
            y_arr = y_cpu.detach().cpu().numpy()
            ids_arr = aligned_ids if aligned_ids else None
            tickers_arr = aligned_tickers if aligned_tickers else None
            for asset_idx, weight_val in enumerate(weights_np):
                record = {
                    "split": name,
                    "date": date_str,
                    "asset_index": int(asset_idx),
                    "weight": float(weight_val),
                    "prediction": float(preds_arr[asset_idx]),
                    "return": float(y_arr[asset_idx]),
                }
                if ids_arr is not None and len(ids_arr) > asset_idx:
                    record["asset_id"] = str(ids_arr[asset_idx])
                if tickers_arr is not None and len(tickers_arr) > asset_idx:
                    record["ticker"] = str(tickers_arr[asset_idx])
                weights_recorder.append(record)

    if not ew_returns:
        return {"split": name, "snapshots": 0}

    ew_np = np.array(ew_returns, dtype=np.float64)
    model_np = np.array(model_returns, dtype=np.float64)
    diff_np = model_np - ew_np

    summary: Dict[str, float] = {
        "split": name,
        "snapshots": len(ew_returns),
        "assets_total": int(total_assets),
        "start_date": timeline_dates[0],
        "end_date": timeline_dates[-1],
        "ew_avg_return": float(ew_np.mean()),
        "ew_std_return": float(ew_np.std(ddof=1) if len(ew_np) > 1 else 0.0),
        "ew_total_return": _geometric_growth(ew_np),
        "model_avg_return": float(model_np.mean()),
        "model_std_return": float(model_np.std(ddof=1) if len(model_np) > 1 else 0.0),
        "model_total_return": _geometric_growth(model_np),
        "model_vs_ew_avg_diff": float(diff_np.mean()),
        "model_vs_ew_win_rate": float(np.mean(diff_np >= 0.0)),
    }

    summary["timeline"] = {
        "dates": timeline_dates,
        "ew_returns": ew_np.tolist(),
        "model_returns": model_np.tolist(),
    }

    return summary


def main() -> None:
    args = parse_args()

    train_path = DATA_DIR / args.train
    valid_path = DATA_DIR / args.valid
    test_path = DATA_DIR / args.test
    model_path = MODELS_DIR / args.model

    train_panel = load_panel(str(train_path))
    valid_panel = load_panel(str(valid_path)) if valid_path.exists() else None
    test_panel = load_panel(str(test_path)) if test_path.exists() else None

    X_train_list, y_train_list, asset_ids_train, tickers_train = clean_missing_xy(
        train_panel.X, train_panel.y, train_panel.asset_ids, train_panel.tickers, name="train"
    )
    X_valid_list: List[np.ndarray] = []
    y_valid_list: List[np.ndarray] = []
    asset_ids_valid: Optional[List[np.ndarray]] = None
    tickers_valid: Optional[List[np.ndarray]] = None
    if valid_panel is not None:
        X_valid_list, y_valid_list, asset_ids_valid, tickers_valid = clean_missing_xy(
            valid_panel.X, valid_panel.y, valid_panel.asset_ids, valid_panel.tickers, name="valid"
        )
    X_test_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []
    asset_ids_test: Optional[List[np.ndarray]] = None
    tickers_test: Optional[List[np.ndarray]] = None
    if test_panel is not None:
        X_test_list, y_test_list, asset_ids_test, tickers_test = clean_missing_xy(
            test_panel.X, test_panel.y, test_panel.asset_ids, test_panel.tickers, name="test"
        )

    ticker_panel_inputs: List[Tuple[np.ndarray, List[np.ndarray], Optional[List[np.ndarray]]]] = [
        (train_panel.dates, y_train_list, tickers_train)
    ]
    if valid_panel is not None:
        ticker_panel_inputs.append((valid_panel.dates, y_valid_list, tickers_valid))
    if test_panel is not None:
        ticker_panel_inputs.append((test_panel.dates, y_test_list, tickers_test))
    ticker_panel = build_ticker_return_panel(ticker_panel_inputs)

    feature_mask_names = load_feature_mask(FEATURE_MASK_PATH)
    feature_names_train = train_panel.feature_names
    if not isinstance(feature_names_train, np.ndarray):
        feature_names_train = np.array([f"f{i}" for i in range(X_train_list[0].shape[1])])

    X_train_list = apply_feature_mask(X_train_list, feature_names_train, feature_mask_names)
    if X_valid_list:
        feature_names_valid = valid_panel.feature_names if valid_panel is not None else feature_names_train
        X_valid_list = apply_feature_mask(X_valid_list, feature_names_valid, feature_mask_names)
    if X_test_list:
        feature_names_test = test_panel.feature_names if test_panel is not None else feature_names_train
        X_test_list = apply_feature_mask(X_test_list, feature_names_test, feature_mask_names)

    X_train_scaled, X_valid_scaled, X_test_scaled, _ = scale_features(
        X_train_list,
        X_valid_list,
        X_test_list,
    )

    model = FNN(input_dim=len(feature_mask_names))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    params = dict(model.named_parameters())

    lambda_ = max(args.risk, 1e-6)
    return_history = build_return_history(train_panel.asset_ids, train_panel.dates, train_panel.y)
    if valid_panel is not None:
        return_history = build_return_history(
            valid_panel.asset_ids,
            valid_panel.dates,
            valid_panel.y,
            history=return_history,
        )
    if test_panel is not None:
        return_history = build_return_history(
            test_panel.asset_ids,
            test_panel.dates,
            test_panel.y,
            history=return_history,
        )

    weights_records: Optional[List[Dict[str, Any]]] = [] if args.weights_out else None

    summaries: List[Dict[str, float]] = []
    summaries.append(
        evaluate_split(
            "train",
            model,
            params,
            lambda_,
            X_train_scaled,
            y_train_list,
            asset_ids_train,
            tickers_train,
            train_panel.dates,
            return_history,
            ticker_panel,
            weights_records,
        )
    )
    if X_valid_scaled:
        summaries.append(
            evaluate_split(
                "valid",
                model,
                params,
                lambda_,
                X_valid_scaled,
                y_valid_list,
                asset_ids_valid,
                tickers_valid,
                valid_panel.dates if valid_panel is not None else None,
                return_history,
                ticker_panel,
                weights_records,
            )
        )
    if X_test_scaled:
        summaries.append(
            evaluate_split(
                "test",
                model,
                params,
                lambda_,
                X_test_scaled,
                y_test_list,
                asset_ids_test,
                tickers_test,
                test_panel.dates if test_panel is not None else None,
                return_history,
                ticker_panel,
                weights_records,
            )
        )

    if args.forward_panel:
        forward_summary: Optional[Dict[str, float]] = None
        panel_name = args.forward_panel
        if panel_name == "train":
            forward_summary = evaluate_forward_split(
                "train_forward",
                model,
                params,
                lambda_,
                X_train_scaled,
                y_train_list,
                asset_ids_train,
                tickers_train,
                train_panel.dates,
                return_history,
                ticker_panel,
                args.forward_start,
                args.forward_count,
                weights_records,
            )
        elif panel_name == "valid" and X_valid_scaled:
            forward_summary = evaluate_forward_split(
                "valid_forward",
                model,
                params,
                lambda_,
                X_valid_scaled,
                y_valid_list,
                asset_ids_valid,
                tickers_valid,
                valid_panel.dates if valid_panel is not None else None,
                return_history,
                ticker_panel,
                args.forward_start,
                args.forward_count,
                weights_records,
            )
        elif panel_name == "test" and X_test_scaled:
            forward_summary = evaluate_forward_split(
                "test_forward",
                model,
                params,
                lambda_,
                X_test_scaled,
                y_test_list,
                asset_ids_test,
                tickers_test,
                test_panel.dates if test_panel is not None else None,
                return_history,
                ticker_panel,
                args.forward_start,
                args.forward_count,
                weights_records,
            )
        else:
            forward_summary = {"split": f"{panel_name}_forward", "snapshots": 0}

        if forward_summary is not None:
            summaries.append(forward_summary)

    for summary in summaries:
        split = summary.get("split", "unknown")
        print(f"[{split}] snapshots={summary.get('snapshots', 0)}, assets_total={summary.get('assets_total', 0)}")
        if "start_date" in summary or "end_date" in summary:
            print(f"  window: {summary.get('start_date', '?')} -> {summary.get('end_date', summary.get('start_date', '?'))}")
        metric_keys = (
            "avg_regret",
            "std_regret",
            "median_regret",
            "p90_regret",
            "max_regret",
            "avg_mse",
            "std_mse",
            "ew_avg_return",
            "ew_std_return",
            "ew_total_return",
            "model_avg_return",
            "model_std_return",
            "model_total_return",
            "model_vs_ew_avg_diff",
            "model_vs_ew_win_rate",
        )
        for key in metric_keys:
            if key in summary:
                print(f"  {key}: {summary[key]:.6f}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f_out:
            json.dump(summaries, f_out, indent=2)
        print(f"[info] Wrote evaluation summary to {out_path}")

    if args.weights_out and weights_records is not None:
        weights_path = Path(args.weights_out)
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with weights_path.open("w", encoding="utf-8") as f_weights:
            json.dump(weights_records, f_weights)
        print(f"[info] Wrote per-asset weights to {weights_path}")


if __name__ == "__main__":
    main()
