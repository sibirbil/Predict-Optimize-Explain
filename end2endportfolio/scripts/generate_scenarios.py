#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate monthly scenarios with MALA using portfolio-aware objectives."""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans

# ---------- project-relative paths ----------
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

from src.model import FNN  # noqa: E402
from src.data_utils import (  # noqa: E402
    build_return_history,
    clean_missing_xy,
    covariance_from_history,
    load_panel,
    scale_features,
)
from src.G_function import G_function_benchmark  # noqa: E402
from src.langevin import torch_MALA_chain  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCENARIOS_DIR = ARTIFACTS_DIR / "scenarios"
FEATURE_MASK_PATH = ARTIFACTS_DIR / "feature_mask.json"

DEFAULT_STEPS = 1000
DEFAULT_ETA = 0.7
DEFAULT_RISK_AVERSION = 100.0
DEFAULT_CLUSTERS = 4
DEFAULT_CENTROID_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario generation via MALA")
    parser.add_argument("--timestamp", type=int, default=0, help="Validation timestamp index to seed from")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of MALA steps")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA, help="Langevin step size")
    parser.add_argument(
        "--risk",
        type=float,
        default=DEFAULT_RISK_AVERSION,
        help="Risk aversion (lambda) for portfolio optimisation",
    )
    parser.add_argument("--benchmark", type=float, default=None, help="Target benchmark return (defaults to sample mean)")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS, help="Number of centroids for feature regularisation")
    parser.add_argument(
        "--centroid-weight",
        type=float,
        default=DEFAULT_CENTROID_WEIGHT,
        dest="centroid_weight",
        help="Weight of centroid proximity penalty",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature (beta) for the MALA sampler",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output filename")
    parser.add_argument("--model", type=str, default="trained_model.pth", help="Relative model checkpoint path")
    parser.add_argument("--train", type=str, default="panel_train.npz", help="Training NPZ filename")
    parser.add_argument("--valid", type=str, default="panel_valid.npz", help="Validation NPZ filename")
    parser.add_argument(
        "--feature-reg",
        type=float,
        default=0.0,
        help="L2 regularisation weight keeping generated features near the seed snapshot.",
    )
    return parser.parse_args()


def compute_centroids(train_features: list[np.ndarray], n_clusters: int) -> torch.Tensor | None:
    usable = [x for x in train_features if len(x)]
    if not usable or n_clusters <= 0:
        return None
    try:
        matrix = np.vstack(usable)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(matrix)
        return torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[warn] Centroid computation failed: {exc}")
        return None


def load_feature_mask() -> list[str] | None:
    if not FEATURE_MASK_PATH.exists():
        return None
    try:
        with FEATURE_MASK_PATH.open("r", encoding="utf-8") as f_mask:
            data = json.load(f_mask)
        kept = data.get("kept_features")
        if kept:
            return [str(name) for name in kept]
    except Exception as exc:
        print(f"[warn] Failed to read feature mask ({exc}); ignoring.")
    return None


def apply_feature_mask(
    X_list: list[np.ndarray],
    feature_names: np.ndarray,
    keep_names: list[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    indices = []
    for name in keep_names:
        if name not in name_to_idx:
            raise RuntimeError(f"Feature '{name}' from mask not found in panel columns.")
        indices.append(name_to_idx[name])
    indices = np.array(indices, dtype=int)
    X_masked = [x[:, indices] for x in X_list]
    return X_masked, feature_names[indices]


def summarize_chain(traj: torch.Tensor, feature_dim: int) -> None:
    traj_cpu = traj.detach().cpu()
    steps = traj_cpu.shape[0]
    if steps <= 1:
        print("[warn] Trajectory has <=1 step; skipping movement diagnostics.")
        return

    flat = traj_cpu.view(steps, -1)
    deltas = flat[1:] - flat[:-1]
    norms = torch.linalg.norm(deltas, dim=1)
    moved_ratio = (norms > 1e-6).float().mean().item()
    print(f"Step movement ratio: {moved_ratio:.3f} (fraction of proposals that changed state)")
    print(
        f"Step delta L2 norms -> min {norms.min().item():.4e}, median {norms.median().item():.4e}, max {norms.max().item():.4e}"
    )

    feature_traj = traj_cpu[:, :, -feature_dim:]
    feature_std = feature_traj.std(dim=0)
    mean_std = feature_std.mean().item()
    max_std = feature_std.max().item()
    print(
        f"Feature variation across chain -> mean per-feature std {mean_std:.4e}, max std {max_std:.4e}"
    )


def summarize_generation(final_state: torch.Tensor, real_features: np.ndarray, real_returns: np.ndarray) -> None:
    final_np = final_state.detach().cpu().numpy()
    if final_np.ndim != 2:
        return
    feature_dim = real_features.shape[1]
    generated_features = final_np[:, -feature_dim:]
    feature_shift = np.linalg.norm(generated_features.mean(axis=0) - real_features.mean(axis=0))
    print(f"Feature mean shift (L2): {feature_shift:.4f}")

    if final_np.shape[1] > feature_dim:
        generated_returns = final_np[:, 0]
        return_shift = float(np.mean(generated_returns) - np.mean(real_returns))
        return_vol_shift = float(np.std(generated_returns, ddof=1) - np.std(real_returns, ddof=1))
        print(f"Return mean shift: {return_shift:.4f}, stdev shift: {return_vol_shift:.4f}")


def main() -> None:
    args = parse_args()

    train_path = DATA_DIR / args.train
    valid_path = DATA_DIR / args.valid
    model_path = MODELS_DIR / args.model

    train_panel = load_panel(str(train_path))
    valid_panel = load_panel(str(valid_path))

    X_train_list, y_train_list, asset_ids_train, _tickers_train = clean_missing_xy(
        train_panel.X, train_panel.y, train_panel.asset_ids, train_panel.tickers, name="train"
    )
    X_valid_list, y_valid_list, asset_ids_valid, _tickers_valid = clean_missing_xy(
        valid_panel.X, valid_panel.y, valid_panel.asset_ids, valid_panel.tickers, name="valid"
    )

    feature_names = train_panel.feature_names
    if not isinstance(feature_names, np.ndarray) or feature_names.shape[0] != X_train_list[0].shape[1]:
        feature_names = np.array([f"f{i}" for i in range(X_train_list[0].shape[1])])
    feature_names_full = feature_names.copy()

    mask_names = load_feature_mask()
    if mask_names:
        X_train_list, feature_names = apply_feature_mask(X_train_list, feature_names_full, mask_names)
        if X_valid_list:
            X_valid_list, _ = apply_feature_mask(X_valid_list, feature_names_full, mask_names)
    else:
        mask_names = feature_names_full.tolist()
        feature_names = feature_names_full

    X_train_scaled, X_valid_scaled, _, _ = scale_features(X_train_list, X_valid_list, [])

    return_history = build_return_history(train_panel.asset_ids, train_panel.dates, train_panel.y)
    return_history = build_return_history(
        valid_panel.asset_ids,
        valid_panel.dates,
        valid_panel.y,
        history=return_history,
    )
    centroids_tensor = compute_centroids(X_train_scaled, args.clusters)

    model = FNN(input_dim=len(feature_names))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    trained_params = dict(model.named_parameters())

    if not X_valid_scaled:
        raise RuntimeError("Validation panel is empty after cleaning; cannot seed scenarios.")
    if args.timestamp < 0 or args.timestamp >= len(X_valid_scaled):
        raise IndexError(
            f"timestamp {args.timestamp} out of range for validation panel (0-{len(X_valid_scaled) - 1})"
        )
    timestamp_idx = int(args.timestamp)
    x0_np = X_valid_scaled[timestamp_idx]
    y0_np = y_valid_list[timestamp_idx]
    asset_ids_snapshot = None
    if asset_ids_valid is not None and len(asset_ids_valid) > timestamp_idx:
        asset_ids_snapshot = asset_ids_valid[timestamp_idx]

    Sigma_np = (
        covariance_from_history(asset_ids_snapshot, return_history)
        if asset_ids_snapshot is not None
        else np.eye(x0_np.shape[0], dtype=float)
    )
    Sigma = torch.from_numpy(Sigma_np.astype(np.float64))

    x0 = torch.tensor(x0_np, dtype=torch.float32)
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    x0_full = torch.cat([y0.view(-1, 1), x0], dim=1)

    benchmark_return = float(args.benchmark) if args.benchmark is not None else float(y0_np.mean())
    lambda_ = max(args.risk, 1e-6)
    centroid_weight = max(args.centroid_weight, 0.0)
    beta = max(args.beta, 1e-6)

    print(f"Timestamp idx={timestamp_idx}, assets={x0_np.shape[0]}")
    print(f"Benchmark return={benchmark_return:.4f}, lambda={lambda_:.3f}, beta={beta:.3f}")
    print(f"Covariance condition number={np.linalg.cond(Sigma_np):.3f}")
    if centroids_tensor is not None and centroid_weight > 0:
        print(f"Centroid regularisation: k={args.clusters}, weight={centroid_weight}")

    G, gradG, portfolio_eval = G_function_benchmark(
        model,
        trained_params,
        x0,
        benchmark_return,
        Sigma,
        lambda_,
        centroids=centroids_tensor,
        centroid_weight=centroid_weight,
        feature_anchor=x0,
        feature_reg=max(args.feature_reg, 0.0),
    )

    eta = max(args.eta, 1e-4)
    steps = max(args.steps, 1)
    hypsG = (G, gradG, eta, beta)

    x0_full.requires_grad_(True)
    x_final, x_traj = torch_MALA_chain(x0_full, hypsG, NSteps=steps)
    summarize_chain(x_traj, len(feature_names))

    summarize_generation(x_final, x0_np, y0_np)
    print("Portfolio eval on final sample:", portfolio_eval(x_final.detach()))

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        custom_path = Path(args.output)
        output_path = custom_path if custom_path.is_absolute() else SCENARIOS_DIR / custom_path
    else:
        output_name = f"generated_scenarios_ts{timestamp_idx:03d}_{x0_np.shape[0]}_assets.pt"
        output_path = SCENARIOS_DIR / output_name

    payload = {
        "trajectory": x_traj,
        "final_state": x_final,
        "timestamp_idx": timestamp_idx,
        "asset_ids": asset_ids_snapshot.tolist() if asset_ids_snapshot is not None else None,
        "benchmark_return": benchmark_return,
        "lambda": lambda_,
        "eta": eta,
        "beta": beta,
        "steps": steps,
        "centroid_weight": centroid_weight,
        "clusters": args.clusters,
        "feature_reg": max(args.feature_reg, 0.0),
        "covariance_source": "history" if asset_ids_snapshot is not None else "identity",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Scenario generation complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
