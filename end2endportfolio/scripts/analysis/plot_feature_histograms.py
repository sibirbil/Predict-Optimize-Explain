#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Produce feature/return histogram overlays for generated vs. real panels."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_panel  # noqa: E402

FILL = -99.99


def _to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, (list, tuple)):
        try:
            return torch.tensor(obj)
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def load_generated(path: Path, prefer_key: str = None) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        scenario = None
        if isinstance(obj, dict):
            candidate_keys = [
                prefer_key,
                "trajectory",
                "scenarios",
                "returns",
                "data",
                "X",
                "matrix",
                "scenario_matrix",
                "final_state",
            ]
            for key in candidate_keys:
                if not key:
                    continue
                if key in obj:
                    scenario = _to_tensor(obj[key])
                    if scenario is not None:
                        break
        else:
            scenario = _to_tensor(obj)
        if scenario is None:
            raise RuntimeError(f"Could not locate scenario tensor inside {path}")
        return scenario.float()
    raise RuntimeError(f"Unsupported scenario file format: {path}")

# -----------------------------------
# Load generated trajectory
# -----------------------------------
def main():
    default_scenario = PROJECT_ROOT / "artifacts" / "scenarios" / "generated_scenarios_full_66_assets.pt"
    default_real = PROJECT_ROOT / "data" / "panel_train.npz"
    default_out = PROJECT_ROOT / "reports" / "figures" / "feature_hist_overlays.pdf"

    parser = argparse.ArgumentParser(description="Histogram overlays for generated vs real panels")
    parser.add_argument("--scenario", default=str(default_scenario), help="Path to generated scenarios (.pt)")
    parser.add_argument("--real", default=str(default_real), help="Real panel NPZ path")
    parser.add_argument("--out-pdf", default=str(default_out), help="Destination PDF path")
    parser.add_argument("--gen-key", default="trajectory", help="Key inside scenario file containing the tensor")
    parser.add_argument(
        "--feature-mask",
        default=str(PROJECT_ROOT / "artifacts" / "feature_mask.json"),
        help="JSON file listing kept features (optional)",
    )
    parser.add_argument("--burn-frac", type=float, default=0.5, help="Burn-in fraction (default 0.5)")
    parser.add_argument("--thin", type=int, default=5, help="Keep every Nth sample (default 5)")
    parser.add_argument("--scenario-step", type=int, default=None, help="Specific scenario step index after burn/thin")
    parser.add_argument("--real-index", type=int, default=None, help="Specific timestamp index in the real panel")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    parser.add_argument("--rows", type=int, default=3, help="Rows per PDF page")
    parser.add_argument("--cols", type=int, default=4, help="Cols per PDF page")
    args = parser.parse_args()

    scenario_tensor = load_generated(Path(args.scenario), prefer_key=args.gen_key)
    traj_np = scenario_tensor.detach().cpu().numpy()
    if traj_np.ndim != 3 or traj_np.shape[-1] < 2:
        raise RuntimeError(f"Expected scenario tensor of shape (steps, assets, 1+features); got {traj_np.shape}")

    NSteps, A, Dp1 = traj_np.shape
    D = Dp1 - 1
    start = int(math.floor(args.burn_frac * NSteps))
    idx = np.arange(start, NSteps, max(args.thin, 1))

    Y_gen = traj_np[idx, :, 0]
    X_gen = traj_np[idx, :, 1:]
    if args.scenario_step is not None:
        if args.scenario_step < 0 or args.scenario_step >= Y_gen.shape[0]:
            raise IndexError(f"scenario-step {args.scenario_step} out of range (0-{Y_gen.shape[0]-1})")
        Y_gen = Y_gen[args.scenario_step : args.scenario_step + 1]
        X_gen = X_gen[args.scenario_step : args.scenario_step + 1]
    y_gen_flat = Y_gen.ravel()
    X_gen_flat = X_gen.reshape(-1, D)

    panel = load_panel(str(args.real))
    X_real_3d = panel.X
    y_real_2d = panel.y
    feat_names = panel.feature_names if isinstance(panel.feature_names, np.ndarray) else np.array([f"f{i}" for i in range(panel.X.shape[-1])])

    kept_features = None
    mask_path = Path(args.feature_mask)
    if mask_path.exists():
        try:
            data = json.loads(mask_path.read_text())
            kept = data.get("kept_features")
            if kept:
                kept_features = [str(f) for f in kept]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to parse feature mask {mask_path}: {exc}")

    if kept_features is not None and feat_names.size:
        indices = [i for i, name in enumerate(feat_names) if name in kept_features]
        if len(indices) == len(kept_features):
            feat_names = feat_names[indices]
            X_real_3d = X_real_3d[..., indices]
        else:
            print("[warn] Feature mask names not fully present in real panel; ignoring mask")
            kept_features = None

    if feat_names.shape[0] != D:
        keep_dim = min(feat_names.shape[0], D)
        print(f"[warn] Feature mismatch: scenario {D} vs real {feat_names.shape[0]} -> truncating to {keep_dim}")
        X_gen_flat = X_gen_flat[:, :keep_dim]
        D = keep_dim
        feat_names = feat_names[:D]
        X_real_3d = X_real_3d[..., :D]

    if args.real_index is not None:
        if args.real_index < 0 or args.real_index >= X_real_3d.shape[0]:
            raise IndexError(f"real-index {args.real_index} out of range (0-{X_real_3d.shape[0]-1})")
        X_real_3d = X_real_3d[args.real_index : args.real_index + 1]
        y_real_2d = y_real_2d[args.real_index : args.real_index + 1]

    Xr = X_real_3d.copy()
    Xr[Xr == FILL] = np.nan
    X_real_flat = Xr.reshape(-1, D)
    row_ok_X = np.isfinite(X_real_flat).all(axis=1)
    X_real_flat = X_real_flat[row_ok_X]

    yr = y_real_2d.copy().reshape(-1)
    row_ok_y = np.isfinite(yr)
    y_real_flat = yr[row_ok_y]

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    per_page = args.rows * args.cols
    n_pages = int(math.ceil(D / per_page))

    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        y_real_clean = y_real_flat[np.isfinite(y_real_flat)]
        y_gen_clean = y_gen_flat[np.isfinite(y_gen_flat)]
        if y_real_clean.size or y_gen_clean.size:
            data_all = np.concatenate([y_real_clean, y_gen_clean]) if y_real_clean.size and y_gen_clean.size else y_real_clean if y_real_clean.size else y_gen_clean
            bins = np.histogram_bin_edges(data_all, bins=args.bins)
            ax.hist(y_real_clean, bins=bins, density=False, alpha=0.55, label="real y", linewidth=0)
            ax.hist(y_gen_clean, bins=bins, density=False, alpha=0.55, label="generated y", linewidth=0)
            ax.set_title("Return distribution (pooled)")
            ax.set_xlabel("y")
            ax.set_ylabel("count")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No finite return data to plot", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for page in range(n_pages):
            lo = page * per_page
            hi = min((page + 1) * per_page, D)
            if lo >= hi:
                break
            fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 4.2, args.rows * 3.2))
            axes = np.asarray(axes).reshape(-1)
            last_ax_idx = -1
            for ax_i, j in enumerate(range(lo, hi)):
                ax = axes[ax_i]
                real_j = X_real_flat[:, j]
                gen_j = X_gen_flat[:, j]
                real_j = real_j[np.isfinite(real_j)]
                gen_j = gen_j[np.isfinite(gen_j)]
                data_all = np.concatenate([real_j, gen_j]) if real_j.size and gen_j.size else real_j if real_j.size else gen_j
                if data_all.size == 0:
                    ax.set_title(f"{feat_names[j]} (no data)")
                    ax.axis("off")
                    continue
                bins = np.histogram_bin_edges(data_all, bins=args.bins)
                ax.hist(real_j, bins=bins, density=False, alpha=0.55, label="real", linewidth=0)
                ax.hist(gen_j, bins=bins, density=False, alpha=0.55, label="generated", linewidth=0)
                ax.set_title(str(feat_names[j]))
                ax.tick_params(labelsize=8)
                last_ax_idx = ax_i
            if last_ax_idx >= 0:
                handles, labels = axes[last_ax_idx].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
            for k in range(last_ax_idx + 1, len(axes)):
                axes[k].axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    main()
