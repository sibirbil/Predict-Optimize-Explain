"""Enhanced E2E diversification story plots.

These figures are meant for choosing and explaining a more visual E2E
diversification story:

- anchor-to-final arrows in PCA/t-SNE macro geography
- post-burn-in centroid and covariance ellipse
- nearest historical analog labels
- smoothed all-macro traces, so movement is readable rather than noisy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.macro_scaler import MacroScaler  # noqa: E402
from src.modules.runtime_regime import MACRO_ORDER  # noqa: E402


DEFAULT_RUN = ROOT / "scenario_outputs" / "scenario_e2e_diversify_202004" / "runs" / "20260509_113131_213629"
DEFAULT_OUT = ROOT / "submission_plots" / "story_figures_v3" / "e2e_diversification_202004" / "enhanced_story"
MACRO_COLS = list(MACRO_ORDER)
REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}
SEED_COLORS = ["#1f6f78", "#5a9f4f", "#b6554d", "#8264a8"]


def newest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file in {run_dir} matching {pattern}")
    return matches[-1]


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return pdf, png


def standardize_raw(raw: np.ndarray, macro_cols: list[str]) -> np.ndarray:
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    mean = scaler.mean.detach().cpu().numpy().astype(np.float64)
    std = scaler.std.detach().cpu().numpy().astype(np.float64)
    order = list(MACRO_ORDER)
    idx = [order.index(col) for col in macro_cols]
    return (raw - mean[idx]) / std[idx]


def anchor_std(historical: pd.DataFrame, anchor: int = 202004) -> np.ndarray:
    row = historical[historical["yyyymm"].astype(int).eq(anchor)]
    if row.empty:
        raise ValueError(f"Anchor {anchor} not found in historical macro panel.")
    return standardize_raw(row[MACRO_COLS].iloc[0].to_numpy(dtype=float), MACRO_COLS)


def load_run(run_dir: Path) -> dict[str, object]:
    final = pd.read_csv(newest(run_dir, "final_state_diagnostics_*.csv"))
    sample = pd.read_csv(newest(run_dir, "generated_macro_sample_*.csv"))
    historical = pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv"))
    standardized_payload = torch.load(newest(run_dir, "trajectories_standardized_3d_*.pt"), map_location="cpu")
    raw_path = newest(run_dir, "trajectories_unstandardized_3d_*.pt") if list(run_dir.glob("trajectories_unstandardized_3d_*.pt")) else newest(run_dir, "trajectories_raw_3d_*.pt")
    raw_payload = torch.load(raw_path, map_location="cpu")
    return {
        "final": final,
        "sample": sample,
        "historical": historical,
        "standardized_tensor": standardized_payload["tensor"].detach().cpu().to(dtype=torch.float32).numpy(),
        "unstandardized_tensor": raw_payload["tensor"].detach().cpu().to(dtype=torch.float32).numpy(),
    }


def covariance_ellipse(points: np.ndarray, n_std: float = 1.0) -> tuple[np.ndarray, float, float, float]:
    center = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    return center, float(width), float(height), angle


def nearest_analogs(
    historical: pd.DataFrame,
    query_std: np.ndarray,
    anchor: int = 202004,
    k: int = 5,
) -> pd.DataFrame:
    hist_std = standardize_raw(historical[MACRO_COLS].to_numpy(dtype=float), MACRO_COLS)
    dates = historical["yyyymm"].astype(int).to_numpy()
    mask = dates != int(anchor)
    d = np.sqrt(((hist_std[mask] - query_std.reshape(1, -1)) ** 2).sum(axis=1))
    order = np.argsort(d)[:k]
    out = historical.loc[mask].iloc[order][["yyyymm"]].copy()
    out["std_euclidean_distance"] = d[order]
    return out


def historical_tsne_with_nn_overlay(
    hist_x: np.ndarray,
    overlays: dict[str, np.ndarray],
    *,
    random_state: int = 41,
    perplexity: int = 22,
    n_iter: int = 750,
    k: int = 8,
) -> dict[str, np.ndarray]:
    """Fit t-SNE on historical data only and overlay scenario points afterward.

    sklearn t-SNE has no out-of-sample transform. Overlays are therefore placed
    as inverse-distance weighted averages of nearest historical neighbors in
    standardized macro space, preventing generated points from reshaping the map.
    """
    hist_x = np.asarray(hist_x, dtype=float)
    n_hist = len(hist_x)
    safe_perplexity = min(perplexity, max(2, (n_hist - 1) // 3))
    hist_emb = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=n_iter,
        random_state=random_state,
        method="exact",
    ).fit_transform(hist_x)
    out = {"hist": hist_emb}
    kk = min(k, n_hist)
    for name, arr in overlays.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            out[name] = np.empty((0, 2))
            continue
        dist = np.linalg.norm(arr[:, None, :] - hist_x[None, :, :], axis=2)
        idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        near_dist = np.take_along_axis(dist, idx, axis=1)
        weights = 1.0 / np.maximum(near_dist, 1e-8)
        coords = (hist_emb[idx] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]
        exact = near_dist.min(axis=1) < 1e-10
        if exact.any():
            nearest = idx[np.arange(len(arr)), np.argmin(near_dist, axis=1)]
            coords[exact] = hist_emb[nearest[exact]]
        out[name] = coords
    return out


def plot_macro_geography_mechanism(run: dict[str, object], out_dir: Path, anchor: int = 202004) -> tuple[Path, Path]:
    final = run["final"]
    sample = run["sample"]
    historical = run["historical"]
    hist_std = standardize_raw(historical[MACRO_COLS].to_numpy(dtype=float), MACRO_COLS)
    sample_std = sample[[f"{c}_std" for c in MACRO_COLS]].to_numpy(dtype=float)
    final_std = final[[f"{c}_std" for c in MACRO_COLS]].to_numpy(dtype=float)
    anchor_vec = anchor_std(historical, anchor)
    centroid_std = np.median(sample_std, axis=0)
    analogs = nearest_analogs(historical, centroid_std, anchor=anchor, k=5)
    analog_std = standardize_raw(
        historical[historical["yyyymm"].astype(int).isin(analogs["yyyymm"].astype(int))][MACRO_COLS].to_numpy(dtype=float),
        MACRO_COLS,
    )

    rng = np.random.default_rng(11)
    sample_idx = rng.choice(len(sample_std), size=min(1600, len(sample_std)), replace=False)
    pca = PCA(n_components=2, random_state=0).fit(hist_std)
    pca_parts = {
        "hist": pca.transform(hist_std),
        "sample": pca.transform(sample_std[sample_idx]),
        "final": pca.transform(final_std),
        "anchor": pca.transform(anchor_vec.reshape(1, -1)),
        "analogs": pca.transform(analog_std),
    }
    tsne_parts = historical_tsne_with_nn_overlay(
        hist_std,
        {
            "sample": sample_std[sample_idx],
            "final": final_std,
            "anchor": anchor_vec,
            "analogs": analog_std,
        },
        random_state=41,
        perplexity=22,
        n_iter=750,
        k=8,
    )
    n_hist = len(hist_std)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, parts, title in zip(axes, [pca_parts, tsne_parts], ["PCA macro geography", "t-SNE neighborhood check"]):
        for regime, color in REGIME_COLORS.items():
            mask = historical["regime"].astype(str).eq(regime).to_numpy() if "regime" in historical.columns else np.zeros(n_hist, dtype=bool)
            if mask.any():
                ax.scatter(parts["hist"][mask, 0], parts["hist"][mask, 1], s=15, color=color, alpha=0.16, linewidth=0, label=regime)
        ax.scatter(parts["sample"][:, 0], parts["sample"][:, 1], s=14, color="#6f6f6f", alpha=0.16, linewidth=0, label="post-burn-in sample")
        center, width, height, angle = covariance_ellipse(parts["sample"], n_std=1.15)
        ax.add_patch(Ellipse(center, width, height, angle=angle, facecolor="#6f6f6f", edgecolor="#222222", alpha=0.12, lw=1.2))
        ax.scatter(center[0], center[1], marker="D", s=88, color="#111111", edgecolor="white", linewidth=1.0, label="sample centroid")
        ax.scatter(parts["anchor"][:, 0], parts["anchor"][:, 1], marker="X", s=150, color="#d62728", edgecolor="white", linewidth=1.2, label=f"{anchor} anchor")
        final_regimes = final["regime"].astype(str).to_list()
        for i, xy in enumerate(parts["final"]):
            color = REGIME_COLORS.get(final_regimes[i], "#333333")
            ax.annotate("", xy=xy, xytext=parts["anchor"][0], arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color, "alpha": 0.74})
            ax.scatter(xy[0], xy[1], s=94, color=color, edgecolor="white", linewidth=1.2, zorder=4)
            ax.text(xy[0], xy[1], str(i + 1), fontsize=8.2, ha="center", va="center", color="white", fontweight="bold", zorder=5)
        for j, row in enumerate(analogs.itertuples(index=False)):
            xy = parts["analogs"][j]
            ax.scatter(xy[0], xy[1], marker="s", s=68, color="#f2c94c", edgecolor="#333333", linewidth=0.8, zorder=4)
            ax.text(xy[0], xy[1], str(int(row.yyyymm)), fontsize=7.4, ha="left", va="bottom")
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.grid(alpha=0.18)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    keep = []
    seen = set()
    for h, lab in zip(handles, labels):
        if lab not in seen:
            keep.append((h, lab))
            seen.add(lab)
    fig.legend([x[0] for x in keep], [x[1] for x in keep], loc="lower center", ncol=5, frameon=False, fontsize=8.3)
    fig.suptitle(f"E2E diversification: movement around the {anchor} macro anchor", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.06, "Arrows connect the anchor to final seeds; gray ellipse summarizes the post-burn-in generated cloud; yellow squares are nearest historical analogs.", ha="center", fontsize=9.0)
    fig.text(0.5, 0.035, "t-SNE is fit on historical macro states only; scenario points are overlaid by nearest-neighbor placement.", ha="center", fontsize=8.0, color="#5f6368")
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    analogs.to_csv(out_dir / "e2e_enhanced_nearest_analogs.csv", index=False)
    return savefig(fig, out_dir, "e2e_macro_geography_mechanism_enhanced")


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def plot_smoothed_traces(
    tensor: np.ndarray,
    out_dir: Path,
    stem: str,
    title: str,
    unit_label: str,
    plot_step: int = 10,
    smooth_window: int = 500,
    burn_in_step: int = 10000,
) -> tuple[Path, Path]:
    steps = np.arange(1, tensor.shape[1] + 1, plot_step)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.2), sharex=True)
    axes = axes.ravel()
    for macro_idx, (ax, macro) in enumerate(zip(axes, MACRO_COLS)):
        smoothed = []
        for seed_idx in range(tensor.shape[0]):
            vals = rolling_mean(tensor[seed_idx, :, macro_idx], smooth_window)
            vals = vals[::plot_step]
            smoothed.append(vals)
            ax.plot(steps, vals, color=SEED_COLORS[seed_idx % len(SEED_COLORS)], alpha=0.58, lw=0.9, label=f"seed {seed_idx + 1}" if macro_idx == 0 else None)
        median = np.median(np.vstack(smoothed), axis=0)
        ax.plot(steps, median, color="#111111", lw=1.8, alpha=0.92, label="seed median" if macro_idx == 0 else None)
        ax.axvline(burn_in_step, color="#888888", linestyle=":", lw=0.8)
        ax.set_title(macro, loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=8)
    for ax in axes[6:]:
        ax.set_xlabel("MALA step", fontsize=9)
    for ax in axes[::3]:
        ax.set_ylabel(unit_label, fontsize=9)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, f"Rolling mean window: {smooth_window} steps. Lines are drawn every {plot_step} steps; underlying tensor contains all iterations.", ha="center", fontsize=8.8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return savefig(fig, out_dir, stem)


def write_manifest(out_dir: Path, records: list[dict[str, str]]) -> None:
    with (out_dir / "enhanced_story_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# E2E Enhanced Story Figures",
        "",
        "Purpose: make the E2E diversification scenario visually storyable without forcing the sampler into Scenario-4-style regime escape.",
        "",
        "- `e2e_macro_geography_mechanism_enhanced`: anchor-to-final arrows, post-burn-in centroid/ellipse, historical regime clouds, and nearest analogs.",
        "- `e2e_smoothed_standardized_traces`: readable rolling traces in sampler z-score coordinates.",
        "- `e2e_smoothed_unstandardized_traces`: same rolling traces in economic macro units.",
        "",
        "Interpretation: E2E moves actively across a broad stress-boundary neighborhood; its story is local exploration with concentration reduction, not one-way regime escape.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create enhanced E2E visual story figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-date", type=int, default=202004)
    parser.add_argument("--smooth-window", type=int, default=500)
    parser.add_argument("--plot-step", type=int, default=10)
    args = parser.parse_args()

    args.run_dir = args.run_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(args.run_dir)
    records: list[dict[str, str]] = []
    for stem, paths, note in [
        (
            "e2e_macro_geography_mechanism_enhanced",
            plot_macro_geography_mechanism(run, args.out_dir, anchor=args.anchor_date),
            "PCA/t-SNE geography with anchor-to-final arrows, generated cloud ellipse, centroid, and nearest analogs.",
        ),
        (
            "e2e_smoothed_standardized_traces",
            plot_smoothed_traces(
                run["standardized_tensor"],
                args.out_dir,
                "e2e_smoothed_standardized_traces",
                "E2E diversification: smoothed all-macro traces in standardized space",
                "z-score",
                plot_step=args.plot_step,
                smooth_window=args.smooth_window,
            ),
            "Rolling trace view in sampler coordinates.",
        ),
        (
            "e2e_smoothed_unstandardized_traces",
            plot_smoothed_traces(
                run["unstandardized_tensor"],
                args.out_dir,
                "e2e_smoothed_unstandardized_traces",
                "E2E diversification: smoothed all-macro traces in economic macro units",
                "raw units",
                plot_step=args.plot_step,
                smooth_window=args.smooth_window,
            ),
            "Rolling trace view in original macro units.",
        ),
    ]:
        records.append(
            {
                "figure": stem,
                "pdf": str(paths[0].relative_to(ROOT)),
                "png": str(paths[1].relative_to(ROOT)),
                "note": note,
            }
        )
    write_manifest(args.out_dir, records)
    print(f"Wrote {len(records)} enhanced E2E story figures -> {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
