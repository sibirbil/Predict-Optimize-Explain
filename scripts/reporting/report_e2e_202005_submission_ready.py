"""Submission-ready E2E diversification figures for the 202005 long run.

The figures here are deliberately narrower than the v3 exploratory set:
they prioritize manuscript readability and economic interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.report_story_figures_v2 import (  # noqa: E402
    MACRO_COLS,
    REGIME_COLORS,
    anchor_std,
    historical_std_frame,
    load_regime_nfci_panel,
    newest,
)
from scripts.reporting.report_story_figures_v3 import historical_tsne_with_nn_overlay  # noqa: E402
from src.utils.plotting import set_publication_style  # noqa: E402


DEFAULT_RUN = ROOT / "scenario_outputs" / "scenario_e2e_diversify_202005" / "runs" / "20260513_145948_075567"
DEFAULT_OUT = ROOT / "submission_plots" / "story_figures_v3" / "e2e_diversification_202005" / "submission_ready"

ANCHOR_COLOR = "#c44e52"
SCENARIO_COLOR = "#157f78"
SAMPLE_COLOR = "#6f7782"
ANALOG_COLOR = "#f2c94c"
GRID_COLOR = "#d8dde3"
TEXT_COLOR = "#1f2328"


def apply_submission_style() -> None:
    set_publication_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.titleweight": "bold",
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return pdf, png


def load_run(run_dir: Path) -> dict[str, object]:
    return {
        "final": pd.read_csv(newest(run_dir, "final_state_diagnostics_*.csv")),
        "sample": pd.read_csv(newest(run_dir, "generated_macro_sample_*.csv")),
        "historical": pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv")),
        "anchor_weights": pd.read_csv(newest(run_dir, "anchor_weights_*.csv")),
        "final_weights": pd.read_csv(newest(run_dir, "final_weights_*.csv")),
        "seed": pd.read_csv(newest(run_dir, "seed_summary_*.csv")),
        "config": json.loads(newest(run_dir, "config_*.json").read_text(encoding="utf-8")),
        "tensor": torch.load(newest(run_dir, "trajectories_postburnin_standardized_3d_*.pt"), map_location="cpu"),
    }


def covariance_ellipse(points: np.ndarray, n_std: float = 1.45) -> tuple[np.ndarray, float, float, float]:
    center = np.nanmean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    return center, float(width), float(height), angle


def add_clean_axis(ax: plt.Axes) -> None:
    ax.grid(color=GRID_COLOR, linewidth=0.55, alpha=0.62)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)


def nearest_analogs(historical: pd.DataFrame, query_std: np.ndarray, anchor: int, k: int = 5) -> pd.DataFrame:
    hist_std = historical_std_frame(historical)
    cols = [f"{c}_std" for c in MACRO_COLS]
    x = hist_std[cols].to_numpy(dtype=float)
    dates = hist_std["yyyymm"].astype(int).to_numpy()
    keep = dates != int(anchor)
    dist = np.linalg.norm(x[keep] - query_std.reshape(1, -1), axis=1)
    order = np.argsort(dist)[:k]
    out = hist_std.loc[keep].iloc[order][["yyyymm", "regime"]].copy()
    out["std_euclidean_distance"] = dist[order]
    return out


def regime_ellipses(ax: plt.Axes, coords: np.ndarray, regimes: pd.Series) -> None:
    for regime in ["expansion", "contraction", "financial_stress"]:
        pts = coords[regimes.astype(str).eq(regime).to_numpy()]
        if len(pts) < 8:
            continue
        center, width, height, angle = covariance_ellipse(pts, n_std=1.15)
        ax.add_patch(
            Ellipse(
                center,
                width,
                height,
                angle=angle,
                facecolor=REGIME_COLORS[regime],
                edgecolor=REGIME_COLORS[regime],
                lw=1.25,
                alpha=0.095,
                zorder=1,
            )
        )
        ax.text(
            center[0],
            center[1],
            regime.replace("_", " "),
            color=REGIME_COLORS[regime],
            fontsize=8.0,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.17", "facecolor": "white", "edgecolor": REGIME_COLORS[regime], "alpha": 0.86},
            zorder=6,
        )


def plot_main_pca(run: dict[str, object], out_dir: Path, anchor: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = run["final"]
    sample = run["sample"]
    historical = run["historical"]
    hist_std = historical_std_frame(historical)
    cols = [f"{c}_std" for c in MACRO_COLS]
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_x = sample[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = anchor_std(historical, anchor)[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(sample_x), size=min(900, len(sample_x)), replace=False)
    pca = PCA(n_components=2, random_state=0).fit(hist_x)
    hist_p = pca.transform(hist_x)
    sample_p = pca.transform(sample_x[sample_idx])
    final_p = pca.transform(final_x)
    anchor_p = pca.transform(anchor_x.reshape(1, -1))[0]
    centroid_x = np.median(sample_x, axis=0)
    centroid_p = pca.transform(centroid_x.reshape(1, -1))[0]
    analogs = nearest_analogs(historical, centroid_x, anchor=anchor, k=5)
    analog_x = hist_std[hist_std["yyyymm"].astype(int).isin(analogs["yyyymm"].astype(int))][cols].to_numpy(dtype=float)
    analog_p = pca.transform(analog_x)

    apply_submission_style()
    fig, ax = plt.subplots(figsize=(7.15, 4.45))
    for regime in ["expansion", "contraction", "financial_stress"]:
        idx = hist_std["regime"].astype(str).eq(regime).to_numpy()
        ax.scatter(hist_p[idx, 0], hist_p[idx, 1], s=10, color=REGIME_COLORS[regime], alpha=0.16, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(sample_p[:, 0], sample_p[:, 1], s=9, color=SAMPLE_COLOR, alpha=0.17, linewidths=0, rasterized=True, zorder=2)
    center, width, height, angle = covariance_ellipse(sample_p, n_std=1.05)
    ax.add_patch(Ellipse(center, width, height, angle=angle, facecolor=SCENARIO_COLOR, edgecolor=SCENARIO_COLOR, alpha=0.10, lw=1.25, zorder=3))
    ax.scatter(centroid_p[0], centroid_p[1], marker="D", s=82, color="#111111", edgecolors="white", linewidth=0.9, zorder=6)
    ax.scatter(anchor_p[0], anchor_p[1], marker="X", s=150, color=ANCHOR_COLOR, edgecolors="white", linewidth=1.0, zorder=8)
    colors = final["regime"].map(REGIME_COLORS).fillna(SCENARIO_COLOR).to_list()
    for idx, xy in enumerate(final_p):
        ax.annotate("", xy=xy, xytext=anchor_p, arrowprops={"arrowstyle": "->", "lw": 1.05, "color": colors[idx], "alpha": 0.80}, zorder=5)
        ax.scatter(xy[0], xy[1], s=76, color=colors[idx], edgecolors="white", linewidth=0.95, zorder=7)
        ax.text(xy[0], xy[1], str(int(final.iloc[idx]["seed"])), color="white", fontsize=7.5, weight="bold", ha="center", va="center", zorder=8)
    for xy, row in zip(analog_p, analogs.itertuples(index=False)):
        ax.scatter(xy[0], xy[1], marker="s", s=58, color=ANALOG_COLOR, edgecolors=TEXT_COLOR, linewidth=0.7, zorder=7)
        ax.text(xy[0] + 0.05, xy[1] + 0.05, str(int(row.yyyymm)), fontsize=7.0, color=TEXT_COLOR, zorder=8)
    local = np.vstack([sample_p, final_p, anchor_p.reshape(1, -1), analog_p])
    lo = np.nanquantile(local, [0.01, 0.99], axis=0)
    span = np.maximum(lo[1] - lo[0], 0.8)
    ax.set_xlim(lo[0, 0] - 0.25 * span[0], lo[1, 0] + 0.25 * span[0])
    ax.set_ylim(lo[0, 1] - 0.30 * span[1], lo[1, 1] + 0.30 * span[1])
    ax.set_title("May 2020 E2E diversification: local macro geography", loc="left")
    ax.set_xlabel(f"PC1 from historical macro panel ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    add_clean_axis(ax)
    note = "PCA is fit on history; the panel is zoomed to the generated neighborhood."
    ax.text(0.015, 0.02, note, transform=ax.transAxes, fontsize=7.0, va="bottom", ha="left", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["expansion"], markersize=4.5, label="Historical expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["contraction"], markersize=4.5, label="Historical contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["financial_stress"], markersize=4.5, label="Historical stress"),
        mlines.Line2D([], [], linestyle="", marker="o", color=SAMPLE_COLOR, markersize=5, label="Generated post-burn-in"),
        mlines.Line2D([], [], linestyle="", marker="D", color="#111111", markersize=6, label="Generated centroid"),
        mlines.Line2D([], [], linestyle="", marker="o", color=SCENARIO_COLOR, markersize=6, label="Final seeds"),
        mlines.Line2D([], [], linestyle="", marker="X", color=ANCHOR_COLOR, markersize=7, label="Anchor"),
        mlines.Line2D([], [], linestyle="", marker="s", color=ANALOG_COLOR, markeredgecolor=TEXT_COLOR, markersize=6, label="Nearest analogs"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=6.3, borderpad=0.35, ncol=1)
    fig.tight_layout()
    analogs.to_csv(out_dir / "sr_top5_macro_analogs.csv", index=False)
    return savefig(fig, out_dir, "sr_main_macro_geography_pca")


def plot_tsne_appendix(run: dict[str, object], out_dir: Path, anchor: int) -> tuple[Path, Path]:
    final = run["final"]
    sample = run["sample"]
    historical = run["historical"]
    hist_std = historical_std_frame(historical)
    cols = [f"{c}_std" for c in MACRO_COLS]
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_x = sample[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = anchor_std(historical, anchor)[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(12)
    sample_idx = rng.choice(len(sample_x), size=min(700, len(sample_x)), replace=False)
    parts = historical_tsne_with_nn_overlay(
        hist_x,
        {"sample": sample_x[sample_idx], "final": final_x, "anchor": anchor_x},
        random_state=41,
        perplexity=22,
        n_iter=750,
        k=8,
    )
    apply_submission_style()
    fig, ax = plt.subplots(figsize=(7.15, 4.45))
    for regime in ["expansion", "contraction", "financial_stress"]:
        idx = hist_std["regime"].astype(str).eq(regime).to_numpy()
        ax.scatter(parts["hist"][idx, 0], parts["hist"][idx, 1], s=13, color=REGIME_COLORS[regime], alpha=0.22, linewidths=0, rasterized=True)
    ax.scatter(parts["sample"][:, 0], parts["sample"][:, 1], s=10, color=SAMPLE_COLOR, alpha=0.18, linewidths=0, rasterized=True)
    ax.scatter(parts["anchor"][:, 0], parts["anchor"][:, 1], marker="X", s=150, color=ANCHOR_COLOR, edgecolors="white", linewidth=1.0, zorder=6)
    for idx, xy in enumerate(parts["final"]):
        color = REGIME_COLORS.get(str(final.iloc[idx]["regime"]), SCENARIO_COLOR)
        ax.scatter(xy[0], xy[1], s=78, color=color, edgecolors="white", linewidth=0.95, zorder=6)
        ax.text(xy[0], xy[1], str(int(final.iloc[idx]["seed"])), color="white", fontsize=7.5, weight="bold", ha="center", va="center", zorder=7)
    ax.set_title("Appendix: fixed-history t-SNE neighborhood check", loc="left")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    add_clean_axis(ax)
    ax.text(0.015, 0.02, "t-SNE is fit only on historical macro states; scenario points are placed afterward.", transform=ax.transAxes, fontsize=7.0, va="bottom", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["expansion"], markersize=5, label="Expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["contraction"], markersize=5, label="Contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["financial_stress"], markersize=5, label="Stress"),
        mlines.Line2D([], [], linestyle="", marker="o", color=SAMPLE_COLOR, markersize=5, label="Generated sample"),
        mlines.Line2D([], [], linestyle="", marker="X", color=ANCHOR_COLOR, markersize=7, label="Anchor"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=7.2)
    fig.tight_layout()
    return savefig(fig, out_dir, "sr_appendix_tsne_fixed_history")


def plot_diversification_outcomes(run: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    sample = run["sample"]
    final = run["final"]
    metrics = [
        ("Entropy", "anchor_entropy", "final_entropy", "Higher is better", True, ""),
        ("Effective N", "anchor_effective_n", "final_effective_n", "Higher is better", True, ""),
        ("HHI", "anchor_hhi", "final_hhi", "Lower is better", False, ""),
        ("Max weight", "anchor_max_weight", "final_max_weight", "Lower is better", False, "%"),
        ("Top-10 weight", "anchor_top10_weight", "final_top10_weight", "Lower is better", False, "%"),
    ]
    apply_submission_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), sharey=False)
    axes = axes.ravel()
    for ax, (label, anchor_col, final_col, direction, higher_good, suffix) in zip(axes, metrics):
        vals = sample[final_col].to_numpy(dtype=float)
        finals = final[final_col].to_numpy(dtype=float)
        anchor_val = float(final[anchor_col].iloc[0])
        scale = 100 if suffix == "%" else 1
        vals_s = vals * scale
        finals_s = finals * scale
        anchor_s = anchor_val * scale
        ax.boxplot([vals_s], positions=[1], widths=0.42, patch_artist=True, showfliers=False, boxprops={"facecolor": "#d9ece9", "edgecolor": SCENARIO_COLOR, "linewidth": 1.0}, medianprops={"color": TEXT_COLOR, "linewidth": 1.2}, whiskerprops={"color": SCENARIO_COLOR}, capprops={"color": SCENARIO_COLOR})
        jitter = np.linspace(-0.12, 0.12, len(finals_s))
        ax.scatter(1 + jitter, finals_s, s=36, color=SCENARIO_COLOR, edgecolors="white", linewidth=0.6, zorder=4)
        ax.axhline(anchor_s, color=ANCHOR_COLOR, linestyle="--", linewidth=1.15)
        median_s = float(np.median(vals_s))
        delta = median_s - anchor_s
        good = delta > 0 if higher_good else delta < 0
        delta_text = f"{delta:+.2f}{suffix}"
        ax.text(0.5, 0.95, delta_text, transform=ax.transAxes, ha="center", va="top", fontsize=8.2, weight="bold", color=SCENARIO_COLOR if good else ANCHOR_COLOR)
        ax.set_title(label, fontsize=9.4, weight="bold")
        ax.set_xticks([1])
        ax.set_xticklabels(["Generated"], fontsize=8.0)
        ax.tick_params(axis="y", labelsize=7.8)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.60)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, -0.18, direction, transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color="#5f6368")
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.78, "Red dashed line: anchor\nBox: post-burn-in generated states\nDots: final seeds", fontsize=8.0, va="top", color=TEXT_COLOR)
    axes[-1].text(0.02, 0.33, "Main result:\nmost generated states reduce concentration.", fontsize=8.0, va="top", weight="bold", color=SCENARIO_COLOR)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Generated states diversify the locked E2E portfolio", fontsize=10.8, weight="bold", y=1.00)
    fig.tight_layout()
    return savefig(fig, out_dir, "sr_diversification_outcome_metrics")


def plot_holdings_concentration(run: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    anchor_w = run["anchor_weights"].copy()
    final_w = run["final_weights"].copy()
    mean_final = final_w.groupby("permno", as_index=False)["weight"].mean()
    merged = anchor_w[["permno", "anchor_weight"]].merge(mean_final, on="permno", how="left").fillna({"weight": 0.0})
    merged["anchor_abs"] = merged["anchor_weight"].abs()
    top = merged.nlargest(10, "anchor_abs").copy()
    top = top.sort_values("anchor_weight", ascending=True)
    anchor_sorted = np.sort(np.abs(merged["anchor_weight"].to_numpy(dtype=float)))[::-1]
    final_sorted = np.sort(np.abs(merged["weight"].to_numpy(dtype=float)))[::-1]
    k = np.arange(1, len(anchor_sorted) + 1)

    apply_submission_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [1.05, 1.0]})
    y = np.arange(len(top))
    axes[0].barh(y + 0.18, 100 * top["anchor_weight"], height=0.34, color=ANCHOR_COLOR, alpha=0.82, label="Anchor")
    axes[0].barh(y - 0.18, 100 * top["weight"], height=0.34, color=SCENARIO_COLOR, alpha=0.86, label="Generated final mean")
    axes[0].axvline(0, color=TEXT_COLOR, linewidth=0.75)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top["permno"].astype(str), fontsize=7.6)
    axes[0].set_xlabel("Portfolio weight (%)", fontsize=8.8)
    axes[0].set_title("Largest anchor positions shrink", loc="left", fontsize=10.4, weight="bold")
    axes[0].legend(frameon=False, fontsize=7.8, loc="lower right")
    axes[0].grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.62)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(k, 100 * np.cumsum(anchor_sorted), color=ANCHOR_COLOR, linewidth=2.0, label="Anchor")
    axes[1].plot(k, 100 * np.cumsum(final_sorted), color=SCENARIO_COLOR, linewidth=2.0, label="Generated final mean")
    axes[1].axvline(10, color="#7a7f86", linestyle="--", linewidth=0.9)
    axes[1].set_xlim(1, min(50, len(k)))
    axes[1].set_ylim(0, 103)
    axes[1].set_xlabel("Top-k absolute positions", fontsize=8.8)
    axes[1].set_ylabel("Cumulative absolute weight (%)", fontsize=8.8)
    axes[1].set_title("Concentration curve flattens", loc="left", fontsize=10.4, weight="bold")
    axes[1].legend(frameon=False, fontsize=7.8, loc="lower right")
    add_clean_axis(axes[1])
    fig.suptitle("Portfolio mechanism: concentration is reduced, not merely relabeled", fontsize=10.8, weight="bold", y=1.01)
    fig.tight_layout()
    return savefig(fig, out_dir, "sr_portfolio_concentration_mechanism")


def split_rhat(y: np.ndarray) -> float:
    n_chain, n_draw = y.shape
    half = n_draw // 2
    z = np.concatenate([y[:, :half], y[:, -half:]], axis=0)
    m, n = z.shape
    chain_means = z.mean(axis=1)
    chain_vars = z.var(axis=1, ddof=1)
    w = chain_vars.mean()
    b = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * w + b / n
    return float(np.sqrt(var_hat / w)) if w > 0 else np.nan


def plot_convergence(run: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    sample = run["sample"].copy()
    seed = run["seed"].copy()
    payload = run["tensor"]
    tensor = payload["tensor"].detach().cpu().numpy().astype(float)
    macros = [str(x) for x in payload["macro_columns"]]
    rhats = np.array([split_rhat(tensor[:, :, j]) for j in range(tensor.shape[2])])
    sample = sample.sort_values(["seed", "step"])
    apply_submission_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for seed_id, group in sample.groupby("seed"):
        g = group.copy()
        # The sample is already thinned; a 75-point rolling mean is readable without hiding trends.
        y = g["delta_entropy"].rolling(75, min_periods=10).mean()
        axes[0].plot(g["step"], y, linewidth=1.2, alpha=0.86, label=f"seed {int(seed_id)}")
    axes[0].axhline(0, color=TEXT_COLOR, linewidth=0.85)
    axes[0].axvline(10000, color="#7a7f86", linestyle="--", linewidth=0.9)
    axes[0].set_title("Post-burn-in target trace", loc="left", fontsize=10.4, weight="bold")
    axes[0].set_xlabel("MALA step", fontsize=8.8)
    axes[0].set_ylabel("Entropy increase vs. anchor", fontsize=8.8)
    axes[0].legend(frameon=False, fontsize=7.1, loc="lower right", ncol=2)
    add_clean_axis(axes[0])
    order = np.argsort(rhats)
    colors = np.where(rhats[order] <= 1.05, SCENARIO_COLOR, ANCHOR_COLOR)
    axes[1].barh(np.array(macros)[order], rhats[order], color=colors, alpha=0.86)
    axes[1].axvline(1.05, color="#7a7f86", linestyle="--", linewidth=0.9, label="1.05")
    axes[1].set_xlim(0.999, max(1.06, float(np.nanmax(rhats)) + 0.004))
    axes[1].set_title("Split-Rhat by macro variable", loc="left", fontsize=10.4, weight="bold")
    axes[1].set_xlabel("Split-Rhat after burn-in", fontsize=8.8)
    axes[1].legend(frameon=False, fontsize=7.3, loc="lower right")
    add_clean_axis(axes[1])
    fig.suptitle("Convergence check for the 4-chain, 20,000-step E2E run", fontsize=10.8, weight="bold", y=1.01)
    fig.text(0.5, -0.01, f"Acceptance rates: {100*seed['accept_rate'].min():.1f}% to {100*seed['accept_rate'].max():.1f}%; max split-Rhat = {np.nanmax(rhats):.4f}.", ha="center", fontsize=8.0, color="#5f6368")
    fig.tight_layout()
    pd.DataFrame({"macro": macros, "split_rhat": rhats}).to_csv(out_dir / "sr_convergence_rhat.csv", index=False)
    return savefig(fig, out_dir, "sr_convergence_appendix")


def write_notes(run: dict[str, object], out_dir: Path, records: list[dict[str, str]], anchor: int) -> None:
    final = run["final"]
    sample = run["sample"]
    seed = run["seed"]
    analog_path = out_dir / "sr_top5_macro_analogs.csv"
    analogs = pd.read_csv(analog_path) if analog_path.exists() else pd.DataFrame()
    lines = [
        "# E2E 202005 Submission-Ready Figures",
        "",
        f"Run: `{DEFAULT_RUN.relative_to(ROOT)}`",
        f"Anchor: `{anchor}`",
        "",
        "## Diagnostic Summary",
        f"- Acceptance range: {100*seed['accept_rate'].min():.1f}% to {100*seed['accept_rate'].max():.1f}%; median {100*seed['accept_rate'].median():.1f}%.",
        f"- Post-burn-in entropy improves in {100*(sample['delta_entropy'] > 0).mean():.1f}% of generated states; median entropy change {sample['delta_entropy'].median():+.3f}.",
        f"- Post-burn-in HHI falls in {100*(sample['delta_hhi'] < 0).mean():.1f}% of generated states; median HHI change {sample['delta_hhi'].median():+.3f}.",
        f"- Final seeds: {dict(final['regime'].value_counts())}; generated sample: {dict((sample['regime'].value_counts(normalize=True) * 100).round(1))} percent.",
        f"- Locality caution: empirical-anchor tail median {final['anchor_empirical_mah_chi2_tail'].median():.3f}; VAR-anchor tail median {final['anchor_mah_chi2_tail'].median():.2e}.",
        "",
        "## Recommended Manuscript Framing",
        "The generated macro states should be interpreted as empirically local transition-neighborhood states around May 2020, not as VAR(1)-innovation-likely states. In this neighborhood, the locked E2E allocation becomes less concentrated: entropy and effective number rise while HHI, maximum weight, and top-10 concentration fall for most post-burn-in samples.",
        "",
        "## Nearest Historical Analogs",
    ]
    if not analogs.empty:
        for row in analogs.itertuples(index=False):
            lines.append(f"- `{int(row.yyyymm)}` ({row.regime}), standardized Euclidean distance {row.std_euclidean_distance:.3f}")
    lines += ["", "## Figure Manifest"]
    for record in records:
        lines.append(f"- `{record['figure']}`: {record['purpose']}")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create submission-ready E2E 202005 figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-date", type=int, default=202005)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    run = load_run(args.run_dir)
    records: list[dict[str, str]] = []
    specs = [
        ("sr_main_macro_geography_pca", plot_main_pca(run, args.out_dir, args.anchor_date), "Main manuscript macro-geography figure; historical PCA is fixed and scenario states are overlaid."),
        ("sr_diversification_outcome_metrics", plot_diversification_outcomes(run, args.out_dir), "Shows direct answer to the question: concentration falls and diversification rises."),
        ("sr_portfolio_concentration_mechanism", plot_holdings_concentration(run, args.out_dir), "Shows portfolio-level mechanism through top holdings and cumulative concentration."),
        ("sr_convergence_appendix", plot_convergence(run, args.out_dir), "Appendix convergence plot: target traces and split-Rhat by macro variable."),
        ("sr_appendix_tsne_fixed_history", plot_tsne_appendix(run, args.out_dir, args.anchor_date), "Appendix nonlinear neighborhood check with t-SNE fit only on historical states."),
    ]
    for figure, paths, purpose in specs:
        records.append(
            {
                "figure": figure,
                "pdf": str(paths[0].relative_to(ROOT)),
                "png": str(paths[1].relative_to(ROOT)),
                "purpose": purpose,
            }
        )
    with (args.out_dir / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "pdf", "png", "purpose"])
        writer.writeheader()
        writer.writerows(records)
    write_notes(run, args.out_dir, records, args.anchor_date)
    print(f"Wrote {len(records)} submission-ready figures -> {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
