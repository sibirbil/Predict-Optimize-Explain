"""Submission-ready Scenario 4 figures for the 202004 long run.

Story: from the April 2020 financial-stress anchor, generated macro states move
into a contraction neighborhood where SummerChild beats WinterWolf.
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
    newest,
)
from scripts.reporting.report_story_figures_v3 import historical_tsne_with_nn_overlay  # noqa: E402
from src.utils.plotting import set_publication_style  # noqa: E402


DEFAULT_RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260509_072049"
DEFAULT_OUT = ROOT / "submission_plots" / "story_figures_v3" / "scenario4_202004" / "submission_ready"

ANCHOR = 202004
ANCHOR_COLOR = "#c44e52"
SCENARIO_COLOR = "#157f78"
SAMPLE_COLOR = "#6f7782"
ANALOG_COLOR = "#f2c94c"
GRID_COLOR = "#d8dde3"
TEXT_COLOR = "#1f2328"
SC_COLOR = "#157f78"
WW_COLOR = "#c44e52"


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
        "sample": pd.read_csv(newest(run_dir, "generated_macro_sample_postburnin_*.csv")),
        "historical": pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv")),
        "seed": pd.read_csv(newest(run_dir, "seed_summary_*.csv")),
        "transitions": pd.read_csv(newest(run_dir, "regime_transitions_*.csv")),
        "config": json.loads(newest(run_dir, "config_*.json").read_text(encoding="utf-8")),
        "tensor": torch.load(newest(run_dir, "trajectories_postburnin_standardized_3d_*.pt"), map_location="cpu"),
    }


def covariance_ellipse(points: np.ndarray, n_std: float = 1.35) -> tuple[np.ndarray, float, float, float]:
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


def load_anchor_probs(anchor: int = ANCHOR) -> dict[str, float]:
    probs = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_probability_panel.csv")
    row = probs[probs["yyyymm"].astype(int).eq(anchor)].iloc[0]
    return {
        "contraction": float(row["contraction"]),
        "expansion": float(row["expansion"]),
        "financial_stress": float(row["financial_stress"]),
    }


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
    rng = np.random.default_rng(9)
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
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.05),
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.18},
    )
    for ax in axes:
        for regime in ["expansion", "contraction", "financial_stress"]:
            idx = hist_std["regime"].astype(str).eq(regime).to_numpy()
            ax.scatter(
                hist_p[idx, 0],
                hist_p[idx, 1],
                s=9,
                color=REGIME_COLORS[regime],
                alpha=0.15,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

    global_ax, local_ax = axes
    global_ax.scatter(sample_p[:, 0], sample_p[:, 1], s=8, color=SAMPLE_COLOR, alpha=0.10, linewidths=0, rasterized=True, zorder=2)
    global_ax.scatter(anchor_p[0], anchor_p[1], marker="X", s=145, color=ANCHOR_COLOR, edgecolors="white", linewidth=1.0, zorder=8)
    global_ax.scatter(centroid_p[0], centroid_p[1], marker="D", s=74, color="#111111", edgecolors="white", linewidth=0.9, zorder=7)
    global_ax.annotate(
        "",
        xy=centroid_p,
        xytext=anchor_p,
        arrowprops={"arrowstyle": "->", "lw": 1.35, "color": TEXT_COLOR, "alpha": 0.88},
        zorder=6,
    )
    for idx, xy in enumerate(final_p):
        color = REGIME_COLORS.get(str(final.iloc[idx]["regime"]), SCENARIO_COLOR)
        global_ax.scatter(xy[0], xy[1], s=56, color=color, edgecolors="white", linewidth=0.85, zorder=7)
    analog_label_offsets = [(-0.38, 0.10), (0.08, 0.08), (-0.28, -0.14)]
    for analog_idx, (xy, row) in enumerate(zip(analog_p, analogs.itertuples(index=False))):
        global_ax.scatter(xy[0], xy[1], marker="s", s=46, color=ANALOG_COLOR, edgecolors=TEXT_COLOR, linewidth=0.65, zorder=7)
        if analog_idx < 3:
            dx, dy = analog_label_offsets[analog_idx]
            global_ax.text(
                xy[0] + dx,
                xy[1] + dy,
                str(int(row.yyyymm)),
                fontsize=6.6,
                color=TEXT_COLOR,
                zorder=8,
                clip_on=True,
            )
    key = np.vstack([anchor_p.reshape(1, -1), centroid_p.reshape(1, -1), final_p])
    lo_hi = np.nanquantile(hist_p, [0.02, 0.98], axis=0)
    lo = np.minimum(lo_hi[0], key.min(axis=0))
    hi = np.maximum(lo_hi[1], key.max(axis=0))
    span = np.maximum(hi - lo, 0.8)
    global_ax.set_xlim(lo[0] - 0.09 * span[0], hi[0] + 0.09 * span[0])
    global_ax.set_ylim(lo[1] - 0.11 * span[1], hi[1] + 0.11 * span[1])
    global_ax.set_title("Global history: stress anchor to contraction region", loc="left")
    global_ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    global_ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    add_clean_axis(global_ax)

    local_ax.scatter(sample_p[:, 0], sample_p[:, 1], s=10, color=SAMPLE_COLOR, alpha=0.18, linewidths=0, rasterized=True, zorder=2)
    center, width, height, angle = covariance_ellipse(sample_p, n_std=1.05)
    local_ax.add_patch(Ellipse(center, width, height, angle=angle, facecolor=SCENARIO_COLOR, edgecolor=SCENARIO_COLOR, alpha=0.11, lw=1.25, zorder=3))
    local_ax.scatter(centroid_p[0], centroid_p[1], marker="D", s=82, color="#111111", edgecolors="white", linewidth=0.9, zorder=6)
    for idx, xy in enumerate(final_p):
        color = REGIME_COLORS.get(str(final.iloc[idx]["regime"]), SCENARIO_COLOR)
        local_ax.scatter(xy[0], xy[1], s=76, color=color, edgecolors="white", linewidth=0.95, zorder=7)
        local_ax.text(xy[0], xy[1], str(int(final.iloc[idx]["seed"])), color="white", fontsize=7.5, weight="bold", ha="center", va="center", zorder=8)
    local = np.vstack([sample_p, final_p])
    lo_hi = np.nanquantile(local, [0.01, 0.99], axis=0)
    span = np.maximum(lo_hi[1] - lo_hi[0], 0.20)
    local_ax.set_xlim(lo_hi[0, 0] - 0.18 * span[0], lo_hi[1, 0] + 0.18 * span[0])
    local_ax.set_ylim(lo_hi[0, 1] - 0.22 * span[1], lo_hi[1, 1] + 0.22 * span[1])
    local_ax.set_title("Local generated neighborhood", loc="left")
    local_ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    local_ax.set_ylabel("")
    add_clean_axis(local_ax)
    global_ax.text(0.015, 0.02, "PCA is fit on history; scenario states are overlaid afterward.", transform=global_ax.transAxes, fontsize=7.0, va="bottom", ha="left", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    local_ax.text(0.015, 0.02, "Zoom excludes the anchor and analogs to show the final contraction cloud.", transform=local_ax.transAxes, fontsize=7.0, va="bottom", ha="left", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["expansion"], markersize=4.5, label="Historical expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["contraction"], markersize=4.5, label="Historical contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["financial_stress"], markersize=4.5, label="Historical stress"),
        mlines.Line2D([], [], linestyle="", marker="o", color=SAMPLE_COLOR, markersize=5, label="Generated post-burn-in"),
        mlines.Line2D([], [], linestyle="", marker="D", color="#111111", markersize=6, label="Generated centroid"),
        mlines.Line2D([], [], linestyle="", marker="o", color=REGIME_COLORS["contraction"], markersize=6, label="Final seeds"),
        mlines.Line2D([], [], linestyle="", marker="X", color=ANCHOR_COLOR, markersize=7, label="Anchor"),
        mlines.Line2D([], [], linestyle="", marker="s", color=ANALOG_COLOR, markeredgecolor=TEXT_COLOR, markersize=6, label="Nearest analogs"),
    ]
    local_ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=6.1, borderpad=0.35)
    fig.suptitle("Scenario 4 macro geography: generated states leave April 2020 stress", fontsize=10.6, weight="bold", y=1.02)
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
        random_state=33,
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
        ax.scatter(xy[0], xy[1], s=78, color=REGIME_COLORS["contraction"], edgecolors="white", linewidth=0.95, zorder=6)
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


def plot_return_regime_outcome(run: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    final = run["final"].copy()
    sample = run["sample"].copy()
    anchor_probs = load_anchor_probs(ANCHOR)
    sorted_final = final.sort_values("return_gap").reset_index(drop=True)
    x = np.arange(1, len(sorted_final) + 1)
    final_stress = 100 * final["prob_financial_stress"].to_numpy(dtype=float)
    sample_stress = 100 * sample["prob_financial_stress"].to_numpy(dtype=float)
    sample_contr = 100 * sample["prob_contraction"].to_numpy(dtype=float)

    apply_submission_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))
    axes[0].bar(x, 100 * sorted_final["return_gap"], color=SC_COLOR, alpha=0.88)
    axes[0].axhline(0, color=TEXT_COLOR, linewidth=0.85)
    axes[0].axhline(100 * sorted_final["return_gap"].median(), color="#495057", linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_final["seed"].astype(int).astype(str))
    axes[0].set_xlabel("Final seed")
    axes[0].set_ylabel("SC - WW realized return (pp)")
    axes[0].set_title("SummerChild wins in all final states", loc="left")
    axes[0].text(0.03, 0.92, f"Median gap: {100*final['return_gap'].median():+.2f}pp", transform=axes[0].transAxes, fontsize=7.4, va="top", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    add_clean_axis(axes[0])

    axes[1].boxplot(
        [sample_stress, sample_contr],
        positions=[1, 2],
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#d9ece9", "edgecolor": SCENARIO_COLOR, "linewidth": 1.0},
        medianprops={"color": TEXT_COLOR, "linewidth": 1.2},
        whiskerprops={"color": SCENARIO_COLOR},
        capprops={"color": SCENARIO_COLOR},
    )
    axes[1].scatter(np.full(len(final_stress), 1) + np.linspace(-0.08, 0.08, len(final_stress)), final_stress, s=35, color=SCENARIO_COLOR, edgecolors="white", linewidth=0.6, zorder=4)
    axes[1].axhline(100 * anchor_probs["financial_stress"], color=ANCHOR_COLOR, linestyle="--", linewidth=1.05, label="Anchor stress prob.")
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Stress prob.", "Contraction prob."])
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_title("Generated states leave stress", loc="left")
    axes[1].text(0.03, 0.92, f"Anchor stress: {100*anchor_probs['financial_stress']:.1f}%\nSample stress median: {np.median(sample_stress):.1f}%", transform=axes[1].transAxes, fontsize=7.3, va="top", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    add_clean_axis(axes[1])
    fig.suptitle("Scenario 4 mechanism: regime escape reverses the model ranking", fontsize=10.8, weight="bold", y=1.01)
    fig.tight_layout()
    return savefig(fig, out_dir, "sr_return_gap_and_regime_escape")


def plot_macro_mechanism(run: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    sample = run["sample"]
    final = run["final"]
    shifts = []
    for col in MACRO_COLS:
        values = sample[f"{col}_std"].to_numpy(dtype=float)
        shifts.append(
            {
                "macro": col,
                "median": float(np.nanmedian(values - anchor_std(run["historical"], ANCHOR)[f"{col}_std"])),
                "lo": float(np.nanquantile(values - anchor_std(run["historical"], ANCHOR)[f"{col}_std"], 0.25)),
                "hi": float(np.nanquantile(values - anchor_std(run["historical"], ANCHOR)[f"{col}_std"], 0.75)),
            }
        )
    frame = pd.DataFrame(shifts).sort_values("median")
    apply_submission_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    y = np.arange(len(frame))
    colors = np.where(frame["median"] >= 0, SCENARIO_COLOR, WW_COLOR)
    ax.barh(y, frame["median"], color=colors, alpha=0.82)
    ax.errorbar(frame["median"], y, xerr=[frame["median"] - frame["lo"], frame["hi"] - frame["median"]], fmt="none", ecolor=TEXT_COLOR, elinewidth=0.85, capsize=2.0)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["macro"])
    ax.set_xlabel("Generated macro state - anchor (z-score units)")
    ax.set_title("Macro movement behind Scenario 4", loc="left")
    add_clean_axis(ax)
    ax.text(0.02, 0.04, "Bars show post-burn-in medians; whiskers are IQR. The generated states move away from the April 2020 stress anchor.", transform=ax.transAxes, fontsize=7.3, va="bottom", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": GRID_COLOR, "alpha": 0.94})
    frame.to_csv(out_dir / "sr_macro_shift_summary.csv", index=False)
    return savefig(fig, out_dir, "sr_macro_mechanism_shift")


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
    sample = run["sample"].sort_values(["seed", "step"]).copy()
    seed = run["seed"].copy()
    payload = run["tensor"]
    tensor = payload["tensor"].detach().cpu().numpy().astype(float)
    macros = [str(x) for x in payload["macro_columns"]]
    rhats = np.array([split_rhat(tensor[:, :, j]) for j in range(tensor.shape[2])])
    apply_submission_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for seed_id, group in sample.groupby("seed"):
        y = group["prob_financial_stress"].rolling(75, min_periods=10).mean() * 100
        axes[0].plot(group["step"], y, linewidth=1.2, alpha=0.86, label=f"seed {int(seed_id)}")
    axes[0].axvline(10000, color="#7a7f86", linestyle="--", linewidth=0.9)
    axes[0].set_title("Post-burn-in stress-probability trace", loc="left")
    axes[0].set_xlabel("MALA step")
    axes[0].set_ylabel("Financial-stress probability (%)")
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper right", ncol=2)
    add_clean_axis(axes[0])

    order = np.argsort(rhats)
    colors = np.where(rhats[order] <= 1.05, SCENARIO_COLOR, ANCHOR_COLOR)
    axes[1].barh(np.array(macros)[order], rhats[order], color=colors, alpha=0.86)
    axes[1].axvline(1.05, color="#7a7f86", linestyle="--", linewidth=0.9, label="1.05")
    axes[1].set_xlim(0.999, max(1.06, float(np.nanmax(rhats)) + 0.004))
    axes[1].set_title("Split-Rhat by macro variable", loc="left")
    axes[1].set_xlabel("Split-Rhat after burn-in")
    axes[1].legend(frameon=False, fontsize=7.0, loc="lower right")
    add_clean_axis(axes[1])
    fig.suptitle("Scenario 4 convergence check", fontsize=10.8, weight="bold", y=1.01)
    fig.text(0.5, -0.01, f"Acceptance rates: {100*seed['accept_rate'].min():.1f}% to {100*seed['accept_rate'].max():.1f}%; max split-Rhat = {np.nanmax(rhats):.4f}.", ha="center", fontsize=8.0, color="#5f6368")
    fig.tight_layout()
    pd.DataFrame({"macro": macros, "split_rhat": rhats}).to_csv(out_dir / "sr_convergence_rhat.csv", index=False)
    return savefig(fig, out_dir, "sr_convergence_appendix")


def write_notes(run: dict[str, object], out_dir: Path, records: list[dict[str, str]], anchor: int) -> None:
    final = run["final"]
    sample = run["sample"]
    seed = run["seed"]
    anchor_probs = load_anchor_probs(anchor)
    analog_path = out_dir / "sr_top5_macro_analogs.csv"
    analogs = pd.read_csv(analog_path) if analog_path.exists() else pd.DataFrame()
    lines = [
        "# Scenario 4 Submission-Ready Figures",
        "",
        f"Run: `{DEFAULT_RUN.relative_to(ROOT)}`",
        f"Anchor: `{anchor}`",
        "",
        "## Diagnostic Summary",
        f"- SummerChild wins {int((final['return_gap'] > 0).sum())}/{len(final)} final states; median SC-WW return gap {100*final['return_gap'].median():+.2f}pp.",
        f"- Anchor financial-stress probability {100*anchor_probs['financial_stress']:.1f}%; generated sample stress-probability median {100*sample['prob_financial_stress'].median():.2f}%.",
        f"- Final regimes: {dict(final['regime'].value_counts())}; generated sample: {dict((sample['regime'].value_counts(normalize=True) * 100).round(1))} percent.",
        f"- Acceptance range: {100*seed['accept_rate'].min():.1f}% to {100*seed['accept_rate'].max():.1f}%; median {100*seed['accept_rate'].median():.1f}%.",
        f"- Locality caution: VAR chi-square tails are tiny; frame as an economically interpretable stress-escape counterfactual, not a VAR(1)-innovation-likely state.",
        "",
        "## Recommended Manuscript Framing",
        "Scenario 4 identifies a regime-escape mechanism. Starting from the April 2020 financial-stress anchor, generated macro states move into a contraction-dominated neighborhood. In those states SummerChild beats WinterWolf, so the model ranking reversal is tied to a clear economic movement rather than an arbitrary perturbation.",
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
    parser = argparse.ArgumentParser(description="Create submission-ready Scenario 4 figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-date", type=int, default=ANCHOR)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    run = load_run(args.run_dir)
    specs = [
        ("sr_main_macro_geography_pca", plot_main_pca(run, args.out_dir, args.anchor_date), "Main manuscript macro-geography figure; historical PCA is fixed and scenario states are overlaid."),
        ("sr_return_gap_and_regime_escape", plot_return_regime_outcome(run, args.out_dir), "Directly links SC-WW return reversal to a fall in financial-stress probability."),
        ("sr_macro_mechanism_shift", plot_macro_mechanism(run, args.out_dir), "Shows the standardized macro shifts behind the stress-escape story."),
        ("sr_convergence_appendix", plot_convergence(run, args.out_dir), "Appendix convergence plot: stress-probability traces and split-Rhat by macro variable."),
        ("sr_appendix_tsne_fixed_history", plot_tsne_appendix(run, args.out_dir, args.anchor_date), "Appendix nonlinear neighborhood check with t-SNE fit only on historical states."),
    ]
    records: list[dict[str, str]] = []
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
    print(f"Wrote {len(records)} Scenario 4 submission-ready figures -> {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
