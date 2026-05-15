"""Story-first v3 paper figures for the three selected scenario generations.

V3 replaces broad diagnostic plots with mechanism-focused figures:

1. Scenario 4: April 2020 stress -> SC beats WW after regime escape.
2. Locked E2E diversification: concentrated anchor -> diversified portfolios.
3. PTO catch-up: PTO overtakes locked E2E inside financial stress.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from report_story_figures_v2 import (  # noqa: E402
    E2E_RUN,
    MACRO_COLS,
    MODEL_COLORS,
    PTO_RUN,
    REGIME_COLORS,
    S4_RUN,
    FigureRecord,
    add_record,
    anchor_std,
    grouped_weight_means,
    historical_std_frame,
    load_regime_nfci_panel,
    load_s4,
    load_scenario5,
    plot_e2e_concentration,
    plot_e2e_metrics,
    plot_pto_portfolio,
    plot_pto_return_gap,
    plot_s4_return_gap,
    save_figure,
    yyyymm_to_datetime,
)
from src.utils.plotting import PAPER_COLORS, macro_density_grid, set_publication_style  # noqa: E402


OUT_ROOT = ROOT / "submission_plots" / "story_figures_v3"
REGIME_ORDER = ["financial_stress", "contraction", "expansion"]
PROB_COLS = ["prob_financial_stress", "prob_contraction", "prob_expansion"]
PROB_LABELS = {
    "prob_financial_stress": "Financial\nstress",
    "prob_contraction": "Contraction",
    "prob_expansion": "Expansion",
}


@dataclass(frozen=True)
class StorySpec:
    key: str
    out_dir: str
    anchor: int
    short: str
    analog_title: str
    embedding_title: str


S4_SPEC = StorySpec(
    key="s4",
    out_dir="scenario4_202004",
    anchor=202004,
    short="Scenario 4",
    analog_title="Scenario 4 NFCI analogs",
    embedding_title="Scenario 4 macro geography",
)
E2E_SPEC = StorySpec(
    key="e2e",
    out_dir="e2e_diversification_202004",
    anchor=202004,
    short="E2E diversification",
    analog_title="E2E diversification NFCI analogs",
    embedding_title="E2E diversification macro geography",
)
PTO_SPEC = StorySpec(
    key="pto",
    out_dir="pto_catchup_202003",
    anchor=202003,
    short="PTO catch-up",
    analog_title="PTO catch-up NFCI analogs",
    embedding_title="PTO catch-up macro geography",
)


def infer_anchor_from_run_dir(run_dir: Path, fallback: int) -> int:
    """Infer anchor yyyymm from scenario output folders such as scenario_e2e_diversify_202005."""
    match = re.search(r"_(\d{6})$", run_dir.parent.parent.name)
    return int(match.group(1)) if match else fallback


def historical_tsne_with_nn_overlay(
    hist_x: np.ndarray,
    overlays: dict[str, np.ndarray],
    *,
    random_state: int,
    perplexity: int = 20,
    n_iter: int = 650,
    k: int = 8,
) -> dict[str, np.ndarray]:
    """Fit t-SNE on historical data only, then place overlays by local historical neighbors.

    sklearn's t-SNE has no out-of-sample transform. For scenario plots we therefore
    learn the nonlinear map only from historical macro states, then place generated
    points as inverse-distance weighted averages of their nearest historical
    neighbors in standardized macro space. This prevents generated points from
    reshaping the historical regime geometry.
    """
    hist_x = np.asarray(hist_x, dtype=float)
    n_hist = len(hist_x)
    if n_hist < 4:
        raise ValueError("Need at least four historical points for t-SNE.")
    safe_perplexity = min(perplexity, max(2, (n_hist - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=n_iter,
        random_state=random_state,
        method="exact",
    )
    hist_emb = tsne.fit_transform(hist_x)
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
        exact = near_dist[:, 0] < 1e-10
        coords = (hist_emb[idx] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]
        if exact.any():
            nearest = idx[np.arange(len(arr)), np.argmin(near_dist, axis=1)]
            coords[exact] = hist_emb[nearest[exact]]
        out[name] = coords
    return out


def prob_panel() -> pd.DataFrame:
    out = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_probability_panel.csv")
    out["date"] = yyyymm_to_datetime(out["yyyymm"])
    return out


def anchor_probs(anchor: int) -> dict[str, float]:
    probs = prob_panel()
    row = probs[probs["yyyymm"].astype(int).eq(anchor)].iloc[0]
    return {
        "prob_financial_stress": float(row["financial_stress"]),
        "prob_contraction": float(row["contraction"]),
        "prob_expansion": float(row["expansion"]),
    }


def raw_anchor(historical: pd.DataFrame, anchor: int) -> np.ndarray:
    row = historical[historical["yyyymm"].astype(int).eq(anchor)]
    if row.empty:
        raise ValueError(f"Anchor {anchor} not found in historical macro panel.")
    return row[MACRO_COLS].iloc[0].to_numpy(dtype=float)


def prob_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PROB_COLS:
        if col not in df.columns:
            continue
        for value in df[col].to_numpy(dtype=float):
            rows.append({"probability": col, "value": value})
    return pd.DataFrame(rows)


def final_box_violation(final: pd.DataFrame) -> float:
    if "final_box_violation" in final.columns:
        return float(final["final_box_violation"].mean())
    if "box_violation" in final.columns:
        return float(final["box_violation"].mean())
    return float("nan")


def plot_macro_density_grid(
    spec: StorySpec,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
) -> tuple[str, str]:
    fig = macro_density_grid(
        historical_df=historical[MACRO_COLS],
        generated_df=sample[MACRO_COLS],
        anchor=raw_anchor(historical, spec.anchor),
        columns=MACRO_COLS,
        title=f"{spec.short}: generated macro density vs historical panel",
    )
    fig.set_size_inches(11.5, 8.0, forward=True)
    return save_figure(fig, out_dir, f"{spec.key}_macro_density_grid")


def nearest_analogs(
    final: pd.DataFrame,
    historical: pd.DataFrame,
    anchor: int,
    *,
    top_k: int = 5,
    start_yyyymm: int | None = None,
    end_yyyymm: int | None = None,
) -> pd.DataFrame:
    hist = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    hist = hist[hist["yyyymm"].astype(int).ne(anchor)].copy()
    if start_yyyymm is not None:
        hist = hist[hist["yyyymm"].astype(int).ge(start_yyyymm)].copy()
    if end_yyyymm is not None:
        hist = hist[hist["yyyymm"].astype(int).le(end_yyyymm)].copy()
    hist = hist.reset_index(drop=True)
    hist_x = hist[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    median_final = np.median(final_x, axis=0)
    dists = np.linalg.norm(hist_x - median_final[None, :], axis=1)
    nearest_idx = np.linalg.norm(final_x[:, None, :] - hist_x[None, :, :], axis=2).argmin(axis=1)
    counts = pd.Series(hist.loc[nearest_idx, "yyyymm"].astype(int)).value_counts()
    top_idx = np.argsort(dists)[:top_k]
    top = hist.iloc[top_idx][["yyyymm", "regime"]].copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["std_euclidean_distance"] = dists[top_idx]
    top["nearest_final_state_count"] = top["yyyymm"].astype(int).map(counts).fillna(0).astype(int)
    panel = load_regime_nfci_panel()
    top = top.merge(panel[["yyyymm", "NFCI"]], on="yyyymm", how="left")
    top["date"] = yyyymm_to_datetime(top["yyyymm"])
    return top


def plot_nfci_analogs(
    spec: StorySpec,
    final: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
    *,
    window: str,
) -> tuple[str, str]:
    set_publication_style()
    if window == "test":
        start, end = 201601, 202412
        stem = f"{spec.key}_nfci_analogs_test_window"
        csv_stem = f"{spec.key}_top5_neighbors_test.csv"
        subtitle = "Test-window view keeps the 2020 neighborhood readable."
    elif window == "full":
        start, end = None, None
        stem = f"{spec.key}_nfci_analogs_full_history_appendix"
        csv_stem = f"{spec.key}_top5_neighbors_full.csv"
        subtitle = "Full-history appendix view shows older macro analogs."
    else:
        raise ValueError(window)

    top = nearest_analogs(final, historical, spec.anchor, start_yyyymm=start, end_yyyymm=end)
    top.to_csv(out_dir / csv_stem, index=False)
    panel = load_regime_nfci_panel()
    if start is not None:
        panel = panel[panel["yyyymm"].astype(int).ge(start)].copy()
    if end is not None:
        panel = panel[panel["yyyymm"].astype(int).le(end)].copy()

    anchor_date = yyyymm_to_datetime([spec.anchor]).iloc[0]
    anchor_nfci = float(load_regime_nfci_panel().loc[lambda x: x["yyyymm"].astype(int).eq(spec.anchor), "NFCI"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.plot(panel["date"], panel["NFCI"], color="#1f1f1f", linewidth=1.25, label="NFCI")
    ax.axhline(0.0, color="#73777d", linestyle="--", linewidth=0.85)
    ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label=f"{spec.anchor} anchor")
    sizes = 58 + 18 * np.sqrt(top["nearest_final_state_count"].clip(lower=1).to_numpy(dtype=float))
    ax.scatter(top["date"], top["NFCI"], s=sizes, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.85, zorder=5, label="Top analogs")
    for row in top.itertuples(index=False):
        y_offset = 14 if float(row.NFCI) <= panel["NFCI"].quantile(0.82) else -24
        ax.annotate(
            f"{int(row.yyyymm)}\n#{int(row.rank)}",
            xy=(row.date, row.NFCI),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94},
        )
    ax.annotate(
        "Anchor",
        xy=(anchor_date, anchor_nfci),
        xytext=(anchor_date + pd.DateOffset(months=5 if window == "test" else 16), anchor_nfci + 0.35),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 0.9},
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.text(
        0.02,
        0.04,
        f"Macro z-score nearest analogs; anchor excluded. {subtitle}",
        transform=ax.transAxes,
        fontsize=7.1,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.set_title(f"{spec.analog_title}: {window.replace('_', ' ')}", loc="left", fontsize=12.2, pad=8)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(1 if window == "test" else 4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.60, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0)
    return save_figure(fig, out_dir, stem)


def plot_regime_probability_story(
    spec: StorySpec,
    final: pd.DataFrame,
    sample: pd.DataFrame,
    out_dir: Path,
    *,
    title: str,
    callout: str,
) -> tuple[str, str]:
    set_publication_style()
    anchor = anchor_probs(spec.anchor)
    sample_share = sample["regime"].value_counts(normalize=True).reindex(REGIME_ORDER).fillna(0.0) * 100.0
    final_counts = final["regime"].value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)
    final_prob = prob_frame(final)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [0.9, 1.35, 1.0]})
    labels = [PROB_LABELS[col] for col in PROB_COLS]
    colors = [REGIME_COLORS["financial_stress"], REGIME_COLORS["contraction"], REGIME_COLORS["expansion"]]
    axes[0].bar(np.arange(3), [100 * anchor[col] for col in PROB_COLS], color=colors, alpha=0.88)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Probability / share (%)")
    axes[0].set_title("Anchor", loc="left", fontsize=10.5, pad=7)

    positions = np.arange(3)
    data = [100 * final_prob.loc[final_prob["probability"].eq(col), "value"].to_numpy(dtype=float) for col in PROB_COLS]
    bp = axes[1].boxplot(data, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.1)
    for median in bp["medians"]:
        median.set(color=PAPER_COLORS["text"], linewidth=1.1)
    rng = np.random.default_rng(13)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.normal(0, 0.035, size=len(vals))
        axes[1].scatter(np.full_like(vals, pos) + jitter, vals, s=18, color=color, alpha=0.72, edgecolors="white", linewidth=0.35)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Final seed probabilities", loc="left", fontsize=10.5, pad=7)

    axes[2].bar(np.arange(3), sample_share.values, color=colors, alpha=0.88)
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels([r.replace("_", "\n") for r in REGIME_ORDER], fontsize=7.5)
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Generated sample", loc="left", fontsize=10.5, pad=7)
    for idx, value in enumerate(sample_share.values):
        axes[2].text(idx, min(value + 3, 101), f"{value:.0f}%", ha="center", fontsize=7.5)
    for idx, value in enumerate(final_counts.values):
        if value:
            axes[1].text(idx, 102, f"n={value}", ha="center", fontsize=7.2)
    for ax in axes:
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)
    fig.suptitle(title, fontsize=13.0, y=1.02)
    fig.text(
        0.02,
        -0.02,
        callout,
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    return save_figure(fig, out_dir, f"{spec.key}_regime_probability_story")


def plot_macro_embedding_pca_tsne(
    spec: StorySpec,
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
    *,
    callout: str,
) -> tuple[str, str]:
    set_publication_style()
    hist_std = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    sample_plot = sample.sample(n=min(350, len(sample)), random_state=41) if len(sample) > 350 else sample.copy()
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_x = sample_plot[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = anchor_std(historical, spec.anchor)[cols].to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=0).fit(hist_x)
    pca_parts = {
        "hist": pca.transform(hist_x),
        "sample": pca.transform(sample_x),
        "final": pca.transform(final_x),
        "anchor": pca.transform(anchor_x[None, :]),
    }

    tsne_parts = historical_tsne_with_nn_overlay(
        hist_x,
        {"sample": sample_x, "final": final_x, "anchor": anchor_x},
        random_state=17,
        perplexity=20,
        n_iter=650,
        k=8,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for ax, parts, name in zip(axes, [pca_parts, tsne_parts], ["PCA", "t-SNE"]):
        for regime in ["expansion", "contraction", "financial_stress"]:
            idx = hist_std["regime"].eq(regime).to_numpy()
            ax.scatter(
                parts["hist"][idx, 0],
                parts["hist"][idx, 1],
                s=15,
                color=REGIME_COLORS[regime],
                alpha=0.18,
                linewidths=0,
                rasterized=True,
            )
        ax.scatter(parts["sample"][:, 0], parts["sample"][:, 1], s=10, color="#8d949b", alpha=0.20, linewidths=0, rasterized=True)
        final_colors = final["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
        ax.scatter(parts["final"][:, 0], parts["final"][:, 1], s=58, color=final_colors, edgecolors="white", linewidth=0.55, zorder=4)
        ax.scatter(parts["anchor"][:, 0], parts["anchor"][:, 1], s=145, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.8, zorder=5)
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.52, linewidth=0.55)
        ax.set_title(name, loc="left", fontsize=11.4, pad=7)
        if name == "PCA":
            ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
        else:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["expansion"], label="Historical expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["contraction"], label="Historical contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["financial_stress"], label="Historical stress"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color="#8d949b", label="Generated sample"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=7, color=PAPER_COLORS["generated"], label="Final states"),
        mlines.Line2D([], [], linestyle="", marker="X", markersize=8, color=PAPER_COLORS["anchor"], label="Anchor"),
    ]
    _ = callout  # The interpretation is recorded in figure_notes.md to keep the map uncluttered.
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(spec.embedding_title, fontsize=13.0, y=0.99)
    fig.text(0.5, 0.055, "t-SNE is fit on historical macro states only; scenario points are overlaid by nearest-neighbor placement.", ha="center", fontsize=7.5, color="#5f6368")
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.94))
    return save_figure(fig, out_dir, f"{spec.key}_macro_embedding_pca_tsne")


def plot_regime_class_cloud_pca_tsne(
    spec: StorySpec,
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
) -> tuple[str, str]:
    """Cleaner class-cloud map: regimes first, scenario overlay second."""
    set_publication_style()
    hist_std = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    sample_plot = sample.sample(n=min(260, len(sample)), random_state=23) if len(sample) > 260 else sample.copy()
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_x = sample_plot[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = anchor_std(historical, spec.anchor)[cols].to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=0).fit(hist_x)
    pca_parts = {
        "hist": pca.transform(hist_x),
        "sample": pca.transform(sample_x),
        "final": pca.transform(final_x),
        "anchor": pca.transform(anchor_x[None, :]),
    }
    tsne_parts = historical_tsne_with_nn_overlay(
        hist_x,
        {"sample": sample_x, "final": final_x, "anchor": anchor_x},
        random_state=29,
        perplexity=20,
        n_iter=650,
        k=8,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1))
    for ax, parts, title in zip(axes, [pca_parts, tsne_parts], ["PCA regime cloud", "t-SNE regime cloud"]):
        for regime in ["expansion", "contraction", "financial_stress"]:
            idx = hist_std["regime"].eq(regime).to_numpy()
            cloud = parts["hist"][idx]
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                s=20,
                color=REGIME_COLORS[regime],
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
            center = np.median(cloud, axis=0)
            ax.text(
                center[0],
                center[1],
                regime.replace("_", " "),
                ha="center",
                va="center",
                fontsize=7.4,
                weight="bold",
                color=REGIME_COLORS[regime],
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": REGIME_COLORS[regime], "alpha": 0.78},
            )
        ax.scatter(parts["sample"][:, 0], parts["sample"][:, 1], s=9, color="#343a40", alpha=0.18, linewidths=0, rasterized=True)
        ax.scatter(parts["final"][:, 0], parts["final"][:, 1], s=74, marker="D", color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.65, zorder=4)
        ax.scatter(parts["anchor"][:, 0], parts["anchor"][:, 1], s=150, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.85, zorder=5)
        ax.set_title(title, loc="left", fontsize=11.3, pad=7)
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.52, linewidth=0.55)
        if title.startswith("PCA"):
            ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
        else:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
    handles = [
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["expansion"], label="Historical expansion"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["contraction"], label="Historical contraction"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=6, color=REGIME_COLORS["financial_stress"], label="Historical stress"),
        mlines.Line2D([], [], linestyle="", marker="o", markersize=5, color="#343a40", label="Generated sample"),
        mlines.Line2D([], [], linestyle="", marker="D", markersize=6, color=PAPER_COLORS["generated"], label="Final scenario states"),
        mlines.Line2D([], [], linestyle="", marker="X", markersize=8, color=PAPER_COLORS["anchor"], label="Anchor"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{spec.short}: historical regime class cloud with scenario overlay", fontsize=13.0, y=0.99)
    fig.text(0.5, 0.055, "t-SNE is fit on historical macro states only; scenario points are overlaid by nearest-neighbor placement.", ha="center", fontsize=7.5, color="#5f6368")
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.94))
    return save_figure(fig, out_dir, f"{spec.key}_regime_class_cloud_pca_tsne")


def load_config(run_dir: Path) -> dict:
    matches = sorted(run_dir.glob("config_*.json"))
    if not matches:
        return {}
    with matches[-1].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wrap_text(text: str, width: int = 55) -> str:
    return "\n".join(textwrap.fill(part, width=width) for part in str(text).split("\n"))


def card_texts(
    spec: StorySpec,
    final: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict,
    *,
    story: str,
) -> list[tuple[str, str]]:
    ap = anchor_probs(spec.anchor)
    anchor_regime = (
        f"{spec.anchor}: financial stress {100*ap['prob_financial_stress']:.1f}%, "
        f"contraction {100*ap['prob_contraction']:.1f}%, expansion {100*ap['prob_expansion']:.1f}%."
    )
    sampler = (
        f"MALA/Gibbs scenario generation; {int(config.get('N_SEEDS', len(final)))} seeds, "
        f"{int(config.get('N_STEPS', 0))} steps, beta={float(config.get('BETA', np.nan)):.3g}, "
        f"eta={float(config.get('ETA', np.nan)):.3g}, L2={float(config.get('L2REG', np.nan)):.3g}, "
        f"{config.get('REG_MODE', 'l2')} + {config.get('CONSTRAINT_MODE', 'box_barrier')}."
    )
    final_reg = final["regime"].value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)
    sample_reg = (sample["regime"].value_counts(normalize=True).reindex(REGIME_ORDER).fillna(0) * 100).round(1)
    regimes = (
        f"Final seeds: stress {final_reg['financial_stress']}, contraction {final_reg['contraction']}, expansion {final_reg['expansion']}. "
        f"Generated sample: stress {sample_reg['financial_stress']:.1f}%, contraction {sample_reg['contraction']:.1f}%, expansion {sample_reg['expansion']:.1f}%."
    )
    if story == "s4":
        question = "Question: from April 2020 stress, can generated macro states make SummerChild beat WinterWolf?"
        g_function = "G(m): SummerChild realized return minus WinterWolf realized return, with macro locality regularization and box-barrier feasibility."
        outcome = f"Outcome: {(final['return_gap'] > 0).sum()}/{len(final)} final seeds favor SummerChild; median SC-WW return gap {100*final['return_gap'].median():+.2f}pp."
        economics = "Economic reading: stress-escape explanation. SC wins when generated macro states move away from acute financial stress into contraction/expansion probability mass."
    elif story == "e2e":
        question = f"Question: can a concentrated locked E2E {spec.anchor} anchor become more diversified in nearby macro states?"
        g_function = "G(m): locked E2E entropy increase / concentration reduction objective, regularized toward the anchor macro state."
        outcome = f"Outcome: median entropy {final['delta_entropy'].median():+.3f}, HHI {final['delta_hhi'].median():+.3f}, max weight {100*final['delta_max_weight'].median():+.2f}pp."
        economics = f"Economic reading: allocation concentration is sensitive around the {spec.anchor} macro boundary; generated states relax extreme single-name exposure."
    else:
        question = "Question: can PTO catch up from behind locked E2E and overtake while staying near March 2020 stress?"
        g_function = "G(m): standardized PTO return advantage over locked E2E, i.e. PTO return minus locked E2E return, with locality and feasibility terms."
        outcome = f"Outcome: anchor E2E-PTO gap {100*final['anchor_return_gap_a_minus_b'].iloc[0]:+.2f}pp; final median gap {100*final['return_gap_a_minus_b'].median():+.2f}pp; PTO wins {final['b_return_matches_or_beats_a'].mean():.1%}."
        economics = "Economic reading: not regime escape. PTO overtakes inside financial stress, so model rankings can reverse along interpretable within-stress macro directions."
    caution = (
        "Caution: analogs are standardized-Euclidean neighbors, not uniqueness proof. "
        "Use empirical-anchor locality; avoid VAR(1)-plausibility claims when VAR tails are tiny."
    )
    return [
        ("What Was Asked", question),
        ("G Function", g_function),
        ("Anchor State", anchor_regime),
        ("Scenario Generation", sampler),
        ("Generated Scenarios", regimes + " " + outcome),
        ("Economic Comment", economics + " " + caution),
    ]


def plot_manuscript_summary_card(
    spec: StorySpec,
    final: pd.DataFrame,
    sample: pd.DataFrame,
    run_dir: Path,
    out_dir: Path,
    *,
    story: str,
) -> tuple[str, str]:
    set_publication_style()
    config = load_config(run_dir)
    rows = card_texts(spec, final, sample, config, story=story)
    fig, ax = plt.subplots(figsize=(11.4, 7.2))
    ax.axis("off")
    ax.text(0.04, 0.965, f"{spec.short}: scenario-generation summary", fontsize=16.5, weight="bold", va="top")
    ax.text(0.04, 0.918, "Question, objective, anchor, generation design, result, and economic reading.", fontsize=9.2, color="#333333", va="top")
    positions = [(0.04, 0.64), (0.52, 0.64), (0.04, 0.39), (0.52, 0.39), (0.04, 0.14), (0.52, 0.14)]
    box_colors = ["#f6f7f8", "#f6f7f8", "#fff8ef", "#f6f7f8", "#eef7f5", "#fff4f4"]
    for (heading, body), (x, y), color in zip(rows, positions, box_colors):
        ax.add_patch(
            plt.Rectangle(
                (x - 0.012, y - 0.015),
                0.43,
                0.205,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=PAPER_COLORS["grid"],
                linewidth=0.9,
                alpha=1.0,
                zorder=0,
            )
        )
        ax.text(
            x,
            y + 0.155,
            heading,
            fontsize=9.7,
            weight="bold",
            va="top",
            color=PAPER_COLORS["text"],
        )
        ax.text(x, y + 0.105, wrap_text(body, 54), fontsize=7.7, va="top", color=PAPER_COLORS["text"], linespacing=1.18)
    return save_figure(fig, out_dir, f"{spec.key}_manuscript_summary_card")


def plot_s4_economic_mechanism(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    gaps = 100 * final["return_gap"].to_numpy(dtype=float)
    stress = 100 * final["prob_financial_stress"].to_numpy(dtype=float)
    colors = final["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].scatter(stress, gaps, s=54, color=colors, edgecolors="white", linewidth=0.55)
    axes[0].axhline(0, color=PAPER_COLORS["text"], linewidth=0.9)
    axes[0].set_xlabel("Final financial-stress probability (%)")
    axes[0].set_ylabel("SC minus WW return (pp)")
    axes[0].set_title("SC wins after stress probability collapses", loc="left", fontsize=11.4, pad=7)
    axes[1].bar(["Anchor\nstress prob.", "Final\nmedian"], [100 * anchor_probs(202004)["prob_financial_stress"], np.median(stress)], color=[PAPER_COLORS["anchor"], PAPER_COLORS["generated"]], alpha=0.88)
    axes[1].set_ylabel("Financial-stress probability (%)")
    axes[1].set_title("Regime channel", loc="left", fontsize=11.4, pad=7)
    axes[1].text(0.02, 0.92, f"Final median gap {np.median(gaps):+.2f}pp", transform=axes[1].transAxes, fontsize=8.2, va="top")
    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)
    fig.suptitle("Economic mechanism: Scenario 4 is a stress-escape explanation", fontsize=13.0, y=1.03)
    return save_figure(fig, out_dir, "s4_economic_mechanism")


def plot_e2e_economic_mechanism(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    stress = 100 * final["prob_financial_stress"].to_numpy(dtype=float)
    delta_entropy = final["delta_entropy"].to_numpy(dtype=float)
    delta_hhi = final["delta_hhi"].to_numpy(dtype=float)
    colors = final["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].scatter(stress, delta_entropy, s=58, color=colors, edgecolors="white", linewidth=0.55)
    axes[0].axhline(0, color=PAPER_COLORS["text"], linewidth=0.9)
    axes[0].set_xlabel("Final financial-stress probability (%)")
    axes[0].set_ylabel("Entropy change")
    axes[0].set_title("Diversification rises across regime boundary", loc="left", fontsize=11.4, pad=7)
    axes[1].scatter(stress, delta_hhi, s=58, color=colors, edgecolors="white", linewidth=0.55)
    axes[1].axhline(0, color=PAPER_COLORS["text"], linewidth=0.9)
    axes[1].set_xlabel("Final financial-stress probability (%)")
    axes[1].set_ylabel("HHI change")
    axes[1].set_title("Concentration falls", loc="left", fontsize=11.4, pad=7)
    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)
    fig.suptitle("Economic mechanism: E2E concentration is regime-sensitive near April 2020", fontsize=13.0, y=1.03)
    return save_figure(fig, out_dir, "e2e_economic_mechanism")


def plot_pto_economic_mechanism(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    gap = 100 * final["return_gap_a_minus_b"].to_numpy(dtype=float)
    anchor_gap = 100 * float(final["anchor_return_gap_a_minus_b"].iloc[0])
    dist_col = "anchor_empirical_mah_dist" if "anchor_empirical_mah_dist" in final.columns else "l2_dist"
    dist = final[dist_col].to_numpy(dtype=float)
    stress = 100 * final["prob_financial_stress"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].scatter(dist, gap, s=58, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.55)
    axes[0].axhline(0, color=PAPER_COLORS["text"], linewidth=0.9)
    axes[0].axhline(anchor_gap, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.0, label=f"Anchor {anchor_gap:+.2f}pp")
    axes[0].set_xlabel("Distance to anchor macro state")
    axes[0].set_ylabel("Locked E2E minus PTO return (pp)")
    axes[0].set_title("PTO overtakes at empirical-local distances", loc="left", fontsize=11.4, pad=7)
    axes[0].legend(frameon=False, fontsize=8.0)
    axes[1].bar(["Anchor\nstress prob.", "Final\nmedian"], [100 * anchor_probs(202003)["prob_financial_stress"], np.median(stress)], color=[PAPER_COLORS["anchor"], PAPER_COLORS["generated"]], alpha=0.88)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Financial-stress probability (%)")
    axes[1].set_title("No regime escape", loc="left", fontsize=11.4, pad=7)
    axes[1].text(0.02, 0.92, f"Final median PTO lead {-np.median(gap):.2f}pp", transform=axes[1].transAxes, fontsize=8.2, va="top")
    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.55)
    fig.suptitle("Economic mechanism: PTO catch-up is a within-stress explanation", fontsize=13.0, y=1.03)
    return save_figure(fig, out_dir, "pto_economic_mechanism")


def top_analog_text(out_dir: Path, stem: str) -> str:
    path = out_dir / stem
    if not path.exists():
        return "not computed"
    df = pd.read_csv(path)
    if df.empty:
        return "not computed"
    return ", ".join(str(int(x)) for x in df["yyyymm"].head(5))


def run_s4(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    spec = S4_SPEC
    out_dir = out_root / spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical = load_s4(run_dir)
    records: list[FigureRecord] = []
    add_record(records, spec.out_dir, "s4_return_gap_reversal", plot_s4_return_gap(final, out_dir), "All final seeds produce positive SC-WW return gaps.")
    add_record(records, spec.out_dir, "s4_regime_probability_story", plot_regime_probability_story(spec, final, sample, out_dir, title="Scenario 4 regime probabilities", callout="The explanation is not only higher returns; it is a move away from acute April 2020 financial stress."), "Final states move from the stress anchor into contraction/expansion probability mass.")
    add_record(records, spec.out_dir, "s4_macro_density_grid", plot_macro_density_grid(spec, sample, historical, out_dir), "Generated macro densities show where the counterfactual distribution moves relative to history.")
    add_record(records, spec.out_dir, "s4_nfci_analogs_test_window", plot_nfci_analogs(spec, final, historical, out_dir, window="test"), "Readable test-window NFCI analogs place Scenario 4 states around recent history.")
    add_record(records, spec.out_dir, "s4_nfci_analogs_full_history_appendix", plot_nfci_analogs(spec, final, historical, out_dir, window="full"), "Full-history NFCI analogs provide appendix context.")
    add_record(records, spec.out_dir, "s4_macro_embedding_pca_tsne", plot_macro_embedding_pca_tsne(spec, final, sample, historical, out_dir, callout="PCA shows the stress anchor and the final contraction/expansion cloud in historical macro space."), "PCA/t-SNE maps show where final states sit relative to historical regime clouds.")
    add_record(records, spec.out_dir, "s4_regime_class_cloud_pca_tsne", plot_regime_class_cloud_pca_tsne(spec, final, sample, historical, out_dir), "Regime class clouds show Scenario 4 final states relative to historical regime labels.")
    add_record(records, spec.out_dir, "s4_economic_mechanism", plot_s4_economic_mechanism(final, out_dir), "SC wins through stress escape rather than arbitrary macro movement.")
    add_record(records, spec.out_dir, "s4_manuscript_summary_card", plot_manuscript_summary_card(spec, final, sample, run_dir, out_dir, story="s4"), "One-page card states the question, G function, anchor, generated scenario, and economic interpretation.")
    notes = {
        "anchor_metric": "April 2020 financial-stress anchor; Scenario 4 asks SummerChild to beat WinterWolf.",
        "final_metric": f"Final win rate {(final['return_gap'] > 0).mean():.1%}; median SC-WW return gap {100*final['return_gap'].median():+.2f}pp.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final_box_violation(final):.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "analogs": f"Test-window analogs: {top_analog_text(out_dir, 's4_top5_neighbors_test.csv')}; full-history analogs: {top_analog_text(out_dir, 's4_top5_neighbors_full.csv')}.",
        "caution": "Nearest analogs are standardized-Euclidean historical analogs, not formal plausibility proof.",
        "economic": "This scenario gives an economic mechanism: SC beats WW when the generated distribution exits the acute financial-stress corner of April 2020.",
    }
    return records, notes


def run_e2e(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    anchor = infer_anchor_from_run_dir(run_dir, E2E_SPEC.anchor)
    spec = replace(E2E_SPEC, anchor=anchor, out_dir=f"e2e_diversification_{anchor}")
    out_dir = out_root / spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical, anchor_weights, final_weights = load_scenario5(run_dir)
    records: list[FigureRecord] = []
    add_record(records, spec.out_dir, "e2e_concentration_before_after", plot_e2e_concentration(anchor_weights, final_weights, out_dir), "Largest anchor holdings shrink and the concentration curve flattens.")
    add_record(records, spec.out_dir, "e2e_diversification_metrics", plot_e2e_metrics(final, out_dir), "Entropy and effective N rise while HHI and large-position concentration fall.")
    add_record(records, spec.out_dir, "e2e_regime_probability_story", plot_regime_probability_story(spec, final, sample, out_dir, title="E2E diversification regime probabilities", callout="Diversification is strongest around a mixed stress/contraction boundary rather than a single arbitrary point."), "Regime probabilities show whether diversification stays in stress or crosses into contraction/expansion.")
    add_record(records, spec.out_dir, "e2e_macro_density_grid", plot_macro_density_grid(spec, sample, historical, out_dir), "Generated macro densities show the distributional movement behind diversification.")
    add_record(records, spec.out_dir, "e2e_nfci_analogs_test_window", plot_nfci_analogs(spec, final, historical, out_dir, window="test"), "Readable test-window NFCI analogs place diversification states around recent history.")
    add_record(records, spec.out_dir, "e2e_nfci_analogs_full_history_appendix", plot_nfci_analogs(spec, final, historical, out_dir, window="full"), "Full-history NFCI analogs provide appendix context.")
    add_record(records, spec.out_dir, "e2e_macro_embedding_pca_tsne", plot_macro_embedding_pca_tsne(spec, final, sample, historical, out_dir, callout="The generated states form a local stress/contraction neighborhood while the portfolio deconcentrates."), "PCA/t-SNE maps show diversification states relative to historical regime clouds.")
    add_record(records, spec.out_dir, "e2e_regime_class_cloud_pca_tsne", plot_regime_class_cloud_pca_tsne(spec, final, sample, historical, out_dir), "Regime class clouds show E2E diversification states relative to historical regime labels.")
    add_record(records, spec.out_dir, "e2e_economic_mechanism", plot_e2e_economic_mechanism(final, out_dir), "Diversification changes are linked to regime probability, not only optimized target movement.")
    add_record(records, spec.out_dir, "e2e_manuscript_summary_card", plot_manuscript_summary_card(spec, final, sample, run_dir, out_dir, story="e2e"), "One-page card states the question, G function, anchor, generated scenario, and economic interpretation.")
    notes = {
        "anchor_metric": f"Anchor entropy {final['anchor_entropy'].iloc[0]:.3f}; HHI {final['anchor_hhi'].iloc[0]:.3f}; max weight {100*final['anchor_max_weight'].iloc[0]:.1f}%.",
        "final_metric": f"Median entropy change {final['delta_entropy'].median():+.3f}; HHI change {final['delta_hhi'].median():+.3f}; max-weight change {100*final['delta_max_weight'].median():+.2f}pp.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final_box_violation(final):.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "analogs": f"Test-window analogs: {top_analog_text(out_dir, 'e2e_top5_neighbors_test.csv')}; full-history analogs: {top_analog_text(out_dir, 'e2e_top5_neighbors_full.csv')}.",
        "caution": f"Empirical-anchor tail median {final['anchor_empirical_mah_chi2_tail'].median():.3f}; VAR-anchor tail median {final['anchor_mah_chi2_tail'].median():.2e}. Frame as empirically local, not VAR(1)-innovation plausible.",
        "economic": f"The mechanism is concentration sensitivity around the {anchor} macro state: nearby generated macro states reduce extreme single-name exposure and raise effective diversification.",
    }
    return records, notes


def run_pto(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    spec = PTO_SPEC
    out_dir = out_root / spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical, anchor_weights, final_weights = load_scenario5(run_dir)
    records: list[FigureRecord] = []
    add_record(records, spec.out_dir, "pto_return_gap_reversal", plot_pto_return_gap(final, anchor_weights, out_dir), "PTO moves from behind locked E2E at anchor to ahead in generated final states.")
    add_record(records, spec.out_dir, "pto_portfolio_mechanism", plot_pto_portfolio(anchor_weights, final_weights, final, out_dir), "PTO wins while remaining more concentrated than locked E2E.")
    add_record(records, spec.out_dir, "pto_regime_probability_story", plot_regime_probability_story(spec, final, sample, out_dir, title="PTO catch-up regime probabilities", callout="Unlike Scenario 4, PTO catch-up does not escape stress; it identifies a within-stress macro direction."), "Regime probabilities show PTO catch-up remains inside financial stress.")
    add_record(records, spec.out_dir, "pto_macro_density_grid", plot_macro_density_grid(spec, sample, historical, out_dir), "Generated macro densities show the within-stress distributional movement behind PTO catch-up.")
    add_record(records, spec.out_dir, "pto_nfci_analogs_test_window", plot_nfci_analogs(spec, final, historical, out_dir, window="test"), "Readable test-window NFCI analogs place PTO states around recent stress history.")
    add_record(records, spec.out_dir, "pto_nfci_analogs_full_history_appendix", plot_nfci_analogs(spec, final, historical, out_dir, window="full"), "Full-history NFCI analogs provide appendix stress analogs.")
    add_record(records, spec.out_dir, "pto_macro_embedding_pca_tsne", plot_macro_embedding_pca_tsne(spec, final, sample, historical, out_dir, callout="Final states remain in the stress cloud while PTO overtakes locked E2E."), "PCA/t-SNE maps show PTO catch-up occurs within the financial-stress regime cloud.")
    add_record(records, spec.out_dir, "pto_regime_class_cloud_pca_tsne", plot_regime_class_cloud_pca_tsne(spec, final, sample, historical, out_dir), "Regime class clouds show PTO catch-up states remain in the historical stress class neighborhood.")
    add_record(records, spec.out_dir, "pto_economic_mechanism", plot_pto_economic_mechanism(final, out_dir), "PTO overtakes through an empirical-local within-stress deviation.")
    add_record(records, spec.out_dir, "pto_manuscript_summary_card", plot_manuscript_summary_card(spec, final, sample, run_dir, out_dir, story="pto"), "One-page card states the question, G function, anchor, generated scenario, and economic interpretation.")
    notes = {
        "anchor_metric": f"Anchor locked E2E minus PTO return gap {100*final['anchor_return_gap_a_minus_b'].iloc[0]:+.2f}pp.",
        "final_metric": f"Final median locked E2E minus PTO gap {100*final['return_gap_a_minus_b'].median():+.2f}pp; PTO win share {final['b_return_matches_or_beats_a'].mean():.1%}.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final_box_violation(final):.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "analogs": f"Test-window analogs: {top_analog_text(out_dir, 'pto_top5_neighbors_test.csv')}; full-history analogs: {top_analog_text(out_dir, 'pto_top5_neighbors_full.csv')}.",
        "caution": f"Empirical-anchor tail median {final['anchor_empirical_mah_chi2_tail'].median():.3f}; VAR-anchor tail median {final['anchor_mah_chi2_tail'].median():.2e}. Frame as empirically local, not VAR(1)-innovation plausible.",
        "economic": "The mechanism is not regime escape: PTO catches up inside financial stress, showing that pipeline rankings can reverse along economically interpretable stress-state directions.",
    }
    return records, notes


def write_manifest(records: list[FigureRecord], notes: dict[str, dict[str, str]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.__dict__ for record in records]).to_csv(out_root / "figure_manifest.csv", index=False)
    lines = ["# Story Figures V3 Notes", ""]
    for scenario, payload in notes.items():
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"- Anchor target metric: {payload['anchor_metric']}",
                f"- Final target metric: {payload['final_metric']}",
                f"- Acceptance/constraints: {payload['acceptance']}",
                f"- Regime movement: {payload['regime']}",
                f"- Nearest analogs: {payload['analogs']}",
                f"- Locality caution: {payload['caution']}",
                f"- Economic interpretation: {payload['economic']}",
                "- Embedding caution: PCA is the interpretable geometry; t-SNE is only a visual neighborhood diagnostic.",
                "",
            ]
        )
    (out_root / "figure_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v3 story-first paper figures for the selected scenario runs.")
    parser.add_argument("--scenario", choices=["all", "s4", "e2e", "pto"], default="all")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--s4-run-dir", type=Path, default=S4_RUN)
    parser.add_argument("--e2e-run-dir", type=Path, default=E2E_RUN)
    parser.add_argument("--pto-run-dir", type=Path, default=PTO_RUN)
    args = parser.parse_args()

    args.out_root = args.out_root.resolve()
    args.s4_run_dir = args.s4_run_dir.resolve()
    args.e2e_run_dir = args.e2e_run_dir.resolve()
    args.pto_run_dir = args.pto_run_dir.resolve()

    records: list[FigureRecord] = []
    notes: dict[str, dict[str, str]] = {}
    if args.scenario in {"all", "s4"}:
        recs, note = run_s4(args.s4_run_dir, args.out_root)
        records.extend(recs)
        notes[S4_SPEC.out_dir] = note
    if args.scenario in {"all", "e2e"}:
        recs, note = run_e2e(args.e2e_run_dir, args.out_root)
        records.extend(recs)
        e2e_anchor = infer_anchor_from_run_dir(args.e2e_run_dir, E2E_SPEC.anchor)
        notes[f"e2e_diversification_{e2e_anchor}"] = note
    if args.scenario in {"all", "pto"}:
        recs, note = run_pto(args.pto_run_dir, args.out_root)
        records.extend(recs)
        notes[PTO_SPEC.out_dir] = note

    write_manifest(records, notes, args.out_root)
    print(f"Wrote {len(records)} figures -> {args.out_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
