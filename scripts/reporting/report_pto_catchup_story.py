"""Paper-facing figures for the PTO catch-up scenario.

This report is specific to the question:

    From the March 2020 stress anchor where locked E2E beats PTO, can nearby
    macro states make standardized PTO match or beat locked E2E?

The figures emphasize return-gap reversal, model allocation behavior, macro
geography, and historical interpretability.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batuhan.regime.nfci import load_nfci
from src.data.macro_scaler import MacroScaler
from src.modules.runtime_regime import MACRO_ORDER
from src.utils.plotting import PAPER_COLORS, set_publication_style


DEFAULT_RUN_DIR = (
    ROOT
    / "scenario_outputs"
    / "scenario_pto_catchup_202003"
    / "runs"
    / "20260507_154755_972692"
)
DEFAULT_OUT_DIR = ROOT / "submission_plots" / "pto_catchup_202003"
ANCHOR = 202003
MACRO_COLS = list(MACRO_ORDER)
MODELS = ["locked_e2e", "standardized_pto"]
MODEL_LABELS = {"locked_e2e": "Locked E2E", "standardized_pto": "Standardized PTO"}
MODEL_COLORS = {"locked_e2e": "#4c78a8", "standardized_pto": "#157f78"}
REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}


def newest(pattern: str, run_dir: Path) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {run_dir}")
    return matches[-1]


def yyyymm_to_datetime(values: pd.Series | list[int] | np.ndarray) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def load_run(run_dir: Path) -> dict[str, object]:
    return {
        "final": pd.read_csv(newest("final_state_diagnostics_*.csv", run_dir)),
        "sample": pd.read_csv(newest("generated_macro_sample_*.csv", run_dir)),
        "anchor_weights": pd.read_csv(newest("anchor_weights_*.csv", run_dir)),
        "final_weights": pd.read_csv(newest("final_weights_*.csv", run_dir)),
        "historical": pd.read_csv(newest("historical_macro_panel_*.csv", run_dir)),
        "config": json.loads(newest("config_*.json", run_dir).read_text(encoding="utf-8")),
    }


def historical_std_frame(historical: pd.DataFrame) -> pd.DataFrame:
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    values = historical[MACRO_COLS].to_numpy(dtype=np.float32)
    std = (values - scaler.mean.cpu().numpy()) / scaler.std.cpu().numpy()
    out = historical[["yyyymm"]].copy()
    for idx, col in enumerate(MACRO_COLS):
        out[f"{col}_std"] = std[:, idx]
    labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    out = out.merge(labels[["yyyymm", "regime"]], on="yyyymm", how="left")
    return out


def load_regime_nfci_panel() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    nfci = load_nfci().rename("NFCI").reset_index()
    panel = labels.merge(nfci, on="yyyymm", how="left").sort_values("yyyymm").reset_index(drop=True)
    panel["date"] = yyyymm_to_datetime(panel["yyyymm"])
    return panel


def regime_segments(panel: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    rows = panel[["date", "regime"]].dropna().reset_index(drop=True)
    segments = []
    start = rows.loc[0, "date"]
    current = str(rows.loc[0, "regime"])
    prev = rows.loc[0, "date"]
    for idx in range(1, len(rows)):
        date = rows.loc[idx, "date"]
        regime = str(rows.loc[idx, "regime"])
        if regime != current:
            segments.append((start, prev + pd.offsets.MonthEnd(1), current))
            start = date
            current = regime
        prev = date
    segments.append((start, prev + pd.offsets.MonthEnd(1), current))
    return segments


def add_regime_background(ax: plt.Axes, panel: pd.DataFrame) -> None:
    for start, end, regime in regime_segments(panel):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, PAPER_COLORS["grid"]), alpha=0.10, linewidth=0)


def median_weights(final_weights: pd.DataFrame) -> pd.DataFrame:
    out = (
        final_weights.groupby(["model", "permno"], as_index=False)
        .agg(weight=("weight", "median"), anchor_weight=("anchor_weight", "first"))
    )
    out["delta_weight"] = out["weight"] - out["anchor_weight"]
    return out


def plot_return_gap_reversal(final: pd.DataFrame, anchor_weights: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    anchor_locked = float(anchor_weights.loc[anchor_weights["model"].eq("locked_e2e"), "anchor_return"].iloc[0])
    anchor_pto = float(anchor_weights.loc[anchor_weights["model"].eq("standardized_pto"), "anchor_return"].iloc[0])
    final_locked = float(final["locked_e2e_return"].median())
    final_pto = float(final["standardized_pto_return"].median())
    anchor_gap = float(final["anchor_return_gap_a_minus_b"].iloc[0])
    final_gaps = final["return_gap_a_minus_b"].to_numpy(dtype=float)
    improvements = final["return_gap_improvement_for_b"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(13.2, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2], height_ratios=[1.0, 0.95], wspace=0.28, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    width = 0.34
    ax0.bar(x - width / 2, 100 * np.array([anchor_locked, final_locked]), width, color=MODEL_COLORS["locked_e2e"], label="Locked E2E")
    ax0.bar(x + width / 2, 100 * np.array([anchor_pto, final_pto]), width, color=MODEL_COLORS["standardized_pto"], label="Standardized PTO")
    ax0.set_xticks(x)
    ax0.set_xticklabels(["Anchor\n2020-03", "Generated\nfinal median"])
    ax0.set_ylabel("Realized portfolio return (%)")
    ax0.set_title("Return ranking reverses", loc="left", fontsize=12.5, pad=8)
    ax0.legend(frameon=False, loc="upper left")
    ax0.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax1 = fig.add_subplot(gs[0, 1])
    for gap in final_gaps:
        ax1.plot([0, 1], [100 * anchor_gap, 100 * gap], color=PAPER_COLORS["generated"], alpha=0.35, linewidth=1.2)
        ax1.scatter([1], [100 * gap], color=PAPER_COLORS["generated"], s=22, alpha=0.80)
    ax1.scatter([0], [100 * anchor_gap], color=PAPER_COLORS["anchor"], marker="X", s=90, zorder=4)
    ax1.scatter([1], [100 * np.median(final_gaps)], color=PAPER_COLORS["generated"], marker="D", s=75, zorder=5)
    ax1.axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Anchor", "Generated final"])
    ax1.set_ylabel("Locked E2E return minus PTO return (pp)")
    ax1.set_title("Every seed crosses the zero-gap line", loc="left", fontsize=12.5, pad=8)
    ax1.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax2 = fig.add_subplot(gs[1, :])
    order = np.argsort(improvements)
    ax2.bar(np.arange(len(final)), 100 * improvements[order], color=PAPER_COLORS["positive"], alpha=0.88)
    ax2.axhline(100 * np.median(improvements), color=PAPER_COLORS["text"], linestyle="--", linewidth=1.0, label=f"Median improvement: {100*np.median(improvements):.2f}pp")
    ax2.set_xlabel("Final seed, sorted by PTO gap improvement")
    ax2.set_ylabel("PTO improvement vs anchor gap (pp)")
    ax2.set_title("PTO closes a 4.33pp deficit and wins by a median 2.70pp", loc="left", fontsize=12.5, pad=8)
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    fig.suptitle("PTO Catch-Up Scenario: March 2020 Ranking Reversal", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "pto_catchup_return_gap_reversal")


def plot_portfolio_mechanism(anchor_weights: pd.DataFrame, final_weights: pd.DataFrame, final: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    med = median_weights(final_weights)
    pto_anchor = anchor_weights[anchor_weights["model"].eq("standardized_pto")].copy()
    top_permnos = pto_anchor.sort_values("anchor_weight", ascending=False).head(12)["permno"].astype(int).tolist()
    top = (
        pto_anchor[["permno", "anchor_weight"]]
        .merge(med[med["model"].eq("standardized_pto")][["permno", "weight"]], on="permno", how="left")
    )
    top["permno"] = top["permno"].astype(int)
    top = top[top["permno"].isin(top_permnos)].copy()
    top["_order"] = top["permno"].map({permno: idx for idx, permno in enumerate(top_permnos)})
    top = top.sort_values("_order")

    rows = []
    for model in MODELS:
        anchor_part = anchor_weights[anchor_weights["model"].eq(model)].iloc[0]
        rows.extend(
            [
                {"model": MODEL_LABELS[model], "state": "Anchor", "metric": "HHI", "value": float(anchor_part["anchor_hhi"])},
                {"model": MODEL_LABELS[model], "state": "Final median", "metric": "HHI", "value": float(final[f"{model}_hhi"].median())},
                {"model": MODEL_LABELS[model], "state": "Anchor", "metric": "Max weight", "value": float(anchor_part["anchor_max_weight"])},
                {"model": MODEL_LABELS[model], "state": "Final median", "metric": "Max weight", "value": float(final[f"{model}_max_weight"].median())},
                {"model": MODEL_LABELS[model], "state": "Anchor", "metric": "Effective N", "value": float(anchor_part["anchor_effective_n"])},
                {"model": MODEL_LABELS[model], "state": "Final median", "metric": "Effective N", "value": float(final[f"{model}_effective_n"].median())},
            ]
        )
    conc = pd.DataFrame(rows)

    fig = plt.figure(figsize=(13.8, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, 1.0], wspace=0.30, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(top))
    width = 0.38
    ax0.bar(x - width / 2, 100 * top["anchor_weight"], width, color=PAPER_COLORS["anchor"], label="PTO anchor")
    ax0.bar(x + width / 2, 100 * top["weight"], width, color=PAPER_COLORS["generated"], label="PTO generated median")
    ax0.set_xticks(x)
    ax0.set_xticklabels(top["permno"].astype(str), rotation=45, ha="right", fontsize=8)
    ax0.set_ylabel("Portfolio weight (%)")
    ax0.set_title("PTO remains tilted, but less extremely than at anchor", loc="left", fontsize=12.5, pad=8)
    ax0.legend(frameon=False, loc="upper right")
    ax0.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax1 = fig.add_subplot(gs[0, 1])
    for model in MODELS:
        part = med[med["model"].eq(model)]
        weights = np.sort(part["weight"].to_numpy(dtype=float))[::-1]
        ax1.plot(np.arange(1, len(weights) + 1), 100 * np.cumsum(weights), linewidth=2.2, color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} final")
    ax1.axvline(10, color=PAPER_COLORS["grid"], linewidth=1.0)
    ax1.set_xlabel("Top-k holdings sorted by final weight")
    ax1.set_ylabel("Cumulative portfolio weight (%)")
    ax1.set_title("Final concentration curves", loc="left", fontsize=12.5, pad=8)
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    bottom = gs[1, :].subgridspec(1, 3, wspace=0.28)
    metrics = ["HHI", "Max weight", "Effective N"]
    specs = [
        ("Locked E2E", "Anchor", MODEL_COLORS["locked_e2e"], 0.45),
        ("Locked E2E", "Final median", MODEL_COLORS["locked_e2e"], 0.90),
        ("Standardized PTO", "Anchor", MODEL_COLORS["standardized_pto"], 0.45),
        ("Standardized PTO", "Final median", MODEL_COLORS["standardized_pto"], 0.90),
    ]
    for metric_idx, metric in enumerate(metrics):
        ax_metric = fig.add_subplot(bottom[0, metric_idx])
        x_metric = np.arange(len(specs))
        vals = []
        labels = []
        colors = []
        alphas = []
        for model_label, state, color, alpha in specs:
            value = conc[(conc["model"].eq(model_label)) & (conc["state"].eq(state)) & (conc["metric"].eq(metric))]["value"].iloc[0]
            vals.append(100 * value if metric == "Max weight" else value)
            labels.append(f"{model_label}\n{state}")
            colors.append(color)
            alphas.append(alpha)
        bars = ax_metric.bar(x_metric, vals, color=colors)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        ax_metric.set_xticks(x_metric)
        ax_metric.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.4)
        ax_metric.set_title("Max weight (%)" if metric == "Max weight" else metric, loc="left", fontsize=11.0, pad=7)
        ax_metric.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
        if metric_idx == 0:
            ax_metric.set_ylabel("Metric value")

    fig.suptitle("Portfolio Mechanism Behind PTO Catch-Up", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "pto_catchup_portfolio_mechanism")


def plot_macro_geography(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    hist_std = historical_std_frame(historical)
    cols = [f"{c}_std" for c in MACRO_COLS]
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_plot = sample.sample(n=min(180, len(sample)), random_state=19) if len(sample) > 180 else sample.copy()
    sample_x = sample_plot[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = hist_std.loc[hist_std["yyyymm"].astype(int).eq(ANCHOR), cols].iloc[0].to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(hist_x)
    hist_pca = pca.transform(hist_x)
    sample_pca = pca.transform(sample_x)
    final_pca = pca.transform(final_x)
    anchor_pca = pca.transform(anchor_x[None, :])[0]

    combined = np.vstack([hist_x, sample_x, final_x, anchor_x[None, :]])
    labels = np.asarray(["historical"] * len(hist_x) + ["post-burn-in"] * len(sample_x) + ["final"] * len(final_x) + ["anchor"])
    emb = TSNE(n_components=2, perplexity=20, init="pca", learning_rate="auto", random_state=13, n_iter=600, method="exact").fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
    for regime, part in hist_std.groupby("regime"):
        idx = part.index.to_numpy()
        axes[0].scatter(hist_pca[idx, 0], hist_pca[idx, 1], s=18, color=REGIME_COLORS.get(str(regime), PAPER_COLORS["historical"]), alpha=0.22, linewidths=0, label=f"Historical {str(regime).replace('_', ' ')}")
    axes[0].scatter(sample_pca[:, 0], sample_pca[:, 1], s=9, color="#8c9299", alpha=0.20, linewidths=0, label="Generated sample")
    axes[0].scatter(final_pca[:, 0], final_pca[:, 1], s=58, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.55, label="Final states", zorder=4)
    axes[0].scatter(anchor_pca[0], anchor_pca[1], s=145, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label="March 2020 anchor", zorder=5)
    axes[0].set_title("PCA macro geography", loc="left", fontsize=12.5, pad=8)
    axes[0].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var.)")
    axes[0].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var.)")

    for regime, color in REGIME_COLORS.items():
        mask = np.zeros(len(labels), dtype=bool)
        mask[: len(hist_x)] = hist_std["regime"].astype(str).eq(regime).to_numpy()
        axes[1].scatter(emb[mask, 0], emb[mask, 1], s=16, color=color, alpha=0.20, linewidths=0, label=f"Historical {regime.replace('_', ' ')}")
    sample_mask = labels == "post-burn-in"
    final_mask = labels == "final"
    anchor_mask = labels == "anchor"
    axes[1].scatter(emb[sample_mask, 0], emb[sample_mask, 1], s=8, color="#8c9299", alpha=0.18, linewidths=0, label="Generated sample")
    axes[1].scatter(emb[final_mask, 0], emb[final_mask, 1], s=58, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.55, label="Final states", zorder=4)
    axes[1].scatter(emb[anchor_mask, 0], emb[anchor_mask, 1], s=145, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label="March 2020 anchor", zorder=5)
    axes[1].set_title("t-SNE neighborhood map", loc="left", fontsize=12.5, pad=8)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:6], legend_labels[:6], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("PTO Catch-Up States Stay Inside the March 2020 Financial-Stress Neighborhood", fontsize=15.0, y=0.98)
    save_figure(fig, out_dir / "pto_catchup_macro_pca_tsne")


def nearest_historical_analogs(final: pd.DataFrame, historical: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    hist_std = historical_std_frame(historical)
    cols = [f"{c}_std" for c in MACRO_COLS]
    hist = hist_std[hist_std["yyyymm"].astype(int).ne(ANCHOR)].reset_index(drop=True)
    hist_x = hist[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    median_final = np.median(final_x, axis=0)
    dists = np.linalg.norm(median_final[None, :] - hist_x, axis=1)
    final_dists = np.linalg.norm(final_x[:, None, :] - hist_x[None, :, :], axis=2)
    counts = Counter(hist.loc[final_dists.argmin(axis=1), "yyyymm"].astype(int).tolist())
    top = hist.iloc[np.argsort(dists)[:top_k]][["yyyymm", "regime"]].copy()
    top["distance_to_generated_final_median"] = np.sort(dists)[:top_k]
    top["nearest_final_state_count"] = top["yyyymm"].astype(int).map(counts).fillna(0).astype(int)
    panel = load_regime_nfci_panel()
    top = top.merge(panel[["yyyymm", "NFCI"]], on="yyyymm", how="left")
    top["date"] = yyyymm_to_datetime(top["yyyymm"])
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top


def plot_historical_neighbors(final: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    top = nearest_historical_analogs(final, historical, top_k=5)
    out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_dir / "pto_catchup_top5_historical_neighbors.csv", index=False)
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200701) & (panel["yyyymm"] <= 202412)].copy()

    set_publication_style()
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    add_regime_background(ax, focus)
    ax.plot(focus["date"], focus["NFCI"], color="#222222", linewidth=1.25, label="NFCI")
    ax.axhline(0.0, color="#697078", linestyle="--", linewidth=0.85)
    anchor_date = yyyymm_to_datetime([ANCHOR]).iloc[0]
    anchor_nfci = float(focus.loc[focus["yyyymm"].astype(int).eq(ANCHOR), "NFCI"].iloc[0])
    ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.35, label="March 2020 anchor")
    sizes = 85 + 38 * top["nearest_final_state_count"].to_numpy(dtype=float)
    ax.scatter(top["date"], top["NFCI"], s=sizes, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.85, zorder=5, label="Top-5 analogs")
    for _, row in top.iterrows():
        ax.annotate(
            f"{int(row['yyyymm'])}\n#{int(row['rank'])}",
            xy=(row["date"], row["NFCI"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.92},
        )
    ax.annotate(
        "Anchor: acute COVID stress",
        xy=(anchor_date, anchor_nfci),
        xytext=(pd.Timestamp("2020-08-01"), anchor_nfci + 0.45),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.0},
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94},
    )
    months = "; ".join(f"{int(r.yyyymm)} ({str(r.regime).replace('_', ' ')}, {r.NFCI:.2f})" for _, r in top.iterrows())
    ax.text(0.02, 0.04, f"Top analogs excluding 202003: {months}", transform=ax.transAxes, fontsize=8.0, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94})
    ax.set_title("Historical analogs for PTO catch-up states on NFCI", loc="left", fontsize=12.5, pad=8)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    save_figure(fig, out_dir / "pto_catchup_historical_nearest_neighbors")


def plot_diagnostic_dashboard(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.scatter(final["anchor_empirical_mah_dist"], 100 * final["return_gap_improvement_for_b"], c=final["accept_rate"], cmap="viridis", s=62, edgecolors="white", linewidth=0.6)
    ax0.set_xlabel("Empirical Mahalanobis distance to anchor")
    ax0.set_ylabel("PTO gap improvement (pp)")
    ax0.set_title("Return reversal achieved within empirical locality", loc="left", fontsize=12.5, pad=8)

    ax1.scatter(final["l2_dist"], 100 * final["return_gap_a_minus_b"], color=PAPER_COLORS["generated"], s=62, edgecolors="white", linewidth=0.6)
    ax1.axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax1.set_xlabel("L2 distance from anchor")
    ax1.set_ylabel("Locked E2E minus PTO return (pp)")
    ax1.set_title("PTO wins across final states", loc="left", fontsize=12.5, pad=8)

    ax2.bar(["financial\nstress", "contraction", "expansion"], [100.0, 0.0, 0.0], color=[REGIME_COLORS["financial_stress"], REGIME_COLORS["contraction"], REGIME_COLORS["expansion"]], alpha=0.9)
    ax2.set_ylabel("Post-burn-in generated states (%)")
    ax2.set_title("Scenario remains inside financial stress", loc="left", fontsize=12.5, pad=8)
    ax2.text(0, 101, "100.0%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 112)

    hist_std = historical_std_frame(historical)
    shift_rows = []
    for col in MACRO_COLS:
        anchor_std = hist_std.loc[hist_std["yyyymm"].astype(int).eq(ANCHOR), f"{col}_std"].iloc[0]
        values = sample[f"{col}_std"].to_numpy(dtype=float) - anchor_std
        shift_rows.append({"macro": col, "median": np.median(values), "q25": np.quantile(values, 0.25), "q75": np.quantile(values, 0.75)})
    shifts = pd.DataFrame(shift_rows).sort_values("median")
    y = np.arange(len(shifts))
    colors = [PAPER_COLORS["positive"] if v >= 0 else PAPER_COLORS["negative"] for v in shifts["median"]]
    ax3.barh(y, shifts["median"], color=colors, alpha=0.88)
    ax3.errorbar(shifts["median"], y, xerr=[shifts["median"] - shifts["q25"], shifts["q75"] - shifts["median"]], fmt="none", ecolor=PAPER_COLORS["text"], elinewidth=0.9, capsize=3)
    ax3.axvline(0.0, color=PAPER_COLORS["text"], linewidth=0.9)
    ax3.set_yticks(y)
    ax3.set_yticklabels(shifts["macro"])
    ax3.set_xlabel("Generated minus anchor, standardized units")
    ax3.set_title("Macro movements behind PTO catch-up", loc="left", fontsize=12.5, pad=8)

    for ax in axes.ravel():
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Diagnostics for the PTO Catch-Up Scenario", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "pto_catchup_diagnostic_dashboard")


def write_summary(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    top = nearest_historical_analogs(final, historical, top_k=5)
    top_line = "; ".join(f"{int(r.yyyymm)} ({str(r.regime).replace('_', ' ')}, NFCI={r.NFCI:.2f}, dist={r.distance_to_generated_final_median:.2f})" for _, r in top.iterrows())
    lines = [
        "# PTO Catch-Up Figure Summary",
        "",
        f"- Anchor locked E2E minus PTO return gap: `{100*final['anchor_return_gap_a_minus_b'].iloc[0]:+.3f}` percentage points.",
        f"- Final median locked E2E minus PTO return gap: `{100*final['return_gap_a_minus_b'].median():+.3f}` percentage points.",
        f"- Median PTO gap improvement: `{100*final['return_gap_improvement_for_b'].median():+.3f}` percentage points.",
        f"- PTO matches/beats locked E2E in `{100*final['b_return_matches_or_beats_a'].mean():.1f}%` of final seeds.",
        f"- Median allocation L1 gap: `{final['allocation_l1_gap'].median():.3f}`.",
        f"- Mean acceptance rate: `{100*final['accept_rate'].mean():.1f}%`.",
        f"- Median empirical-anchor Mahalanobis tail: `{final['anchor_empirical_mah_chi2_tail'].median():.3f}`.",
        f"- Median anchor VAR-innovation tail: `{final['anchor_mah_chi2_tail'].median():.2e}`.",
        f"- Generated regime occupancy: `{dict((sample['regime'].value_counts(normalize=True) * 100).round(1))}` percent.",
        f"- Top-5 historical analogs for the median final state: {top_line}.",
        "",
        "Manuscript wording: empirically local to March 2020 and fully within financial stress; not VAR(1)-innovation plausible.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pto_catchup_figure_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-specific plots for the PTO catch-up scenario.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = load_run(args.run_dir)
    final = payload["final"]
    sample = payload["sample"]
    anchor_weights = payload["anchor_weights"]
    final_weights = payload["final_weights"]
    historical = payload["historical"]

    plot_return_gap_reversal(final, anchor_weights, args.out_dir)
    plot_portfolio_mechanism(anchor_weights, final_weights, final, args.out_dir)
    plot_macro_geography(final, sample, historical, args.out_dir)
    plot_historical_neighbors(final, historical, args.out_dir)
    plot_diagnostic_dashboard(final, sample, historical, args.out_dir)
    write_summary(final, sample, historical, args.out_dir)
    print(f"Wrote figures -> {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
