"""Paper-facing figures for the locked E2E diversification scenario.

This report is specific to the question:

    From the April 2020 stress anchor, can nearby macro states make the locked
    E2E policy move from a concentrated portfolio to a more diversified one?

The figures intentionally emphasize portfolio concentration and macro geography,
not generic Scenario 5 diagnostics.
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
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
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
    / "scenario_e2e_diversify_202004"
    / "runs"
    / "20260507_130143_361581"
)
DEFAULT_OUT_DIR = ROOT / "submission_plots" / "e2e_diversification_202004"
MACRO_COLS = list(MACRO_ORDER)
ANCHOR = 202004
REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}


def yyyymm_to_datetime(values: pd.Series | list[int] | np.ndarray) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def newest(pattern: str, run_dir: Path) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {run_dir}")
    return matches[-1]


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def load_run(run_dir: Path) -> dict[str, object]:
    final = pd.read_csv(newest("final_state_diagnostics_*.csv", run_dir))
    sample = pd.read_csv(newest("generated_macro_sample_*.csv", run_dir))
    anchor_weights = pd.read_csv(newest("anchor_weights_*.csv", run_dir))
    final_weights = pd.read_csv(newest("final_weights_*.csv", run_dir))
    historical = pd.read_csv(newest("historical_macro_panel_*.csv", run_dir))
    config = json.loads(newest("config_*.json", run_dir).read_text(encoding="utf-8"))
    return {
        "final": final,
        "sample": sample,
        "anchor_weights": anchor_weights,
        "final_weights": final_weights,
        "historical": historical,
        "config": config,
    }


def median_final_weights(final_weights: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        final_weights.groupby("permno", as_index=False)
        .agg(weight=("weight", "median"), anchor_weight=("anchor_weight", "first"))
    )
    grouped["delta_weight"] = grouped["weight"] - grouped["anchor_weight"]
    return grouped


def concentration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "entropy_anchor": float(frame["anchor_entropy"].iloc[0]),
        "entropy_median": float(frame["locked_e2e_entropy"].median()),
        "hhi_anchor": float(frame["anchor_hhi"].iloc[0]),
        "hhi_median": float(frame["locked_e2e_hhi"].median()),
        "effective_n_anchor": float(frame["anchor_effective_n"].iloc[0]),
        "effective_n_median": float(frame["locked_e2e_effective_n"].median()),
        "max_weight_anchor": float(frame["anchor_max_weight"].iloc[0]),
        "max_weight_median": float(frame["locked_e2e_max_weight"].median()),
        "top10_anchor": float(frame["anchor_top10_weight"].iloc[0]),
        "top10_median": float(frame["locked_e2e_top10_weight"].median()),
    }


def plot_concentration_before_after(
    final: pd.DataFrame,
    anchor_weights: pd.DataFrame,
    final_weights: pd.DataFrame,
    out_dir: Path,
) -> None:
    set_publication_style()
    median_weights = median_final_weights(final_weights)
    top_permnos = (
        anchor_weights.sort_values("anchor_weight", ascending=False)
        .head(15)["permno"]
        .astype(int)
        .tolist()
    )
    top = (
        anchor_weights[["permno", "anchor_weight"]]
        .merge(median_weights[["permno", "weight"]], on="permno", how="left")
        .loc[lambda df: df["permno"].astype(int).isin(top_permnos)]
    )
    top["permno"] = top["permno"].astype(int)
    top["_order"] = top["permno"].map({permno: idx for idx, permno in enumerate(top_permnos)})
    top = top.sort_values("_order")

    anchor_sorted = np.sort(anchor_weights["anchor_weight"].to_numpy(dtype=float))[::-1]
    median_sorted = np.sort(median_weights["weight"].to_numpy(dtype=float))[::-1]
    metrics = concentration_metrics(final)

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.15, 1.0], wspace=0.28, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(top))
    width = 0.39
    ax0.bar(x - width / 2, 100.0 * top["anchor_weight"], width, color=PAPER_COLORS["anchor"], label="Anchor")
    ax0.bar(x + width / 2, 100.0 * top["weight"], width, color=PAPER_COLORS["generated"], label="Generated final median")
    ax0.set_xticks(x)
    ax0.set_xticklabels(top["permno"].astype(str), rotation=45, ha="right", fontsize=8)
    ax0.set_ylabel("Portfolio weight (%)")
    ax0.set_title("Largest anchor positions shrink in generated states", loc="left", fontsize=12.5, pad=8)
    ax0.legend(frameon=False, ncol=2, loc="upper right")
    ax0.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax1 = fig.add_subplot(gs[0, 1])
    k = np.arange(1, len(anchor_sorted) + 1)
    ax1.plot(k, 100.0 * np.cumsum(anchor_sorted), color=PAPER_COLORS["anchor"], linewidth=2.2, label="Anchor")
    ax1.plot(k, 100.0 * np.cumsum(median_sorted), color=PAPER_COLORS["generated"], linewidth=2.2, label="Generated final median")
    ax1.axvline(10, color=PAPER_COLORS["grid"], linewidth=1.0)
    ax1.set_xlabel("Top-k holdings sorted by weight")
    ax1.set_ylabel("Cumulative portfolio weight (%)")
    ax1.set_title("Concentration curve", loc="left", fontsize=12.5, pad=8)
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax2 = fig.add_subplot(gs[1, :])
    labels = ["Entropy", "HHI", "Effective N", "Max weight", "Top-10 weight"]
    changes = [
        100.0 * (metrics["entropy_median"] / metrics["entropy_anchor"] - 1.0),
        100.0 * (metrics["hhi_median"] / metrics["hhi_anchor"] - 1.0),
        100.0 * (metrics["effective_n_median"] / metrics["effective_n_anchor"] - 1.0),
        100.0 * (metrics["max_weight_median"] / metrics["max_weight_anchor"] - 1.0),
        100.0 * (metrics["top10_median"] / metrics["top10_anchor"] - 1.0),
    ]
    display = pd.DataFrame({"metric": labels, "Relative change": changes})
    positions = np.arange(len(labels))
    colors = [
        PAPER_COLORS["positive"] if metric in {"Entropy", "Effective N"} and value >= 0 else PAPER_COLORS["negative"]
        for metric, value in zip(labels, changes)
    ]
    colors = [
        PAPER_COLORS["positive"] if metric in {"HHI", "Max weight", "Top-10 weight"} and value <= 0 else color
        for metric, value, color in zip(labels, changes, colors)
    ]
    ax2.bar(positions, display["Relative change"], color=colors, alpha=0.88)
    ax2.axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax2.set_ylim(min(changes) - 8.0, max(changes) + 10.0)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Relative change (%)")
    ax2.set_title("Relative changes: diversification rises while concentration falls", loc="left", fontsize=12.5, pad=8)
    for x_pos, value in zip(positions, changes):
        va = "bottom" if value >= 0 else "top"
        y_text = value + (1.2 if value >= 0 else -1.2)
        ax2.text(x_pos, y_text, f"{value:+.1f}%", ha="center", va=va, fontsize=9)
    ax2.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Locked E2E Diversifies Away From the April 2020 Concentrated Anchor", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "e2e_diversification_concentration_before_after")


def plot_seed_level_diversification(final: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 5, figsize=(14.2, 4.2), sharex=False)
    specs = [
        ("locked_e2e_entropy", "anchor_entropy", "Entropy", ""),
        ("locked_e2e_hhi", "anchor_hhi", "HHI", ""),
        ("locked_e2e_effective_n", "anchor_effective_n", "Effective N", ""),
        ("locked_e2e_max_weight", "anchor_max_weight", "Max weight", "%"),
        ("locked_e2e_top10_weight", "anchor_top10_weight", "Top-10 weight", "%"),
    ]
    for ax, (final_col, anchor_col, title, unit) in zip(axes, specs):
        anchor = float(final[anchor_col].iloc[0])
        values = final[final_col].to_numpy(dtype=float)
        if unit == "%":
            anchor *= 100.0
            values = 100.0 * values
        for value in values:
            color = PAPER_COLORS["positive"] if title in {"Entropy", "Effective N"} and value >= anchor else PAPER_COLORS["negative"]
            if title in {"HHI", "Max weight", "Top-10 weight"}:
                color = PAPER_COLORS["positive"] if value <= anchor else PAPER_COLORS["negative"]
            ax.plot([0, 1], [anchor, value], color=color, alpha=0.45, linewidth=1.2)
            ax.scatter([1], [value], color=color, alpha=0.75, s=18)
        ax.scatter([0], [anchor], color=PAPER_COLORS["anchor"], s=45, marker="X", zorder=4)
        ax.scatter([1], [np.median(values)], color=PAPER_COLORS["generated"], s=55, marker="D", zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Anchor", "Final"], rotation=0)
        ax.set_title(title, fontsize=11, pad=7)
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Every Chain Reports the Same Economic Direction: More Diversified Final Portfolios", fontsize=14.0, y=1.02)
    save_figure(fig, out_dir / "e2e_diversification_seed_level_deltas")


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


def nearest_historical_summary(
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist_std = historical_std_frame(historical)
    hist_cols = [f"{col}_std" for col in MACRO_COLS]
    hist_x = hist_std[hist_cols].to_numpy(dtype=float)
    final_x = final[hist_cols].to_numpy(dtype=float)
    sample_x = sample[hist_cols].to_numpy(dtype=float)
    anchor_x = hist_std.loc[hist_std["yyyymm"].astype(int).eq(ANCHOR), hist_cols].iloc[0].to_numpy(dtype=float)

    eligible = hist_std["yyyymm"].astype(int).ne(ANCHOR).to_numpy()
    hist_x_ex = hist_x[eligible]
    hist_ex = hist_std.loc[eligible].reset_index(drop=True)

    final_dists = np.linalg.norm(final_x[:, None, :] - hist_x_ex[None, :, :], axis=2)
    sample_dists = np.linalg.norm(sample_x[:, None, :] - hist_x_ex[None, :, :], axis=2)
    anchor_dists = np.linalg.norm(anchor_x[None, :] - hist_x_ex, axis=1)
    anchor_nearest_dist = float(np.min(anchor_dists))
    median_final_x = np.median(final_x, axis=0)
    median_final_dists = np.linalg.norm(median_final_x[None, :] - hist_x_ex, axis=1)

    final_argmin = final_dists.argmin(axis=1)
    sample_argmin = sample_dists.argmin(axis=1)
    final_nearest_dist = final_dists.min(axis=1)
    sample_nearest_dist = sample_dists.min(axis=1)

    counts = Counter(hist_ex.loc[final_argmin, "yyyymm"].astype(int).tolist())
    top_rows = []
    for rank, hist_idx in enumerate(np.argsort(median_final_dists)[:top_k], start=1):
        yyyymm = int(hist_ex.loc[hist_idx, "yyyymm"])
        top_rows.append(
            {
                "rank": rank,
                "yyyymm": yyyymm,
                "distance_to_generated_final_median": float(median_final_dists[hist_idx]),
                "nearest_final_state_count": int(counts.get(yyyymm, 0)),
                "regime": str(hist_ex.loc[hist_idx, "regime"]),
            }
        )
    top = pd.DataFrame(top_rows)
    panel = load_regime_nfci_panel()
    top = top.merge(panel[["yyyymm", "NFCI"]], on="yyyymm", how="left")
    top["date"] = yyyymm_to_datetime(top["yyyymm"])

    distance_rows = []
    for label, values in [
        ("Generated final states", final_nearest_dist),
        ("Generated post-burn-in states", sample_nearest_dist),
        ("Anchor nearest analog", np.asarray([anchor_nearest_dist])),
    ]:
        distance_rows.append(
            {
                "distribution": label,
                "median_nearest_distance": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    distance_summary = pd.DataFrame(distance_rows)
    final_nn = final[["seed", "regime", "delta_entropy", "delta_hhi", "delta_effective_n"]].copy()
    final_nn["nearest_historical_yyyymm_ex_anchor"] = hist_ex.loc[final_argmin, "yyyymm"].astype(int).to_numpy()
    final_nn["nearest_historical_distance_ex_anchor"] = final_nearest_dist
    final_nn = final_nn.merge(
        panel[["yyyymm", "NFCI"]].rename(columns={"yyyymm": "nearest_historical_yyyymm_ex_anchor", "NFCI": "nearest_historical_NFCI"}),
        on="nearest_historical_yyyymm_ex_anchor",
        how="left",
    )
    return top, distance_summary, final_nn


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical multivariate energy distance."""
    xy = float(cdist(x, y).mean())
    xx = float(cdist(x, x).mean())
    yy = float(cdist(y, y).mean())
    return max(0.0, 2.0 * xy - xx - yy)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, *, n_projections: int = 256, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vals = [
        wasserstein_distance(x @ direction, y @ direction)
        for direction in directions
    ]
    return float(np.mean(vals))


def distribution_distance_summary(
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    n_boot: int = 400,
) -> pd.DataFrame:
    hist_std = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    hist = hist_std[hist_std["yyyymm"].astype(int).ne(ANCHOR)].reset_index(drop=True)
    hist_x = hist[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    sample_x = sample[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(23)

    observed = [
        {
            "comparison": "Final states vs history",
            "n_generated": len(final_x),
            "sliced_wasserstein": sliced_wasserstein(final_x, hist_x, seed=31),
            "energy_distance": energy_distance_multivariate(final_x, hist_x),
        },
        {
            "comparison": "Post-burn-in states vs history",
            "n_generated": len(sample_x),
            "sliced_wasserstein": sliced_wasserstein(sample_x, hist_x, seed=37),
            "energy_distance": energy_distance_multivariate(sample_x, hist_x),
        },
    ]

    boot_rows = []
    for n in sorted({len(final_x), len(sample_x)}):
        for idx in range(n_boot):
            draw_idx = rng.choice(len(hist_x), size=n, replace=False if n <= len(hist_x) else True)
            other_idx = rng.choice(len(hist_x), size=n, replace=False if n <= len(hist_x) else True)
            x = hist_x[draw_idx]
            y = hist_x[other_idx]
            boot_rows.append(
                {
                    "n_generated": n,
                    "sliced_wasserstein": sliced_wasserstein(x, y, seed=1000 + idx),
                    "energy_distance": energy_distance_multivariate(x, y),
                }
            )
    boot = pd.DataFrame(boot_rows)
    rows = []
    for item in observed:
        n = item["n_generated"]
        ref = boot[boot["n_generated"].eq(n)]
        row = dict(item)
        for metric in ["sliced_wasserstein", "energy_distance"]:
            row[f"{metric}_hist_p50"] = float(ref[metric].median())
            row[f"{metric}_hist_p90"] = float(ref[metric].quantile(0.90))
            row[f"{metric}_hist_percentile"] = float((ref[metric] <= item[metric]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_macro_geography(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    hist_std = historical_std_frame(historical)
    hist_x = hist_std[[f"{c}_std" for c in MACRO_COLS]].to_numpy(dtype=float)
    sample_plot = sample.sample(n=min(180, len(sample)), random_state=17) if len(sample) > 180 else sample.copy()
    sample_x = sample_plot[[f"{c}_std" for c in MACRO_COLS]].to_numpy(dtype=float)
    final_x = final[[f"{c}_std" for c in MACRO_COLS]].to_numpy(dtype=float)
    # The anchor standardized state is stored implicitly through the historical
    # macro row for 202004.
    anchor_row = hist_std.loc[hist_std["yyyymm"].astype(int).eq(202004), [f"{c}_std" for c in MACRO_COLS]]
    anchor_x = anchor_row.iloc[0].to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(hist_x)
    hist_pca = pca.transform(hist_x)
    sample_pca = pca.transform(sample_x)
    final_pca = pca.transform(final_x)
    anchor_pca = pca.transform(anchor_x[None, :])[0]

    combined = np.vstack([hist_x, sample_x, final_x, anchor_x[None, :]])
    labels = (
        ["historical"] * len(hist_x)
        + ["post-burn-in"] * len(sample_x)
        + ["final"] * len(final_x)
        + ["anchor"]
    )
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        init="pca",
        learning_rate="auto",
        random_state=13,
        n_iter=600,
        method="exact",
    )
    emb = tsne.fit_transform(combined)
    labels = np.asarray(labels)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
    for regime, part in hist_std.groupby("regime"):
        idx = part.index.to_numpy()
        axes[0].scatter(
            hist_pca[idx, 0],
            hist_pca[idx, 1],
            s=18,
            color=REGIME_COLORS.get(str(regime), PAPER_COLORS["historical"]),
            alpha=0.25,
            linewidths=0,
            label=f"Historical {str(regime).replace('_', ' ')}",
        )
    axes[0].scatter(sample_pca[:, 0], sample_pca[:, 1], s=9, color="#8c9299", alpha=0.20, linewidths=0, label="Generated sample")
    final_colors = final["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    axes[0].scatter(final_pca[:, 0], final_pca[:, 1], s=54, color=final_colors, edgecolors="white", linewidth=0.55, label="Final states", zorder=4)
    axes[0].scatter(anchor_pca[0], anchor_pca[1], s=140, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label="April 2020 anchor", zorder=5)
    axes[0].set_title("PCA macro geography", loc="left", fontsize=12.5, pad=8)
    axes[0].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var.)")
    axes[0].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var.)")

    for regime, color in REGIME_COLORS.items():
        hist_mask = np.zeros(len(labels), dtype=bool)
        hist_mask[: len(hist_x)] = hist_std["regime"].astype(str).eq(regime).to_numpy()
        axes[1].scatter(emb[hist_mask, 0], emb[hist_mask, 1], s=16, color=color, alpha=0.20, linewidths=0, label=f"Historical {regime.replace('_', ' ')}")
    sample_mask = labels == "post-burn-in"
    final_mask = labels == "final"
    anchor_mask = labels == "anchor"
    axes[1].scatter(emb[sample_mask, 0], emb[sample_mask, 1], s=8, color="#8c9299", alpha=0.18, linewidths=0, label="Generated sample")
    axes[1].scatter(emb[final_mask, 0], emb[final_mask, 1], s=54, color=final_colors, edgecolors="white", linewidth=0.55, label="Final states", zorder=4)
    axes[1].scatter(emb[anchor_mask, 0], emb[anchor_mask, 1], s=140, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label="April 2020 anchor", zorder=5)
    axes[1].set_title("t-SNE neighborhood map", loc="left", fontsize=12.5, pad=8)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles[:6], labels_[:6], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Generated Diversification States Stay Near the April 2020 Macro Neighborhood", fontsize=15.0, y=0.98)
    save_figure(fig, out_dir / "e2e_diversification_macro_pca_tsne")


def plot_diagnostic_dashboard(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.1))

    ax0 = axes[0, 0]
    scatter = ax0.scatter(
        final["anchor_empirical_mah_dist"],
        final["delta_entropy"],
        c=final["locked_e2e_hhi"],
        cmap="viridis_r",
        s=62,
        edgecolors="white",
        linewidth=0.6,
    )
    ax0.axhline(0.0, color=PAPER_COLORS["text"], linewidth=0.9)
    ax0.set_xlabel("Empirical Mahalanobis distance to anchor")
    ax0.set_ylabel("Entropy change from anchor")
    ax0.set_title("Target achieved within empirical locality", loc="left", fontsize=12.5, pad=8)
    fig.colorbar(scatter, ax=ax0, label="Final HHI")

    ax1 = axes[0, 1]
    ax1.scatter(final["l2_dist"], final["delta_hhi"], color=PAPER_COLORS["generated"], s=62, edgecolors="white", linewidth=0.6)
    ax1.axhline(0.0, color=PAPER_COLORS["text"], linewidth=0.9)
    ax1.set_xlabel("L2 distance from anchor")
    ax1.set_ylabel("HHI change from anchor")
    ax1.set_title("Concentration falls across final states", loc="left", fontsize=12.5, pad=8)

    ax2 = axes[1, 0]
    regimes = ["financial_stress", "contraction", "expansion"]
    counts = sample["regime"].value_counts(normalize=True).reindex(regimes).fillna(0.0) * 100.0
    ax2.bar([r.replace("_", "\n") for r in regimes], counts.values, color=[REGIME_COLORS[r] for r in regimes], alpha=0.90)
    ax2.set_ylabel("Post-burn-in generated states (%)")
    ax2.set_title("Scenario remains mostly stress/contraction-like", loc="left", fontsize=12.5, pad=8)
    for idx, value in enumerate(counts.values):
        ax2.text(idx, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    ax3 = axes[1, 1]
    hist_std = historical_std_frame(historical)
    shift_rows = []
    for col in MACRO_COLS:
        anchor_std = hist_std.loc[
            lambda df: df["yyyymm"].astype(int).eq(202004), f"{col}_std"
        ].iloc[0]
        values = sample[f"{col}_std"].to_numpy(dtype=float) - anchor_std
        shift_rows.append({"macro": col, "median": np.median(values), "q25": np.quantile(values, 0.25), "q75": np.quantile(values, 0.75)})
    shifts = pd.DataFrame(shift_rows).sort_values("median")
    y = np.arange(len(shifts))
    colors = [PAPER_COLORS["positive"] if v >= 0 else PAPER_COLORS["negative"] for v in shifts["median"]]
    ax3.barh(y, shifts["median"], color=colors, alpha=0.88)
    ax3.errorbar(
        shifts["median"],
        y,
        xerr=[shifts["median"] - shifts["q25"], shifts["q75"] - shifts["median"]],
        fmt="none",
        ecolor=PAPER_COLORS["text"],
        elinewidth=0.9,
        capsize=3,
    )
    ax3.axvline(0.0, color=PAPER_COLORS["text"], linewidth=0.9)
    ax3.set_yticks(y)
    ax3.set_yticklabels(shifts["macro"])
    ax3.set_xlabel("Generated minus anchor, standardized units")
    ax3.set_title("Macro movements behind diversification", loc="left", fontsize=12.5, pad=8)

    for ax in axes.ravel():
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Diagnostics for the E2E Diversification Scenario", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "e2e_diversification_diagnostic_dashboard")


def plot_nearest_neighbors(
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
) -> None:
    top, distance_summary, final_nn = nearest_historical_summary(final, sample, historical, top_k=5)
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200701) & (panel["yyyymm"] <= 202412)].copy()
    anchor_date = yyyymm_to_datetime([ANCHOR]).iloc[0]
    anchor_nfci = float(focus.loc[focus["yyyymm"].astype(int).eq(ANCHOR), "NFCI"].iloc[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_dir / "e2e_diversification_top5_historical_neighbors.csv", index=False)
    distance_summary.to_csv(out_dir / "e2e_diversification_nearest_neighbor_distance_summary.csv", index=False)
    final_nn.to_csv(out_dir / "e2e_diversification_final_state_nearest_neighbors.csv", index=False)

    set_publication_style()
    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, 0.95], wspace=0.28, hspace=0.34)

    ax0 = fig.add_subplot(gs[0, :])
    add_regime_background(ax0, focus)
    ax0.plot(focus["date"], focus["NFCI"], color="#222222", linewidth=1.25, label="NFCI")
    ax0.axhline(0.0, color="#697078", linestyle="--", linewidth=0.85)
    ax0.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.35, label="April 2020 anchor")
    sizes = 85 + 38 * top["nearest_final_state_count"].to_numpy(dtype=float)
    ax0.scatter(
        top["date"],
        top["NFCI"],
        s=sizes,
        color=PAPER_COLORS["generated"],
        edgecolors="white",
        linewidth=0.85,
        zorder=5,
        label="Top-5 historical analogs",
    )
    for _, row in top.iterrows():
        label = f"{int(row['yyyymm'])}\n#{int(row['rank'])}"
        ax0.annotate(
            label,
            xy=(row["date"], row["NFCI"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.92},
        )
    ax0.annotate(
        "Anchor: acute COVID stress",
        xy=(anchor_date, anchor_nfci),
        xytext=(pd.Timestamp("2020-11-01"), anchor_nfci + 0.45),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.0},
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94},
    )
    ax0.set_title("Nearest historical macro analogs for diversified generated states", loc="left", fontsize=12.5, pad=8)
    ax0.set_ylabel("NFCI")
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.65)
    ax0.legend(loc="upper right", frameon=False, fontsize=8.5)

    ax1 = fig.add_subplot(gs[1, 0])
    rows = distance_summary.copy()
    y = np.arange(len(rows))
    ax1.barh(y, rows["median_nearest_distance"], color=[PAPER_COLORS["generated"], "#8c9299", PAPER_COLORS["anchor"]], alpha=0.88)
    ax1.errorbar(
        rows["median_nearest_distance"],
        y,
        xerr=[rows["median_nearest_distance"] - rows["q25"], rows["q75"] - rows["median_nearest_distance"]],
        fmt="none",
        ecolor=PAPER_COLORS["text"],
        elinewidth=0.9,
        capsize=3,
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(rows["distribution"])
    ax1.set_xlabel("Euclidean distance in standardized macro space")
    ax1.set_title("Distance to the historical macro distribution", loc="left", fontsize=12.5, pad=8)
    ax1.grid(axis="x", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.65)

    ax2 = fig.add_subplot(gs[1, 1])
    table = top[["rank", "yyyymm", "regime", "NFCI", "distance_to_generated_final_median"]].copy()
    table["NFCI"] = table["NFCI"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    table["distance_to_generated_final_median"] = table["distance_to_generated_final_median"].map(lambda x: f"{x:.2f}")
    table["regime"] = table["regime"].str.replace("_", " ")
    ax2.axis("off")
    ax2.set_title("Top-5 analog months", loc="left", fontsize=12.5, pad=8)
    tbl = ax2.table(
        cellText=table.values,
        colLabels=["Rank", "Month", "Regime", "NFCI", "Dist."],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.04, 1.0, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(PAPER_COLORS["grid"])
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f3f4")

    fig.suptitle("Historical Nearest Neighbors Make the Diversification Scenario Interpretable", fontsize=15.0, y=0.985)
    save_figure(fig, out_dir / "e2e_diversification_historical_nearest_neighbors")


def plot_distribution_distances(
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    out_dir: Path,
) -> None:
    summary = distribution_distance_summary(final, sample, historical, n_boot=400)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "e2e_diversification_distribution_distance_summary.csv", index=False)

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), sharey=False)
    metrics = [
        ("sliced_wasserstein", "Sliced Wasserstein distance", "Average 1D Wasserstein over random macro projections"),
        ("energy_distance", "Energy distance", "Full multivariate two-sample distribution distance"),
    ]
    x = np.arange(len(summary))
    labels = ["Final\nstates", "Post-burn-in\nstates"]
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        observed = summary[metric].to_numpy(dtype=float)
        p50 = summary[f"{metric}_hist_p50"].to_numpy(dtype=float)
        p90 = summary[f"{metric}_hist_p90"].to_numpy(dtype=float)
        width = 0.34
        ax.bar(x - width / 2, observed, width, color=PAPER_COLORS["generated"], alpha=0.90, label="Generated vs history")
        ax.bar(x + width / 2, p50, width, color="#9ba1a6", alpha=0.85, label="Historical bootstrap median")
        ax.scatter(x + width / 2, p90, color=PAPER_COLORS["anchor"], marker="D", s=42, label="Historical bootstrap 90th pct.")
        for idx, row in summary.iterrows():
            ax.text(
                idx - width / 2,
                observed[idx] + 0.025 * max(observed.max(), p90.max()),
                f"hist pct={100.0 * row[f'{metric}_hist_percentile']:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontsize=12.5, pad=8)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=8.2, color="#4d545a")
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    axes[0].set_ylabel("Distance")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8.7)
    fig.suptitle("Distribution Distances: Generated Macro States Compared With Historical Macro States", fontsize=14.5, y=1.02)
    save_figure(fig, out_dir / "e2e_diversification_distribution_distances")


def write_summary(final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    top, distance_summary, _ = nearest_historical_summary(final, sample, historical, top_k=5)
    distribution_distances = distribution_distance_summary(final, sample, historical, n_boot=400)
    top_line = "; ".join(
        f"{int(row.yyyymm)} ({str(row.regime).replace('_', ' ')}, NFCI={row.NFCI:.2f}, dist={row.distance_to_generated_final_median:.2f})"
        for _, row in top.iterrows()
    )
    dist_lookup = distance_summary.set_index("distribution")["median_nearest_distance"].to_dict()
    final_dist = distribution_distances[distribution_distances["comparison"].eq("Final states vs history")].iloc[0]
    lines = [
        "# E2E Diversification Figure Summary",
        "",
        f"- Median entropy change: `{final['delta_entropy'].median():+.4f}`.",
        f"- Median HHI change: `{final['delta_hhi'].median():+.4f}`.",
        f"- Median effective-N change: `{final['delta_effective_n'].median():+.2f}`.",
        f"- Median max-weight change: `{100.0 * final['delta_max_weight'].median():+.2f}` percentage points.",
        f"- Median top-10 weight change: `{100.0 * final['delta_top10_weight'].median():+.2f}` percentage points.",
        f"- Median empirical-anchor Mahalanobis tail: `{final['anchor_empirical_mah_chi2_tail'].median():.3f}`.",
        f"- Median anchor VAR-innovation tail: `{final['anchor_mah_chi2_tail'].median():.2e}`.",
        f"- Median final-state nearest-neighbor distance to history, excluding anchor: `{dist_lookup['Generated final states']:.3f}`.",
        f"- Final-state sliced Wasserstein vs history: `{final_dist['sliced_wasserstein']:.3f}`; historical-bootstrap percentile `{100.0 * final_dist['sliced_wasserstein_hist_percentile']:.1f}`.",
        f"- Final-state energy distance vs history: `{final_dist['energy_distance']:.3f}`; historical-bootstrap percentile `{100.0 * final_dist['energy_distance_hist_percentile']:.1f}`.",
        f"- Top-5 historical analogs for the median generated final state: {top_line}.",
        "",
        "Economic reading: the generated states do not need to become a new, unhistorical macro region; they move from the acute April 2020 stress anchor toward historically observed stress/contraction-like neighborhoods where the locked E2E allocation is less dominated by the largest anchor positions.",
        "",
        "Manuscript wording: empirically local to April 2020; not VAR(1)-innovation plausible.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_diversification_figure_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-specific plots for the E2E diversification scenario.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = load_run(args.run_dir)
    final = payload["final"]
    sample = payload["sample"]
    anchor_weights = payload["anchor_weights"]
    final_weights = payload["final_weights"]
    historical = payload["historical"]

    plot_concentration_before_after(final, anchor_weights, final_weights, args.out_dir)
    plot_seed_level_diversification(final, args.out_dir)
    plot_macro_geography(final, sample, historical, args.out_dir)
    plot_diagnostic_dashboard(final, sample, historical, args.out_dir)
    plot_nearest_neighbors(final, sample, historical, args.out_dir)
    plot_distribution_distances(final, sample, historical, args.out_dir)
    write_summary(final, sample, historical, args.out_dir)
    print(f"Wrote figures -> {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
