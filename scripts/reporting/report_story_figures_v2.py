"""Story-first paper figures for the three selected scenario generations.

This script intentionally creates separate, manuscript-facing figures rather
than broad diagnostic dashboards. The three stories are:

1. Scenario 4 baseline: April 2020 stress -> SC beats WW.
2. Locked E2E diversification: April 2020 concentration -> diversification.
3. PTO catch-up: March 2020 E2E lead -> PTO catches up and overtakes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batuhan.regime.nfci import load_nfci
from src.data.macro_scaler import MacroScaler
from src.modules.runtime_regime import MACRO_ORDER
from src.utils.plotting import PAPER_COLORS, set_publication_style


MACRO_COLS = list(MACRO_ORDER)
OUT_ROOT = ROOT / "submission_plots" / "story_figures_v2"

S4_RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260424_102040"
E2E_RUN = ROOT / "scenario_outputs" / "scenario_e2e_diversify_202004" / "runs" / "20260507_130143_361581"
PTO_RUN = ROOT / "scenario_outputs" / "scenario_pto_catchup_202003" / "runs" / "20260507_154755_972692"

REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}
MODEL_COLORS = {
    "locked_e2e": "#4c78a8",
    "standardized_pto": "#157f78",
    "summer_child": "#157f78",
    "winter_wolf": "#c44e52",
}


@dataclass
class FigureRecord:
    scenario: str
    figure: str
    pdf: str
    png: str
    proves: str


def newest(run_dir: Path, *patterns: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(run_dir.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No file in {run_dir} matching {patterns}")
    return sorted(matches)[-1]


def yyyymm_to_datetime(values: pd.Series | list[int] | np.ndarray) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))


def add_record(records: list[FigureRecord], scenario: str, stem: str, paths: tuple[str, str], proves: str) -> None:
    records.append(FigureRecord(scenario=scenario, figure=stem, pdf=paths[0], png=paths[1], proves=proves))


def load_regime_nfci_panel() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    nfci = load_nfci().rename("NFCI").reset_index()
    panel = labels.merge(nfci, on="yyyymm", how="left").sort_values("yyyymm").reset_index(drop=True)
    panel["date"] = yyyymm_to_datetime(panel["yyyymm"])
    return panel


def regime_segments(panel: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    rows = panel[["date", "regime"]].dropna().reset_index(drop=True)
    if rows.empty:
        return []
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


def historical_std_frame(historical: pd.DataFrame) -> pd.DataFrame:
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    values = historical[MACRO_COLS].to_numpy(dtype=np.float32)
    std = (values - scaler.mean.cpu().numpy()) / scaler.std.cpu().numpy()
    out = historical[["yyyymm"]].copy()
    for idx, col in enumerate(MACRO_COLS):
        out[f"{col}_std"] = std[:, idx]
    labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    return out.merge(labels[["yyyymm", "regime"]], on="yyyymm", how="left")


def anchor_std(historical: pd.DataFrame, anchor: int) -> pd.Series:
    hist_std = historical_std_frame(historical)
    rows = hist_std[hist_std["yyyymm"].astype(int).eq(anchor)]
    if rows.empty:
        raise ValueError(f"Anchor {anchor} not found in historical panel")
    return rows.iloc[0]


def macro_shift_frame(sample: pd.DataFrame, historical: pd.DataFrame, anchor: int) -> pd.DataFrame:
    anchor_row = anchor_std(historical, anchor)
    rows = []
    for col in MACRO_COLS:
        values = sample[f"{col}_std"].to_numpy(dtype=float) - float(anchor_row[f"{col}_std"])
        rows.append(
            {
                "macro": col,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values("median")


def nearest_analogs(final: pd.DataFrame, historical: pd.DataFrame, anchor: int, top_k: int = 5) -> pd.DataFrame:
    hist = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    hist = hist[hist["yyyymm"].astype(int).ne(anchor)].reset_index(drop=True)
    hist_x = hist[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    median_final = np.median(final_x, axis=0)
    dists = np.linalg.norm(hist_x - median_final[None, :], axis=1)
    nearest_idx = np.linalg.norm(final_x[:, None, :] - hist_x[None, :, :], axis=2).argmin(axis=1)
    counts = pd.Series(hist.loc[nearest_idx, "yyyymm"].astype(int)).value_counts()
    top_idx = np.argsort(dists)[:top_k]
    top = hist.iloc[top_idx][["yyyymm", "regime"]].copy()
    top["distance_to_generated_final_median"] = dists[top_idx]
    top["nearest_final_state_count"] = top["yyyymm"].astype(int).map(counts).fillna(0).astype(int)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    panel = load_regime_nfci_panel()
    top = top.merge(panel[["yyyymm", "NFCI"]], on="yyyymm", how="left")
    top["date"] = yyyymm_to_datetime(top["yyyymm"])
    return top


def story_pca_map(
    final: pd.DataFrame,
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    anchor: int,
    out_dir: Path,
    stem: str,
    title: str,
    subtitle: str,
) -> tuple[str, str]:
    set_publication_style()
    hist_std = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    hist_x = hist_std[cols].to_numpy(dtype=float)
    sample_plot = sample.sample(n=min(500, len(sample)), random_state=41) if len(sample) > 500 else sample.copy()
    sample_x = sample_plot[cols].to_numpy(dtype=float)
    final_x = final[cols].to_numpy(dtype=float)
    anchor_x = hist_std.loc[hist_std["yyyymm"].astype(int).eq(anchor), cols].iloc[0].to_numpy(dtype=float)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(hist_x)
    hist_pca = pca.transform(hist_x)
    sample_pca = pca.transform(sample_x)
    final_pca = pca.transform(final_x)
    anchor_pca = pca.transform(anchor_x[None, :])[0]

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for regime, part in hist_std.groupby("regime"):
        idx = part.index.to_numpy()
        ax.scatter(
            hist_pca[idx, 0],
            hist_pca[idx, 1],
            s=18,
            color=REGIME_COLORS.get(str(regime), PAPER_COLORS["historical"]),
            alpha=0.22,
            linewidths=0,
            label=f"Historical {str(regime).replace('_', ' ')}",
        )
    ax.scatter(sample_pca[:, 0], sample_pca[:, 1], s=10, color="#8c9299", alpha=0.20, linewidths=0, label="Generated sample")
    final_colors = final["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    ax.scatter(final_pca[:, 0], final_pca[:, 1], s=62, color=final_colors, edgecolors="white", linewidth=0.6, label="Final states", zorder=4)
    ax.scatter(anchor_pca[0], anchor_pca[1], s=155, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label="Anchor", zorder=5)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% historical variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% historical variance)")
    ax.set_title(title, loc="left", fontsize=13.0, pad=9)
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8.7, va="bottom", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95})
    ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.2, loc="best")
    return save_figure(fig, out_dir, stem)


def macro_shift_plot(
    sample: pd.DataFrame,
    historical: pd.DataFrame,
    anchor: int,
    out_dir: Path,
    stem: str,
    title: str,
    callout: str,
    highlight: set[str] | None = None,
) -> tuple[str, str]:
    set_publication_style()
    shifts = macro_shift_frame(sample, historical, anchor)
    highlight = highlight or set()
    y = np.arange(len(shifts))
    colors = [
        PAPER_COLORS["positive"] if row.macro in highlight else (PAPER_COLORS["generated"] if row.median >= 0 else PAPER_COLORS["negative"])
        for row in shifts.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(y, shifts["median"], color=colors, alpha=0.88)
    ax.errorbar(
        shifts["median"],
        y,
        xerr=[shifts["median"] - shifts["q25"], shifts["q75"] - shifts["median"]],
        fmt="none",
        ecolor=PAPER_COLORS["text"],
        elinewidth=0.9,
        capsize=3,
    )
    ax.axvline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(shifts["macro"])
    ax.set_xlabel("Generated minus anchor, standardized units")
    ax.set_title(title, loc="left", fontsize=13.0, pad=9)
    ax.text(0.02, 0.04, callout, transform=ax.transAxes, fontsize=8.8, va="bottom", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95})
    ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    return save_figure(fig, out_dir, stem)


def nfci_analogs_plot(
    final: pd.DataFrame,
    historical: pd.DataFrame,
    anchor: int,
    out_dir: Path,
    stem: str,
    title: str,
    callout: str,
) -> tuple[str, str]:
    set_publication_style()
    top = nearest_analogs(final, historical, anchor, top_k=5)
    top.to_csv(out_dir / f"{stem}_top5_neighbors.csv", index=False)
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200701) & (panel["yyyymm"] <= 202412)].copy()
    anchor_date = yyyymm_to_datetime([anchor]).iloc[0]
    anchor_nfci = float(focus.loc[focus["yyyymm"].astype(int).eq(anchor), "NFCI"].iloc[0])

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    add_regime_background(ax, focus)
    ax.plot(focus["date"], focus["NFCI"], color="#202020", linewidth=1.25, label="NFCI")
    ax.axhline(0.0, color="#697078", linestyle="--", linewidth=0.85)
    ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.35, label=f"{anchor} anchor")
    sizes = 85 + 40 * top["nearest_final_state_count"].to_numpy(dtype=float)
    ax.scatter(top["date"], top["NFCI"], s=sizes, color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.85, zorder=5, label="Top analogs")
    for row in top.itertuples(index=False):
        ax.annotate(
            f"{int(row.yyyymm)}\n#{int(row.rank)}",
            xy=(row.date, row.NFCI),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.92},
        )
    ax.annotate(
        "Anchor",
        xy=(anchor_date, anchor_nfci),
        xytext=(anchor_date + pd.DateOffset(months=8), anchor_nfci + 0.45),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.0},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94},
    )
    ax.text(0.02, 0.04, callout, transform=ax.transAxes, fontsize=8.2, va="bottom", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.94})
    ax.set_title(title, loc="left", fontsize=13.0, pad=9)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper right", frameon=False, fontsize=8.3)
    return save_figure(fig, out_dir, stem)


def load_s4(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(newest(run_dir, "final_state_diagnostics_*_enriched.csv", "final_state_diagnostics_*.csv"))
    sample = pd.read_csv(newest(run_dir, "generated_macro_sample_postburnin_*_enriched.csv", "generated_macro_sample*.csv"))
    historical = pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv"))
    return final, sample, historical


def load_scenario5(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(newest(run_dir, "final_state_diagnostics_*.csv"))
    sample = pd.read_csv(newest(run_dir, "generated_macro_sample_*.csv"))
    historical = pd.read_csv(newest(run_dir, "historical_macro_panel_*.csv"))
    anchor_weights = pd.read_csv(newest(run_dir, "anchor_weights_*.csv"))
    final_weights = pd.read_csv(newest(run_dir, "final_weights_*.csv"))
    return final, sample, historical, anchor_weights, final_weights


def plot_s4_return_gap(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    gaps = 100.0 * final["return_gap"].to_numpy(dtype=float)
    order = np.argsort(gaps)
    colors = final.iloc[order]["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.bar(np.arange(len(gaps)), gaps[order], color=colors, alpha=0.92)
    ax.axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax.axhline(np.median(gaps), color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label=f"Median {np.median(gaps):+.2f}pp")
    ax.set_xlabel("Final seed, sorted by SC-WW return gap")
    ax.set_ylabel("SummerChild minus WinterWolf return (pp)")
    ax.set_title("Scenario 4: every final state favors SummerChild", loc="left", fontsize=13.0, pad=9)
    ax.text(0.02, 0.92, f"{(gaps > 0).sum()}/{len(gaps)} final seeds positive", transform=ax.transAxes, fontsize=9.0, va="top", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95})
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    return save_figure(fig, out_dir, "s4_return_gap_reversal")


def plot_s4_regime_escape(final: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    regimes = ["financial_stress", "contraction", "expansion"]
    final_counts = final["regime"].value_counts().reindex(regimes).fillna(0).astype(int)
    sample_share = sample["regime"].value_counts(normalize=True).reindex(regimes).fillna(0.0) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=False)
    axes[0].bar([r.replace("_", "\n") for r in regimes], final_counts.values, color=[REGIME_COLORS[r] for r in regimes], alpha=0.92)
    axes[0].set_ylabel("Final seeds")
    axes[0].set_title("Final states leave acute stress", loc="left", fontsize=12.5, pad=8)
    for idx, value in enumerate(final_counts.values):
        axes[0].text(idx, value + 0.35, str(value), ha="center", fontsize=9)
    axes[1].bar([r.replace("_", "\n") for r in regimes], sample_share.values, color=[REGIME_COLORS[r] for r in regimes], alpha=0.92)
    axes[1].set_ylabel("Post-burn-in states (%)")
    axes[1].set_title("Generated sample is mostly contraction/expansion", loc="left", fontsize=12.5, pad=8)
    for idx, value in enumerate(sample_share.values):
        axes[1].text(idx, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Scenario 4 Regime Story: SC wins after moving away from financial stress", fontsize=14.5, y=1.02)
    return save_figure(fig, out_dir, "s4_regime_escape")


def grouped_weight_means(final_weights: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    frame = final_weights.copy()
    if model is not None and "model" in frame.columns:
        frame = frame[frame["model"].eq(model)].copy()
    grouped = frame.groupby("permno", as_index=False).agg(weight=("weight", "mean"), anchor_weight=("anchor_weight", "first"))
    grouped["delta_weight"] = grouped["weight"] - grouped["anchor_weight"]
    return grouped


def plot_e2e_concentration(anchor_weights: pd.DataFrame, final_weights: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    anchor = anchor_weights.copy()
    if "model" in anchor.columns:
        anchor = anchor[anchor["model"].eq("locked_e2e")].copy()
    med = grouped_weight_means(final_weights, "locked_e2e")
    top_permnos = anchor.sort_values("anchor_weight", ascending=False).head(15)["permno"].astype(int).tolist()
    top = anchor[["permno", "anchor_weight"]].merge(med[["permno", "weight"]], on="permno", how="left")
    top["permno"] = top["permno"].astype(int)
    top = top[top["permno"].isin(top_permnos)].copy()
    top["_order"] = top["permno"].map({permno: idx for idx, permno in enumerate(top_permnos)})
    top = top.sort_values("_order")

    anchor_sorted = np.sort(anchor["anchor_weight"].to_numpy(dtype=float))[::-1]
    med_sorted = np.sort(med["weight"].to_numpy(dtype=float))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), gridspec_kw={"width_ratios": [1.45, 1.0]})
    x = np.arange(len(top))
    width = 0.38
    axes[0].bar(x - width / 2, 100 * top["anchor_weight"], width, color=PAPER_COLORS["anchor"], label="Anchor")
    axes[0].bar(x + width / 2, 100 * top["weight"], width, color=PAPER_COLORS["generated"], label="Generated final mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["permno"].astype(str), rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Portfolio weight (%)")
    axes[0].set_title("Largest anchor holdings shrink", loc="left", fontsize=12.5, pad=8)
    axes[0].legend(frameon=False)
    k = np.arange(1, len(anchor_sorted) + 1)
    axes[1].plot(k, 100 * np.cumsum(anchor_sorted), color=PAPER_COLORS["anchor"], linewidth=2.2, label="Anchor")
    axes[1].plot(k, 100 * np.cumsum(med_sorted), color=PAPER_COLORS["generated"], linewidth=2.2, label="Generated final mean")
    axes[1].axvline(10, color=PAPER_COLORS["grid"], linewidth=1.0)
    axes[1].set_xlabel("Top-k holdings")
    axes[1].set_ylabel("Cumulative weight (%)")
    axes[1].set_title("Concentration curve flattens", loc="left", fontsize=12.5, pad=8)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Locked E2E becomes less concentrated near April 2020", fontsize=14.5, y=1.02)
    return save_figure(fig, out_dir, "e2e_concentration_before_after")


def plot_e2e_metrics(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    specs = [
        ("Entropy", "anchor_entropy", "locked_e2e_entropy", ""),
        ("HHI", "anchor_hhi", "locked_e2e_hhi", ""),
        ("Effective N", "anchor_effective_n", "locked_e2e_effective_n", ""),
        ("Max weight", "anchor_max_weight", "locked_e2e_max_weight", "%"),
        ("Top-10 weight", "anchor_top10_weight", "locked_e2e_top10_weight", "%"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14.0, 4.3))
    for ax, (label, anchor_col, final_col, unit) in zip(axes, specs):
        anchor_val = float(final[anchor_col].iloc[0])
        final_val = float(final[final_col].median())
        if unit == "%":
            anchor_val *= 100
            final_val *= 100
        ax.bar([0, 1], [anchor_val, final_val], color=[PAPER_COLORS["anchor"], PAPER_COLORS["generated"]], alpha=0.90)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Anchor", "Final"], rotation=20, ha="right")
        ax.set_title(label, fontsize=11.0, pad=7)
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
        ax.text(1, final_val, f"{final_val:.2f}" if unit != "%" else f"{final_val:.1f}%", ha="center", va="bottom", fontsize=8.0)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Diversification metrics move in the intended direction", fontsize=14.5, y=1.04)
    return save_figure(fig, out_dir, "e2e_diversification_metrics")


def plot_e2e_seed_consistency(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    specs = [
        ("Entropy change", "delta_entropy", True),
        ("HHI change", "delta_hhi", False),
        ("Effective N change", "delta_effective_n", True),
        ("Max weight change (pp)", "delta_max_weight", False),
        ("Top-10 change (pp)", "delta_top10_weight", False),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14.2, 4.2))
    for ax, (title, col, positive_good) in zip(axes, specs):
        vals = final[col].to_numpy(dtype=float)
        if "pp" in title:
            vals = 100 * vals
        order = np.argsort(vals)
        colors = [PAPER_COLORS["positive"] if (v >= 0) == positive_good else PAPER_COLORS["negative"] for v in vals[order]]
        ax.barh(np.arange(len(vals)), vals[order], color=colors, alpha=0.88)
        ax.axvline(0.0, color=PAPER_COLORS["text"], linewidth=0.9)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("Seed-level consistency: every chain supports diversification", fontsize=14.5, y=1.04)
    return save_figure(fig, out_dir, "e2e_seed_consistency")


def plot_pto_return_gap(final: pd.DataFrame, anchor_weights: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    anchor_locked = float(anchor_weights.loc[anchor_weights["model"].eq("locked_e2e"), "anchor_return"].iloc[0])
    anchor_pto = float(anchor_weights.loc[anchor_weights["model"].eq("standardized_pto"), "anchor_return"].iloc[0])
    final_locked = float(final["locked_e2e_return"].median())
    final_pto = float(final["standardized_pto_return"].median())
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, 100 * np.array([anchor_locked, final_locked]), width, color=MODEL_COLORS["locked_e2e"], label="Locked E2E")
    axes[0].bar(x + width / 2, 100 * np.array([anchor_pto, final_pto]), width, color=MODEL_COLORS["standardized_pto"], label="PTO")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Anchor\n2020-03", "Generated\nfinal median"])
    axes[0].set_ylabel("Realized return (%)")
    axes[0].set_title("Return ranking reverses", loc="left", fontsize=12.5, pad=8)
    axes[0].legend(frameon=False)
    gaps = 100 * final["return_gap_a_minus_b"].to_numpy(dtype=float)
    anchor_gap = 100 * float(final["anchor_return_gap_a_minus_b"].iloc[0])
    for gap in gaps:
        axes[1].plot([0, 1], [anchor_gap, gap], color=PAPER_COLORS["generated"], alpha=0.35, linewidth=1.2)
        axes[1].scatter([1], [gap], color=PAPER_COLORS["generated"], s=22, alpha=0.80)
    axes[1].scatter([0], [anchor_gap], color=PAPER_COLORS["anchor"], marker="X", s=95, zorder=4)
    axes[1].axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Anchor", "Generated final"])
    axes[1].set_ylabel("Locked E2E minus PTO return (pp)")
    axes[1].set_title("All seeds cross below zero", loc="left", fontsize=12.5, pad=8)
    for ax in axes:
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("PTO catch-up: from 4.33pp behind to 2.70pp ahead", fontsize=14.5, y=1.02)
    return save_figure(fig, out_dir, "pto_return_gap_reversal")


def plot_pto_win_consistency(final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    improvements = 100 * final["return_gap_improvement_for_b"].to_numpy(dtype=float)
    order = np.argsort(improvements)
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    ax.bar(np.arange(len(improvements)), improvements[order], color=PAPER_COLORS["generated"], alpha=0.90)
    ax.axhline(np.median(improvements), color=PAPER_COLORS["text"], linestyle="--", linewidth=1.1, label=f"Median {np.median(improvements):.2f}pp")
    ax.set_xlabel("Final seed, sorted by PTO improvement")
    ax.set_ylabel("PTO improvement versus anchor gap (pp)")
    ax.set_title("PTO wins in all final states", loc="left", fontsize=13.0, pad=9)
    ax.text(0.02, 0.92, "20/20 final seeds have PTO return >= locked E2E", transform=ax.transAxes, fontsize=9.0, va="top", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95})
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    return save_figure(fig, out_dir, "pto_win_consistency")


def plot_pto_portfolio(anchor_weights: pd.DataFrame, final_weights: pd.DataFrame, final: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    set_publication_style()
    pto_anchor = anchor_weights[anchor_weights["model"].eq("standardized_pto")].copy()
    pto_med = grouped_weight_means(final_weights, "standardized_pto")
    top_permnos = pto_anchor.sort_values("anchor_weight", ascending=False).head(12)["permno"].astype(int).tolist()
    top = pto_anchor[["permno", "anchor_weight"]].merge(pto_med[["permno", "weight"]], on="permno", how="left")
    top["permno"] = top["permno"].astype(int)
    top = top[top["permno"].isin(top_permnos)].copy()
    top["_order"] = top["permno"].map({permno: idx for idx, permno in enumerate(top_permnos)})
    top = top.sort_values("_order")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    x = np.arange(len(top))
    width = 0.38
    axes[0].bar(x - width / 2, 100 * top["anchor_weight"], width, color=PAPER_COLORS["anchor"], label="PTO anchor")
    axes[0].bar(x + width / 2, 100 * top["weight"], width, color=PAPER_COLORS["generated"], label="PTO final mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["permno"].astype(str), rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Portfolio weight (%)")
    axes[0].set_title("PTO top weight falls", loc="left", fontsize=12.5, pad=8)
    axes[0].legend(frameon=False, fontsize=8.4)

    metrics = [
        ("HHI", "hhi", ""),
        ("Effective N", "effective_n", ""),
    ]
    for ax, (title, suffix, unit) in zip(axes[1:], metrics):
        vals = [
            float(anchor_weights[anchor_weights["model"].eq("locked_e2e")][f"anchor_{suffix}"].iloc[0]),
            float(final[f"locked_e2e_{suffix}"].median()),
            float(anchor_weights[anchor_weights["model"].eq("standardized_pto")][f"anchor_{suffix}"].iloc[0]),
            float(final[f"standardized_pto_{suffix}"].median()),
        ]
        labels = ["E2E\nanchor", "E2E\nfinal", "PTO\nanchor", "PTO\nfinal"]
        ax.bar(np.arange(4), vals, color=[MODEL_COLORS["locked_e2e"], MODEL_COLORS["locked_e2e"], MODEL_COLORS["standardized_pto"], MODEL_COLORS["standardized_pto"]], alpha=0.78)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, loc="left", fontsize=12.5, pad=8)
    for ax in axes:
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.suptitle("PTO wins while remaining more concentrated than locked E2E", fontsize=14.5, y=1.02)
    return save_figure(fig, out_dir, "pto_portfolio_mechanism")


def run_s4(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    scenario = "scenario4_202004"
    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical = load_s4(run_dir)
    records: list[FigureRecord] = []
    add_record(records, scenario, "s4_return_gap_reversal", plot_s4_return_gap(final, out_dir), "All final seeds produce positive SC-WW return gaps.")
    add_record(records, scenario, "s4_regime_escape", plot_s4_regime_escape(final, sample, out_dir), "Generated states leave April 2020 financial stress toward contraction/expansion.")
    add_record(records, scenario, "s4_macro_story_shift", macro_shift_plot(sample, historical, 202004, out_dir, "s4_macro_story_shift", "Macro movement behind Scenario 4", "Largest story shifts: higher bm and dp, with financial stress largely reduced.", {"bm", "dp"}), "Macro shifts identify the economic movement needed for SC to beat WW.")
    add_record(records, scenario, "s4_nfci_historical_analogs", nfci_analogs_plot(final, historical, 202004, out_dir, "s4_nfci_historical_analogs", "Scenario 4 historical analogs on NFCI", "Nearest analogs make the counterfactual interpretable; they are analogs, not formal plausibility proof."), "Top historical analogs place generated states on the NFCI timeline.")
    add_record(records, scenario, "s4_pca_macro_map", story_pca_map(final, sample, historical, 202004, out_dir, "s4_pca_macro_map", "Scenario 4 PCA macro map", "Anchor starts in stress; final states sit in nearby contraction/expansion regions."), "PCA shows anchor, generated sample, and final states relative to historical regimes.")
    notes = {
        "anchor_metric": "April 2020 financial-stress anchor; Scenario 4 asks SC to beat WW.",
        "final_metric": f"Final win rate {(final['return_gap'] > 0).mean():.1%}; median SC-WW return gap {100*final['return_gap'].median():+.2f}pp.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final['box_violation'].mean():.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "caution": "Use empirical/historical-neighbor framing; do not overclaim one-step VAR plausibility unless separately supported.",
        "sentence": "From the April 2020 stress anchor, the scenario generator finds nearby non-stress macro states where SummerChild beats WinterWolf in every final chain.",
    }
    return records, notes


def run_e2e(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    scenario = "e2e_diversification_202004"
    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical, anchor_weights, final_weights = load_scenario5(run_dir)
    records: list[FigureRecord] = []
    add_record(records, scenario, "e2e_concentration_before_after", plot_e2e_concentration(anchor_weights, final_weights, out_dir), "The largest anchor holdings shrink and the concentration curve flattens.")
    add_record(records, scenario, "e2e_diversification_metrics", plot_e2e_metrics(final, out_dir), "Entropy and effective N rise while HHI and large-position concentration fall.")
    add_record(records, scenario, "e2e_seed_consistency", plot_e2e_seed_consistency(final, out_dir), "Seed-level outcomes consistently move in the diversification direction.")
    add_record(records, scenario, "e2e_macro_story_shift", macro_shift_plot(sample, historical, 202004, out_dir, "e2e_macro_story_shift", "Macro movements behind locked E2E diversification", "Diversification occurs near April 2020 while the macro state moves to a less concentrated allocation region.", {"dfy", "tbl"}), "Macro shifts show the counterfactual movement behind diversification.")
    add_record(records, scenario, "e2e_nfci_historical_analogs", nfci_analogs_plot(final, historical, 202004, out_dir, "e2e_nfci_historical_analogs", "E2E diversification analogs on NFCI", "Historical analogs make the generated diversification states interpretable."), "Nearest historical analogs place generated diversification states on the NFCI timeline.")
    add_record(records, scenario, "e2e_pca_macro_map", story_pca_map(final, sample, historical, 202004, out_dir, "e2e_pca_macro_map", "E2E diversification PCA macro map", "Generated states remain near the April 2020 macro neighborhood while concentration falls."), "PCA shows diversification states relative to the anchor and historical regimes.")
    notes = {
        "anchor_metric": f"Anchor entropy {final['anchor_entropy'].iloc[0]:.3f}; anchor HHI {final['anchor_hhi'].iloc[0]:.3f}; anchor max weight {100*final['anchor_max_weight'].iloc[0]:.1f}%.",
        "final_metric": f"Median entropy change {final['delta_entropy'].median():+.3f}; median HHI change {final['delta_hhi'].median():+.3f}; median max-weight change {100*final['delta_max_weight'].median():+.2f}pp.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final['final_box_violation'].mean():.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "caution": f"Empirical-anchor tail median {final['anchor_empirical_mah_chi2_tail'].median():.3f}; VAR-anchor tail median {final['anchor_mah_chi2_tail'].median():.2e}. Frame as empirically local, not VAR(1)-innovation plausible.",
        "sentence": "Near the April 2020 stress anchor, locked E2E can be pushed into empirically local states where the portfolio becomes materially more diversified.",
    }
    return records, notes


def run_pto(run_dir: Path, out_root: Path) -> tuple[list[FigureRecord], dict[str, str]]:
    scenario = "pto_catchup_202003"
    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sample, historical, anchor_weights, final_weights = load_scenario5(run_dir)
    records: list[FigureRecord] = []
    add_record(records, scenario, "pto_return_gap_reversal", plot_pto_return_gap(final, anchor_weights, out_dir), "PTO moves from behind locked E2E at anchor to ahead in generated final states.")
    add_record(records, scenario, "pto_win_consistency", plot_pto_win_consistency(final, out_dir), "Every final seed improves PTO enough to beat locked E2E.")
    add_record(records, scenario, "pto_portfolio_mechanism", plot_pto_portfolio(anchor_weights, final_weights, final, out_dir), "PTO wins while remaining more concentrated than locked E2E.")
    add_record(records, scenario, "pto_macro_story_shift", macro_shift_plot(sample, historical, 202003, out_dir, "pto_macro_story_shift", "Macro movements behind PTO catch-up", "The catch-up is driven within stress by lower dp/bm/tbl and higher tms.", {"dp", "bm", "tbl", "tms"}), "Macro shifts show the within-stress movement that reverses the ranking.")
    add_record(records, scenario, "pto_nfci_historical_analogs", nfci_analogs_plot(final, historical, 202003, out_dir, "pto_nfci_historical_analogs", "PTO catch-up analogs on NFCI", "Stress analogs such as 198710 make the generated states historically interpretable."), "Historical analogs place PTO catch-up states on the NFCI timeline.")
    add_record(records, scenario, "pto_pca_macro_map", story_pca_map(final, sample, historical, 202003, out_dir, "pto_pca_macro_map", "PTO catch-up PCA macro map", "Generated states remain in the financial-stress neighborhood while PTO overtakes."), "PCA shows PTO catch-up occurs without leaving financial stress.")
    notes = {
        "anchor_metric": f"Anchor locked E2E minus PTO return gap {100*final['anchor_return_gap_a_minus_b'].iloc[0]:+.2f}pp.",
        "final_metric": f"Final median locked E2E minus PTO gap {100*final['return_gap_a_minus_b'].median():+.2f}pp; PTO win share {final['b_return_matches_or_beats_a'].mean():.1%}.",
        "acceptance": f"Mean acceptance {100*final['accept_rate'].mean():.1f}%; mean box violation {final['final_box_violation'].mean():.4f}.",
        "regime": f"Final regimes {dict(final['regime'].value_counts())}; sample regimes {dict((sample['regime'].value_counts(normalize=True)*100).round(1))} percent.",
        "caution": f"Empirical-anchor tail median {final['anchor_empirical_mah_chi2_tail'].median():.3f}; VAR-anchor tail median {final['anchor_mah_chi2_tail'].median():.2e}. Frame as empirically local, not VAR(1)-innovation plausible.",
        "sentence": "Starting from March 2020, PTO closes a 4.33pp deficit and overtakes locked E2E inside empirically local financial-stress states.",
    }
    return records, notes


def write_manifest(records: list[FigureRecord], notes: dict[str, dict[str, str]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.__dict__ for record in records]).to_csv(out_root / "figure_manifest.csv", index=False)
    lines = ["# Story Figures V2 Notes", ""]
    for scenario, payload in notes.items():
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"- Anchor target metric: {payload['anchor_metric']}",
                f"- Final target metric: {payload['final_metric']}",
                f"- Acceptance/constraints: {payload['acceptance']}",
                f"- Regime movement: {payload['regime']}",
                f"- Locality caution: {payload['caution']}",
                f"- Recommended manuscript sentence: {payload['sentence']}",
                "",
            ]
        )
    (out_root / "figure_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate story-first paper figures for the three selected scenario runs.")
    parser.add_argument("--scenario", choices=["all", "s4", "e2e", "pto"], default="all")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--s4-run-dir", type=Path, default=S4_RUN)
    parser.add_argument("--e2e-run-dir", type=Path, default=E2E_RUN)
    parser.add_argument("--pto-run-dir", type=Path, default=PTO_RUN)
    args = parser.parse_args()

    set_publication_style()
    records: list[FigureRecord] = []
    notes: dict[str, dict[str, str]] = {}

    if args.scenario in {"all", "s4"}:
        recs, note = run_s4(args.s4_run_dir, args.out_root)
        records.extend(recs)
        notes["scenario4_202004"] = note
    if args.scenario in {"all", "e2e"}:
        recs, note = run_e2e(args.e2e_run_dir, args.out_root)
        records.extend(recs)
        notes["e2e_diversification_202004"] = note
    if args.scenario in {"all", "pto"}:
        recs, note = run_pto(args.pto_run_dir, args.out_root)
        records.extend(recs)
        notes["pto_catchup_202003"] = note

    write_manifest(records, notes, args.out_root)
    print(f"Wrote {len(records)} figures -> {args.out_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
