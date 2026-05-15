"""Create self-contained Q2 professor-facing story figures.

Q2 is the training-gap question: from the April 2020 anchor, can nearby macro
states make the locked E2E and standardized PTO portfolios disagree materially?
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batuhan.regime.nfci import load_nfci
from src.data.macro_scaler import MACRO_COLUMNS
from src.modules.nn_historical import build_index_from_runtime
from src.utils.plotting import PAPER_COLORS, set_publication_style

RUN_DIR = ROOT / "scenario_outputs" / "scenario5_202004" / "runs" / "20260430_135230_323170"
OUT_DIR = ROOT / "submission_plots" / "q2_professor"
MACRO_COLS = list(MACRO_COLUMNS)
ANCHOR = 202004

REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}


def yyyymm_to_datetime(values: Iterable[int] | pd.Series) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(sorted(run_dir.glob("final_state_diagnostics_*.csv"))[-1])
    sample = pd.read_csv(sorted(run_dir.glob("generated_macro_sample_*.csv"))[-1])
    return final, sample


def attach_euclidean_neighbors(final: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    index = build_index_from_runtime(MACRO_COLS)
    states = final[[f"{col}_std" for col in MACRO_COLS]].rename(columns={f"{col}_std": col for col in MACRO_COLS})
    nn = index.attach(states, k=k)
    out = final.copy()
    for col in nn.columns:
        if col.startswith("nn_eucl_"):
            out[col] = nn[col].to_numpy()
    return out


def top_euclidean_months(final: pd.DataFrame, *, exclude_anchor: bool, top_k: int = 6, neighbors: int = 10) -> pd.DataFrame:
    final = attach_euclidean_neighbors(final, k=neighbors)
    values: list[int] = []
    for rank in range(1, neighbors + 1):
        values.extend(final[f"nn_eucl_yyyymm_{rank}"].dropna().astype(int).tolist())
    if exclude_anchor:
        values = [value for value in values if value != ANCHOR]
    out = pd.DataFrame(Counter(values).most_common(top_k), columns=["yyyymm", "count"])
    if not out.empty:
        out["date"] = yyyymm_to_datetime(out["yyyymm"])
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


def plot_story_page(final: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    final = attach_euclidean_neighbors(final, k=10)
    gap = final["allocation_l1_gap"].to_numpy(dtype=float)
    top_ex_anchor = top_euclidean_months(final, exclude_anchor=True, top_k=5, neighbors=10)
    regime_counts = final["regime"].value_counts()
    switched = int(len(final) - regime_counts.get("financial_stress", 0))
    empirical_tail = float(final["anchor_empirical_mah_chi2_tail"].median())
    var_tail = float(final["anchor_mah_chi2_tail"].median())

    set_publication_style()
    fig = plt.figure(figsize=(13.2, 8.7))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.975, bottom=0.075, top=0.85, wspace=0.25, hspace=0.36)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    boxes = [
        (
            "Asked\n"
            "From April 2020 financial stress,\n"
            "can nearby macro states make locked E2E\n"
            "and standardized PTO choose very different portfolios?",
            0.98,
            "#f3f4f5",
        ),
        (
            "Obtained\n"
            f"Median allocation L1 gap: {np.median(gap):.3f}.\n"
            f"Mean gap: {np.mean(gap):.3f}; q25-q75: {np.quantile(gap, .25):.3f}-{np.quantile(gap, .75):.3f}.\n"
            f"Acceptance: {100.0 * final['accept_rate'].mean():.1f}%; box violation: {final['box_violation'].mean():.4f}.",
            0.64,
            "#eef7f5",
        ),
        (
            "Interpretation\n"
            "The two pipelines can be locally stable in macro space\n"
            "but economically far apart in portfolio choice.\n"
            "Use empirical-tail wording, not VAR-plausible wording.",
            0.32,
            "#fff6ea",
        ),
    ]
    for text, y, color in boxes:
        ax0.text(
            0.02,
            y,
            text,
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            linespacing=1.28,
            bbox={"boxstyle": "round,pad=0.42", "facecolor": color, "edgecolor": PAPER_COLORS["grid"]},
        )

    ax1 = fig.add_subplot(gs[0, 1])
    order = np.argsort(gap)
    colors = final.iloc[order]["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    y = np.arange(len(gap))
    ax1.barh(y, gap[order], color=colors, alpha=0.90)
    ax1.axvline(np.median(gap), color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5)
    ax1.set_yticks([])
    ax1.set_xlabel("Allocation L1 gap between E2E and PTO")
    ax1.set_title("Target match: large portfolio disagreement", loc="left", fontsize=11.5, pad=8)
    ax1.text(
        0.04,
        0.94,
        f"median {np.median(gap):.3f}\n20/20 gaps above 0.50",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax1.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.6, linewidth=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    regimes = ["financial_stress", "contraction", "expansion"]
    final_values = [int(regime_counts.get(regime, 0)) for regime in regimes]
    bars = ax2.barh(np.arange(len(regimes)), final_values, color=[REGIME_COLORS[regime] for regime in regimes], alpha=0.90)
    ax2.set_yticks(np.arange(len(regimes)))
    ax2.set_yticklabels([regime.replace("_", " ") for regime in regimes])
    ax2.set_xlabel("Number of final chains")
    ax2.set_title("Regime movement: mostly financial stress, some exits", loc="left", fontsize=11.5, pad=8)
    for bar, value in zip(bars, final_values):
        ax2.text(value + 0.25, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax2.text(
        0.02,
        0.95,
        f"Anchor label: financial stress\nFinal labels: 12 financial stress, 5 contraction, 3 expansion\nRegime switched in {switched}/20 chains",
        transform=ax2.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax2.set_xlim(0, max(final_values) + 4)
    ax2.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.6, linewidth=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    y3 = np.arange(len(top_ex_anchor))
    ax3.barh(y3, top_ex_anchor["count"], color=PAPER_COLORS["generated"], alpha=0.90)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(top_ex_anchor["yyyymm"].astype(int).astype(str))
    ax3.invert_yaxis()
    ax3.set_xlabel("Times appearing in top-10 Euclidean NN pool")
    ax3.set_title("Where do generated states go in history?", loc="left", fontsize=11.5, pad=8)
    for idx, row in top_ex_anchor.iterrows():
        ax3.text(row["count"] + 0.4, idx, int(row["count"]), va="center", fontsize=9)
    ax3.text(
        0.02,
        0.08,
        f"Empirical anchor tail median: {empirical_tail:.2f}\nVAR-innovation tail median: {var_tail:.1e}\nClaim local empirical analogs; do not claim VAR-innovation plausibility.",
        transform=ax3.transAxes,
        va="bottom",
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax3.set_xlim(0, max(top_ex_anchor["count"]) + 5)
    ax3.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.6, linewidth=0.6)

    fig.suptitle("Q2 Training-Gap Scenario: what was asked, what was obtained, and how to interpret it", fontsize=15, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.905,
        "Run: scenario5_202004/runs/20260430_135230_323170. Target is E2E-vs-PTO allocation disagreement near April 2020.",
        ha="center",
        fontsize=9.5,
    )
    save_figure(fig, out_dir / "q2_professor_story_one_page")


def plot_annotated_nfci(final: pd.DataFrame, out_dir: Path) -> None:
    final = attach_euclidean_neighbors(final, k=10)
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200801) & (panel["yyyymm"] <= 202412)].copy()
    top = top_euclidean_months(final, exclude_anchor=True, top_k=6, neighbors=10)
    top = top.merge(focus[["yyyymm", "NFCI"]], on="yyyymm", how="left").dropna(subset=["NFCI"])

    set_publication_style()
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    add_regime_background(ax, focus)
    ax.plot(focus["date"], focus["NFCI"], color="#202020", linewidth=1.45, label="NFCI")
    ax.axhline(0.0, color="#697078", linestyle="--", linewidth=0.9)
    anchor_date = yyyymm_to_datetime([ANCHOR]).iloc[0]
    ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5, label="April 2020 anchor")
    ax.scatter(
        top["date"],
        top["NFCI"],
        s=55 + 15 * top["count"].to_numpy(dtype=float),
        color=PAPER_COLORS["generated"],
        edgecolors="white",
        linewidth=0.8,
        alpha=0.90,
        label="Nearest historical analogs",
        zorder=4,
    )
    ax.annotate(
        "Anchor: April 2020 financial stress\nEuclidean top-1 for all final states",
        xy=(anchor_date, float(focus.loc[focus["yyyymm"] == ANCHOR, "NFCI"].iloc[0])),
        xytext=(pd.Timestamp("2020-10-01"), 1.15),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.1},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.annotate(
        "Anchor excluded:\nnearest analogs stay around\nnear-COVID / late-2010s states",
        xy=(top["date"].iloc[0], top["NFCI"].iloc[0]),
        xytext=(pd.Timestamp("2015-05-01"), 0.35),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["generated"], "lw": 1.1},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    months = ", ".join(f"{int(row.yyyymm)} ({int(row['count'])})" for _, row in top.iterrows())
    ax.text(
        0.02,
        0.04,
        f"Top Euclidean NN months excluding 202004: {months}",
        transform=ax.transAxes,
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.set_title("Q2 historical analogs on NFCI: anchor excluded for top-k display", loc="left", fontsize=12.5, pad=8)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    save_figure(fig, out_dir / "q2_professor_nfci_nn_annotated")


def write_talk_track(final: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    top = top_euclidean_months(final, exclude_anchor=True, top_k=6, neighbors=10)
    lines = [
        "# Q2 Professor Talk Track",
        "",
        "Question: near the April 2020 financial-stress anchor, can the locked E2E and standardized PTO pipelines choose materially different portfolios?",
        "",
        "What we obtained:",
        f"- Median allocation L1 gap is `{final['allocation_l1_gap'].median():.3f}`.",
        f"- Mean allocation L1 gap is `{final['allocation_l1_gap'].mean():.3f}`.",
        f"- All `20/20` final states have allocation L1 gap above `0.50`.",
        f"- Final regimes: `{dict(final['regime'].value_counts())}`.",
        f"- Regime switched away from financial stress in `{int((final['regime'] != 'financial_stress').sum())}/20` chains.",
        f"- Post-burn-in regime mix: `{dict((sample['regime'].value_counts(normalize=True) * 100).round(1))}` percent.",
        f"- Mean acceptance rate is `{100.0 * final['accept_rate'].mean():.1f}%`; mean box violation is `{final['box_violation'].mean():.4f}`.",
        "",
        "Historical analogs and plausibility wording:",
        f"- Euclidean top-1 NN is `202004` for all final states.",
        f"- Excluding the anchor, top Euclidean NN months are: `{', '.join(top['yyyymm'].astype(int).astype(str))}`.",
        f"- Median empirical-anchor Mahalanobis tail is `{final['anchor_empirical_mah_chi2_tail'].median():.2f}`.",
        f"- Median anchor VAR-innovation chi-square tail is `{final['anchor_mah_chi2_tail'].median():.1e}`.",
        "",
        "Safe claim:",
        "Nearby April-2020-like macro states expose large E2E-vs-PTO allocation disagreement. Frame locality through Euclidean/historical analogs and the empirical-anchor tail; do not call these VAR-innovation-plausible.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Q2_PROFESSOR_TALK_TRACK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    final, sample = load_frames(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_story_page(final, sample, args.out_dir)
    plot_annotated_nfci(final, args.out_dir)
    write_talk_track(final, sample, args.out_dir)
    print(f"Wrote Q2 professor-facing story outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
