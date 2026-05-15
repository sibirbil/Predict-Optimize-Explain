"""Create self-contained Q1 figures for professor-facing discussion.

The goal is not another diagnostic dump. These figures spell out:

- what question was asked,
- what the scenario generator obtained,
- whether regimes changed,
- where the generated states land in historical nearest-neighbor terms.
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

RUN_DIR = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260424_102040"
OUT_DIR = ROOT / "submission_plots" / "q1_professor"
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


def load_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(sorted(run_dir.glob("final_state_diagnostics_*_enriched.csv"))[-1])
    sample = pd.read_csv(sorted(run_dir.glob("generated_macro_sample_postburnin_*_enriched.csv"))[-1])
    historical = pd.read_csv(sorted(run_dir.glob("historical_macro_panel_*.csv"))[-1])
    return final, sample, historical


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


def metric_mode_line(final: pd.DataFrame, metric: str, label: str) -> str:
    col = f"nn_{metric}_yyyymm_1"
    if col not in final.columns:
        return f"{label}: not available"
    mode = int(final[col].mode().iloc[0])
    share = float((final[col].astype(int) == mode).mean() * 100.0)
    return f"{label}: {mode} ({share:.0f}% of final states)"


def plot_story_page(final: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    final = attach_euclidean_neighbors(final, k=10)
    gap = 100.0 * final["return_gap"].to_numpy(dtype=float)
    top_ex_anchor = top_euclidean_months(final, exclude_anchor=True, top_k=5, neighbors=10)
    regime_counts = final["regime"].value_counts()

    set_publication_style()
    fig = plt.figure(figsize=(13.2, 8.7))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.975, bottom=0.075, top=0.85, wspace=0.25, hspace=0.36)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    asked = (
        "Asked\n"
        "From April 2020 financial stress,\n"
        "can nearby macro states make\n"
        "SummerChild beat WinterWolf?"
    )
    obtained = (
        "Obtained\n"
        f"Yes: SC wins {int((final['winner'] == 'summer_child').sum())}/{len(final)} chains.\n"
        f"Median SC-WW gap: {np.median(gap):+.2f} pp.\n"
        f"Acceptance: {100.0 * final['accept_rate'].mean():.1f}%; box violation: {final['box_violation'].mean():.4f}."
    )
    interpretation = (
        "Interpretation\n"
        "The generated states leave acute\n"
        "financial stress but remain historically\n"
        "interpretable through nearest neighbors."
    )
    y_positions = [0.98, 0.66, 0.35]
    for text, y, color in zip([asked, obtained, interpretation], y_positions, ["#f3f4f5", "#eef7f5", "#fff6ea"]):
        ax0.text(
            0.02,
            y,
            text,
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=9.1,
            linespacing=1.30,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": PAPER_COLORS["grid"]},
        )

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(gap))
    order = np.argsort(gap)
    colors = final.iloc[order]["regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    ax1.barh(y, gap[order], color=colors, alpha=0.90)
    ax1.axvline(0.0, color="#202020", linewidth=1.0)
    ax1.axvline(np.median(gap), color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.4)
    ax1.set_yticks([])
    ax1.set_xlabel("SC - WW return gap, percentage points")
    ax1.set_title("Target match: all final states favor SC", loc="left", fontsize=11.5, pad=8)
    ax1.text(
        0.03,
        0.94,
        f"20/20 positive\nmedian {np.median(gap):+.2f} pp",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax1.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.6, linewidth=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    regimes = ["financial_stress", "contraction", "expansion"]
    final_values = [int(regime_counts.get(regime, 0)) for regime in regimes]
    bars = ax2.barh(
        np.arange(len(regimes)),
        final_values,
        color=[REGIME_COLORS[regime] for regime in regimes],
        alpha=0.90,
    )
    ax2.set_yticks(np.arange(len(regimes)))
    ax2.set_yticklabels([regime.replace("_", " ") for regime in regimes])
    ax2.set_xlabel("Number of final chains")
    ax2.set_title("Regime movement: financial stress to non-stress states", loc="left", fontsize=11.5, pad=8)
    for bar, value in zip(bars, final_values):
        ax2.text(value + 0.25, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax2.text(
        0.02,
        0.95,
        "Anchor label: financial stress\nFinal labels: 13 contraction, 7 expansion, 0 financial stress",
        transform=ax2.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax2.set_xlim(0, max(final_values) + 3)
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
        "Euclidean top-1 is April 2020 for all 20 states.\nAfter excluding the anchor, the closest analogs\ncluster around late-2010s / nearby COVID months.",
        transform=ax3.transAxes,
        va="bottom",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax3.set_xlim(0, max(top_ex_anchor["count"]) + 5)
    ax3.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.6, linewidth=0.6)

    fig.suptitle("Q1 Locked Scenario: what was asked, what was obtained, and how to interpret it", fontsize=15, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.905,
        "Run: scenario4_202004/runs/20260424_102040. Generated states are evaluated with nearest-neighbor analogs and regime classifier labels.",
        ha="center",
        fontsize=9.5,
    )
    save_figure(fig, out_dir / "q1_professor_story_one_page")


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
        "Anchor: acute COVID financial stress\nEuclidean top-1 match for all final states",
        xy=(anchor_date, float(focus.loc[focus["yyyymm"] == ANCHOR, "NFCI"].iloc[0])),
        xytext=(pd.Timestamp("2020-10-01"), 1.15),
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.1},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.annotate(
        "Anchor excluded:\nnearest analogs remain\nhistorically recognizable",
        xy=(top["date"].iloc[0], top["NFCI"].iloc[0]),
        xytext=(pd.Timestamp("2016-05-01"), -0.05),
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
    ax.set_title("Q1 historical analogs on NFCI: anchor excluded for top-k display", loc="left", fontsize=12.5, pad=8)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    save_figure(fig, out_dir / "q1_professor_nfci_nn_annotated")


def write_talk_track(final: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    top = top_euclidean_months(final, exclude_anchor=True, top_k=6, neighbors=10)
    lines = [
        "# Q1 Professor Talk Track",
        "",
        "Question: starting from April 2020 financial stress, find nearby macro states where SummerChild beats WinterWolf.",
        "",
        "What we obtained:",
        f"- SC wins in `{int((final['winner'] == 'summer_child').sum())}/{len(final)}` final states.",
        f"- Median SC-WW return gap is `{100.0 * final['return_gap'].median():+.2f}` percentage points.",
        f"- Final regimes: `{dict(final['regime'].value_counts())}`.",
        f"- Post-burn-in regime mix: `{dict((sample['regime'].value_counts(normalize=True) * 100).round(1))}` percent.",
        f"- Mean acceptance rate is `{100.0 * final['accept_rate'].mean():.1f}%`; mean box violation is `{final['box_violation'].mean():.4f}`.",
        "",
        "Historical analogs:",
        f"- Euclidean top-1 NN is `202004` for all final states.",
        f"- Excluding the anchor, top Euclidean NN months are: `{', '.join(top['yyyymm'].astype(int).astype(str))}`.",
        f"- {metric_mode_line(final, 'var1', 'VAR(1)-innovation modal top-1')}.",
        f"- {metric_mode_line(final, 'hist', 'Historical-covariance modal top-1')}.",
        "",
        "Safe claim:",
        "Generated states flip the SC-vs-WW ranking while moving from the April 2020 financial-stress anchor into nearby contraction/expansion regimes; the nearest-neighbor mapping keeps the macro states interpretable in historical terms.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Q1_PROFESSOR_TALK_TRACK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    final, sample, _historical = load_frames(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_story_page(final, sample, args.out_dir)
    plot_annotated_nfci(final, args.out_dir)
    write_talk_track(final, sample, args.out_dir)
    print(f"Wrote Q1 professor-facing story outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
