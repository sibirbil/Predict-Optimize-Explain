"""Create simple submission-facing plots for the locked Q1 run.

These figures are intentionally one-message-at-a-time:

1. top historical NN months on the NFCI/regime timeline,
2. seed-level SC minus WW return gaps,
3. macro shifts from the April 2020 anchor,
4. generated regime composition.
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
from src.data.macro_scaler import MACRO_COLUMNS, MacroScaler
from src.utils.plotting import PAPER_COLORS, set_publication_style

Q1_RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260424_102040"
OUT_DIR = ROOT / "submission_plots" / "q1_simple"

REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}
METRIC_STYLES = {
    "var1": ("VAR(1) innovation", "#4c78a8", "o"),
    "hist": ("Historical covariance", "#287c71", "s"),
    "eucl": ("Euclidean z-score", "#8f63a8", "^"),
}
MACRO_LABELS = {
    "dp": "Dividend-price",
    "ep": "Earnings-price",
    "bm": "Book-to-market",
    "ntis": "Net equity issuance",
    "tbl": "T-bill",
    "tms": "Term spread",
    "dfy": "Default yield spread",
    "svar": "Stock variance",
    "infl": "Inflation",
}


def yyyymm_to_datetime(values: Iterable[int] | pd.Series) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def newest_csv(run_dir: Path, pattern: str) -> Path:
    paths = sorted(run_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No file matching {pattern} in {run_dir}")
    return paths[-1]


def load_q1(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(newest_csv(run_dir, "seed_summary_*_enriched.csv"))
    final = pd.read_csv(newest_csv(run_dir, "final_state_diagnostics_*_enriched.csv"))
    sample = pd.read_csv(newest_csv(run_dir, "generated_macro_sample_postburnin_*_enriched.csv"))
    return seed, final, sample


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
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
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


def nn_month_counts(final: pd.DataFrame, metric: str, neighbors: int = 3, top_k: int = 3) -> pd.DataFrame:
    values: list[int] = []
    for rank in range(1, neighbors + 1):
        col = f"nn_{metric}_yyyymm_{rank}"
        if col in final.columns:
            values.extend(final[col].dropna().astype(int).tolist())
    counts = Counter(values).most_common(top_k)
    out = pd.DataFrame(counts, columns=["yyyymm", "count"])
    if not out.empty:
        out["date"] = yyyymm_to_datetime(out["yyyymm"])
    return out


def anchor_std_state(anchor: int = 202004) -> dict[str, float]:
    macro = pd.read_parquet(ROOT / "runtime_universe500" / "data" / "macro_final.parquet")
    row = macro.loc[macro["yyyymm"].astype(int) == int(anchor)]
    if row.empty:
        raise ValueError(f"Anchor {anchor} not found in runtime macro panel")
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    raw = row.iloc[0][list(MACRO_COLUMNS)].to_numpy(dtype=float)
    mean = scaler.mean.detach().cpu().numpy().astype(float)
    std = scaler.std.detach().cpu().numpy().astype(float)
    z = (raw - mean) / std
    return dict(zip(MACRO_COLUMNS, z))


def add_regime_background(ax: plt.Axes, panel: pd.DataFrame) -> None:
    for start, end, regime in regime_segments(panel):
        ax.axvspan(
            start,
            end,
            color=REGIME_COLORS.get(regime, PAPER_COLORS["grid"]),
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )


def plot_nn_on_nfci(final: pd.DataFrame, out_dir: Path) -> None:
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200801) & (panel["yyyymm"] <= 202312)].copy()
    set_publication_style()
    fig, (ax, note_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 5.3),
        gridspec_kw={"height_ratios": [4.0, 0.85], "hspace": 0.16},
    )

    add_regime_background(ax, focus)
    ax.plot(focus["date"], focus["NFCI"], color="#202020", linewidth=1.4, label="NFCI")
    ax.axhline(0.0, color="#697078", linestyle="--", linewidth=0.9)
    ax.axvline(yyyymm_to_datetime([202004]).iloc[0], color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5)

    table_lines = []
    for metric, (label, color, marker) in METRIC_STYLES.items():
        counts = nn_month_counts(final, metric)
        counts = counts.merge(focus[["yyyymm", "NFCI"]], on="yyyymm", how="left")
        counts = counts.dropna(subset=["NFCI"])
        if counts.empty:
            continue
        sizes = 48 + counts["count"].to_numpy(dtype=float) * 18
        ax.scatter(
            counts["date"],
            counts["NFCI"],
            s=sizes,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidth=0.8,
            alpha=0.90,
            label=label,
            zorder=4,
        )
        months = ", ".join(f"{int(row.yyyymm)} ({int(row['count'])})" for _, row in counts.iterrows())
        table_lines.append(f"{label}: {months}")

    ax.set_title("Q1 historical analogs on the NFCI regime timeline", loc="left", pad=8, fontsize=12.5)
    ax.set_ylabel("NFCI")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", ncols=2, frameon=False, fontsize=9)

    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.78,
        "Top 3 months aggregate the 3 nearest neighbors across 20 generated final states; counts in parentheses.",
        ha="left",
        va="top",
        fontsize=8.4,
        color="#333333",
    )
    note_ax.text(0.0, 0.26, "\n".join(table_lines), ha="left", va="top", fontsize=8.0, color="#111111")
    save_figure(fig, out_dir / "q1_nfci_top_nn")


def plot_return_gaps(seed: pd.DataFrame, out_dir: Path) -> None:
    frame = seed.sort_values("return_gap", ascending=True).reset_index(drop=True)
    colors = frame["final_regime"].map(REGIME_COLORS).fillna(PAPER_COLORS["generated"])
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    y = np.arange(len(frame))
    ax.barh(y, frame["return_gap"] * 100.0, color=colors, alpha=0.88)
    median = float(frame["return_gap"].median() * 100.0)
    ax.axvline(median, color="#202020", linestyle="--", linewidth=1.2, label=f"Median: {median:.2f} pp")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["seed"].astype(int).astype(str))
    ax.set_xlabel("SC minus WW return gap, percentage points")
    ax.set_ylabel("Seed")
    ax.set_title("Q1: SC beats WW in every generated final state", loc="left", pad=8, fontsize=12.5)
    ax.grid(axis="x", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markeredgecolor="none", markersize=8, label=regime)
        for regime, color in REGIME_COLORS.items()
        if regime in set(frame["final_regime"])
    ]
    handles.append(plt.Line2D([0], [0], color="#202020", linestyle="--", linewidth=1.2, label=f"Median: {median:.2f} pp"))
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save_figure(fig, out_dir / "q1_return_gap_by_seed")


def plot_macro_shifts(final: pd.DataFrame, out_dir: Path) -> None:
    anchor = anchor_std_state()
    rows = []
    for macro in MACRO_LABELS:
        col = f"{macro}_std"
        if col not in final.columns:
            continue
        vals = final[col].astype(float) - anchor[macro]
        rows.append(
            {
                "macro": macro,
                "label": MACRO_LABELS[macro],
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            }
        )
    frame = pd.DataFrame(rows).sort_values("median")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(frame))
    xerr = np.vstack([frame["median"] - frame["q25"], frame["q75"] - frame["median"]])
    colors = np.where(frame["median"] >= 0.0, PAPER_COLORS["positive"], PAPER_COLORS["negative"])
    ax.errorbar(frame["median"], y, xerr=xerr, fmt="none", ecolor="#7c858c", elinewidth=1.8, capsize=3, zorder=2)
    ax.scatter(frame["median"], y, s=58, color=colors, edgecolors="white", linewidth=0.8, zorder=3)
    ax.axvline(0.0, color="#202020", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlabel("Standardized shift from April 2020 anchor")
    ax.set_title("Q1 macro movement: valuation variables rise", loc="left", pad=8, fontsize=12.5)
    ax.grid(axis="x", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    save_figure(fig, out_dir / "q1_macro_shift_iqr")


def plot_regime_mix(seed: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for label, frame, col in [
        ("Post-burn-in states", sample, "regime"),
        ("Final states", seed, "final_regime"),
    ]:
        counts = frame[col].value_counts(normalize=True)
        row = {"source": label}
        row.update({regime: float(counts.get(regime, 0.0)) for regime in REGIME_COLORS})
        rows.append(row)
    mix = pd.DataFrame(rows)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    left = np.zeros(len(mix))
    y = np.arange(len(mix))
    for regime, color in REGIME_COLORS.items():
        values = mix[regime].to_numpy(dtype=float)
        ax.barh(y, values * 100.0, left=left * 100.0, color=color, label=regime, alpha=0.88)
        for idx, value in enumerate(values):
            if value >= 0.08:
                ax.text((left[idx] + value / 2.0) * 100.0, idx, f"{value * 100:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(mix["source"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of generated states")
    ax.set_title("Q1 generated states move out of April 2020 financial stress", loc="left", pad=8, fontsize=12.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncols=3, frameon=False)
    ax.grid(axis="x", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.7)
    save_figure(fig, out_dir / "q1_regime_mix")


def write_readme(out_dir: Path, seed: pd.DataFrame, final: pd.DataFrame) -> None:
    modal_lines = []
    for metric, (label, _, _) in METRIC_STYLES.items():
        col = f"nn_{metric}_yyyymm_1"
        if col in final.columns:
            modal = int(final[col].mode().iloc[0])
            modal_lines.append(f"- {label}: `{modal}`")
    text = f"""# Q1 Simple Submission Plots

Generated by `python scripts/reporting/report_q1_simple_plots.py`.

## Files

- `q1_nfci_top_nn.pdf/png`: professor-requested NFCI/regime timeline with top historical NN months.
- `q1_return_gap_by_seed.pdf/png`: seed-level SC minus WW return gaps.
- `q1_macro_shift_iqr.pdf/png`: median macro shifts from April 2020 with interquartile ranges.
- `q1_regime_mix.pdf/png`: generated regime composition for post-burn-in and final states.

## Locked Q1 Facts

- Seeds: `{len(seed)}`
- SC win rate: `{(seed["return_gap"] > 0).mean() * 100:.1f}%`
- Median return gap: `{seed["return_gap"].median() * 100:.2f}` percentage points
- Mean acceptance rate: `{seed["accept_rate"].mean() * 100:.1f}%`
- Mean box violation: `{seed["final_box_violation"].mean():.4f}`

## Modal Top-1 NN Months

{chr(10).join(modal_lines)}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Q1_RUN)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    seed, final, sample = load_q1(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_nn_on_nfci(final, args.out_dir)
    plot_return_gaps(seed, args.out_dir)
    plot_macro_shifts(final, args.out_dir)
    plot_regime_mix(seed, sample, args.out_dir)
    write_readme(args.out_dir, seed, final)
    print(f"Wrote simple Q1 submission plots to {args.out_dir}")


if __name__ == "__main__":
    main()
