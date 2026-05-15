"""Generate submission-ready story plots for the scenario manuscript.

Outputs are derived from locked/candidate run artifacts only; no scenario is
re-sampled. The key professor-requested plot overlays top-k historical nearest
neighbors from generated scenarios on the NFCI/regime timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batuhan.regime.nfci import load_nfci
from src.data.macro_scaler import MacroScaler
from src.modules.runtime_regime import MACRO_ORDER
from src.utils.plotting import PAPER_COLORS, set_publication_style

Q1_RUN = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260424_102040"
Q2_RUN = ROOT / "scenario_outputs" / "scenario5_202004" / "runs" / "20260430_135230_323170"
OUT_DIR = ROOT / "submission_plots"
MACRO_COLS = list(MACRO_ORDER)
REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}
METRIC_LABELS = {
    "var1": "VAR(1)-innovation Mahalanobis NN",
    "hist": "Historical-covariance Mahalanobis NN",
    "eucl": "Euclidean z-score NN",
}


def yyyymm_to_datetime(values: pd.Series | Sequence[int]) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_config(run_dir: Path) -> dict:
    configs = sorted(run_dir.glob("config_*.json"))
    if not configs:
        return {}
    return json.loads(configs[0].read_text(encoding="utf-8"))


def load_run_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    config = load_config(run_dir)
    final_paths = sorted(run_dir.glob("final_state_diagnostics_*_enriched.csv")) or sorted(run_dir.glob("final_state_diagnostics_*.csv"))
    sample_paths = (
        sorted(run_dir.glob("generated_macro_sample_postburnin_*_enriched.csv"))
        or sorted(run_dir.glob("generated_macro_sample_postburnin_*.csv"))
        or sorted(run_dir.glob("generated_macro_sample_*.csv"))
    )
    hist_paths = sorted(run_dir.glob("historical_macro_panel_*.csv"))
    if not final_paths:
        raise FileNotFoundError(f"No final diagnostics found in {run_dir}")
    if not sample_paths:
        raise FileNotFoundError(f"No generated sample found in {run_dir}")
    if not hist_paths:
        raise FileNotFoundError(f"No historical macro panel found in {run_dir}")
    final = pd.read_csv(final_paths[0])
    sample = pd.read_csv(sample_paths[0])
    hist = pd.read_csv(hist_paths[0])
    return final, sample, hist, config


def load_regime_nfci_panel() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    probs = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_probability_panel.csv")
    nfci = load_nfci().rename("NFCI").reset_index()
    panel = labels.merge(probs, on="yyyymm", how="left").merge(nfci, on="yyyymm", how="left")
    panel = panel.sort_values("yyyymm").reset_index(drop=True)
    panel["date"] = yyyymm_to_datetime(panel["yyyymm"])
    return panel


def anchor_std_for_date(date: int) -> dict[str, float]:
    macro = pd.read_parquet(ROOT / "runtime_universe500" / "data" / "macro_final.parquet")
    row = macro.loc[macro["yyyymm"].astype(int) == int(date)]
    if row.empty:
        raise ValueError(f"Anchor date {date} not found in runtime macro panel")
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    mean = scaler.mean.detach().cpu().numpy().astype(float)
    std = scaler.std.detach().cpu().numpy().astype(float)
    raw = row.iloc[0][MACRO_COLS].to_numpy(dtype=float)
    z = (raw - mean) / std
    return dict(zip(MACRO_COLS, z))


def contiguous_regime_segments(panel: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    rows = panel[["date", "regime"]].dropna().reset_index(drop=True)
    if rows.empty:
        return []
    segments = []
    start = rows.loc[0, "date"]
    current = rows.loc[0, "regime"]
    prev = rows.loc[0, "date"]
    for idx in range(1, len(rows)):
        regime = rows.loc[idx, "regime"]
        date = rows.loc[idx, "date"]
        if regime != current:
            segments.append((start, prev + pd.offsets.MonthEnd(1), str(current)))
            start = date
            current = regime
        prev = date
    segments.append((start, prev + pd.offsets.MonthEnd(1), str(current)))
    return segments


def top_nn_counts(frame: pd.DataFrame, metric: str, k_neighbors: int, top_k_months: int) -> pd.DataFrame:
    cols = [f"nn_{metric}_yyyymm_{rank}" for rank in range(1, k_neighbors + 1)]
    cols = [col for col in cols if col in frame.columns]
    if not cols:
        return pd.DataFrame(columns=["yyyymm", "count"])
    values = []
    for col in cols:
        values.extend(frame[col].dropna().astype(int).tolist())
    counts = Counter(values)
    out = pd.DataFrame(counts.most_common(top_k_months), columns=["yyyymm", "count"])
    if not out.empty:
        out["date"] = yyyymm_to_datetime(out["yyyymm"])
    return out


def plot_nfci_regime_nn(
    q1_final: pd.DataFrame,
    q2_final: pd.DataFrame,
    out_dir: Path,
    *,
    source_label: str = "final states",
    top_k_months: int = 5,
    k_neighbors: int = 3,
) -> None:
    panel = load_regime_nfci_panel()
    set_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 9.2), sharex=True)
    y_min = float(panel["NFCI"].min() - 0.35)
    y_max = float(panel["NFCI"].max() + 0.55)
    anchor_date = yyyymm_to_datetime([202004]).iloc[0]
    combined = pd.concat(
        [
            q1_final.assign(_scenario="Q1: SC beats WW"),
            q2_final.assign(_scenario="Q2: E2E vs PTO gap"),
        ],
        ignore_index=True,
        sort=False,
    )
    scenario_marker = {
        "Q1: SC beats WW": "o",
        "Q2: E2E vs PTO gap": "s",
    }
    scenario_color = {
        "Q1: SC beats WW": PAPER_COLORS["generated"],
        "Q2: E2E vs PTO gap": "#6f4e7c",
    }

    for ax, metric in zip(axes, ("var1", "hist", "eucl")):
        inset_lines = []
        for start, end, regime in contiguous_regime_segments(panel):
            ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#d8dadd"), alpha=0.10, linewidth=0)
        ax.plot(panel["date"], panel["NFCI"], color="#222222", linewidth=1.25, label="NFCI")
        ax.axhline(0.0, color="#6f757c", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.axvline(anchor_date, color=PAPER_COLORS["anchor"], linewidth=1.6, linestyle="--", label="April 2020 anchor")

        for scenario, part in combined.groupby("_scenario"):
            counts = top_nn_counts(part, metric, k_neighbors=k_neighbors, top_k_months=top_k_months)
            if counts.empty:
                continue
            inset_lines.append(f"{scenario.split(':', 1)[0]}: " + ", ".join(str(int(v)) for v in counts["yyyymm"].head(top_k_months)))
            counts = counts.merge(panel[["yyyymm", "NFCI", "regime"]], on="yyyymm", how="left")
            sizes = 42 + 22 * counts["count"].to_numpy(dtype=float)
            y = counts["NFCI"].fillna(y_min + 0.15)
            ax.scatter(
                counts["date"],
                y,
                s=sizes,
                marker=scenario_marker[scenario],
                color=scenario_color[scenario],
                alpha=0.82,
                edgecolors="white",
                linewidth=0.8,
                zorder=5,
                label=scenario,
            )

        ax.set_ylabel("NFCI")
        ax.set_ylim(y_min, y_max)
        ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
        if inset_lines:
            ax.text(
                0.012,
                0.94,
                "\n".join(inset_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.90},
            )

    handles = [
        plt.Line2D([0], [0], color="#222222", linewidth=1.25, label="NFCI"),
        plt.Line2D([0], [0], color=PAPER_COLORS["anchor"], linewidth=1.6, linestyle="--", label="April 2020 anchor"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_COLORS["generated"], markeredgecolor="white", markersize=8, label="Q1 top-k NN"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#6f4e7c", markeredgecolor="white", markersize=8, label="Q2 top-k NN"),
    ]
    regime_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.18, label=label.replace("_", " "))
        for label, color in REGIME_COLORS.items()
    ]
    fig.legend(
        handles=handles + regime_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.915),
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.suptitle(
        "Top-k Historical Nearest Neighbors on the NFCI Regime Timeline",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        f"Top {top_k_months} historical months from top-{k_neighbors} NN columns across generated {source_label}; marker size is frequency.",
        ha="center",
        fontsize=9,
    )
    axes[-1].set_xlabel("Month")
    fig.tight_layout(rect=(0.03, 0.04, 0.99, 0.875))
    save_figure(fig, out_dir / "submission_nfci_regime_topk_nn")


def macro_shift_frame(sample: pd.DataFrame, anchor_std: dict[str, float], final: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    frame = sample if not sample.empty else final
    for col in MACRO_COLS:
        std_col = f"{col}_std"
        if std_col not in frame.columns:
            continue
        anchor = float(anchor_std[col])
        values = frame[std_col].to_numpy(dtype=float)
        rows.append(
            {
                "macro": col,
                "median": float(np.nanmedian(values - anchor)),
                "q25": float(np.nanquantile(values - anchor, 0.25)),
                "q75": float(np.nanquantile(values - anchor, 0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values("median")


def plot_q1_story(final: pd.DataFrame, sample: pd.DataFrame, anchor_std: dict[str, float], out_dir: Path) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.09, top=0.86, wspace=0.28, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    gap = 100.0 * final["return_gap"].to_numpy(dtype=float)
    colors = [REGIME_COLORS.get(reg, PAPER_COLORS["generated"]) for reg in final["regime"]]
    ax.bar(np.arange(len(gap)) + 1, gap, color=colors, alpha=0.88)
    ax.axhline(0.0, color="#111111", linewidth=1.0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("SC minus WW return gap (%)")
    ax.set_title("Q1 target: SummerChild wins in every final state", fontsize=11, fontweight="bold")
    ax.text(
        0.03,
        0.94,
        f"win rate: {100.0 * final['winner'].eq('summer_child').mean():.0f}%\nmedian gap: {np.nanmedian(gap):+.2f} pp",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.96},
    )
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[0, 1])
    counts = final["regime"].value_counts(normalize=True).reindex(["financial_stress", "contraction", "expansion"]).fillna(0.0)
    ax.bar([name.replace("_", "\n") for name in counts.index], 100.0 * counts.values, color=[REGIME_COLORS.get(name, "#999999") for name in counts.index], alpha=0.88)
    ax.set_ylabel("Final states (%)")
    ax.set_title("Generated states move out of acute stress", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[1, 0])
    shift = macro_shift_frame(sample, anchor_std=anchor_std, final=final)
    y = np.arange(len(shift))
    colors = [PAPER_COLORS["positive"] if value >= 0 else PAPER_COLORS["negative"] for value in shift["median"]]
    ax.barh(y, shift["median"], color=colors, alpha=0.88)
    ax.errorbar(
        shift["median"],
        y,
        xerr=[shift["median"] - shift["q25"], shift["q75"] - shift["median"]],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0.0, color="#111111", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(shift["macro"])
    ax.set_xlabel("Generated minus anchor, standardized macro units")
    ax.set_title("Macro signature of generated Q1 states", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[1, 1])
    metrics = ["var1", "hist", "eucl"]
    modal_labels = []
    med_dists = []
    for metric in metrics:
        modal = final[f"nn_{metric}_yyyymm_1"].dropna().astype(int).mode().iloc[0]
        modal_labels.append(f"{METRIC_LABELS[metric].split()[0]}\n{int(modal)}")
        med_dists.append(float(final[f"nn_{metric}_dist_1"].median()))
    ax.bar(modal_labels, med_dists, color=["#5b8fa8", "#7b9e6b", "#c28e4b"], alpha=0.88)
    ax.set_ylabel("Median top-1 NN distance")
    ax.set_title("Same scenarios, different historical analogs", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    fig.suptitle("Q1 Submission Story: Recession-Trained Policy Wins Under Interpretable Macro Analogs", fontsize=15, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.91,
        "The regime classifier gives the broad macro label; historical NN gives concrete economic episodes.",
        ha="center",
        fontsize=10,
    )
    save_figure(fig, out_dir / "submission_q1_story")


def plot_q2_story(final: pd.DataFrame, sample: pd.DataFrame, anchor_std: dict[str, float], out_dir: Path) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.09, top=0.86, wspace=0.28, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    values = final["allocation_l1_gap"].to_numpy(dtype=float)
    ax.hist(values, bins=10, color="#6f4e7c", alpha=0.84)
    ax.axvline(np.nanmedian(values), color=PAPER_COLORS["anchor"], linewidth=1.6, linestyle="--")
    ax.set_xlabel("E2E vs PTO allocation L1 gap")
    ax.set_ylabel("Seeds")
    ax.set_title("Q2 target: training paradigms disagree", fontsize=11, fontweight="bold")
    ax.text(
        0.04,
        0.94,
        f"median: {np.nanmedian(values):.3f}\nmean: {np.nanmean(values):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.96},
    )
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[0, 1])
    tail_cols = [
        ("anchor_mah_chi2_tail", "VAR innovation"),
        ("anchor_empirical_mah_chi2_tail", "Empirical anchor"),
    ]
    data = [final[col].clip(lower=1e-12).to_numpy(dtype=float) for col, _ in tail_cols if col in final.columns]
    labels = [label for col, label in tail_cols if col in final.columns]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.axhline(0.05, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label="0.05 threshold")
    ax.set_ylabel("Chi-square tail probability, log scale")
    ax.set_title("Plausibility metric must be labeled", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[1, 0])
    shift = macro_shift_frame(sample, anchor_std=anchor_std, final=final)
    y = np.arange(len(shift))
    colors = [PAPER_COLORS["positive"] if value >= 0 else PAPER_COLORS["negative"] for value in shift["median"]]
    ax.barh(y, shift["median"], color=colors, alpha=0.88)
    ax.errorbar(
        shift["median"],
        y,
        xerr=[shift["median"] - shift["q25"], shift["q75"] - shift["median"]],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0.0, color="#111111", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(shift["macro"])
    ax.set_xlabel("Generated minus anchor, standardized macro units")
    ax.set_title("Macro signature of Q2 disagreement states", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    ax = fig.add_subplot(gs[1, 1])
    regimes = final["regime"].value_counts(normalize=True).reindex(["financial_stress", "contraction", "expansion"]).fillna(0.0)
    ax.bar([name.replace("_", "\n") for name in regimes.index], 100.0 * regimes.values, color=[REGIME_COLORS.get(name, "#999999") for name in regimes.index], alpha=0.88)
    ax.set_ylabel("Final states (%)")
    ax.set_title("Regime classifier anchors the economic label", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    fig.suptitle("Q2 Submission Story: E2E and PTO Diverge in Empirically Local Macro States", fontsize=15, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.91,
        "This figure supports empirical-locality language; it deliberately separates empirical and VAR-innovation tails.",
        ha="center",
        fontsize=10,
    )
    save_figure(fig, out_dir / "submission_q2_story")


def write_plot_index(out_dir: Path, q1_final: pd.DataFrame, q2_final: pd.DataFrame) -> None:
    q1_modes = {
        metric: int(q1_final[f"nn_{metric}_yyyymm_1"].dropna().astype(int).mode().iloc[0])
        for metric in ("var1", "hist", "eucl")
    }
    q2_modes = {
        metric: int(q2_final[f"nn_{metric}_yyyymm_1"].dropna().astype(int).mode().iloc[0])
        for metric in ("var1", "hist", "eucl")
    }
    lines = [
        "# Submission Plots",
        "",
        "Generated by `python scripts/reporting/report_submission_plots.py`.",
        "",
        "## Files",
        "",
        "- `submission_nfci_regime_topk_nn.pdf/png`: NFCI timeline with regime shading and top-k historical NN months from Q1/Q2.",
        "- `submission_q1_story.pdf/png`: Q1 target, regimes, macro signature, and metric-specific NN analogs.",
        "- `submission_q2_story.pdf/png`: Q2 allocation gap, plausibility tails, macro signature, and regime mix.",
        "",
        "## Modal Top-1 NN Months",
        "",
        f"- Q1 VAR(1)-innovation: `{q1_modes['var1']}`; historical-covariance: `{q1_modes['hist']}`; Euclidean: `{q1_modes['eucl']}`.",
        f"- Q2 VAR(1)-innovation: `{q2_modes['var1']}`; historical-covariance: `{q2_modes['hist']}`; Euclidean: `{q2_modes['eucl']}`.",
        "",
        "## Manuscript Wording Guardrail",
        "",
        "Use the NFCI/NN plot to say that generated macro states map to concrete historical months under explicitly labeled metrics. Do not call Q2 VAR(1)-innovation plausible; its empirical-anchor tail is the manuscript-safe locality diagnostic.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create submission-ready plots for Q1/Q2 scenario story.")
    parser.add_argument("--q1-run", type=Path, default=Q1_RUN)
    parser.add_argument("--q2-run", type=Path, default=Q2_RUN)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k-months", type=int, default=5)
    parser.add_argument("--k-neighbors", type=int, default=3)
    args = parser.parse_args()

    q1_run = args.q1_run if args.q1_run.is_absolute() else ROOT / args.q1_run
    q2_run = args.q2_run if args.q2_run.is_absolute() else ROOT / args.q2_run
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_final, q1_sample, _, q1_config = load_run_frames(q1_run)
    q2_final, q2_sample, _, q2_config = load_run_frames(q2_run)
    q1_anchor_std = anchor_std_for_date(int(q1_config.get("DATE", 202004)))
    q2_anchor_std = anchor_std_for_date(int(q2_config.get("DATE", 202004)))
    plot_nfci_regime_nn(
        q1_final=q1_final,
        q2_final=q2_final,
        out_dir=out_dir,
        top_k_months=args.top_k_months,
        k_neighbors=args.k_neighbors,
    )
    plot_q1_story(q1_final, q1_sample, q1_anchor_std, out_dir)
    plot_q2_story(q2_final, q2_sample, q2_anchor_std, out_dir)
    write_plot_index(out_dir, q1_final, q2_final)
    print(f"Wrote submission plots to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
