"""Generate the requested submission plots for Q1/Q2/Q3.

The script reads locked/candidate scenario outputs and writes new figure files
under ``submission_plots/requested``. It does not mutate scenario run folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
from src.utils.plotting import PAPER_COLORS, macro_density_grid, set_publication_style

MACRO_COLS = list(MACRO_COLUMNS)
OUT_DIR = ROOT / "submission_plots" / "requested"

REGIME_COLORS = {
    "expansion": "#4c78a8",
    "contraction": "#f28e2b",
    "financial_stress": "#c44e52",
}


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    label: str
    question: str
    run_dir: Path
    anchor: int
    target_col: str
    target_label: str
    target_scale: float = 1.0
    target_suffix: str = ""


SCENARIOS = {
    "q1": ScenarioSpec(
        key="q1",
        label="Q1",
        question="SC beats WW",
        run_dir=ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260424_102040",
        anchor=202004,
        target_col="return_gap",
        target_label="SC - WW return gap",
        target_scale=100.0,
        target_suffix=" pp",
    ),
    "q2": ScenarioSpec(
        key="q2",
        label="Q2",
        question="E2E vs PTO training gap",
        run_dir=ROOT / "scenario_outputs" / "scenario5_202004" / "runs" / "20260430_135230_323170",
        anchor=202004,
        target_col="allocation_l1_gap",
        target_label="Allocation L1 gap",
    ),
    "q3": ScenarioSpec(
        key="q3",
        label="Q3",
        question="Decision fragility",
        run_dir=ROOT / "scenario_outputs" / "scenario5_202201" / "runs" / "20260425_045603_982167",
        anchor=202201,
        target_col="allocation_l1_from_anchor",
        target_label="Allocation L1 from anchor",
    ),
}


def yyyymm_to_datetime(values: Iterable[int] | pd.Series) -> pd.Series:
    vals = pd.Series(values).astype(int)
    return pd.to_datetime(vals.astype(str) + "01", format="%Y%m%d")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def newest_csv(run_dir: Path, patterns: Sequence[str]) -> Path:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(run_dir.glob(pattern)))
    paths = [p for p in paths if p.stat().st_size > 0]
    if not paths:
        raise FileNotFoundError(f"No non-empty CSV for {patterns} in {run_dir}")
    return sorted(paths)[-1]


def load_config(run_dir: Path) -> dict:
    paths = sorted(run_dir.glob("config_*.json"))
    if not paths:
        return {}
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def load_scenario(spec: ScenarioSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    final = pd.read_csv(newest_csv(spec.run_dir, ["final_state_diagnostics_*_enriched.csv", "final_state_diagnostics_*.csv"]))
    sample = pd.read_csv(
        newest_csv(
            spec.run_dir,
            [
                "generated_macro_sample_postburnin_*_enriched.csv",
                "generated_macro_sample_postburnin_*.csv",
                "generated_macro_sample_*.csv",
            ],
        )
    )
    historical = pd.read_csv(newest_csv(spec.run_dir, ["historical_macro_panel_*.csv"]))
    return final, sample, historical, load_config(spec.run_dir)


def anchor_raw(historical: pd.DataFrame, anchor: int) -> np.ndarray:
    row = historical.loc[historical["yyyymm"].astype(int) == int(anchor)]
    if row.empty:
        macro = pd.read_parquet(ROOT / "runtime_universe500" / "data" / "macro_final.parquet")
        row = macro.loc[macro["yyyymm"].astype(int) == int(anchor)]
    if row.empty:
        raise ValueError(f"Anchor {anchor} is missing from historical/runtime macro panel")
    return row.iloc[0][MACRO_COLS].to_numpy(dtype=float)


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
        ax.axvspan(
            start,
            end,
            color=REGIME_COLORS.get(regime, PAPER_COLORS["grid"]),
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )


def attach_euclidean_nn_if_needed(final: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    cols = [f"nn_eucl_yyyymm_{rank}" for rank in range(1, k + 1)]
    if all(col in final.columns for col in cols):
        return final
    std_cols = [f"{col}_std" for col in MACRO_COLS]
    if not all(col in final.columns for col in std_cols):
        raise ValueError("Final diagnostics need either nn_eucl columns or standardized macro columns.")
    index = build_index_from_runtime(MACRO_COLS)
    states = final[std_cols].rename(columns={f"{col}_std": col for col in MACRO_COLS})
    nn = index.attach(states, k=k)
    out = final.copy()
    for col in nn.columns:
        if col.startswith("nn_eucl_"):
            out[col] = nn[col].to_numpy()
    return out


def euclidean_topk(final: pd.DataFrame, anchor: int, top_k: int = 8, neighbors: int = 10) -> pd.DataFrame:
    final = attach_euclidean_nn_if_needed(final, k=neighbors)
    values: list[int] = []
    for rank in range(1, neighbors + 1):
        col = f"nn_eucl_yyyymm_{rank}"
        if col in final.columns:
            values.extend(final[col].dropna().astype(int).tolist())
    counts = Counter(v for v in values if int(v) != int(anchor)).most_common(top_k)
    out = pd.DataFrame(counts, columns=["yyyymm", "count"])
    if not out.empty:
        out["date"] = yyyymm_to_datetime(out["yyyymm"])
    return out


def plot_nfci_euclidean_topk(spec: ScenarioSpec, final: pd.DataFrame, out_dir: Path) -> None:
    panel = load_regime_nfci_panel()
    focus = panel[(panel["yyyymm"] >= 200801) & (panel["yyyymm"] <= 202412)].copy()
    counts = euclidean_topk(final, anchor=spec.anchor, top_k=8, neighbors=10)
    counts = counts.merge(focus[["yyyymm", "NFCI", "regime"]], on="yyyymm", how="left").dropna(subset=["NFCI"])

    set_publication_style()
    fig, (ax, note_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 5.0),
        gridspec_kw={"height_ratios": [4.0, 0.72], "hspace": 0.16},
    )
    add_regime_background(ax, focus)
    ax.plot(focus["date"], focus["NFCI"], color="#202020", linewidth=1.4, label="NFCI")
    ax.axhline(0.0, color="#697078", linestyle="--", linewidth=0.9)
    ax.axvline(
        yyyymm_to_datetime([spec.anchor]).iloc[0],
        color=PAPER_COLORS["anchor"],
        linestyle="--",
        linewidth=1.4,
        label=f"Anchor {spec.anchor}",
    )
    if not counts.empty:
        ax.scatter(
            counts["date"],
            counts["NFCI"],
            s=52 + 15 * counts["count"].to_numpy(dtype=float),
            color=PAPER_COLORS["generated"],
            marker="o",
            edgecolors="white",
            linewidth=0.8,
            alpha=0.88,
            label="Top Euclidean NN months",
            zorder=4,
        )
    ax.set_title(f"{spec.label}: Euclidean top-k historical analogs, anchor excluded", loc="left", fontsize=12.5, pad=8)
    ax.set_ylabel("NFCI")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=PAPER_COLORS["grid"], linewidth=0.6, alpha=0.70)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    note_ax.axis("off")
    months = ", ".join(f"{int(row.yyyymm)} ({int(row['count'])})" for _, row in counts.iterrows())
    note_ax.text(
        0.0,
        0.82,
        f"{spec.question}. Top-k uses Euclidean z-score distance only; anchor month {spec.anchor} is excluded.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    note_ax.text(0.0, 0.28, f"Top months: {months}", ha="left", va="top", fontsize=8.2)
    save_figure(fig, out_dir / f"{spec.key}_nfci_euclidean_topk")


def fit_pca(reference: pd.DataFrame, values: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = reference[list(columns)].mean(axis=0)
    std = reference[list(columns)].std(axis=0).replace(0.0, 1.0)
    ref_z = ((reference[list(columns)] - mean) / std).to_numpy(dtype=float)
    val_z = ((values[list(columns)] - mean) / std).to_numpy(dtype=float)
    _, singular_values, vh = np.linalg.svd(ref_z - ref_z.mean(axis=0, keepdims=True), full_matrices=False)
    components = vh[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return val_z @ components, explained[:2]


def plot_pca_regime(spec: ScenarioSpec, final: pd.DataFrame, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    regime_labels = pd.read_csv(ROOT / "runtime_universe500" / "regime" / "regime_label_panel.csv")
    hist = historical.merge(regime_labels[["yyyymm", "regime"]], on="yyyymm", how="left")
    hist_plot = hist.dropna(subset=MACRO_COLS).copy()
    if len(hist_plot) > 900:
        hist_plot = hist_plot.sample(900, random_state=7)
    sample_plot = sample.dropna(subset=MACRO_COLS).copy()
    if len(sample_plot) > 1000:
        sample_plot = sample_plot.sample(1000, random_state=7)

    anchor_frame = pd.DataFrame([anchor_raw(historical, spec.anchor)], columns=MACRO_COLS)
    final_median_frame = pd.DataFrame([final[MACRO_COLS].median(axis=0)], columns=MACRO_COLS)
    all_points = pd.concat(
        [
            hist_plot[MACRO_COLS],
            sample_plot[MACRO_COLS],
            anchor_frame,
            final_median_frame,
        ],
        ignore_index=True,
    )
    xy, explained = fit_pca(historical, all_points, MACRO_COLS)
    hist_xy = xy[: len(hist_plot)]
    sample_xy = xy[len(hist_plot) : len(hist_plot) + len(sample_plot)]
    anchor_xy = xy[len(hist_plot) + len(sample_plot)]
    final_median_xy = xy[len(hist_plot) + len(sample_plot) + 1]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for regime, part_idx in hist_plot.groupby("regime").groups.items():
        idx = list(part_idx)
        positions = hist_plot.index.get_indexer(idx)
        ax.scatter(
            hist_xy[positions, 0],
            hist_xy[positions, 1],
            s=13,
            color=REGIME_COLORS.get(str(regime), PAPER_COLORS["historical"]),
            alpha=0.30,
            edgecolors="none",
            rasterized=True,
            label=f"Historical: {str(regime).replace('_', ' ')}",
        )
    ax.scatter(
        sample_xy[:, 0],
        sample_xy[:, 1],
        s=13,
        color="#202020",
        alpha=0.20,
        edgecolors="none",
        rasterized=True,
        label="Scenario generation",
    )
    ax.scatter(anchor_xy[0], anchor_xy[1], s=130, marker="X", color=PAPER_COLORS["anchor"], edgecolors="white", linewidth=0.9, label=f"Anchor {spec.anchor}", zorder=5)
    ax.scatter(final_median_xy[0], final_median_xy[1], s=100, marker="D", color=PAPER_COLORS["generated"], edgecolors="white", linewidth=0.9, label="Final median", zorder=5)
    ax.set_title(f"{spec.label}: PCA macro geography with regime labels", loc="left", fontsize=12.5, pad=8)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% historical variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% historical variance)")
    ax.grid(color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    ax.legend(loc="best", frameon=False, fontsize=7.5)
    save_figure(fig, out_dir / f"{spec.key}_pca_regime_anchor_generated_final_median")


def plot_density_grid(spec: ScenarioSpec, sample: pd.DataFrame, historical: pd.DataFrame, out_dir: Path) -> None:
    fig = macro_density_grid(
        historical[MACRO_COLS],
        sample[MACRO_COLS],
        anchor_raw(historical, spec.anchor),
        columns=MACRO_COLS,
        title=f"{spec.label}: generated macro density vs historical panel",
    )
    save_figure(fig, out_dir / f"{spec.key}_macro_density_grid")


def target_values(spec: ScenarioSpec, final: pd.DataFrame) -> np.ndarray:
    if spec.target_col not in final.columns:
        if "target_metric" not in final.columns:
            raise ValueError(f"{spec.key} has no {spec.target_col!r} or target_metric column")
        values = final["target_metric"].to_numpy(dtype=float)
    else:
        values = final[spec.target_col].to_numpy(dtype=float)
    return values * spec.target_scale


def plot_target_match(spec: ScenarioSpec, final: pd.DataFrame, out_dir: Path) -> None:
    values = target_values(spec, final)
    median = float(np.nanmedian(values))
    mean = float(np.nanmean(values))
    q25, q75 = np.nanquantile(values, [0.25, 0.75])
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=min(10, max(4, len(values))), color=PAPER_COLORS["generated"], alpha=0.84)
    ax.axvline(median, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5, label="Median")
    if spec.key == "q1":
        ax.axvline(0.0, color="#202020", linewidth=1.0, label="No SC-WW gap")
    ax.text(
        0.04,
        0.94,
        f"mean: {mean:.3f}{spec.target_suffix}\nmedian: {median:.3f}{spec.target_suffix}\nq25-q75: {q25:.3f}-{q75:.3f}{spec.target_suffix}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    ax.set_title(f"{spec.label}: target match for {spec.question}", loc="left", fontsize=12.5, pad=8)
    ax.set_xlabel(f"{spec.target_label}{spec.target_suffix}")
    ax.set_ylabel("Number of chains")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    save_figure(fig, out_dir / f"{spec.key}_target_match")


def write_index(out_dir: Path, written: list[str]) -> None:
    lines = [
        "# Requested Submission Plots",
        "",
        "Generated by `python scripts/reporting/report_requested_submission_plots.py`.",
        "",
        "All NFCI top-k plots use Euclidean z-score distance only and exclude the anchor month.",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in sorted(written))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scenarios", default="q1,q2,q3", help="Comma-separated subset: q1,q2,q3")
    parser.add_argument("--include-density-q1", action="store_true", help="Also regenerate a Q1 density grid in the requested output folder.")
    args = parser.parse_args()

    requested = [item.strip().lower() for item in args.scenarios.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for key in requested:
        spec = SCENARIOS[key]
        final, sample, historical, _ = load_scenario(spec)
        plot_nfci_euclidean_topk(spec, final, args.out_dir)
        written.extend([f"{spec.key}_nfci_euclidean_topk.pdf", f"{spec.key}_nfci_euclidean_topk.png"])
        plot_pca_regime(spec, final, sample, historical, args.out_dir)
        written.extend([f"{spec.key}_pca_regime_anchor_generated_final_median.pdf", f"{spec.key}_pca_regime_anchor_generated_final_median.png"])
        plot_target_match(spec, final, args.out_dir)
        written.extend([f"{spec.key}_target_match.pdf", f"{spec.key}_target_match.png"])
        if key != "q1" or args.include_density_q1:
            plot_density_grid(spec, sample, historical, args.out_dir)
            written.extend([f"{spec.key}_macro_density_grid.pdf", f"{spec.key}_macro_density_grid.png"])

    write_index(args.out_dir, written)
    print(f"Wrote requested submission plots to {args.out_dir}")


if __name__ == "__main__":
    main()
