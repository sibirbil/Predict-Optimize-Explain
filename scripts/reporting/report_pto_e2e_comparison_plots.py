"""
Wealth path and allocation plots: standardized PTO vs locked E2E (stdz_exact_layer_focused_v1).
Test period only.

Run from repo root:
    PYTHONPATH=. python3 pipelines/report_pto_e2e_comparison_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.portfolio_diagnostics.report import build_industry_palette, load_roster
from src.utils.paths import resolve_repo_path


MODEL_SPECS = {
    "standardized_pto": {
        "run_dir": "batuhan/artifacts/pto/standardized_pto",
        "label": "PTO",
        "color": "#12436D",
        "weights_file": "test_monthly_weights.csv",
        "weights_format": "csv",
    },
    "stdz_exact_layer_focused_v1": {
        "run_dir": "batuhan/artifacts/e2e/stdz_exact_layer_focused_v1",
        "label": "Locked E2E",
        "color": "#B85C2E",
        "weights_file": "test_monthly_weights.parquet",
        "weights_format": "parquet",
    },
    "summer_child": {
        "run_dir": "batuhan/artifacts/e2e_contrastive/standardized_sc/summerchild_balanced_v2",
        "label": "SummerChild (stable)",
        "color": "#F4A261",
        "weights_file": "test_monthly_weights.parquet",
        "weights_format": "parquet",
    },
    "winter_wolf": {
        "run_dir": "batuhan/artifacts/e2e_contrastive/standardized_ww/winterwolf_balanced_v2",
        "label": "WinterWolf (stress)",
        "color": "#3B7A57",
        "weights_file": "test_monthly_weights.parquet",
        "weights_format": "parquet",
    },
}


def main() -> None:
    output_dir = resolve_repo_path(
        "batuhan/artifacts/portfolio_diagnostics/four_model_comparison_20260415"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster(resolve_repo_path("batuhan/universe_500/universe_500_roster.parquet"))
    wealth = build_test_wealth_frame()
    broad_alloc = build_monthly_allocations(roster=roster, industry_column="industry_division")
    sic2_alloc = build_monthly_allocations(roster=roster, industry_column="industry_sic2")

    broad_grouped, broad_selected = collapse_to_top_groups(broad_alloc, "industry", top_n=8)
    sic2_grouped, sic2_selected = collapse_to_top_groups(sic2_alloc, "industry", top_n=10)

    wealth.to_csv(output_dir / "test_wealth_paths.csv", index=False)
    broad_alloc.to_csv(output_dir / "broad_industry_allocations_monthly.csv", index=False)
    broad_grouped.to_csv(output_dir / "broad_industry_allocations_top_grouped_monthly.csv", index=False)
    sic2_alloc.to_csv(output_dir / "sic2_allocations_monthly.csv", index=False)
    sic2_grouped.to_csv(output_dir / "sic2_allocations_top_grouped_monthly.csv", index=False)

    wealth_fig = build_wealth_figure(wealth)
    wealth_fig.savefig(output_dir / "wealth_paths_test_period.png", dpi=260, bbox_inches="tight")
    wealth_fig.savefig(output_dir / "wealth_paths_test_period.pdf", bbox_inches="tight")
    plt.close(wealth_fig)

    broad_fig = build_allocation_figure(
        allocations=broad_grouped,
        selected_industries=broad_selected,
        title="Broad Industry Allocation — Test Period",
        subtitle="Standardized PTO vs Locked E2E | monthly test-period weights aggregated to broad industries",
    )
    broad_fig.savefig(output_dir / "broad_industry_allocation_test_period.png", dpi=260, bbox_inches="tight")
    broad_fig.savefig(output_dir / "broad_industry_allocation_test_period.pdf", bbox_inches="tight")
    plt.close(broad_fig)

    sic2_fig = build_allocation_figure(
        allocations=sic2_grouped,
        selected_industries=sic2_selected,
        title="SIC2 Allocation — Test Period",
        subtitle="Top SIC2 buckets by average weight; remainder grouped as Other",
    )
    sic2_fig.savefig(output_dir / "sic2_allocation_test_period.png", dpi=260, bbox_inches="tight")
    sic2_fig.savefig(output_dir / "sic2_allocation_test_period.pdf", bbox_inches="tight")
    plt.close(sic2_fig)

    (output_dir / "source_artifacts.json").write_text(
        json.dumps(
            {model: {"run_dir": spec["run_dir"], "label": spec["label"]} for model, spec in MODEL_SPECS.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Print summary table
    final = (
        wealth.sort_values(["model", "date"])
        .groupby(["model", "label"], as_index=False)["wealth"]
        .last()
        .sort_values("wealth", ascending=False)
    )
    print("\nFinal test-period wealth (start=$1):")
    for row in final.itertuples():
        print(f"  {row.label}: ${row.wealth:.3f}")
    print(f"\nOutput: {output_dir}")


def load_test_monthly_portfolio(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "test_monthly_portfolio.csv").copy()
    frame["yyyymm"] = frame["yyyymm"].astype(int)
    frame["date"] = pd.to_datetime(frame["yyyymm"].astype(str) + "01", format="%Y%m%d")
    return frame.sort_values("yyyymm").reset_index(drop=True)


def load_test_weights(run_dir: Path, weights_file: str, weights_format: str) -> pd.DataFrame:
    if weights_format == "csv":
        frame = pd.read_csv(run_dir / weights_file).copy()
    else:
        frame = pd.read_parquet(run_dir / weights_file).copy()
    frame["yyyymm"] = frame["yyyymm"].astype(int)
    frame["permno"] = frame["permno"].astype(int)
    frame["date"] = pd.to_datetime(frame["yyyymm"].astype(str) + "01", format="%Y%m%d")
    return frame[["yyyymm", "date", "permno", "weight"]].copy()


def build_test_wealth_frame() -> pd.DataFrame:
    frames = []
    for model, spec in MODEL_SPECS.items():
        monthly = load_test_monthly_portfolio(resolve_repo_path(spec["run_dir"]))
        monthly["wealth"] = (1.0 + monthly["portfolio_return"].astype(float)).cumprod()
        monthly["model"] = model
        monthly["label"] = spec["label"]
        frames.append(monthly[["yyyymm", "date", "model", "label", "portfolio_return", "wealth"]])
    return pd.concat(frames, ignore_index=True)


def build_monthly_allocations(roster: pd.DataFrame, industry_column: str) -> pd.DataFrame:
    frames = []
    roster_frame = roster[["permno", industry_column]].copy()
    roster_frame[industry_column] = roster_frame[industry_column].fillna("Unknown / Other").astype(str)
    roster_frame[industry_column] = roster_frame[industry_column].replace("", "Unknown / Other")
    for model, spec in MODEL_SPECS.items():
        weights = load_test_weights(
            resolve_repo_path(spec["run_dir"]),
            spec["weights_file"],
            spec["weights_format"],
        )
        merged = weights.merge(roster_frame, on="permno", how="left")
        merged[industry_column] = merged[industry_column].fillna("Unknown / Other")
        monthly = (
            merged.groupby(["yyyymm", "date", industry_column], as_index=False)["weight"]
            .sum()
            .rename(columns={industry_column: "industry"})
            .sort_values(["yyyymm", "industry"])
            .reset_index(drop=True)
        )
        monthly["model"] = model
        monthly["label"] = spec["label"]
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True)


def collapse_to_top_groups(
    allocations: pd.DataFrame,
    industry_column: str,
    top_n: int,
) -> tuple[pd.DataFrame, list[str]]:
    mean_weights = (
        allocations.groupby(industry_column, as_index=False)["weight"]
        .mean()
        .sort_values("weight", ascending=False)
    )
    selected = mean_weights.head(top_n)[industry_column].astype(str).tolist()
    grouped = allocations.copy()
    grouped[industry_column] = np.where(grouped[industry_column].isin(selected), grouped[industry_column], "Other")
    grouped = (
        grouped.groupby(["yyyymm", "date", "model", "label", industry_column], as_index=False)["weight"]
        .sum()
        .sort_values(["model", "yyyymm", industry_column])
        .reset_index(drop=True)
    )
    if "Other" in grouped[industry_column].values and "Other" not in selected:
        selected = selected + ["Other"]
    return grouped, selected


def build_wealth_figure(wealth: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for model, spec in MODEL_SPECS.items():
        subset = wealth.loc[wealth["model"] == model].sort_values("date")
        ax.plot(
            subset["date"],
            subset["wealth"],
            label=spec["label"],
            color=spec["color"],
            linewidth=2.4,
        )
    ax.set_title("Test-Period Total-Return Wealth", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("Wealth of $1")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    return fig


def build_allocation_figure(
    allocations: pd.DataFrame,
    selected_industries: list[str],
    title: str,
    subtitle: str,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12.5, 13.0), sharex=True, sharey=True)
    palette = build_industry_palette(selected_industries)
    for ax, (model, spec) in zip(axes, MODEL_SPECS.items()):
        subset = allocations.loc[allocations["model"] == model].copy()
        pivot = subset.pivot(index="date", columns="industry", values="weight").fillna(0.0)
        for industry in selected_industries:
            if industry not in pivot.columns:
                pivot[industry] = 0.0
        pivot = pivot[selected_industries].sort_index()
        x = mdates.date2num(pivot.index.to_pydatetime())
        y = [pivot[industry].to_numpy(dtype=float) for industry in selected_industries]
        ax.stackplot(x, *y, labels=selected_industries, colors=[palette[i] for i in selected_industries], alpha=0.95)
        ax.set_title(spec["label"], fontsize=12.5, fontweight="bold", loc="left", color=spec["color"])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes:
        ax.set_ylabel("Portfolio Weight")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[industry]) for industry in selected_industries]
    fig.legend(
        handles, selected_industries,
        loc="lower center", ncol=min(5, len(selected_industries)),
        frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.06, ha="left", y=0.99)
    fig.text(0.06, 0.955, subtitle, fontsize=10.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


if __name__ == "__main__":
    main()
