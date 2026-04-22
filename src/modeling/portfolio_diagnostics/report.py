from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from plotly.subplots import make_subplots

from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class DiagnosticsPaths:
    pto_run_dir: Path
    e2e_run_dir: Path
    pto_benchmark_run_dir: Path
    roster_path: Path
    output_dir: Path


def build_portfolio_diagnostics_package(
    pto_run_name: str,
    e2e_run_name: str,
    output_run_name: str,
    pto_output_root: str = "batuhan/artifacts/pto",
    e2e_output_root: str = "batuhan/artifacts/e2e",
    roster_path: str = "batuhan/universe_500/universe_500_roster.parquet",
    benchmark_run_name: str | None = None,
) -> Path:
    benchmark_run_name = benchmark_run_name or pto_run_name
    paths = DiagnosticsPaths(
        pto_run_dir=resolve_repo_path(pto_output_root) / pto_run_name,
        e2e_run_dir=resolve_repo_path(e2e_output_root) / e2e_run_name,
        pto_benchmark_run_dir=resolve_repo_path(pto_output_root) / benchmark_run_name,
        roster_path=resolve_repo_path(roster_path),
        output_dir=resolve_repo_path("batuhan/artifacts/portfolio_diagnostics") / output_run_name,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster(paths.roster_path)
    wealth_frame = build_wealth_frame(paths)
    pto_alloc = build_yearly_industry_allocations(
        weights_frame=load_pto_test_weights(paths.pto_run_dir),
        roster=roster,
        strategy_label="pto",
    )
    e2e_alloc = build_yearly_industry_allocations(
        weights_frame=load_e2e_test_weights(paths.e2e_run_dir),
        roster=roster,
        strategy_label="e2e",
    )
    pto_alloc_specific = build_yearly_industry_allocations(
        weights_frame=load_pto_test_weights(paths.pto_run_dir),
        roster=roster,
        strategy_label="pto",
        industry_column="industry_specific_division",
    )
    e2e_alloc_specific = build_yearly_industry_allocations(
        weights_frame=load_e2e_test_weights(paths.e2e_run_dir),
        roster=roster,
        strategy_label="e2e",
        industry_column="industry_specific_division",
    )
    summary = build_summary_dict(wealth_frame=wealth_frame, pto_alloc=pto_alloc, e2e_alloc=e2e_alloc)

    wealth_frame.to_csv(paths.output_dir / "test_wealth_paths.csv", index=False)
    wealth_frame.to_parquet(paths.output_dir / "test_wealth_paths.parquet", index=False)
    pto_alloc.to_csv(paths.output_dir / "pto_yearly_industry_allocations.csv", index=False)
    pto_alloc.to_parquet(paths.output_dir / "pto_yearly_industry_allocations.parquet", index=False)
    e2e_alloc.to_csv(paths.output_dir / "e2e_yearly_industry_allocations.csv", index=False)
    e2e_alloc.to_parquet(paths.output_dir / "e2e_yearly_industry_allocations.parquet", index=False)
    pto_alloc_specific.to_csv(paths.output_dir / "pto_yearly_industry_specific_allocations.csv", index=False)
    pto_alloc_specific.to_parquet(paths.output_dir / "pto_yearly_industry_specific_allocations.parquet", index=False)
    e2e_alloc_specific.to_csv(paths.output_dir / "e2e_yearly_industry_specific_allocations.csv", index=False)
    e2e_alloc_specific.to_parquet(paths.output_dir / "e2e_yearly_industry_specific_allocations.parquet", index=False)
    (paths.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (paths.output_dir / "diagnostic_summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")

    figure = build_yearly_figure(wealth_frame=wealth_frame, pto_alloc=pto_alloc, e2e_alloc=e2e_alloc)
    figure.savefig(paths.output_dir / "wealth_and_industry_diagnostics.png", dpi=220, bbox_inches="tight")
    figure.savefig(paths.output_dir / "wealth_and_industry_diagnostics.pdf", bbox_inches="tight")
    plt.close(figure)

    specific_figure = build_yearly_specific_figure(
        wealth_frame=wealth_frame,
        pto_alloc=pto_alloc_specific,
        e2e_alloc=e2e_alloc_specific,
    )
    specific_figure.savefig(
        paths.output_dir / "wealth_and_specific_industry_diagnostics.png",
        dpi=220,
        bbox_inches="tight",
    )
    specific_figure.savefig(
        paths.output_dir / "wealth_and_specific_industry_diagnostics.pdf",
        bbox_inches="tight",
    )
    plt.close(specific_figure)

    return paths.output_dir


def build_monthly_portfolio_diagnostics_package(
    pto_run_name: str,
    e2e_run_name: str,
    output_run_name: str,
    pto_output_root: str = "batuhan/artifacts/pto",
    e2e_output_root: str = "batuhan/artifacts/e2e",
    roster_path: str = "batuhan/universe_500/universe_500_roster.parquet",
    benchmark_run_name: str | None = None,
) -> Path:
    benchmark_run_name = benchmark_run_name or pto_run_name
    paths = DiagnosticsPaths(
        pto_run_dir=resolve_repo_path(pto_output_root) / pto_run_name,
        e2e_run_dir=resolve_repo_path(e2e_output_root) / e2e_run_name,
        pto_benchmark_run_dir=resolve_repo_path(pto_output_root) / benchmark_run_name,
        roster_path=resolve_repo_path(roster_path),
        output_dir=resolve_repo_path("batuhan/artifacts/portfolio_diagnostics") / output_run_name,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster(paths.roster_path)
    wealth_frame = build_wealth_frame(paths)
    wealth_tidy = build_wealth_tidy_frame(wealth_frame)
    pto_monthly_alloc = build_monthly_industry_allocations(load_pto_test_weights(paths.pto_run_dir), roster, "pto")
    e2e_monthly_alloc = build_monthly_industry_allocations(load_e2e_test_weights(paths.e2e_run_dir), roster, "e2e")
    industry_order = determine_industry_order(pto_monthly_alloc, e2e_monthly_alloc)
    pto_matrix = build_heatmap_matrix(pto_monthly_alloc, industry_order)
    e2e_matrix = build_heatmap_matrix(e2e_monthly_alloc, industry_order)
    color_limits = determine_heatmap_limits(pto_matrix, e2e_matrix)
    summary = build_monthly_summary_dict(wealth_tidy, pto_monthly_alloc, e2e_monthly_alloc)

    wealth_tidy.to_csv(paths.output_dir / "wealth_paths_test_total_return.csv", index=False)
    pto_monthly_alloc.to_csv(paths.output_dir / "industry_allocation_pto_monthly.csv", index=False)
    e2e_monthly_alloc.to_csv(paths.output_dir / "industry_allocation_e2e_monthly.csv", index=False)
    (paths.output_dir / "monthly_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (paths.output_dir / "monthly_diagnostics_summary.md").write_text(
        build_monthly_markdown_summary(summary), encoding="utf-8"
    )

    figure = build_monthly_static_figure(
        wealth_frame=wealth_frame,
        pto_matrix=pto_matrix,
        e2e_matrix=e2e_matrix,
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    figure.savefig(paths.output_dir / "monthly_diagnostics_static.png", dpi=240, bbox_inches="tight")
    figure.savefig(paths.output_dir / "monthly_diagnostics_static.pdf", bbox_inches="tight")
    plt.close(figure)

    dashboard = build_monthly_plotly_dashboard(
        wealth_frame=wealth_frame,
        pto_matrix=pto_matrix,
        e2e_matrix=e2e_matrix,
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    dashboard.write_html(
        paths.output_dir / "monthly_diagnostics_interactive.html",
        include_plotlyjs="cdn",
        full_html=True,
    )

    return paths.output_dir


def build_difference_portfolio_diagnostics_package(
    pto_run_name: str,
    e2e_run_name: str,
    output_run_name: str,
    pto_output_root: str = "batuhan/artifacts/pto",
    e2e_output_root: str = "batuhan/artifacts/e2e",
    roster_path: str = "batuhan/universe_500/universe_500_roster.parquet",
    benchmark_run_name: str | None = None,
) -> Path:
    benchmark_run_name = benchmark_run_name or pto_run_name
    paths = DiagnosticsPaths(
        pto_run_dir=resolve_repo_path(pto_output_root) / pto_run_name,
        e2e_run_dir=resolve_repo_path(e2e_output_root) / e2e_run_name,
        pto_benchmark_run_dir=resolve_repo_path(pto_output_root) / benchmark_run_name,
        roster_path=resolve_repo_path(roster_path),
        output_dir=resolve_repo_path("batuhan/artifacts/portfolio_diagnostics") / output_run_name,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster(paths.roster_path)
    pto_weights = load_pto_test_weights(paths.pto_run_dir)
    e2e_weights = load_e2e_test_weights(paths.e2e_run_dir)
    merged_weights = build_active_weight_frame(pto_weights=pto_weights, e2e_weights=e2e_weights)
    active_weight_industry = build_active_weight_aggregations(merged_weights=merged_weights, roster=roster, level="broad")
    active_contribution_industry = build_active_contribution_aggregations(
        merged_weights=merged_weights,
        roster=roster,
        level="broad",
    )
    cumulative_active_contribution = build_cumulative_contribution_frame(active_contribution_industry)
    overlap_metrics = build_active_share_overlap_frame(merged_weights)
    divergence_frame = build_top_divergence_months_frame(
        merged_weights=merged_weights,
        active_contribution_industry=active_contribution_industry,
        top_n=10,
    )
    sic2_active_weight = build_active_weight_aggregations(merged_weights=merged_weights, roster=roster, level="sic2")
    sic2_contribution = build_active_contribution_aggregations(
        merged_weights=merged_weights,
        roster=roster,
        level="sic2",
    )
    sic2_summary, sic2_filtered = build_sic2_summary_and_filtered_frame(sic2_active_weight, sic2_contribution, top_n=12)
    summary = build_difference_summary_dict(
        active_weight_industry=active_weight_industry,
        active_contribution_industry=active_contribution_industry,
        overlap_metrics=overlap_metrics,
        divergence_frame=divergence_frame,
        sic2_summary=sic2_summary,
    )

    active_weight_industry.to_csv(paths.output_dir / "active_weight_industry_monthly.csv", index=False)
    active_contribution_industry.to_csv(paths.output_dir / "active_contribution_industry_monthly.csv", index=False)
    cumulative_active_contribution.to_csv(paths.output_dir / "cumulative_active_contribution_industry.csv", index=False)
    overlap_metrics.to_csv(paths.output_dir / "active_share_monthly.csv", index=False)
    divergence_frame.to_csv(paths.output_dir / "top_divergence_months.csv", index=False)
    sic2_summary.to_csv(paths.output_dir / "sic2_contribution_summary.csv", index=False)
    (paths.output_dir / "top_divergence_months.md").write_text(
        build_top_divergence_markdown(divergence_frame),
        encoding="utf-8",
    )
    (paths.output_dir / "difference_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (paths.output_dir / "difference_diagnostics_summary.md").write_text(
        build_difference_markdown_summary(summary),
        encoding="utf-8",
    )

    active_weight_matrix = build_active_heatmap_matrix(active_weight_industry)
    sic2_matrix = build_active_heatmap_matrix(sic2_filtered)
    active_vmax = active_heatmap_limit(active_weight_industry["active_weight"])
    sic2_vmax = active_heatmap_limit(sic2_filtered["active_weight"])

    active_weight_figure = build_active_weight_heatmap_figure(active_weight_matrix, active_vmax)
    active_weight_figure.savefig(paths.output_dir / "active_weight_heatmap_industry.png", dpi=240, bbox_inches="tight")
    active_weight_figure.savefig(paths.output_dir / "active_weight_heatmap_industry.pdf", bbox_inches="tight")
    plt.close(active_weight_figure)

    cumulative_contribution_figure = build_cumulative_contribution_figure(cumulative_active_contribution)
    cumulative_contribution_figure.savefig(
        paths.output_dir / "cumulative_active_contribution_industry.png",
        dpi=240,
        bbox_inches="tight",
    )
    cumulative_contribution_figure.savefig(
        paths.output_dir / "cumulative_active_contribution_industry.pdf",
        bbox_inches="tight",
    )
    plt.close(cumulative_contribution_figure)

    overlap_figure = build_active_share_overlap_figure(overlap_metrics)
    overlap_figure.savefig(paths.output_dir / "active_share_and_overlap.png", dpi=240, bbox_inches="tight")
    overlap_figure.savefig(paths.output_dir / "active_share_and_overlap.pdf", bbox_inches="tight")
    plt.close(overlap_figure)

    sic2_figure = build_sic2_active_weight_heatmap_figure(sic2_matrix, sic2_vmax)
    sic2_figure.savefig(paths.output_dir / "sic2_active_weight_heatmap.png", dpi=240, bbox_inches="tight")
    sic2_figure.savefig(paths.output_dir / "sic2_active_weight_heatmap.pdf", bbox_inches="tight")
    plt.close(sic2_figure)

    dashboard = build_difference_plotly_dashboard(
        active_weight_matrix=active_weight_matrix,
        cumulative_active_contribution=cumulative_active_contribution,
        overlap_metrics=overlap_metrics,
        sic2_matrix=sic2_matrix,
        active_vmax=active_vmax,
        sic2_vmax=sic2_vmax,
    )
    dashboard.write_html(
        paths.output_dir / "difference_diagnostics_interactive.html",
        include_plotlyjs="cdn",
        full_html=True,
    )

    return paths.output_dir


def build_presentation_portfolio_diagnostics_package(
    pto_run_name: str,
    e2e_run_name: str,
    output_run_name: str,
    pto_output_root: str = "batuhan/artifacts/pto",
    e2e_output_root: str = "batuhan/artifacts/e2e",
    roster_path: str = "batuhan/universe_500/universe_500_roster.parquet",
    benchmark_run_name: str | None = None,
) -> Path:
    benchmark_run_name = benchmark_run_name or pto_run_name
    paths = DiagnosticsPaths(
        pto_run_dir=resolve_repo_path(pto_output_root) / pto_run_name,
        e2e_run_dir=resolve_repo_path(e2e_output_root) / e2e_run_name,
        pto_benchmark_run_dir=resolve_repo_path(pto_output_root) / benchmark_run_name,
        roster_path=resolve_repo_path(roster_path),
        output_dir=resolve_repo_path("batuhan/artifacts/portfolio_diagnostics") / output_run_name,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster(paths.roster_path)
    wealth_frame = build_wealth_frame(paths)
    pto_weights = load_pto_test_weights(paths.pto_run_dir)
    e2e_weights = load_e2e_test_weights(paths.e2e_run_dir)
    merged_weights = build_active_weight_frame(pto_weights=pto_weights, e2e_weights=e2e_weights)
    active_weight_industry = build_active_weight_aggregations(merged_weights=merged_weights, roster=roster, level="broad")
    active_contribution_industry = build_active_contribution_aggregations(
        merged_weights=merged_weights,
        roster=roster,
        level="broad",
    )
    cumulative_active_contribution = build_cumulative_contribution_frame(active_contribution_industry)
    overlap_metrics = build_active_share_overlap_frame(merged_weights)
    divergence_frame = build_top_divergence_months_frame(
        merged_weights=merged_weights,
        active_contribution_industry=active_contribution_industry,
        top_n=10,
    )
    sic2_active_weight = build_active_weight_aggregations(merged_weights=merged_weights, roster=roster, level="sic2")
    sic2_contribution = build_active_contribution_aggregations(
        merged_weights=merged_weights,
        roster=roster,
        level="sic2",
    )
    sic2_summary, sic2_filtered = build_sic2_summary_and_filtered_frame(sic2_active_weight, sic2_contribution, top_n=10)

    executive_tables = prepare_executive_summary_tables(
        wealth_frame=wealth_frame,
        cumulative_active_contribution=cumulative_active_contribution,
        overlap_metrics=overlap_metrics,
        active_weight_industry=active_weight_industry,
        top_n_industries=7,
    )
    decomposition_tables = prepare_difference_decomposition_tables(
        active_weight_industry=active_weight_industry,
        cumulative_active_contribution=cumulative_active_contribution,
        divergence_frame=divergence_frame,
    )
    sic2_tables = prepare_sic2_presentation_tables(
        sic2_summary=sic2_summary,
        sic2_filtered=sic2_filtered,
    )
    note_text = build_presentation_note_markdown(
        executive_tables=executive_tables,
        decomposition_tables=decomposition_tables,
        sic2_tables=sic2_tables,
    )

    executive_tables["wealth_paths"].to_csv(paths.output_dir / "executive_summary_wealth_paths.csv", index=False)
    executive_tables["top_industry_contributions"].to_csv(
        paths.output_dir / "executive_summary_top_industries.csv", index=False
    )
    executive_tables["active_share"].to_csv(paths.output_dir / "executive_summary_active_share.csv", index=False)
    executive_tables["heatmap"].to_csv(paths.output_dir / "executive_summary_heatmap.csv", index=False)
    decomposition_tables["active_weights"].to_csv(
        paths.output_dir / "difference_decomposition_active_weights.csv", index=False
    )
    decomposition_tables["active_contributions"].to_csv(
        paths.output_dir / "difference_decomposition_active_contributions.csv", index=False
    )
    decomposition_tables["top_divergence_months"].to_csv(
        paths.output_dir / "difference_decomposition_top_divergence_months.csv", index=False
    )
    sic2_tables["top_buckets"].to_csv(paths.output_dir / "sic2_drilldown_top_buckets.csv", index=False)
    sic2_tables["heatmap"].to_csv(paths.output_dir / "sic2_drilldown_heatmap.csv", index=False)
    (paths.output_dir / "presentation_figure_note.md").write_text(note_text, encoding="utf-8")

    executive_figure = build_executive_summary_figure(executive_tables)
    executive_figure.savefig(paths.output_dir / "executive_summary_figure.png", dpi=260, bbox_inches="tight")
    executive_figure.savefig(paths.output_dir / "executive_summary_figure.pdf", bbox_inches="tight")
    plt.close(executive_figure)

    decomposition_figure = build_difference_decomposition_presentation_figure(decomposition_tables)
    decomposition_figure.savefig(paths.output_dir / "difference_decomposition_figure.png", dpi=260, bbox_inches="tight")
    decomposition_figure.savefig(paths.output_dir / "difference_decomposition_figure.pdf", bbox_inches="tight")
    plt.close(decomposition_figure)

    sic2_figure = build_sic2_drilldown_presentation_figure(sic2_tables)
    sic2_figure.savefig(paths.output_dir / "sic2_drilldown_figure.png", dpi=260, bbox_inches="tight")
    sic2_figure.savefig(paths.output_dir / "sic2_drilldown_figure.pdf", bbox_inches="tight")
    plt.close(sic2_figure)

    return paths.output_dir


def load_roster(path: Path) -> pd.DataFrame:
    roster = pd.read_parquet(path).copy()
    industry_fallback = roster["industry_broad_stat"] if "industry_broad_stat" in roster.columns else None
    roster["industry_division"] = roster["industry_broad"].fillna(industry_fallback).fillna("Unknown / Other")
    roster["industry_division"] = roster["industry_division"].replace("", "Unknown / Other")
    industry_specific_fallback = roster["industry_specific_stat"] if "industry_specific_stat" in roster.columns else None
    roster["industry_specific_division"] = (
        roster["industry_specific"].fillna(industry_specific_fallback).fillna("Unknown / Other")
    )
    roster["industry_specific_division"] = roster["industry_specific_division"].replace("", "Unknown / Other")
    return roster[["permno", "industry_division", "industry_specific_division", "industry_sic2"]].drop_duplicates("permno")


def load_pto_test_weights(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "test_monthly_weights.csv")
    frame = frame.rename(columns={"realized_return": "ret_tplus1"})
    return frame[["yyyymm", "permno", "weight", "ret_tplus1"]].copy()


def load_e2e_test_weights(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / "test_asset_predictions.parquet")
    return frame[["yyyymm", "permno", "weight", "ret_tplus1"]].copy()


def load_equal_weight_test_weights(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "pto_benchmark_weights.csv")
    frame = frame.loc[(frame["strategy"] == "equal_weight") & (frame["split"] == "test")].copy()
    frame = frame.rename(columns={"realized_return": "ret_tplus1"})
    return frame[["yyyymm", "permno", "weight", "ret_tplus1"]].copy()


def build_wealth_frame(paths: DiagnosticsPaths) -> pd.DataFrame:
    portfolios = {
        "pto": load_pto_test_weights(paths.pto_run_dir),
        "e2e": load_e2e_test_weights(paths.e2e_run_dir),
        "equal_weight": load_equal_weight_test_weights(paths.pto_benchmark_run_dir),
    }

    rows: list[pd.DataFrame] = []
    for label, frame in portfolios.items():
        monthly = (
            frame.groupby("yyyymm", as_index=False)
            .apply(lambda part: pd.Series({"portfolio_return": float((part["weight"] * part["ret_tplus1"]).sum())}))
            .reset_index(drop=True)
        )
        monthly["date"] = monthly["yyyymm"].apply(yyyymm_to_timestamp)
        monthly[f"{label}_return"] = monthly["portfolio_return"]
        monthly[f"{label}_wealth"] = (1.0 + monthly["portfolio_return"]).cumprod()
        rows.append(monthly[["yyyymm", "date", f"{label}_return", f"{label}_wealth"]])

    wealth = rows[0]
    for frame in rows[1:]:
        wealth = wealth.merge(frame, on=["yyyymm", "date"], how="inner")

    start_row = {
        "yyyymm": int(previous_yyyymm(int(wealth["yyyymm"].min()))),
        "date": wealth["date"].min() - pd.offsets.MonthEnd(1),
        "pto_return": 0.0,
        "e2e_return": 0.0,
        "equal_weight_return": 0.0,
        "pto_wealth": 1.0,
        "e2e_wealth": 1.0,
        "equal_weight_wealth": 1.0,
    }
    wealth = pd.concat([pd.DataFrame([start_row]), wealth], ignore_index=True)
    return wealth.sort_values("date").reset_index(drop=True)


def build_wealth_tidy_frame(wealth_frame: pd.DataFrame) -> pd.DataFrame:
    monthly = wealth_frame.loc[wealth_frame["pto_return"] != 0.0].copy()
    tidy_frames = []
    for strategy in ("pto", "e2e", "equal_weight"):
        tidy_frames.append(
            pd.DataFrame(
                {
                    "yyyymm": monthly["yyyymm"].astype(int),
                    "date": monthly["date"],
                    "strategy": strategy,
                    "portfolio_return": monthly[f"{strategy}_return"].astype(float),
                    "wealth_index": monthly[f"{strategy}_wealth"].astype(float),
                }
            )
        )
    return pd.concat(tidy_frames, ignore_index=True)


def build_yearly_industry_allocations(
    weights_frame: pd.DataFrame,
    roster: pd.DataFrame,
    strategy_label: str,
    industry_column: str = "industry_division",
) -> pd.DataFrame:
    merged = weights_frame.merge(roster, on="permno", how="left")
    merged[industry_column] = merged[industry_column].fillna("Unknown / Other")
    merged["year"] = (merged["yyyymm"] // 100).astype(int)
    monthly = merged.groupby(["yyyymm", "year", industry_column], as_index=False)["weight"].sum()
    yearly = monthly.groupby(["year", industry_column], as_index=False)["weight"].mean()
    yearly = yearly.rename(columns={"weight": "average_weight"})
    if industry_column != "industry_division":
        yearly = yearly.rename(columns={industry_column: "industry_division"})
    yearly["strategy"] = strategy_label
    return yearly.sort_values(["year", "industry_division"]).reset_index(drop=True)


def build_monthly_industry_allocations(
    weights_frame: pd.DataFrame,
    roster: pd.DataFrame,
    strategy_label: str,
) -> pd.DataFrame:
    merged = weights_frame.merge(roster, on="permno", how="left")
    merged["industry_division"] = merged["industry_division"].fillna("Unknown / Other")
    monthly = merged.groupby(["yyyymm", "industry_division"], as_index=False)["weight"].sum()
    monthly["date"] = monthly["yyyymm"].apply(yyyymm_to_timestamp)
    monthly["strategy"] = strategy_label
    return monthly.sort_values(["yyyymm", "industry_division"]).reset_index(drop=True)


def build_active_weight_frame(pto_weights: pd.DataFrame, e2e_weights: pd.DataFrame) -> pd.DataFrame:
    pto = pto_weights.rename(columns={"weight": "pto_weight", "ret_tplus1": "pto_ret_tplus1"})
    e2e = e2e_weights.rename(columns={"weight": "e2e_weight", "ret_tplus1": "e2e_ret_tplus1"})
    merged = pto.merge(e2e, on=["yyyymm", "permno"], how="outer")
    merged["pto_weight"] = merged["pto_weight"].fillna(0.0)
    merged["e2e_weight"] = merged["e2e_weight"].fillna(0.0)
    merged["ret_tplus1"] = merged["pto_ret_tplus1"].combine_first(merged["e2e_ret_tplus1"]).fillna(0.0)
    merged["date"] = merged["yyyymm"].apply(yyyymm_to_timestamp)
    merged["active_weight"] = merged["pto_weight"] - merged["e2e_weight"]
    merged["active_contribution"] = merged["active_weight"] * merged["ret_tplus1"]
    return merged[["yyyymm", "date", "permno", "pto_weight", "e2e_weight", "ret_tplus1", "active_weight", "active_contribution"]].sort_values(
        ["yyyymm", "permno"]
    )


def build_active_weight_aggregations(
    merged_weights: pd.DataFrame,
    roster: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    label_column = "industry_division" if level == "broad" else "industry_sic2"
    output_column = "industry_division" if level == "broad" else "industry_sic2"
    merged = merged_weights.merge(roster[["permno", "industry_division", "industry_sic2"]], on="permno", how="left")
    merged[output_column] = merged[label_column].fillna("Unknown / Other")
    frame = (
        merged.groupby(["yyyymm", "date", output_column], as_index=False)["active_weight"].sum()
        .sort_values(["yyyymm", output_column])
        .reset_index(drop=True)
    )
    return frame


def build_active_contribution_aggregations(
    merged_weights: pd.DataFrame,
    roster: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    label_column = "industry_division" if level == "broad" else "industry_sic2"
    output_column = "industry_division" if level == "broad" else "industry_sic2"
    merged = merged_weights.merge(roster[["permno", "industry_division", "industry_sic2"]], on="permno", how="left")
    merged[output_column] = merged[label_column].fillna("Unknown / Other")
    frame = (
        merged.groupby(["yyyymm", "date", output_column], as_index=False)["active_contribution"].sum()
        .sort_values(["yyyymm", output_column])
        .reset_index(drop=True)
    )
    return frame


def build_cumulative_contribution_frame(active_contribution_industry: pd.DataFrame) -> pd.DataFrame:
    frame = active_contribution_industry.copy()
    frame["cumulative_active_contribution"] = frame.groupby("industry_division")["active_contribution"].cumsum()
    return frame


def build_active_share_overlap_frame(merged_weights: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (yyyymm, date), month in merged_weights.groupby(["yyyymm", "date"], sort=True):
        pto_weights = month["pto_weight"].to_numpy(dtype=float)
        e2e_weights = month["e2e_weight"].to_numpy(dtype=float)
        active_share = 0.5 * float(np.abs(pto_weights - e2e_weights).sum())
        pto_positions = set(month.loc[month["pto_weight"] > 0.0, "permno"].astype(int))
        e2e_positions = set(month.loc[month["e2e_weight"] > 0.0, "permno"].astype(int))
        union = pto_positions.union(e2e_positions)
        jaccard = float(len(pto_positions.intersection(e2e_positions)) / len(union)) if union else 1.0
        cosine_denominator = float(np.linalg.norm(pto_weights) * np.linalg.norm(e2e_weights))
        cosine_similarity = float(np.dot(pto_weights, e2e_weights) / cosine_denominator) if cosine_denominator > 0 else 0.0
        pto_top = set(month.nlargest(top_k, "pto_weight")["permno"].astype(int))
        e2e_top = set(month.nlargest(top_k, "e2e_weight")["permno"].astype(int))
        top_k_union = pto_top.union(e2e_top)
        top_k_overlap = float(len(pto_top.intersection(e2e_top)) / max(top_k, 1))
        top_k_jaccard = float(len(pto_top.intersection(e2e_top)) / len(top_k_union)) if top_k_union else 1.0
        rows.append(
            {
                "yyyymm": int(yyyymm),
                "date": date,
                "active_share": active_share,
                "jaccard_active_positions": jaccard,
                "cosine_similarity": cosine_similarity,
                f"top_{top_k}_overlap": top_k_overlap,
                f"top_{top_k}_jaccard": top_k_jaccard,
                "pto_active_positions": int((month["pto_weight"] > 0.0).sum()),
                "e2e_active_positions": int((month["e2e_weight"] > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("yyyymm").reset_index(drop=True)


def build_top_divergence_months_frame(
    merged_weights: pd.DataFrame,
    active_contribution_industry: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    monthly_returns = (
        merged_weights.groupby(["yyyymm", "date"], as_index=False)
        .agg(
            pto_return=("pto_weight", lambda s: float(np.dot(s.to_numpy(dtype=float), merged_weights.loc[s.index, "ret_tplus1"].to_numpy(dtype=float)))),
            e2e_return=("e2e_weight", lambda s: float(np.dot(s.to_numpy(dtype=float), merged_weights.loc[s.index, "ret_tplus1"].to_numpy(dtype=float)))),
        )
    )
    monthly_returns["return_difference"] = monthly_returns["pto_return"] - monthly_returns["e2e_return"]
    top_months = monthly_returns.reindex(
        monthly_returns["return_difference"].abs().sort_values(ascending=False).index
    ).head(top_n)

    contribution_lookup = active_contribution_industry.groupby("yyyymm")
    rows: list[dict[str, object]] = []
    for _, row in top_months.iterrows():
        month_contrib = contribution_lookup.get_group(int(row["yyyymm"])).copy()
        month_contrib = month_contrib.reindex(month_contrib["active_contribution"].abs().sort_values(ascending=False).index)
        top_items = [
            f"{contrib_row['industry_division']} ({contrib_row['active_contribution']:+.4f})"
            for _, contrib_row in month_contrib.head(3).iterrows()
        ]
        rows.append(
            {
                "yyyymm": int(row["yyyymm"]),
                "date": row["date"],
                "pto_realized_return": float(row["pto_return"]),
                "e2e_realized_return": float(row["e2e_return"]),
                "pto_minus_e2e_return": float(row["return_difference"]),
                "top_contributing_industries": "; ".join(top_items),
            }
        )
    return pd.DataFrame(rows)


def build_sic2_summary_and_filtered_frame(
    sic2_active_weight: pd.DataFrame,
    sic2_contribution: pd.DataFrame,
    top_n: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_abs_weight = (
        sic2_active_weight.groupby("industry_sic2", as_index=False)["active_weight"]
        .agg(average_absolute_active_weight=lambda s: float(np.abs(s).mean()))
    )
    contribution_summary = sic2_contribution.groupby("industry_sic2", as_index=False)["active_contribution"].agg(
        cumulative_active_contribution="sum",
        cumulative_absolute_active_contribution=lambda s: float(np.abs(s).sum()),
    )
    summary = avg_abs_weight.merge(contribution_summary, on="industry_sic2", how="outer").fillna(0.0)
    summary["selection_score"] = (
        summary["average_absolute_active_weight"].rank(ascending=False, method="min")
        + summary["cumulative_absolute_active_contribution"].rank(ascending=False, method="min")
    )
    summary = summary.sort_values(
        ["selection_score", "cumulative_absolute_active_contribution"],
        ascending=[True, False],
    ).reset_index(drop=True)
    selected = summary.head(top_n)["industry_sic2"].tolist()
    filtered = sic2_active_weight.loc[sic2_active_weight["industry_sic2"].isin(selected)].copy()
    ordered = summary.head(top_n)["industry_sic2"].tolist()
    filtered["industry_sic2"] = pd.Categorical(filtered["industry_sic2"], categories=ordered, ordered=True)
    filtered = filtered.sort_values(["industry_sic2", "yyyymm"]).reset_index(drop=True)
    return summary, filtered


def determine_industry_order(pto_monthly_alloc: pd.DataFrame, e2e_monthly_alloc: pd.DataFrame) -> list[str]:
    combined = pd.concat([pto_monthly_alloc, e2e_monthly_alloc], ignore_index=True)
    average = (
        combined.groupby("industry_division", as_index=False)["weight"].mean()
        .sort_values(["weight", "industry_division"], ascending=[False, True])
    )
    return average["industry_division"].astype(str).tolist()


def build_heatmap_matrix(monthly_alloc: pd.DataFrame, industry_order: list[str]) -> pd.DataFrame:
    pivot = monthly_alloc.pivot(index="industry_division", columns="date", values="weight").fillna(0.0)
    pivot = pivot.reindex(industry_order)
    return pivot.sort_index(axis=1)


def select_top_yearly_specific_industries(
    pto_alloc: pd.DataFrame,
    e2e_alloc: pd.DataFrame,
    top_n: int,
) -> list[str]:
    combined = pd.concat([pto_alloc, e2e_alloc], ignore_index=True)
    average = (
        combined.groupby("industry_division", as_index=False)["average_weight"]
        .mean()
        .sort_values(["average_weight", "industry_division"], ascending=[False, True])
    )
    return average.head(top_n)["industry_division"].astype(str).tolist()


def build_yearly_heatmap_matrix(yearly_alloc: pd.DataFrame, industry_order: list[str]) -> pd.DataFrame:
    pivot = yearly_alloc.pivot(index="industry_division", columns="year", values="average_weight").fillna(0.0)
    pivot = pivot.reindex(industry_order)
    pivot = pivot.sort_index(axis=1)
    pivot.columns = pd.to_datetime(pivot.columns.astype(int).astype(str) + "-12-31")
    return pivot


def determine_heatmap_limits(pto_matrix: pd.DataFrame, e2e_matrix: pd.DataFrame) -> tuple[float, float]:
    combined_max = float(
        max(
            pto_matrix.to_numpy(dtype=float).max(initial=0.0),
            e2e_matrix.to_numpy(dtype=float).max(initial=0.0),
        )
    )
    upper = max(combined_max, 0.05)
    return 0.0, upper


def build_summary_dict(
    wealth_frame: pd.DataFrame,
    pto_alloc: pd.DataFrame,
    e2e_alloc: pd.DataFrame,
) -> dict[str, object]:
    wealth_last = wealth_frame.iloc[-1]
    final_gap = float(wealth_last["pto_wealth"] - wealth_last["e2e_wealth"])
    pto_top = top_industries(pto_alloc)
    e2e_top = top_industries(e2e_alloc)
    tilt_diff = compare_industry_tilts(pto_alloc, e2e_alloc)
    return {
        "test_period": {
            "start_yyyymm": int(wealth_frame.loc[wealth_frame["pto_return"] != 0.0, "yyyymm"].min()),
            "end_yyyymm": int(wealth_frame["yyyymm"].max()),
        },
        "wealth_summary": {
            "final_pto_wealth": float(wealth_last["pto_wealth"]),
            "final_e2e_wealth": float(wealth_last["e2e_wealth"]),
            "final_equal_weight_wealth": float(wealth_last["equal_weight_wealth"]),
            "pto_minus_e2e_final_gap": final_gap,
            "pto_vs_e2e_material_divergence": bool(abs(final_gap) >= 0.10),
        },
        "pto_top_industries": pto_top,
        "e2e_top_industries": e2e_top,
        "largest_tilt_differences": tilt_diff,
        "different_industry_tilts": bool(len(tilt_diff) > 0 and abs(tilt_diff[0]["difference"]) >= 0.02),
    }


def build_monthly_summary_dict(
    wealth_tidy: pd.DataFrame,
    pto_monthly_alloc: pd.DataFrame,
    e2e_monthly_alloc: pd.DataFrame,
) -> dict[str, object]:
    wealth_tail = wealth_tidy.pivot(index="date", columns="strategy", values="wealth_index").sort_index()
    final_gap = float(wealth_tail["pto"].iloc[-1] - wealth_tail["e2e"].iloc[-1])
    monthly_tilt = compare_monthly_industry_tilts(pto_monthly_alloc, e2e_monthly_alloc)
    persistent = find_persistent_industry_shifts(pto_monthly_alloc, e2e_monthly_alloc)
    monthly_reveals_hidden = any(item["share_of_months_over_1pct"] >= 0.35 for item in persistent[:3])
    return {
        "test_period": {
            "start_yyyymm": int(wealth_tidy["yyyymm"].min()),
            "end_yyyymm": int(wealth_tidy["yyyymm"].max()),
        },
        "wealth_summary": {
            "final_pto_wealth": float(wealth_tail["pto"].iloc[-1]),
            "final_e2e_wealth": float(wealth_tail["e2e"].iloc[-1]),
            "final_equal_weight_wealth": float(wealth_tail["equal_weight"].iloc[-1]),
            "pto_minus_e2e_final_gap": final_gap,
            "pto_vs_e2e_material_divergence": bool(abs(final_gap) >= 0.10),
        },
        "monthly_tilt_differences": monthly_tilt,
        "persistent_industry_shifts": persistent,
        "meaningfully_different_month_by_month": bool(
            len(monthly_tilt) > 0 and monthly_tilt[0]["average_absolute_difference"] >= 0.01
        ),
        "monthly_view_reveals_more_than_yearly_average": bool(monthly_reveals_hidden),
    }


def build_difference_summary_dict(
    active_weight_industry: pd.DataFrame,
    active_contribution_industry: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    divergence_frame: pd.DataFrame,
    sic2_summary: pd.DataFrame,
) -> dict[str, object]:
    persistent_broad = summarize_persistent_broad_differences(active_weight_industry, active_contribution_industry)
    top_month_share = float(
        divergence_frame["pto_minus_e2e_return"].abs().sum()
        / max(active_contribution_industry.groupby("yyyymm")["active_contribution"].sum().abs().sum(), 1e-12)
    )
    concentrated = top_month_share >= 0.45
    top_sic2 = sic2_summary.head(5)
    return {
        "test_period": {
            "start_yyyymm": int(active_weight_industry["yyyymm"].min()),
            "end_yyyymm": int(active_weight_industry["yyyymm"].max()),
        },
        "persistent_broad_industry_differences": persistent_broad,
        "active_share_summary": {
            "average_active_share": float(overlap_metrics["active_share"].mean()),
            "max_active_share": float(overlap_metrics["active_share"].max()),
            "average_jaccard_active_positions": float(overlap_metrics["jaccard_active_positions"].mean()),
            "average_cosine_similarity": float(overlap_metrics["cosine_similarity"].mean()),
            "average_top_50_overlap": float(overlap_metrics["top_50_overlap"].mean()),
        },
        "top_divergence_months_share_of_absolute_gap": top_month_share,
        "return_gap_concentrated_in_few_months": bool(concentrated),
        "top_sic2_buckets": [
            {
                "industry_sic2": str(row["industry_sic2"]),
                "average_absolute_active_weight": float(row["average_absolute_active_weight"]),
                "cumulative_active_contribution": float(row["cumulative_active_contribution"]),
                "cumulative_absolute_active_contribution": float(row["cumulative_absolute_active_contribution"]),
            }
            for _, row in top_sic2.iterrows()
        ],
        "models_substantially_different_month_to_month": bool(overlap_metrics["active_share"].mean() >= 0.15),
    }


def top_industries(yearly_alloc: pd.DataFrame, top_n: int = 5) -> list[dict[str, object]]:
    average = (
        yearly_alloc.groupby("industry_division", as_index=False)["average_weight"].mean()
        .sort_values("average_weight", ascending=False)
        .head(top_n)
    )
    return [
        {"industry": str(row["industry_division"]), "average_weight": float(row["average_weight"])}
        for _, row in average.iterrows()
    ]


def compare_industry_tilts(
    pto_alloc: pd.DataFrame,
    e2e_alloc: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, object]]:
    pto_avg = pto_alloc.groupby("industry_division", as_index=False)["average_weight"].mean().rename(
        columns={"average_weight": "pto_average_weight"}
    )
    e2e_avg = e2e_alloc.groupby("industry_division", as_index=False)["average_weight"].mean().rename(
        columns={"average_weight": "e2e_average_weight"}
    )
    merged = pto_avg.merge(e2e_avg, on="industry_division", how="outer").fillna(0.0)
    merged["difference"] = merged["pto_average_weight"] - merged["e2e_average_weight"]
    merged = merged.reindex(merged["difference"].abs().sort_values(ascending=False).index).head(top_n)
    return [
        {
            "industry": str(row["industry_division"]),
            "pto_average_weight": float(row["pto_average_weight"]),
            "e2e_average_weight": float(row["e2e_average_weight"]),
            "difference": float(row["difference"]),
        }
        for _, row in merged.iterrows()
    ]


def compare_monthly_industry_tilts(
    pto_monthly_alloc: pd.DataFrame,
    e2e_monthly_alloc: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, object]]:
    pto = pto_monthly_alloc.rename(columns={"weight": "pto_weight"})
    e2e = e2e_monthly_alloc.rename(columns={"weight": "e2e_weight"})
    merged = pto.merge(e2e, on=["yyyymm", "date", "industry_division"], how="outer").fillna(0.0)
    merged["difference"] = merged["pto_weight"] - merged["e2e_weight"]
    summary = (
        merged.groupby("industry_division", as_index=False)["difference"]
        .agg(average_absolute_difference=lambda s: float(np.abs(s).mean()), max_absolute_difference=lambda s: float(np.abs(s).max()))
    )
    summary = summary.sort_values("average_absolute_difference", ascending=False).head(top_n)
    return [
        {
            "industry": str(row["industry_division"]),
            "average_absolute_difference": float(row["average_absolute_difference"]),
            "max_absolute_difference": float(row["max_absolute_difference"]),
        }
        for _, row in summary.iterrows()
    ]


def find_persistent_industry_shifts(
    pto_monthly_alloc: pd.DataFrame,
    e2e_monthly_alloc: pd.DataFrame,
    threshold: float = 0.01,
    top_n: int = 5,
) -> list[dict[str, object]]:
    pto = pto_monthly_alloc.rename(columns={"weight": "pto_weight"})
    e2e = e2e_monthly_alloc.rename(columns={"weight": "e2e_weight"})
    merged = pto.merge(e2e, on=["yyyymm", "date", "industry_division"], how="outer").fillna(0.0)
    merged["difference"] = merged["pto_weight"] - merged["e2e_weight"]
    summary = (
        merged.groupby("industry_division", as_index=False)
        .agg(
            average_difference=("difference", "mean"),
            average_absolute_difference=("difference", lambda s: float(np.abs(s).mean())),
            share_of_months_over_1pct=("difference", lambda s: float((np.abs(s) >= threshold).mean())),
        )
        .sort_values(["share_of_months_over_1pct", "average_absolute_difference"], ascending=[False, False])
        .head(top_n)
    )
    results = []
    for _, row in summary.iterrows():
        direction = "pto_overweight" if row["average_difference"] > 0 else "e2e_overweight"
        results.append(
            {
                "industry": str(row["industry_division"]),
                "direction": direction,
                "average_difference": float(row["average_difference"]),
                "average_absolute_difference": float(row["average_absolute_difference"]),
                "share_of_months_over_1pct": float(row["share_of_months_over_1pct"]),
            }
        )
    return results


def summarize_persistent_broad_differences(
    active_weight_industry: pd.DataFrame,
    active_contribution_industry: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, object]]:
    avg_weight = active_weight_industry.groupby("industry_division", as_index=False)["active_weight"].agg(
        average_active_weight="mean",
        average_absolute_active_weight=lambda s: float(np.abs(s).mean()),
        share_over_1pct=lambda s: float((np.abs(s) >= 0.01).mean()),
    )
    contrib = active_contribution_industry.groupby("industry_division", as_index=False)["active_contribution"].agg(
        cumulative_active_contribution="sum",
        cumulative_absolute_active_contribution=lambda s: float(np.abs(s).sum()),
    )
    merged = avg_weight.merge(contrib, on="industry_division", how="left").fillna(0.0)
    merged = merged.sort_values(
        ["average_absolute_active_weight", "cumulative_absolute_active_contribution"],
        ascending=[False, False],
    ).head(top_n)
    results: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        direction = "pto_overweight" if row["average_active_weight"] > 0 else "e2e_overweight"
        results.append(
            {
                "industry": str(row["industry_division"]),
                "direction": direction,
                "average_active_weight": float(row["average_active_weight"]),
                "average_absolute_active_weight": float(row["average_absolute_active_weight"]),
                "share_over_1pct": float(row["share_over_1pct"]),
                "cumulative_active_contribution": float(row["cumulative_active_contribution"]),
            }
        )
    return results


def build_yearly_figure(
    wealth_frame: pd.DataFrame,
    pto_alloc: pd.DataFrame,
    e2e_alloc: pd.DataFrame,
    wealth_title: str = "Test-Period Wealth Paths",
    pto_title: str = "PTO Yearly Average Industry Allocation",
    e2e_title: str = "E2E Yearly Average Industry Allocation",
    legend_columns: int = 3,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 14))
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.1, 1.0, 1.0], hspace=0.28)
    wealth_ax = fig.add_subplot(grid[0, 0])
    pto_ax = fig.add_subplot(grid[1, 0])
    e2e_ax = fig.add_subplot(grid[2, 0], sharex=pto_ax)

    palette = portfolio_palette()
    wealth_ax.plot(wealth_frame["date"], wealth_frame["pto_wealth"], label="PTO", color=palette["pto"], linewidth=2.6)
    wealth_ax.plot(wealth_frame["date"], wealth_frame["e2e_wealth"], label="E2E", color=palette["e2e"], linewidth=2.4)
    wealth_ax.plot(
        wealth_frame["date"],
        wealth_frame["equal_weight_wealth"],
        label="Equal-Weight",
        color=palette["equal_weight"],
        linewidth=2.0,
        linestyle="--",
    )
    wealth_ax.set_title(wealth_title, fontsize=15, fontweight="bold")
    wealth_ax.set_ylabel("Wealth Index")
    wealth_ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    wealth_ax.legend(loc="upper left", frameon=False, ncol=3)
    wealth_ax.spines["top"].set_visible(False)
    wealth_ax.spines["right"].set_visible(False)

    industry_colors = build_industry_palette(sorted(set(pto_alloc["industry_division"]).union(e2e_alloc["industry_division"])))
    plot_industry_panel(pto_ax, pto_alloc, pto_title, industry_colors)
    plot_industry_panel(e2e_ax, e2e_alloc, e2e_title, industry_colors)
    e2e_ax.set_xlabel("Year")
    e2e_ax.xaxis.set_major_locator(mdates.YearLocator(2))
    e2e_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (pto_ax, e2e_ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = e2e_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=legend_columns, frameon=False)
    return fig


def build_yearly_specific_figure(
    wealth_frame: pd.DataFrame,
    pto_alloc: pd.DataFrame,
    e2e_alloc: pd.DataFrame,
    top_n: int = 18,
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.15, 1.15], hspace=0.32)
    wealth_ax = fig.add_subplot(grid[0, 0])
    pto_ax = fig.add_subplot(grid[1, 0])
    e2e_ax = fig.add_subplot(grid[2, 0])

    palette = portfolio_palette()
    wealth_ax.plot(wealth_frame["date"], wealth_frame["pto_wealth"], label="PTO", color=palette["pto"], linewidth=2.6)
    wealth_ax.plot(wealth_frame["date"], wealth_frame["e2e_wealth"], label="E2E", color=palette["e2e"], linewidth=2.4)
    wealth_ax.plot(
        wealth_frame["date"],
        wealth_frame["equal_weight_wealth"],
        label="Equal-Weight",
        color=palette["equal_weight"],
        linewidth=2.0,
        linestyle="--",
    )
    wealth_ax.set_title("Test-Period Wealth Paths", fontsize=15, fontweight="bold")
    wealth_ax.set_ylabel("Wealth Index")
    wealth_ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    wealth_ax.legend(loc="upper left", frameon=False, ncol=3)
    wealth_ax.spines["top"].set_visible(False)
    wealth_ax.spines["right"].set_visible(False)

    selected_industries = select_top_yearly_specific_industries(pto_alloc, e2e_alloc, top_n=top_n)
    pto_matrix = build_yearly_heatmap_matrix(pto_alloc, selected_industries)
    e2e_matrix = build_yearly_heatmap_matrix(e2e_alloc, selected_industries)
    vmin, vmax = determine_heatmap_limits(pto_matrix, e2e_matrix)

    pto_im = plot_heatmap_panel(
        ax=pto_ax,
        matrix=pto_matrix,
        title=f"PTO Yearly Average Specific-Industry Allocation (Top {top_n})",
        cmap=build_heatmap_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    e2e_im = plot_heatmap_panel(
        ax=e2e_ax,
        matrix=e2e_matrix,
        title=f"E2E Yearly Average Specific-Industry Allocation (Top {top_n})",
        cmap=build_heatmap_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    pto_ax.set_xlabel("")
    e2e_ax.set_xlabel("Year")

    cbar = fig.colorbar(e2e_im, ax=[pto_ax, e2e_ax], pad=0.012, fraction=0.025)
    cbar.set_label("Average Weight", rotation=270, labelpad=16)
    return fig


def build_monthly_static_figure(
    wealth_frame: pd.DataFrame,
    pto_matrix: pd.DataFrame,
    e2e_matrix: pd.DataFrame,
    vmin: float,
    vmax: float,
) -> plt.Figure:
    palette = portfolio_palette()
    heatmap_cmap = build_heatmap_cmap()
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(18, 12.5))
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.05, 1.3, 1.3], hspace=0.26)
    wealth_ax = fig.add_subplot(grid[0, 0])
    pto_ax = fig.add_subplot(grid[1, 0])
    e2e_ax = fig.add_subplot(grid[2, 0], sharex=pto_ax)

    wealth_ax.plot(
        wealth_frame["date"],
        wealth_frame["pto_wealth"],
        label="Frozen Robust PTO",
        color=palette["pto"],
        linewidth=2.8,
    )
    wealth_ax.plot(
        wealth_frame["date"],
        wealth_frame["e2e_wealth"],
        label="Frozen E2E v2",
        color=palette["e2e"],
        linewidth=2.4,
    )
    wealth_ax.plot(
        wealth_frame["date"],
        wealth_frame["equal_weight_wealth"],
        label="Equal-Weight",
        color=palette["equal_weight"],
        linewidth=1.9,
        linestyle=(0, (4, 2)),
    )
    wealth_ax.set_title("Test-Period Monthly Wealth Paths And Industry Allocation Heatmaps", fontsize=16, fontweight="bold")
    wealth_ax.set_ylabel("Wealth Index")
    wealth_ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    wealth_ax.legend(loc="upper left", frameon=False, ncol=3)
    wealth_ax.spines["top"].set_visible(False)
    wealth_ax.spines["right"].set_visible(False)

    pto_im = plot_heatmap_panel(
        ax=pto_ax,
        matrix=pto_matrix,
        title="PTO Monthly Industry Allocation",
        cmap=heatmap_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plot_heatmap_panel(
        ax=e2e_ax,
        matrix=e2e_matrix,
        title="E2E Monthly Industry Allocation",
        cmap=heatmap_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    e2e_ax.set_xlabel("Month")

    cbar = fig.colorbar(pto_im, ax=[pto_ax, e2e_ax], pad=0.012, fraction=0.025)
    cbar.set_label("Portfolio Weight", rotation=270, labelpad=16)

    return fig


def build_monthly_plotly_dashboard(
    wealth_frame: pd.DataFrame,
    pto_matrix: pd.DataFrame,
    e2e_matrix: pd.DataFrame,
    vmin: float,
    vmax: float,
) -> go.Figure:
    palette = portfolio_palette()
    colorscale = [
        [0.0, "#f4efe5"],
        [0.25, "#dcc6a0"],
        [0.5, "#c59b5e"],
        [0.75, "#9f6a2d"],
        [1.0, "#5d3412"],
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.08,
        row_heights=[0.32, 0.34, 0.34],
        subplot_titles=(
            "Test-Period Wealth Paths",
            "PTO Monthly Industry Allocation",
            "E2E Monthly Industry Allocation",
        ),
    )

    for strategy, label in (
        ("pto", "Frozen Robust PTO"),
        ("e2e", "Frozen E2E v2"),
        ("equal_weight", "Equal-Weight"),
    ):
        fig.add_trace(
            go.Scatter(
                x=wealth_frame["date"],
                y=wealth_frame[f"{strategy}_wealth"],
                mode="lines",
                name=label,
                line=dict(
                    color=palette[strategy],
                    width=3 if strategy == "pto" else 2.4,
                    dash="dash" if strategy == "equal_weight" else "solid",
                ),
                hovertemplate="%{x|%Y-%m}<br>Wealth=%{y:.3f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=pto_matrix.to_numpy(dtype=float),
            x=list(pto_matrix.columns),
            y=list(pto_matrix.index),
            zmin=vmin,
            zmax=vmax,
            colorscale=colorscale,
            colorbar=dict(title="Weight", len=0.58, y=0.33),
            hovertemplate="Month=%{x|%Y-%m}<br>Industry=%{y}<br>Weight=%{z:.3%}<extra>PTO</extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=e2e_matrix.to_numpy(dtype=float),
            x=list(e2e_matrix.columns),
            y=list(e2e_matrix.index),
            zmin=vmin,
            zmax=vmax,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Month=%{x|%Y-%m}<br>Industry=%{y}<br>Weight=%{z:.3%}<extra>E2E</extra>",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=1180,
        width=1400,
        title=dict(text="Monthly Portfolio Diagnostics", x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
        margin=dict(l=80, r=45, t=85, b=55),
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(showgrid=False, tickformat="%Y")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_yaxes(title_text="Wealth Index", row=1, col=1)
    return fig


def build_active_heatmap_matrix(active_frame: pd.DataFrame) -> pd.DataFrame:
    label_column = "industry_division" if "industry_division" in active_frame.columns else "industry_sic2"
    pivot = active_frame.pivot(index=label_column, columns="date", values="active_weight").fillna(0.0)
    order = (
        active_frame.groupby(label_column, observed=False)["active_weight"]
        .apply(lambda s: float(np.abs(s).mean()))
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot = pivot.reindex(order)
    return pivot.sort_index(axis=1)


def active_heatmap_limit(series: pd.Series) -> float:
    vmax = float(np.abs(series.to_numpy(dtype=float)).max(initial=0.0))
    return max(vmax, 0.02)


def build_active_weight_heatmap_figure(matrix: pd.DataFrame, vmax: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 7.5))
    cmap = build_active_heatmap_cmap()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = plot_heatmap_panel(
        ax=ax,
        matrix=matrix,
        title="PTO Minus E2E Active Weight By Broad Industry",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cbar.set_label("Active Weight", rotation=270, labelpad=16)
    ax.set_xlabel("Month")
    return fig


def build_cumulative_contribution_figure(cumulative_active_contribution: pd.DataFrame, top_n: int = 8) -> plt.Figure:
    summary = (
        cumulative_active_contribution.groupby("industry_division", as_index=False)["cumulative_active_contribution"]
        .last()
        .assign(abs_final=lambda df: df["cumulative_active_contribution"].abs())
        .sort_values("abs_final", ascending=False)
        .head(top_n)
    )
    selected = summary["industry_division"].tolist()
    frame = cumulative_active_contribution.loc[cumulative_active_contribution["industry_division"].isin(selected)].copy()
    fig, ax = plt.subplots(figsize=(16, 7.5))
    palette = build_industry_palette(selected)
    for industry in selected:
        subset = frame.loc[frame["industry_division"] == industry].sort_values("date")
        ax.plot(
            subset["date"],
            subset["cumulative_active_contribution"],
            label=industry,
            linewidth=2.2,
            color=palette[industry],
        )
    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.set_title("Cumulative PTO Minus E2E Active Return Contribution By Broad Industry", fontsize=15, fontweight="bold")
    ax.set_ylabel("Cumulative Active Contribution")
    ax.set_xlabel("Month")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def build_active_share_overlap_figure(overlap_metrics: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.22)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    palette = portfolio_palette()

    ax1.plot(overlap_metrics["date"], overlap_metrics["active_share"], color=palette["pto"], linewidth=2.6)
    ax1.set_title("Monthly Active Share And Portfolio Overlap", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Active Share")
    ax1.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(
        overlap_metrics["date"],
        overlap_metrics["jaccard_active_positions"],
        label="Jaccard Active Positions",
        color="#6a7f38",
        linewidth=2.0,
    )
    ax2.plot(
        overlap_metrics["date"],
        overlap_metrics["cosine_similarity"],
        label="Cosine Similarity",
        color=palette["e2e"],
        linewidth=2.0,
    )
    ax2.plot(
        overlap_metrics["date"],
        overlap_metrics["top_50_overlap"],
        label="Top-50 Overlap",
        color="#7d6db3",
        linewidth=2.0,
    )
    ax2.set_ylabel("Overlap / Similarity")
    ax2.set_xlabel("Month")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=False, ncol=3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return fig


def build_sic2_active_weight_heatmap_figure(matrix: pd.DataFrame, vmax: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 7.0))
    cmap = build_active_heatmap_cmap()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = plot_heatmap_panel(
        ax=ax,
        matrix=matrix,
        title="PTO Minus E2E Active Weight By Selected SIC2 Buckets",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cbar.set_label("Active Weight", rotation=270, labelpad=16)
    ax.set_xlabel("Month")
    return fig


def build_difference_plotly_dashboard(
    active_weight_matrix: pd.DataFrame,
    cumulative_active_contribution: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    sic2_matrix: pd.DataFrame,
    active_vmax: float,
    sic2_vmax: float,
) -> go.Figure:
    palette = portfolio_palette()
    colorscale = [
        [0.0, "#7f2704"],
        [0.5, "#f7f7f7"],
        [1.0, "#084081"],
    ]
    fig = make_subplots(
        rows=4,
        cols=1,
        vertical_spacing=0.07,
        row_heights=[0.28, 0.26, 0.22, 0.24],
        subplot_titles=(
            "PTO Minus E2E Active Weight By Broad Industry",
            "Cumulative PTO Minus E2E Active Contribution By Broad Industry",
            "Active Share And Overlap",
            "Selected SIC2 Active Weight Heatmap",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=active_weight_matrix.to_numpy(dtype=float),
            x=list(active_weight_matrix.columns),
            y=list(active_weight_matrix.index),
            zmin=-active_vmax,
            zmax=active_vmax,
            zmid=0.0,
            colorscale=colorscale,
            colorbar=dict(title="Active Weight", len=0.38, y=0.82),
            hovertemplate="Month=%{x|%Y-%m}<br>Industry=%{y}<br>Active weight=%{z:+.3%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    top_broad = (
        cumulative_active_contribution.groupby("industry_division", as_index=False)["cumulative_active_contribution"]
        .last()
        .assign(abs_final=lambda df: df["cumulative_active_contribution"].abs())
        .sort_values("abs_final", ascending=False)
        .head(8)["industry_division"]
        .tolist()
    )
    color_map = build_industry_palette(top_broad)
    for industry in top_broad:
        subset = cumulative_active_contribution.loc[
            cumulative_active_contribution["industry_division"] == industry
        ].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["cumulative_active_contribution"],
                mode="lines",
                name=industry,
                line=dict(color=color_map[industry], width=2.3),
                hovertemplate="%{x|%Y-%m}<br>Cumulative contribution=%{y:+.4f}<extra>" + industry + "</extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=overlap_metrics["date"],
            y=overlap_metrics["active_share"],
            mode="lines",
            name="Active Share",
            line=dict(color=palette["pto"], width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Active share=%{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    for column, label, color in (
        ("jaccard_active_positions", "Jaccard Active Positions", "#6a7f38"),
        ("cosine_similarity", "Cosine Similarity", palette["e2e"]),
        ("top_50_overlap", "Top-50 Overlap", "#7d6db3"),
    ):
        fig.add_trace(
            go.Scatter(
                x=overlap_metrics["date"],
                y=overlap_metrics[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.0),
                hovertemplate="%{x|%Y-%m}<br>" + label + "=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    fig.add_trace(
        go.Heatmap(
            z=sic2_matrix.to_numpy(dtype=float),
            x=list(sic2_matrix.columns),
            y=list(sic2_matrix.index),
            zmin=-sic2_vmax,
            zmax=sic2_vmax,
            zmid=0.0,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Month=%{x|%Y-%m}<br>SIC2=%{y}<br>Active weight=%{z:+.3%}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=1450,
        width=1450,
        title=dict(text="PTO Versus E2E Difference Diagnostics", x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
        margin=dict(l=90, r=55, t=90, b=55),
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(showgrid=False, tickformat="%Y")
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def prepare_executive_summary_tables(
    wealth_frame: pd.DataFrame,
    cumulative_active_contribution: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    active_weight_industry: pd.DataFrame,
    top_n_industries: int = 7,
) -> dict[str, pd.DataFrame]:
    wealth_paths = build_wealth_tidy_frame(wealth_frame)
    top_industries = (
        active_weight_industry.groupby("industry_division", as_index=False)["active_weight"]
        .agg(average_absolute_active_weight=lambda s: float(np.abs(s).mean()))
        .sort_values("average_absolute_active_weight", ascending=False)
        .head(top_n_industries)["industry_division"]
        .tolist()
    )
    contribution_tail = (
        cumulative_active_contribution.groupby("industry_division", as_index=False)["cumulative_active_contribution"]
        .last()
        .assign(abs_final=lambda df: df["cumulative_active_contribution"].abs())
        .sort_values("abs_final", ascending=False)
    )
    top_contribution_industries = contribution_tail.head(5)["industry_division"].tolist()
    contribution_frame = cumulative_active_contribution.loc[
        cumulative_active_contribution["industry_division"].isin(top_contribution_industries)
    ].copy()
    heatmap = active_weight_industry.loc[active_weight_industry["industry_division"].isin(top_industries)].copy()
    heatmap["industry_division"] = pd.Categorical(heatmap["industry_division"], categories=top_industries, ordered=True)
    heatmap = heatmap.sort_values(["industry_division", "yyyymm"]).reset_index(drop=True)
    return {
        "wealth_paths": wealth_paths,
        "top_industry_contributions": contribution_frame,
        "active_share": overlap_metrics.copy(),
        "heatmap": heatmap,
    }


def prepare_difference_decomposition_tables(
    active_weight_industry: pd.DataFrame,
    cumulative_active_contribution: pd.DataFrame,
    divergence_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    active_weights = (
        active_weight_industry.groupby("industry_division", as_index=False)["active_weight"]
        .agg(
            average_active_weight="mean",
            average_absolute_active_weight=lambda s: float(np.abs(s).mean()),
        )
        .sort_values("average_absolute_active_weight", ascending=False)
        .reset_index(drop=True)
    )
    active_contributions = (
        cumulative_active_contribution.groupby("industry_division", as_index=False)["cumulative_active_contribution"]
        .last()
        .assign(absolute_cumulative_active_contribution=lambda df: df["cumulative_active_contribution"].abs())
        .sort_values("absolute_cumulative_active_contribution", ascending=False)
        .reset_index(drop=True)
    )
    top_divergence = divergence_frame.copy()
    top_divergence["absolute_return_difference"] = top_divergence["pto_minus_e2e_return"].abs()
    top_divergence = top_divergence.sort_values("absolute_return_difference", ascending=False).reset_index(drop=True)
    return {
        "active_weights": active_weights,
        "active_contributions": active_contributions,
        "top_divergence_months": top_divergence,
    }


def prepare_sic2_presentation_tables(
    sic2_summary: pd.DataFrame,
    sic2_filtered: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    top_buckets = sic2_summary.head(10).copy()
    selected = top_buckets["industry_sic2"].astype(str).tolist()
    heatmap = sic2_filtered.loc[sic2_filtered["industry_sic2"].astype(str).isin(selected)].copy()
    heatmap["industry_sic2"] = pd.Categorical(heatmap["industry_sic2"].astype(str), categories=selected, ordered=True)
    heatmap = heatmap.sort_values(["industry_sic2", "yyyymm"]).reset_index(drop=True)
    top_buckets["industry_sic2"] = top_buckets["industry_sic2"].astype(str)
    return {"top_buckets": top_buckets, "heatmap": heatmap}


def build_executive_summary_figure(executive_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    wealth = executive_tables["wealth_paths"]
    contributions = executive_tables["top_industry_contributions"]
    active_share = executive_tables["active_share"]
    heatmap_frame = executive_tables["heatmap"]
    palette = portfolio_palette()
    fig = plt.figure(figsize=(17, 13))
    grid = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 0.95, 1.2], hspace=0.30, wspace=0.18)

    wealth_ax = fig.add_subplot(grid[0, :])
    contribution_ax = fig.add_subplot(grid[1, 0])
    active_share_ax = fig.add_subplot(grid[1, 1])
    heatmap_ax = fig.add_subplot(grid[2, :])

    for strategy, label, color, style, width in (
        ("pto", "Frozen Robust PTO", palette["pto"], "-", 2.8),
        ("e2e", "Frozen E2E v2", palette["e2e"], "-", 2.4),
        ("equal_weight", "Equal-Weight", palette["equal_weight"], "--", 2.0),
    ):
        subset = wealth.loc[wealth["strategy"] == strategy].sort_values("date")
        wealth_ax.plot(subset["date"], subset["wealth_index"], color=color, linestyle=style, linewidth=width)
        wealth_ax.text(subset["date"].iloc[-1], subset["wealth_index"].iloc[-1], f"  {label}", color=color, va="center", fontsize=10)
    wealth_ax.set_title("Executive Summary: How PTO Differs From E2E", fontsize=17, fontweight="bold", loc="left")
    wealth_ax.set_ylabel("Wealth Index")
    wealth_ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    wealth_ax.spines["top"].set_visible(False)
    wealth_ax.spines["right"].set_visible(False)
    wealth_ax.xaxis.set_major_locator(mdates.YearLocator(2))
    wealth_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    selected_industries = (
        contributions.groupby("industry_division")["cumulative_active_contribution"].last().abs().sort_values(ascending=False).index.tolist()
    )
    color_map = build_industry_palette(selected_industries)
    for industry in selected_industries:
        subset = contributions.loc[contributions["industry_division"] == industry].sort_values("date")
        contribution_ax.plot(subset["date"], subset["cumulative_active_contribution"], color=color_map[industry], linewidth=2.2)
        contribution_ax.text(
            subset["date"].iloc[-1],
            subset["cumulative_active_contribution"].iloc[-1],
            f"  {industry}",
            color=color_map[industry],
            va="center",
            fontsize=9,
        )
    contribution_ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.55)
    contribution_ax.set_title("Cumulative PTO Minus E2E Active Contribution\nTop 5 Broad Industries", fontsize=12, loc="left")
    contribution_ax.set_ylabel("Cumulative Contribution")
    contribution_ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    contribution_ax.spines["top"].set_visible(False)
    contribution_ax.spines["right"].set_visible(False)

    active_share_ax.plot(active_share["date"], active_share["active_share"], color=palette["pto"], linewidth=2.6)
    active_share_ax.fill_between(active_share["date"], active_share["active_share"], 0.0, color=palette["pto"], alpha=0.10)
    active_share_ax.set_title("Monthly Active Share\nPTO vs E2E", fontsize=12, loc="left")
    active_share_ax.set_ylabel("Active Share")
    active_share_ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    active_share_ax.spines["top"].set_visible(False)
    active_share_ax.spines["right"].set_visible(False)
    active_share_ax.xaxis.set_major_locator(mdates.YearLocator(2))
    active_share_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    heatmap_matrix = build_active_heatmap_matrix(heatmap_frame)
    vmax = active_heatmap_limit(heatmap_frame["active_weight"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = plot_heatmap_panel(
        ax=heatmap_ax,
        matrix=heatmap_matrix,
        title="Monthly Active Weights By Broad Industry\nPositive = PTO Overweight Relative To E2E",
        cmap=build_active_heatmap_cmap(),
        vmin=-vmax,
        vmax=vmax,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=heatmap_ax, pad=0.012, fraction=0.025)
    cbar.set_label("Active Weight", rotation=270, labelpad=16)
    heatmap_ax.set_xlabel("Month")

    return fig


def build_difference_decomposition_presentation_figure(decomposition_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    weights = decomposition_tables["active_weights"].copy()
    contributions = decomposition_tables["active_contributions"].copy()
    divergence = decomposition_tables["top_divergence_months"].copy()
    palette = portfolio_palette()
    fig = plt.figure(figsize=(17, 12))
    grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.28)
    weight_ax = fig.add_subplot(grid[0, 0])
    contribution_ax = fig.add_subplot(grid[0, 1])
    divergence_ax = fig.add_subplot(grid[1, :])

    top_weights = weights.head(10).sort_values("average_active_weight")
    weight_colors = np.where(top_weights["average_active_weight"] >= 0.0, palette["pto"], palette["e2e"])
    weight_ax.barh(top_weights["industry_division"], top_weights["average_active_weight"], color=weight_colors, alpha=0.90)
    weight_ax.axvline(0.0, color="#666666", linewidth=1.0)
    weight_ax.set_title("Average Active Weights By Broad Industry", fontsize=13, loc="left")
    weight_ax.set_xlabel("Average Active Weight")
    weight_ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    weight_ax.spines["top"].set_visible(False)
    weight_ax.spines["right"].set_visible(False)
    for _, row in top_weights.iterrows():
        align = "left" if row["average_active_weight"] >= 0 else "right"
        offset = 0.0015 if row["average_active_weight"] >= 0 else -0.0015
        weight_ax.text(row["average_active_weight"] + offset, row["industry_division"], f"{row['average_active_weight']:+.3f}", va="center", ha=align, fontsize=9)

    top_contrib = contributions.head(10).sort_values("cumulative_active_contribution")
    contrib_colors = np.where(top_contrib["cumulative_active_contribution"] >= 0.0, palette["pto"], palette["e2e"])
    contribution_ax.barh(top_contrib["industry_division"], top_contrib["cumulative_active_contribution"], color=contrib_colors, alpha=0.90)
    contribution_ax.axvline(0.0, color="#666666", linewidth=1.0)
    contribution_ax.set_title("Cumulative Active Return Contribution By Broad Industry", fontsize=13, loc="left")
    contribution_ax.set_xlabel("Cumulative Contribution")
    contribution_ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    contribution_ax.spines["top"].set_visible(False)
    contribution_ax.spines["right"].set_visible(False)
    for _, row in top_contrib.iterrows():
        align = "left" if row["cumulative_active_contribution"] >= 0 else "right"
        offset = 0.0025 if row["cumulative_active_contribution"] >= 0 else -0.0025
        contribution_ax.text(row["cumulative_active_contribution"] + offset, row["industry_division"], f"{row['cumulative_active_contribution']:+.3f}", va="center", ha=align, fontsize=9)

    divergence = divergence.sort_values("absolute_return_difference", ascending=True)
    y_positions = np.arange(len(divergence))
    divergence_ax.hlines(y_positions, divergence["e2e_realized_return"], divergence["pto_realized_return"], color="#c7c7c7", linewidth=2.0)
    divergence_ax.scatter(divergence["pto_realized_return"], y_positions, color=palette["pto"], s=55, label="PTO")
    divergence_ax.scatter(divergence["e2e_realized_return"], y_positions, color=palette["e2e"], s=55, label="E2E")
    divergence_ax.set_yticks(y_positions)
    divergence_ax.set_yticklabels(divergence["yyyymm"].astype(str))
    divergence_ax.set_title("Top Divergence Months: PTO Return vs E2E Return", fontsize=13, loc="left")
    divergence_ax.set_xlabel("Monthly Realized Return")
    divergence_ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    divergence_ax.spines["top"].set_visible(False)
    divergence_ax.spines["right"].set_visible(False)
    divergence_ax.legend(frameon=False, loc="lower right")
    for pos, (_, row) in enumerate(divergence.iterrows()):
        divergence_ax.text(
            max(row["pto_realized_return"], row["e2e_realized_return"]) + 0.002,
            pos,
            row["top_contributing_industries"].split(";")[0],
            fontsize=8.5,
            va="center",
            color="#444444",
        )

    return fig


def build_sic2_drilldown_presentation_figure(sic2_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    top_buckets = sic2_tables["top_buckets"].copy()
    heatmap = sic2_tables["heatmap"].copy()
    palette = portfolio_palette()
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.9, 1.15], hspace=0.28)
    bar_ax = fig.add_subplot(grid[0, 0])
    heatmap_ax = fig.add_subplot(grid[1, 0])

    bars = top_buckets.sort_values("cumulative_active_contribution")
    colors = np.where(bars["cumulative_active_contribution"] >= 0.0, palette["pto"], palette["e2e"])
    bar_ax.barh(bars["industry_sic2"], bars["cumulative_active_contribution"], color=colors, alpha=0.92)
    bar_ax.axvline(0.0, color="#666666", linewidth=1.0)
    bar_ax.set_title("SIC2 Drill-Down: Top 10 Buckets By Absolute Importance", fontsize=14, loc="left")
    bar_ax.set_xlabel("Cumulative Active Contribution")
    bar_ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    for _, row in bars.iterrows():
        align = "left" if row["cumulative_active_contribution"] >= 0 else "right"
        offset = 0.002 if row["cumulative_active_contribution"] >= 0 else -0.002
        label = f"SIC2 {row['industry_sic2']}  {row['cumulative_active_contribution']:+.3f}"
        bar_ax.text(row["cumulative_active_contribution"] + offset, row["industry_sic2"], label, va="center", ha=align, fontsize=9)

    heatmap_matrix = build_active_heatmap_matrix(heatmap.rename(columns={"industry_sic2": "industry_division"}))
    vmax = active_heatmap_limit(heatmap["active_weight"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = plot_heatmap_panel(
        ax=heatmap_ax,
        matrix=heatmap_matrix,
        title="Monthly Active Weights For Selected SIC2 Buckets\nPositive = PTO Overweight Relative To E2E",
        cmap=build_active_heatmap_cmap(),
        vmin=-vmax,
        vmax=vmax,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=heatmap_ax, pad=0.012, fraction=0.025)
    cbar.set_label("Active Weight", rotation=270, labelpad=16)
    heatmap_ax.set_xlabel("Month")
    return fig


def plot_industry_panel(
    ax: plt.Axes,
    yearly_alloc: pd.DataFrame,
    title: str,
    industry_colors: dict[str, str],
) -> None:
    pivot = yearly_alloc.pivot(index="year", columns="industry_division", values="average_weight").fillna(0.0)
    pivot = pivot.sort_index()
    years = pd.to_datetime(pivot.index.astype(str) + "-12-31")
    bottom = np.zeros(len(pivot), dtype=float)
    for industry in pivot.columns:
        values = pivot[industry].to_numpy(dtype=float)
        ax.bar(
            years,
            values,
            bottom=bottom,
            width=250,
            color=industry_colors[industry],
            edgecolor="white",
            linewidth=0.35,
            label=industry,
        )
        bottom += values
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)


def plot_heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    norm: TwoSlopeNorm | None = None,
):
    im = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    tick_positions = np.linspace(0, matrix.shape[1] - 1, num=min(10, matrix.shape[1]), dtype=int)
    tick_positions = np.unique(tick_positions)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([matrix.columns[pos].strftime("%Y-%m") for pos in tick_positions], rotation=0, fontsize=9)
    ax.tick_params(axis="x", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return im


def portfolio_palette() -> dict[str, str]:
    return {
        "pto": "#12436d",
        "e2e": "#b85c2e",
        "equal_weight": "#8a8f98",
    }


def build_industry_palette(industries: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab20")
    return {industry: to_hex(cmap(idx % 20)) for idx, industry in enumerate(industries)}


def build_heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "portfolio_heatmap",
        ["#f7f3eb", "#e8d3aa", "#d3ad69", "#9d6b2a", "#5c3312"],
    )


def build_active_heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "portfolio_active_heatmap",
        ["#7f2704", "#f7f7f7", "#084081"],
    )


def build_markdown_summary(summary: dict[str, object]) -> str:
    wealth = summary["wealth_summary"]
    pto_top = summary["pto_top_industries"]
    e2e_top = summary["e2e_top_industries"]
    tilt_diff = summary["largest_tilt_differences"]
    lines = [
        "# Wealth And Industry Diagnostics",
        "",
        "## Construction",
        "- Wealth paths use test-period monthly rebalancing and one-month holding-period realized total returns from `ret_tplus1`.",
        "- PTO uses saved `test_monthly_weights.csv`, E2E uses saved `test_asset_predictions.parquet`, and equal-weight uses the saved benchmark weights.",
        "- Industry panels aggregate monthly permno weights to broad industry divisions from `universe_500_roster.parquet`, then average within each calendar year.",
        "",
        "## Wealth",
        f"- Final wealth: PTO `{wealth['final_pto_wealth']:.3f}`, E2E `{wealth['final_e2e_wealth']:.3f}`, Equal-Weight `{wealth['final_equal_weight_wealth']:.3f}`.",
        f"- PTO and E2E materially diverge over the test period: `{wealth['pto_vs_e2e_material_divergence']}`.",
        "",
        "## PTO Dominant Industries",
    ]
    for item in pto_top:
        lines.append(f"- {item['industry']}: average weight `{item['average_weight']:.3f}`.")
    lines.append("")
    lines.append("## E2E Dominant Industries")
    for item in e2e_top:
        lines.append(f"- {item['industry']}: average weight `{item['average_weight']:.3f}`.")
    lines.append("")
    lines.append("## Largest PTO vs E2E Tilts")
    for item in tilt_diff:
        direction = "PTO overweight" if item["difference"] > 0 else "E2E overweight"
        lines.append(
            f"- {item['industry']}: {direction}, difference `{abs(item['difference']):.3f}` "
            f"(PTO `{item['pto_average_weight']:.3f}` vs E2E `{item['e2e_average_weight']:.3f}`)."
        )
    lines.append("")
    lines.append(f"- The two pipelines appear to express meaningfully different industry tilts: `{summary['different_industry_tilts']}`.")
    return "\n".join(lines)


def build_monthly_markdown_summary(summary: dict[str, object]) -> str:
    wealth = summary["wealth_summary"]
    monthly_tilts = summary["monthly_tilt_differences"]
    persistent = summary["persistent_industry_shifts"]
    lines = [
        "# Monthly Portfolio Diagnostics",
        "",
        "## Construction",
        "- Wealth paths use realized total return `ret_tplus1`, monthly rebalancing, and one-month holding periods.",
        "- Monthly industry heatmaps aggregate saved permno weights into broad industry divisions from `universe_500_roster.parquet`.",
        "- PTO and E2E heatmaps use the same color scale so weight intensity is directly comparable month by month.",
        "",
        "## Wealth",
        f"- Final wealth: PTO `{wealth['final_pto_wealth']:.3f}`, E2E `{wealth['final_e2e_wealth']:.3f}`, Equal-Weight `{wealth['final_equal_weight_wealth']:.3f}`.",
        f"- PTO and E2E materially diverge over the test period: `{wealth['pto_vs_e2e_material_divergence']}`.",
        "",
        "## Largest Month-By-Month Tilt Differences",
    ]
    for item in monthly_tilts:
        lines.append(
            f"- {item['industry']}: average monthly absolute tilt difference `{item['average_absolute_difference']:.3f}`, "
            f"max monthly difference `{item['max_absolute_difference']:.3f}`."
        )
    lines.append("")
    lines.append("## Persistent Industry Shifts")
    for item in persistent:
        direction = "PTO overweight" if item["direction"] == "pto_overweight" else "E2E overweight"
        lines.append(
            f"- {item['industry']}: {direction}, average difference `{abs(item['average_difference']):.3f}`, "
            f"share of months above 1 pct `{item['share_of_months_over_1pct']:.1%}`."
        )
    lines.append("")
    lines.append(
        f"- PTO and E2E show meaningfully different industry tilts month by month: `{summary['meaningfully_different_month_by_month']}`."
    )
    lines.append(
        f"- The monthly view reveals differences that yearly averages partially hide: `{summary['monthly_view_reveals_more_than_yearly_average']}`."
    )
    return "\n".join(lines)


def build_top_divergence_markdown(divergence_frame: pd.DataFrame) -> str:
    lines = [
        "# Top Divergence Months",
        "",
        "- Sign convention throughout: `PTO minus E2E`.",
        "- Returns use realized total return from `ret_tplus1`.",
        "",
        "| Month | PTO Return | E2E Return | PTO - E2E | Top Contributing Industries |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in divergence_frame.iterrows():
        lines.append(
            f"| {int(row['yyyymm'])} | {row['pto_realized_return']:+.4f} | {row['e2e_realized_return']:+.4f} | "
            f"{row['pto_minus_e2e_return']:+.4f} | {row['top_contributing_industries']} |"
        )
    return "\n".join(lines)


def build_difference_markdown_summary(summary: dict[str, object]) -> str:
    lines = [
        "# PTO Versus E2E Difference Diagnostics",
        "",
        "- Sign convention throughout: `PTO minus E2E`.",
        "",
        "## Persistent Broad-Industry Differences",
    ]
    for item in summary["persistent_broad_industry_differences"]:
        direction = "PTO overweight" if item["direction"] == "pto_overweight" else "E2E overweight"
        lines.append(
            f"- {item['industry']}: {direction}, average active weight `{item['average_active_weight']:+.3f}`, "
            f"share of months above 1 pct `{item['share_over_1pct']:.1%}`, cumulative active contribution `{item['cumulative_active_contribution']:+.4f}`."
        )
    active_share = summary["active_share_summary"]
    lines.extend(
        [
            "",
            "## Similarity Metrics",
            f"- Average active share: `{active_share['average_active_share']:.3f}`.",
            f"- Average Jaccard overlap of active positions: `{active_share['average_jaccard_active_positions']:.3f}`.",
            f"- Average cosine similarity: `{active_share['average_cosine_similarity']:.3f}`.",
            f"- Average top-50 overlap: `{active_share['average_top_50_overlap']:.3f}`.",
            "",
            "## Concentration Of Return Gap",
            f"- Top 10 divergence months explain `{summary['top_divergence_months_share_of_absolute_gap']:.1%}` of the absolute PTO-versus-E2E return gap.",
            f"- Return gap concentrated in a few key months: `{summary['return_gap_concentrated_in_few_months']}`.",
            "",
            "## Key SIC2 Buckets",
        ]
    )
    for item in summary["top_sic2_buckets"]:
        lines.append(
            f"- SIC2 `{item['industry_sic2']}`: average absolute active weight `{item['average_absolute_active_weight']:.3f}`, "
            f"cumulative active contribution `{item['cumulative_active_contribution']:+.4f}`, "
            f"cumulative absolute active contribution `{item['cumulative_absolute_active_contribution']:.4f}`."
        )
    lines.append("")
    lines.append(
        f"- Models substantially different month to month: `{summary['models_substantially_different_month_to_month']}`."
    )
    return "\n".join(lines)


def build_presentation_note_markdown(
    executive_tables: dict[str, pd.DataFrame],
    decomposition_tables: dict[str, pd.DataFrame],
    sic2_tables: dict[str, pd.DataFrame],
) -> str:
    top_broad = decomposition_tables["active_contributions"].head(3)["industry_division"].tolist()
    top_sic2 = sic2_tables["top_buckets"].head(5)["industry_sic2"].astype(str).tolist()
    lines = [
        "# Presentation Figure Package",
        "",
        "## Design choices",
        "- The figure package reuses the existing frozen PTO, frozen E2E, and saved diagnostics inputs rather than adding a new raw-analysis branch.",
        "- Main-text figures are filtered to the most economically important industries so the sign convention stays readable: positive always means `PTO overweight relative to E2E`.",
        "- Direct labels are used where practical to reduce legend clutter, while benchmarks and context lines stay grey.",
        "",
        "## Figure intent",
        "- `executive_summary_figure`: the best main-paper figure. It combines performance, active share, top contribution paths, and the most important broad-industry active tilts in one compact story.",
        "- `difference_decomposition_figure`: best for appendix or slides when the audience needs ranked decomposition and concrete divergence months.",
        "- `sic2_drilldown_figure`: appendix-style detail that sharpens the broad-industry story and shows which narrower SIC2 buckets carry the signal.",
        "",
        "## Main takeaways",
        f"- The broad-industry story is driven most by: {', '.join(top_broad)}.",
        f"- The SIC2 drill-down sharpens that story through buckets such as: {', '.join(top_sic2)}.",
        "- The executive figure is the best choice for the main paper text; the decomposition and SIC2 figures are better suited for appendix pages or presentation backup slides.",
    ]
    return "\n".join(lines)


def yyyymm_to_timestamp(yyyymm: int) -> pd.Timestamp:
    year = yyyymm // 100
    month = yyyymm % 100
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def previous_yyyymm(yyyymm: int) -> int:
    year = yyyymm // 100
    month = yyyymm % 100
    if month == 1:
        return (year - 1) * 100 + 12
    return year * 100 + (month - 1)
