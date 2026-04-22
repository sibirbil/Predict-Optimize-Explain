from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configs.e2e import ContrastiveE2EConfig, E2EV1Config
from src.modeling.e2e.contrastive import is_recession_month, load_nfci_monthly, parse_recession_intervals
from src.modeling.portfolio_diagnostics.report import load_roster, previous_yyyymm, yyyymm_to_timestamp
from src.utils.paths import resolve_repo_path


PALETTE = {
    "summer_child": "#d62728",
    "winter_wolf": "#1f77b4",
    "pto": "#2f4f4f",
    "exact_e2e": "#6c757d",
    "equal_weight": "#c08a00",
}

DISPLAY_NAMES = {
    "summer_child": "SummerChild",
    "winter_wolf": "WinterWolf",
    "pto": "PTO",
    "exact_e2e": "Exact-layer E2E",
    "equal_weight": "Equal-weight",
}

# Strategies to include in the main wealth-path and rebased-window plots.
WEALTH_PATH_STRATEGIES = ("summer_child", "winter_wolf", "pto", "equal_weight")


def format_yyyymm(yyyymm: int) -> str:
    year = int(yyyymm) // 100
    month = int(yyyymm) % 100
    return f"{year:04d}-{month:02d}"


@dataclass(frozen=True)
class ContrastiveEventStudyPaths:
    contrastive_run_dir: Path
    summer_child_dir: Path
    winter_wolf_dir: Path
    pto_run_dir: Path
    exact_e2e_run_dir: Path
    frozen_e2e_run_dir: Path
    output_dir: Path
    roster_path: Path


def build_contrastive_event_study_package(
    contrastive_run_name: str = "summerchild_winterwolf_exact_layer_20260409",
    output_run_name: str = "summerchild_winterwolf_broad_event_study_20260409",
    contrastive_output_root: str = "batuhan/artifacts/e2e_contrastive",
    output_root: str = "batuhan/artifacts/portfolio_diagnostics",
    roster_path: str = "batuhan/universe_500/universe_500_roster.parquet",
) -> Path:
    contrastive_run_dir = resolve_repo_path(contrastive_output_root) / contrastive_run_name
    config = load_contrastive_config(contrastive_run_dir)
    paths = ContrastiveEventStudyPaths(
        contrastive_run_dir=contrastive_run_dir,
        summer_child_dir=contrastive_run_dir / "summer_child",
        winter_wolf_dir=contrastive_run_dir / "winter_wolf",
        pto_run_dir=config.resolve_pto_run_dir(),
        exact_e2e_run_dir=config.resolve_base_e2e_run_dir(),
        frozen_e2e_run_dir=config.resolve_frozen_e2e_run_dir() or config.resolve_base_e2e_run_dir(),
        output_dir=resolve_repo_path(output_root) / output_run_name,
        roster_path=resolve_repo_path(roster_path),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = load_json(paths.contrastive_run_dir / "train_month_mask_summary.json")
    roster = load_roster(paths.roster_path)
    monthly_returns = build_monthly_return_panel(paths)
    stress_tags = build_test_stress_tags(
        contrastive_config=config,
        mask_summary=summary_payload,
        yyyymms=monthly_returns["yyyymm"].tolist(),
    )
    monthly_gap = build_monthly_gap_frame(monthly_returns, stress_tags)
    wealth_paths = build_full_wealth_paths(monthly_returns)
    rebased_windows = build_rebased_windows(wealth_paths, monthly_gap)
    divergence_months, divergence_events = build_top_divergence_tables(paths, roster, monthly_gap, top_n=10, event_radius=3)
    rolling_metrics = build_rolling_metrics(monthly_returns, windows=(12, 36))
    stress_summary = build_stress_summary(monthly_returns, stress_tags)
    allocation_package = build_allocation_difference_tables(paths, roster, stress_tags)
    narrative = build_narrative_summary(
        monthly_gap=monthly_gap,
        stress_summary=stress_summary,
        divergence_months=divergence_months,
        allocation_package=allocation_package,
    )

    wealth_paths.to_csv(paths.output_dir / "wealth_paths_full.csv", index=False)
    rebased_windows.to_csv(paths.output_dir / "wealth_paths_rebased_windows.csv", index=False)
    monthly_gap.to_csv(paths.output_dir / "monthly_return_gap.csv", index=False)
    divergence_months.to_csv(paths.output_dir / "top_divergence_months.csv", index=False)
    divergence_events.to_csv(paths.output_dir / "top_divergence_event_windows.csv", index=False)
    stress_summary.to_csv(paths.output_dir / "stress_vs_nonstress_summary.csv", index=False)
    rolling_metrics.to_csv(paths.output_dir / "rolling_metrics.csv", index=False)
    allocation_package["active_share"].to_csv(paths.output_dir / "summer_vs_winter_active_share.csv", index=False)
    allocation_package["industry_active_weights"].to_csv(
        paths.output_dir / "summer_vs_winter_industry_active_weights.csv",
        index=False,
    )
    allocation_package["industry_summary"].to_csv(
        paths.output_dir / "summer_vs_winter_industry_summary.csv",
        index=False,
    )
    allocation_package["stress_comparison"].to_csv(
        paths.output_dir / "summer_vs_winter_active_share_stress_comparison.csv",
        index=False,
    )
    (paths.output_dir / "summer_vs_winter_summary.md").write_text(narrative, encoding="utf-8")

    full_wealth_figure = plot_full_wealth_paths(wealth_paths)
    full_wealth_figure.savefig(paths.output_dir / "summer_vs_winter_full_wealth.png", dpi=240, bbox_inches="tight")
    full_wealth_figure.savefig(paths.output_dir / "summer_vs_winter_full_wealth.pdf", bbox_inches="tight")
    plt.close(full_wealth_figure)

    rebased_figure = plot_rebased_windows(rebased_windows)
    rebased_figure.savefig(paths.output_dir / "summer_vs_winter_rebased_windows.png", dpi=240, bbox_inches="tight")
    rebased_figure.savefig(paths.output_dir / "summer_vs_winter_rebased_windows.pdf", bbox_inches="tight")
    plt.close(rebased_figure)

    gap_figure = plot_monthly_return_gap(monthly_gap)
    gap_figure.savefig(paths.output_dir / "summer_vs_winter_monthly_return_gap.png", dpi=240, bbox_inches="tight")
    gap_figure.savefig(paths.output_dir / "summer_vs_winter_monthly_return_gap.pdf", bbox_inches="tight")
    plt.close(gap_figure)

    rolling_figure = plot_rolling_metrics(rolling_metrics)
    rolling_figure.savefig(paths.output_dir / "summer_vs_winter_rolling_metrics.png", dpi=240, bbox_inches="tight")
    rolling_figure.savefig(paths.output_dir / "summer_vs_winter_rolling_metrics.pdf", bbox_inches="tight")
    plt.close(rolling_figure)

    divergence_figure = plot_top_divergence_windows(divergence_events)
    divergence_figure.savefig(
        paths.output_dir / "summer_vs_winter_top_divergence_windows.png",
        dpi=240,
        bbox_inches="tight",
    )
    divergence_figure.savefig(
        paths.output_dir / "summer_vs_winter_top_divergence_windows.pdf",
        bbox_inches="tight",
    )
    plt.close(divergence_figure)

    return paths.output_dir


def load_contrastive_config(run_dir: Path) -> ContrastiveE2EConfig:
    return ContrastiveE2EConfig.from_json(run_dir / "contrastive_config.json")


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_monthly_portfolio(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["yyyymm"] = frame["yyyymm"].astype(int)
    frame["date"] = frame["yyyymm"].apply(yyyymm_to_timestamp)
    output = pd.DataFrame(
        {
            "yyyymm": frame["yyyymm"].astype(int),
            "date": frame["date"],
            "strategy": label,
            "portfolio_return": frame["portfolio_return"].astype(float),
            "portfolio_excess_return": frame["portfolio_excess_return"].astype(float),
        }
    )
    return output


def build_equal_weight_monthly_returns(pto_run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(pto_run_dir / "pto_benchmark_weights.csv")
    frame = frame.loc[(frame["strategy"] == "equal_weight") & (frame["split"] == "test")].copy()
    grouped = (
        frame.groupby("yyyymm", as_index=False)
        .agg(
            portfolio_return=("realized_return", lambda s: float(np.dot(s.to_numpy(), frame.loc[s.index, "weight"].to_numpy()))),
            portfolio_excess_return=(
                "realized_excess_return",
                lambda s: float(np.dot(s.to_numpy(), frame.loc[s.index, "weight"].to_numpy())),
            ),
        )
        .sort_values("yyyymm")
        .reset_index(drop=True)
    )
    grouped["date"] = grouped["yyyymm"].astype(int).apply(yyyymm_to_timestamp)
    grouped["strategy"] = "equal_weight"
    return grouped[["yyyymm", "date", "strategy", "portfolio_return", "portfolio_excess_return"]]


def build_monthly_return_panel(paths: ContrastiveEventStudyPaths) -> pd.DataFrame:
    frames = [
        load_monthly_portfolio(paths.summer_child_dir / "test_monthly_portfolio.csv", "summer_child"),
        load_monthly_portfolio(paths.winter_wolf_dir / "test_monthly_portfolio.csv", "winter_wolf"),
        load_monthly_portfolio(paths.pto_run_dir / "test_monthly_portfolio.csv", "pto"),
        load_monthly_portfolio(paths.exact_e2e_run_dir / "test_monthly_portfolio.csv", "exact_e2e"),
        build_equal_weight_monthly_returns(paths.pto_run_dir),
    ]
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", "strategy"]).reset_index(drop=True)


def build_test_stress_tags(
    contrastive_config: ContrastiveE2EConfig,
    mask_summary: dict[str, object],
    yyyymms: list[int],
) -> pd.DataFrame:
    base_config = E2EV1Config.from_json(contrastive_config.resolve_base_e2e_config_path())
    loader = base_config.load_data_config().build_loader()
    macro_state = loader.load_macro_state().copy()
    macro_state["yyyymm"] = macro_state["yyyymm"].astype(int)
    macro_state = macro_state.loc[macro_state["yyyymm"].isin(sorted(set(int(v) for v in yyyymms)))].copy()
    macro_state = macro_state.sort_values("yyyymm").reset_index(drop=True)

    z_stats = mask_summary["z_score_stats"]
    for column in ("dfy", "svar", "tms"):
        mean = float(z_stats[column]["mean"])
        std = float(z_stats[column]["std"]) or 1.0
        macro_state[f"{column}_z"] = (macro_state[column].astype(float) - mean) / std

    macro_state["stress_score"] = (
        float(contrastive_config.dfy_weight) * macro_state["dfy_z"]
        + float(contrastive_config.svar_weight) * macro_state["svar_z"]
        + float(contrastive_config.tms_weight) * macro_state["tms_z"]
    )
    stress_threshold = float(mask_summary["stress_score_threshold"])
    macro_state["macro_stress_flag"] = (
        (macro_state["stress_score"] > stress_threshold)
        | (macro_state["dfy_z"] > float(contrastive_config.dfy_z_cap))
        | (macro_state["svar_z"] > float(contrastive_config.svar_z_cap))
    )

    macro_state["nfci_stress_flag"] = False
    if contrastive_config.use_nfci and mask_summary.get("nfci_threshold") is not None:
        nfci_df = load_nfci_monthly(resolve_repo_path(contrastive_config.nfci_csv_path))
        macro_state = macro_state.merge(nfci_df, on="yyyymm", how="left")
        macro_state["nfci_stress_flag"] = macro_state["nfci"].fillna(-np.inf) >= float(mask_summary["nfci_threshold"])
    else:
        macro_state["nfci"] = np.nan

    macro_state["recession_flag"] = False
    if contrastive_config.use_recession:
        intervals = parse_recession_intervals(resolve_repo_path(contrastive_config.recession_md_path))
        macro_state["recession_flag"] = macro_state["yyyymm"].apply(lambda v: is_recession_month(int(v), intervals))

    macro_state["stress_flag"] = (
        macro_state["macro_stress_flag"] | macro_state["nfci_stress_flag"] | macro_state["recession_flag"]
    )
    macro_state["stress_label"] = np.where(macro_state["stress_flag"], "stress", "nonstress")
    macro_state["date"] = macro_state["yyyymm"].apply(yyyymm_to_timestamp)
    return macro_state[
        [
            "yyyymm",
            "date",
            "stress_label",
            "stress_flag",
            "macro_stress_flag",
            "nfci_stress_flag",
            "recession_flag",
            "stress_score",
            "nfci",
        ]
    ].copy()


def build_monthly_gap_frame(monthly_returns: pd.DataFrame, stress_tags: pd.DataFrame) -> pd.DataFrame:
    pivot = monthly_returns.pivot(index=["yyyymm", "date"], columns="strategy", values="portfolio_return").reset_index()
    pivot.columns.name = None
    pivot = pivot.merge(stress_tags, on=["yyyymm", "date"], how="left")
    pivot["winter_minus_summer"] = pivot["winter_wolf"] - pivot["summer_child"]
    pivot["cumulative_gap"] = pivot["winter_minus_summer"].cumsum()
    pivot["relative_gap_wealth"] = ((1.0 + pivot["winter_wolf"]) / (1.0 + pivot["summer_child"])).cumprod() - 1.0
    pivot["winner"] = np.where(pivot["winter_minus_summer"] > 0.0, "WinterWolf", "SummerChild")
    pivot.loc[pivot["winter_minus_summer"].abs() < 1e-12, "winner"] = "Tie"
    return pivot.sort_values("date").reset_index(drop=True)


def build_full_wealth_paths(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy, frame in monthly_returns.groupby("strategy", sort=False):
        ordered = frame.sort_values("date").copy()
        ordered["wealth"] = (1.0 + ordered["portfolio_return"]).cumprod()
        ordered["display_name"] = DISPLAY_NAMES[strategy]
        frames.append(ordered)
    wealth = pd.concat(frames, ignore_index=True)
    start_date = wealth["date"].min() - pd.offsets.MonthEnd(1)
    start_yyyymm = previous_yyyymm(int(wealth["yyyymm"].min()))
    start_rows = pd.DataFrame(
        {
            "yyyymm": [start_yyyymm] * len(DISPLAY_NAMES),
            "date": [start_date] * len(DISPLAY_NAMES),
            "strategy": list(DISPLAY_NAMES.keys()),
            "portfolio_return": [0.0] * len(DISPLAY_NAMES),
            "portfolio_excess_return": [0.0] * len(DISPLAY_NAMES),
            "wealth": [1.0] * len(DISPLAY_NAMES),
            "display_name": [DISPLAY_NAMES[k] for k in DISPLAY_NAMES],
        }
    )
    wealth = pd.concat([start_rows, wealth], ignore_index=True)
    return wealth.sort_values(["strategy", "date"]).reset_index(drop=True)


def month_distance(a: int, b: int) -> int:
    ay, am = divmod(a, 100)
    by, bm = divmod(b, 100)
    return abs((ay * 12 + am) - (by * 12 + bm))


def discover_additional_windows(monthly_gap: pd.DataFrame, required: list[int], extra_count: int = 2) -> list[int]:
    selected = list(required)
    ranked = monthly_gap.reindex(monthly_gap["winter_minus_summer"].abs().sort_values(ascending=False).index)
    for yyyymm in ranked["yyyymm"].astype(int):
        if any(month_distance(int(yyyymm), int(existing)) <= 9 for existing in selected):
            continue
        selected.append(int(yyyymm))
        if len(selected) >= len(required) + extra_count:
            break
    return selected


def build_rebased_windows(wealth_paths: pd.DataFrame, monthly_gap: pd.DataFrame) -> pd.DataFrame:
    required = [int(monthly_gap["yyyymm"].min()), 202001, 202201]
    available = set(monthly_gap["yyyymm"].astype(int))
    required = [ym for ym in required if ym in available]
    windows = discover_additional_windows(monthly_gap, required=required, extra_count=2)
    rows: list[pd.DataFrame] = []
    for start_yyyymm in windows:
        start_label = format_yyyymm(int(start_yyyymm))
        window_frame = wealth_paths.loc[wealth_paths["yyyymm"] >= int(start_yyyymm)].copy()
        for strategy, part in window_frame.groupby("strategy", sort=False):
            base = float(part.iloc[0]["wealth"])
            tmp = part.copy()
            tmp["window_start_yyyymm"] = int(start_yyyymm)
            tmp["window_label"] = f"Start {start_label}"
            tmp["wealth_rebased"] = tmp["wealth"] / base
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True).sort_values(["window_start_yyyymm", "strategy", "date"]).reset_index(drop=True)


def build_top_divergence_tables(
    paths: ContrastiveEventStudyPaths,
    roster: pd.DataFrame,
    monthly_gap: pd.DataFrame,
    top_n: int,
    event_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summer_assets = pd.read_parquet(paths.summer_child_dir / "test_asset_predictions.parquet")[
        ["yyyymm", "permno", "weight", "ret_tplus1"]
    ].rename(columns={"weight": "summer_weight"})
    winter_assets = pd.read_parquet(paths.winter_wolf_dir / "test_asset_predictions.parquet")[
        ["yyyymm", "permno", "weight", "ret_tplus1"]
    ].rename(columns={"weight": "winter_weight"})
    merged = summer_assets.merge(
        winter_assets.rename(columns={"ret_tplus1": "ret_tplus1_winter"}),
        on=["yyyymm", "permno"],
        how="outer",
    )
    merged["summer_weight"] = merged["summer_weight"].fillna(0.0)
    merged["winter_weight"] = merged["winter_weight"].fillna(0.0)
    merged["ret_tplus1"] = merged["ret_tplus1"].combine_first(merged["ret_tplus1_winter"]).fillna(0.0)
    merged["active_weight"] = merged["winter_weight"] - merged["summer_weight"]
    merged["active_contribution"] = merged["active_weight"] * merged["ret_tplus1"]
    merged["date"] = merged["yyyymm"].astype(int).apply(yyyymm_to_timestamp)
    merged = merged.merge(roster[["permno", "industry_division"]], on="permno", how="left")
    merged["industry_division"] = merged["industry_division"].fillna("Unknown / Other")

    industry_contrib = (
        merged.groupby(["yyyymm", "industry_division"], as_index=False)["active_contribution"].sum().sort_values(["yyyymm", "active_contribution"])
    )
    ranked = monthly_gap.reindex(monthly_gap["winter_minus_summer"].abs().sort_values(ascending=False).index).head(top_n)
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    ranked_months = ranked["yyyymm"].astype(int).tolist()
    for rank, yyyymm in enumerate(ranked_months, start=1):
        month_row = ranked.loc[ranked["yyyymm"] == yyyymm].iloc[0]
        month_contrib = industry_contrib.loc[industry_contrib["yyyymm"] == yyyymm].copy()
        positive = month_contrib.sort_values("active_contribution", ascending=False).head(3)
        negative = month_contrib.sort_values("active_contribution", ascending=True).head(3)
        rows.append(
            {
                "rank": rank,
                "month": format_yyyymm(int(yyyymm)),
                "yyyymm": int(yyyymm),
                "summer_child_return": float(month_row["summer_child"]),
                "winter_wolf_return": float(month_row["winter_wolf"]),
                "winter_minus_summer": float(month_row["winter_minus_summer"]),
                "pto_return": float(month_row["pto"]),
                "exact_e2e_return": float(month_row["exact_e2e"]),
                "equal_weight_return": float(month_row["equal_weight"]),
                "stress_label": month_row["stress_label"],
                "macro_stress_flag": bool(month_row["macro_stress_flag"]),
                "nfci_stress_flag": bool(month_row["nfci_stress_flag"]),
                "recession_flag": bool(month_row["recession_flag"]),
                "top_positive_industries": "; ".join(
                    f"{row.industry_division} ({row.active_contribution:+.3%})"
                    for row in positive.itertuples()
                ),
                "top_negative_industries": "; ".join(
                    f"{row.industry_division} ({row.active_contribution:+.3%})"
                    for row in negative.itertuples()
                ),
            }
        )
        event_min = shift_yyyymm(int(yyyymm), -event_radius)
        event_max = shift_yyyymm(int(yyyymm), event_radius)
        event_slice = monthly_gap.loc[
            (monthly_gap["yyyymm"] >= event_min) & (monthly_gap["yyyymm"] <= event_max),
            ["yyyymm", "date", "summer_child", "winter_wolf", "pto", "exact_e2e", "equal_weight", "winter_minus_summer", "stress_label"],
        ].copy()
        if event_slice.empty:
            continue
        base_rows = []
        for column in ("summer_child", "winter_wolf", "pto", "exact_e2e", "equal_weight"):
            wealth = (1.0 + event_slice[column]).cumprod()
            wealth = wealth / wealth.iloc[0]
            tmp = pd.DataFrame(
                {
                    "rank": rank,
                    "event_month": int(yyyymm),
                    "event_month_label": format_yyyymm(int(yyyymm)),
                    "yyyymm": event_slice["yyyymm"].astype(int),
                    "date": event_slice["date"],
                    "strategy": column,
                    "wealth_rebased": wealth,
                    "month_offset": np.arange(len(event_slice)) - np.where(event_slice["yyyymm"].astype(int).to_numpy() == int(yyyymm))[0][0],
                    "winter_minus_summer": event_slice["winter_minus_summer"].to_numpy(),
                    "stress_label": event_slice["stress_label"].to_numpy(),
                }
            )
            base_rows.append(tmp)
        event_rows.extend(pd.concat(base_rows, ignore_index=True).to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def shift_yyyymm(yyyymm: int, offset: int) -> int:
    year = yyyymm // 100
    month = yyyymm % 100
    total = year * 12 + (month - 1) + offset
    new_year = total // 12
    new_month = total % 12 + 1
    return new_year * 100 + new_month


def build_rolling_metrics(monthly_returns: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for strategy in ("summer_child", "winter_wolf"):
        part = monthly_returns.loc[monthly_returns["strategy"] == strategy].sort_values("date").copy()
        for window in windows:
            excess = part["portfolio_excess_return"].astype(float).rolling(window)
            total = part["portfolio_return"].astype(float).rolling(window)
            out = part[["yyyymm", "date"]].copy()
            out["strategy"] = strategy
            out["window_months"] = window
            out["rolling_excess_return"] = excess.mean() * 12.0
            out["rolling_volatility"] = total.std(ddof=0) * np.sqrt(12.0)
            out["rolling_sharpe"] = np.where(
                out["rolling_volatility"] > 0,
                excess.mean() / total.std(ddof=0) * np.sqrt(12.0),
                np.nan,
            )
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize_regime_subset(returns: pd.Series) -> tuple[float, float]:
    mean_monthly = float(returns.mean()) if not returns.empty else np.nan
    std_monthly = float(returns.std(ddof=0)) if returns.shape[0] > 1 else np.nan
    sharpe = np.nan
    if std_monthly and std_monthly > 0:
        sharpe = mean_monthly / std_monthly * np.sqrt(12.0)
    return mean_monthly, sharpe


def build_stress_summary(monthly_returns: pd.DataFrame, stress_tags: pd.DataFrame) -> pd.DataFrame:
    merged = monthly_returns.merge(stress_tags[["yyyymm", "stress_label"]], on="yyyymm", how="left")
    rows: list[dict[str, object]] = []
    for strategy in ("summer_child", "winter_wolf"):
        for stress_label in ("stress", "nonstress"):
            subset = merged.loc[
                (merged["strategy"] == strategy) & (merged["stress_label"] == stress_label),
                "portfolio_return",
            ].astype(float)
            mean_monthly, sharpe = summarize_regime_subset(subset)
            rows.append(
                {
                    "strategy": strategy,
                    "display_name": DISPLAY_NAMES[strategy],
                    "stress_label": stress_label,
                    "n_months": int(subset.shape[0]),
                    "average_monthly_return": mean_monthly,
                    "annualized_return_proxy": mean_monthly * 12.0 if pd.notna(mean_monthly) else np.nan,
                    "sharpe_ratio": sharpe,
                }
            )
    return pd.DataFrame(rows)


def build_allocation_difference_tables(
    paths: ContrastiveEventStudyPaths,
    roster: pd.DataFrame,
    stress_tags: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summer = pd.read_parquet(paths.summer_child_dir / "test_monthly_weights.parquet").rename(columns={"weight": "summer_weight"})
    winter = pd.read_parquet(paths.winter_wolf_dir / "test_monthly_weights.parquet").rename(columns={"weight": "winter_weight"})
    merged = summer.merge(winter, on=["yyyymm", "permno"], how="outer")
    merged["summer_weight"] = merged["summer_weight"].fillna(0.0)
    merged["winter_weight"] = merged["winter_weight"].fillna(0.0)
    merged["active_weight"] = merged["winter_weight"] - merged["summer_weight"]
    merged["abs_active_weight"] = merged["active_weight"].abs()
    merged["date"] = merged["yyyymm"].astype(int).apply(yyyymm_to_timestamp)
    merged = merged.merge(roster[["permno", "industry_division"]], on="permno", how="left")
    merged["industry_division"] = merged["industry_division"].fillna("Unknown / Other")
    merged = merged.merge(stress_tags[["yyyymm", "stress_label"]], on="yyyymm", how="left")

    active_share = (
        merged.groupby(["yyyymm", "date", "stress_label"], as_index=False)["abs_active_weight"].sum().rename(columns={"abs_active_weight": "active_share_twice"})
    )
    active_share["active_share"] = 0.5 * active_share["active_share_twice"]
    active_share = active_share.drop(columns=["active_share_twice"])

    industry_active = (
        merged.groupby(["yyyymm", "date", "industry_division", "stress_label"], as_index=False)["active_weight"].sum().sort_values(["yyyymm", "industry_division"])
    )
    industry_summary = (
        industry_active.groupby("industry_division", as_index=False)
        .agg(
            mean_active_weight=("active_weight", "mean"),
            mean_abs_active_weight=("active_weight", lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float))))),
            mean_active_weight_stress=(
                "active_weight",
                lambda s: float(np.mean(np.abs(s[industry_active.loc[s.index, "stress_label"] == "stress"].to_numpy(dtype=float))))
                if (industry_active.loc[s.index, "stress_label"] == "stress").any()
                else np.nan,
            ),
            mean_active_weight_nonstress=(
                "active_weight",
                lambda s: float(np.mean(np.abs(s[industry_active.loc[s.index, "stress_label"] == "nonstress"].to_numpy(dtype=float))))
                if (industry_active.loc[s.index, "stress_label"] == "nonstress").any()
                else np.nan,
            ),
        )
        .sort_values("mean_abs_active_weight", ascending=False)
        .reset_index(drop=True)
    )
    industry_summary["stress_minus_nonstress_abs_active_weight"] = (
        industry_summary["mean_active_weight_stress"] - industry_summary["mean_active_weight_nonstress"]
    )
    stress_comparison = (
        active_share.groupby("stress_label", as_index=False)
        .agg(
            n_months=("active_share", "count"),
            average_active_share=("active_share", "mean"),
            median_active_share=("active_share", "median"),
        )
        .reset_index(drop=True)
    )
    return {
        "active_share": active_share,
        "industry_active_weights": industry_active,
        "industry_summary": industry_summary,
        "stress_comparison": stress_comparison,
    }


def format_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def build_narrative_summary(
    monthly_gap: pd.DataFrame,
    stress_summary: pd.DataFrame,
    divergence_months: pd.DataFrame,
    allocation_package: dict[str, pd.DataFrame],
) -> str:
    stress_pivot = stress_summary.pivot(index="strategy", columns="stress_label", values="sharpe_ratio")
    mean_gap = float(monthly_gap["winter_minus_summer"].mean())
    positive_share = float((monthly_gap["winter_minus_summer"] > 0.0).mean())
    stress_months = monthly_gap.loc[monthly_gap["stress_flag"]].copy()
    nonstress_months = monthly_gap.loc[~monthly_gap["stress_flag"]].copy()
    stress_gap = float(stress_months["winter_minus_summer"].mean()) if not stress_months.empty else np.nan
    nonstress_gap = float(nonstress_months["winter_minus_summer"].mean()) if not nonstress_months.empty else np.nan
    top_windows = divergence_months.head(5)
    top_industries = allocation_package["industry_summary"].head(5)["industry_division"].tolist()

    lines = [
        "# SummerChild vs WinterWolf Broad Event Study",
        "",
        "## Direct answers",
        "",
        f"1. **Does WinterWolf outperform specifically in stress / crisis periods?** "
        f"Average monthly return gap `WinterWolf - SummerChild` is {format_pct(stress_gap)} in tagged stress months versus {format_pct(nonstress_gap)} in non-stress months. "
        f"Stress-month Sharpe is `{stress_pivot.loc['winter_wolf', 'stress']:.3f}` for WinterWolf versus `{stress_pivot.loc['summer_child', 'stress']:.3f}` for SummerChild.",
        f"2. **Is the WinterWolf advantage concentrated or persistent?** WinterWolf beats SummerChild in {positive_share:.1%} of test months, with mean monthly gap {format_pct(mean_gap)}. The cumulative gap plot and divergence windows show whether the edge comes from a few bursts rather than a smooth persistent lead.",
        f"3. **Where does SummerChild outperform?** SummerChild wins in the months where `WinterWolf - SummerChild < 0`, especially the largest negative-gap windows listed below and in the lower-volatility stretches highlighted by the rolling Sharpe plots.",
        "4. **Is the pair meaningful before scenario generation?** Yes: the pair has visibly different realized paths, rolling risk-adjusted performance, and industry tilts, but they still share the same exact-layer recipe and benchmark context. That makes the contrast economically interpretable without changing training doctrine.",
        "5. **Which windows should be prioritized later in scenario generation?** The strongest candidates are the top absolute-gap windows below, especially those that are also stress-tagged.",
        "",
        "## Priority windows",
        "",
    ]
    for row in top_windows.itertuples():
        lines.append(
            f"- {row.month}: gap {row.winter_minus_summer:+.2%}, stress={row.stress_label}, "
            f"WinterWolf={row.winter_wolf_return:+.2%}, SummerChild={row.summer_child_return:+.2%}"
        )
    lines += [
        "",
        "## Allocation note",
        "",
        f"- The largest average broad-industry differences are in: {', '.join(top_industries)}.",
        "- Positive active weight means WinterWolf overweight relative to SummerChild.",
        "- Stress-vs-nonstress active-share summaries are saved alongside this note for later scenario targeting.",
    ]
    return "\n".join(lines)


def plot_full_wealth_paths(wealth_paths: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for strategy in WEALTH_PATH_STRATEGIES:
        part = wealth_paths.loc[wealth_paths["strategy"] == strategy].sort_values("date")
        linewidth = 2.8 if strategy in {"summer_child", "winter_wolf"} else 2.0
        alpha = 1.0 if strategy in {"summer_child", "winter_wolf"} else 0.9
        ax.plot(part["date"], part["wealth"], color=PALETTE[strategy], linewidth=linewidth, alpha=alpha)
        label_last_point(ax, part["date"].iloc[-1], part["wealth"].iloc[-1], DISPLAY_NAMES[strategy], PALETTE[strategy])
    ax.set_title("SummerChild vs WinterWolf: Full Test-Period Wealth", fontsize=15, fontweight="bold")
    ax.set_ylabel("Wealth index")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_rebased_windows(rebased_windows: pd.DataFrame) -> plt.Figure:
    window_labels = list(dict.fromkeys(rebased_windows["window_label"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(window_labels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 3.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, label in zip(axes_flat, window_labels):
        subset = rebased_windows.loc[rebased_windows["window_label"] == label]
        for strategy in WEALTH_PATH_STRATEGIES:
            part = subset.loc[subset["strategy"] == strategy].sort_values("date")
            ax.plot(part["date"], part["wealth_rebased"], color=PALETTE[strategy], linewidth=2.3 if strategy in {"summer_child", "winter_wolf"} else 1.8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(window_labels):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=PALETTE[s], lw=2.5, label=DISPLAY_NAMES[s]) for s in WEALTH_PATH_STRATEGIES]
    fig.legend(handles=handles, loc="upper center", ncol=len(WEALTH_PATH_STRATEGIES), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Rebased Wealth Paths From Key Windows", fontsize=16, fontweight="bold", y=1.06)
    fig.tight_layout()
    return fig


def plot_monthly_return_gap(monthly_gap: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(monthly_gap["date"], monthly_gap["summer_child"], color=PALETTE["summer_child"], linewidth=2.2)
    axes[0].plot(monthly_gap["date"], monthly_gap["winter_wolf"], color=PALETTE["winter_wolf"], linewidth=2.2)
    axes[0].set_title("Monthly Portfolio Returns", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Return")

    bar_colors = np.where(monthly_gap["winter_minus_summer"] >= 0.0, PALETTE["winter_wolf"], PALETTE["summer_child"])
    axes[1].bar(monthly_gap["date"], monthly_gap["winter_minus_summer"], color=bar_colors, width=20, alpha=0.85)
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_title("Monthly Return Gap: WinterWolf - SummerChild", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Gap")

    axes[2].plot(monthly_gap["date"], monthly_gap["cumulative_gap"], color="#111111", linewidth=2.3, label="Cumulative simple gap")
    axes[2].plot(monthly_gap["date"], monthly_gap["relative_gap_wealth"], color="#6c757d", linewidth=1.8, linestyle="--", label="Relative wealth gap")
    axes[2].axhline(0.0, color="#333333", linewidth=1.0)
    axes[2].set_title("Cumulative Gap Process", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Cumulative gap")
    axes[2].legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def plot_rolling_metrics(rolling_metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    metric_specs = [
        ("rolling_sharpe", 12, "12m Rolling Sharpe"),
        ("rolling_sharpe", 36, "36m Rolling Sharpe"),
        ("rolling_excess_return", 12, "12m Rolling Excess Return"),
        ("rolling_excess_return", 36, "36m Rolling Excess Return"),
        ("rolling_volatility", 12, "12m Rolling Volatility"),
        ("rolling_volatility", 36, "36m Rolling Volatility"),
    ]
    for ax, (metric, window, title) in zip(axes.ravel(), metric_specs):
        subset = rolling_metrics.loc[rolling_metrics["window_months"] == window]
        for strategy in ("summer_child", "winter_wolf"):
            part = subset.loc[subset["strategy"] == strategy].sort_values("date")
            ax.plot(part["date"], part[metric], color=PALETTE[strategy], linewidth=2.2)
        ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [plt.Line2D([0], [0], color=PALETTE[s], lw=2.5, label=DISPLAY_NAMES[s]) for s in ("summer_child", "winter_wolf")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def plot_top_divergence_windows(divergence_events: pd.DataFrame) -> plt.Figure:
    event_labels = list(dict.fromkeys(divergence_events["event_month_label"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(event_labels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, event_label in zip(axes_flat, event_labels):
        subset = divergence_events.loc[divergence_events["event_month_label"] == event_label]
        for strategy in ("summer_child", "winter_wolf", "pto", "exact_e2e"):
            part = subset.loc[subset["strategy"] == strategy].sort_values("month_offset")
            ax.plot(
                part["month_offset"],
                part["wealth_rebased"],
                color=PALETTE[strategy],
                linewidth=2.2 if strategy in {"summer_child", "winter_wolf"} else 1.7,
            )
        ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        ax.set_title(event_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Months around event")
        ax.set_ylabel("Rebased wealth")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(event_labels):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=PALETTE[s], lw=2.5, label=DISPLAY_NAMES[s]) for s in ("summer_child", "winter_wolf", "pto", "exact_e2e")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Top Divergence Event Windows", fontsize=16, fontweight="bold", y=1.06)
    fig.tight_layout()
    return fig


def label_last_point(ax: plt.Axes, x_value: pd.Timestamp, y_value: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x_value, y_value),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        fontsize=10.5,
        va="center",
        fontweight="bold",
    )
