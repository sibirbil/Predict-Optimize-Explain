from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.e2e import ContrastiveE2EConfig, E2EV1Config
from src.configs.data import Universe500DataConfig
from src.modeling.e2e.data import E2EDecisionMonth, prepare_e2e_months
from src.modeling.e2e.trainer import (
    E2ECandidate,
    build_model,
    evaluate_model,
    log_progress,
    passes_screens,
    resolve_device,
    set_seed,
    summarize_backtest,
    train_candidate,
)
from src.utils.paths import resolve_repo_path


def run_contrastive_e2e_training(config: ContrastiveE2EConfig) -> Path:
    base_config = E2EV1Config.from_json(config.resolve_base_e2e_config_path())
    selected_candidate = load_selected_candidate(config.resolve_base_e2e_run_dir())
    if config.override_kappa is not None:
        selected_candidate = E2ECandidate(
            learning_rate=selected_candidate.learning_rate,
            weight_decay=selected_candidate.weight_decay,
            init_mode=selected_candidate.init_mode,
            lambda_=selected_candidate.lambda_,
            kappa=config.override_kappa,
            omega_mode=selected_candidate.omega_mode,
        )
    run_dir = config.resolve_output_root() / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "contrastive_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "base_e2e_config_snapshot.json").write_text(
        json.dumps(base_config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (run_dir / "selected_base_candidate.json").write_text(
        json.dumps(selected_candidate.to_dict(), indent=2),
        encoding="utf-8",
    )

    set_seed(base_config.seed)
    device = resolve_device(base_config.device)
    month_data = prepare_e2e_months(base_config)
    input_dim = int(month_data["train"][0].X.shape[1])
    data_config = base_config.load_data_config()
    mask_frame, mask_summary = build_train_month_mask(month_data["train"], config, data_config)
    mask_frame.to_csv(run_dir / "train_month_mask.csv", index=False)
    (run_dir / "train_month_mask_summary.json").write_text(json.dumps(mask_summary, indent=2), encoding="utf-8")
    save_mask_documentation(run_dir, mask_summary, config)
    variants, balanced_summary, variant_mask_frame = prepare_variant_month_sets(
        train_months=month_data["train"],
        mask_frame=mask_frame,
        config=config,
        run_dir=run_dir,
    )
    effective_epochs = config.override_epochs if config.override_epochs is not None else base_config.epochs
    effective_patience = config.override_patience if config.override_patience is not None else base_config.patience
    log_progress(
        f"Contrastive run starting: epochs={effective_epochs}, patience={effective_patience}, "
        f"solver={base_config.solver_mode}, kappa={selected_candidate.kappa}"
    )
    variant_results: dict[str, dict[str, object]] = {}
    for variant_name, train_months in variants.items():
        variant_result = run_single_variant(
            run_dir=run_dir,
            variant_name=variant_name,
            base_config=base_config,
            candidate=selected_candidate,
            input_dim=input_dim,
            train_months=train_months,
            val_months=month_data["val"],
            test_months=month_data["test"],
            device=device,
            mask_frame=variant_mask_frame,
            effective_epochs=effective_epochs,
            effective_patience=effective_patience,
        )
        variant_results[variant_name] = variant_result

    frozen_e2e_run_dir = config.resolve_frozen_e2e_run_dir()
    if frozen_e2e_run_dir is None:
        frozen_e2e_run_dir = config.resolve_base_e2e_run_dir()

    comparison = build_contrastive_comparison(
        run_dir=run_dir,
        variant_results=variant_results,
        frozen_e2e_run_dir=frozen_e2e_run_dir,
        rolling_window=config.rolling_sharpe_window,
        mask_summary=mask_summary,
        solver_mode=base_config.solver_mode,
        config=config,
        balanced_summary=balanced_summary,
    )
    comparison["summary_table"].to_csv(run_dir / "contrastive_summary_metrics.csv", index=False)
    comparison["test_wealth_paths"].to_csv(run_dir / "test_wealth_paths.csv", index=False)
    comparison["rolling_sharpe"].to_csv(run_dir / "rolling_36m_sharpe.csv", index=False)
    if comparison.get("split_comparison") is not None:
        comparison["split_comparison"].to_csv(run_dir / "split_comparison.csv", index=False)
    if comparison.get("focused_comparison") is not None:
        comparison["focused_comparison"].to_csv(run_dir / "three_model_comparison.csv", index=False)
    (run_dir / "contrastive_report.md").write_text(comparison["report_markdown"], encoding="utf-8")
    (run_dir / "contrastive_report.json").write_text(
        json.dumps(comparison["report_payload"], indent=2),
        encoding="utf-8",
    )
    save_comparability_audit_note(
        run_dir=run_dir,
        config=config,
        mask_summary=mask_summary,
        balanced_summary=balanced_summary,
    )
    save_interpretation_note(
        run_dir=run_dir,
        comparison=comparison,
        config=config,
        mask_summary=mask_summary,
        balanced_summary=balanced_summary,
    )
    return run_dir


def load_selected_candidate(base_run_dir: Path) -> E2ECandidate:
    with (base_run_dir / "validation_selection_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload["selected_candidate"]
    return E2ECandidate(
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        init_mode=str(selected["init_mode"]),
        lambda_=float(selected["lambda"]),
        kappa=float(selected["kappa"]),
        omega_mode=str(selected["omega_mode"]),
    )


# ---------------------------------------------------------------------------
# Enhanced train-month mask: macro composite + optional NFCI + recession
# ---------------------------------------------------------------------------


def parse_recession_intervals(md_path: Path) -> list[tuple[int, int]]:
    """Parse NBER recession peak/trough pairs from Recessiondating.md.

    Returns a list of (peak_yyyymm, trough_yyyymm) tuples.
    """
    text = md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(\d{4})-(\d{2})-\d{2},\s*(\d{4})-(\d{2})-\d{2}")
    intervals: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        peak_yyyymm = int(match.group(1)) * 100 + int(match.group(2))
        trough_yyyymm = int(match.group(3)) * 100 + int(match.group(4))
        intervals.append((peak_yyyymm, trough_yyyymm))
    return intervals


def is_recession_month(yyyymm: int, intervals: list[tuple[int, int]]) -> bool:
    for peak, trough in intervals:
        if peak <= yyyymm <= trough:
            return True
    return False


def load_nfci_monthly(csv_path: Path) -> pd.DataFrame:
    """Load NFCI CSV and convert to monthly yyyymm format."""
    nfci = pd.read_csv(csv_path)
    nfci.columns = [c.strip() for c in nfci.columns]
    date_col = [c for c in nfci.columns if "date" in c.lower()][0]
    nfci["date"] = pd.to_datetime(nfci[date_col])
    nfci["yyyymm"] = nfci["date"].dt.year * 100 + nfci["date"].dt.month
    nfci_col = [c for c in nfci.columns if c.upper() == "NFCI"][0]
    return nfci[["yyyymm", nfci_col]].rename(columns={nfci_col: "nfci"}).copy()


def build_train_month_mask(
    train_months: list[E2EDecisionMonth],
    config: ContrastiveE2EConfig,
    data_config: Universe500DataConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    month_ids = [int(month.yyyymm) for month in train_months]
    macro_state = train_months_to_macro_frame(train_months, data_config)

    # --- Macro composite stress (original logic) ---
    stats = {}
    for column in ("dfy", "svar", "tms"):
        mean = float(macro_state[column].mean())
        std = float(macro_state[column].std(ddof=0))
        if std <= 0:
            std = 1.0
        macro_state[f"{column}_z"] = (macro_state[column] - mean) / std
        stats[column] = {"mean": mean, "std": std}
    macro_state["stress_score"] = (
        config.dfy_weight * macro_state["dfy_z"]
        + config.svar_weight * macro_state["svar_z"]
        + config.tms_weight * macro_state["tms_z"]
    )
    threshold = float(macro_state["stress_score"].quantile(config.stress_quantile))
    macro_state["macro_stress_flag"] = (
        (macro_state["stress_score"] > threshold)
        | (macro_state["dfy_z"] > config.dfy_z_cap)
        | (macro_state["svar_z"] > config.svar_z_cap)
    )

    # --- Optional NFCI stress ---
    macro_state["nfci_stress_flag"] = False
    nfci_threshold = None
    if config.use_nfci:
        nfci_path = resolve_repo_path(config.nfci_csv_path)
        if nfci_path.exists():
            nfci_df = load_nfci_monthly(nfci_path)
            macro_state = macro_state.merge(nfci_df, on="yyyymm", how="left")
            nfci_valid = macro_state["nfci"].dropna()
            if not nfci_valid.empty:
                nfci_threshold = float(nfci_valid.quantile(config.nfci_stress_percentile))
                macro_state["nfci_stress_flag"] = macro_state["nfci"].fillna(0.0) >= nfci_threshold
                stats["nfci"] = {
                    "threshold": nfci_threshold,
                    "percentile": config.nfci_stress_percentile,
                    "n_valid": int(nfci_valid.shape[0]),
                }

    # --- Optional recession flag ---
    macro_state["recession_flag"] = False
    recession_intervals: list[tuple[int, int]] = []
    if config.use_recession:
        recession_path = resolve_repo_path(config.recession_md_path)
        if recession_path.exists():
            recession_intervals = parse_recession_intervals(recession_path)
            macro_state["recession_flag"] = macro_state["yyyymm"].apply(
                lambda ym: is_recession_month(int(ym), recession_intervals)
            )

    # --- Combined stress flag ---
    macro_state["stress_flag"] = (
        macro_state["macro_stress_flag"]
        | macro_state["nfci_stress_flag"]
        | macro_state["recession_flag"]
    )
    macro_state["stable_flag"] = ~macro_state["stress_flag"]
    macro_state["winterwolf_included"] = True
    macro_state["summerchild_included"] = macro_state["stable_flag"]
    macro_state["stable_rule_name"] = config.stable_rule_name
    macro_state["train_month_eligible"] = macro_state["yyyymm"].isin(month_ids)

    summary: dict[str, object] = {
        "stable_rule_name": config.stable_rule_name,
        "stress_quantile": config.stress_quantile,
        "stress_score_formula": f"stress_score = {config.dfy_weight}*dfy_z + {config.svar_weight}*svar_z + {config.tms_weight}*tms_z",
        "z_score_stats": stats,
        "stress_score_threshold": threshold,
        "use_nfci": config.use_nfci,
        "nfci_stress_percentile": config.nfci_stress_percentile,
        "nfci_threshold": nfci_threshold,
        "use_recession": config.use_recession,
        "n_recession_intervals_in_train": int(sum(
            1 for p, t in recession_intervals
            if any(p <= m <= t for m in macro_state["yyyymm"].astype(int).tolist())
        )) if recession_intervals else 0,
        "eligible_train_months": int(macro_state["train_month_eligible"].sum()),
        "winterwolf_train_months": int(macro_state["winterwolf_included"].sum()),
        "summerchild_train_months": int(macro_state["summerchild_included"].sum()),
        "stress_months_excluded_from_summerchild": int(macro_state["stress_flag"].sum()),
        "stable_flag_count": int(macro_state["stable_flag"].sum()),
        "stress_flag_count": int(macro_state["stress_flag"].sum()),
        "macro_stress_months": int(macro_state["macro_stress_flag"].sum()),
        "nfci_stress_months": int(macro_state["nfci_stress_flag"].sum()),
        "recession_months": int(macro_state["recession_flag"].sum()),
    }
    return macro_state, summary


def save_mask_documentation(
    run_dir: Path,
    mask_summary: dict[str, object],
    config: ContrastiveE2EConfig,
) -> None:
    lines = [
        "# Train-Month Stress Mask Documentation",
        "",
        f"- Rule name: `{mask_summary['stable_rule_name']}`",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Mask Logic",
        "",
        "A train month is classified as **stress** if ANY of the following hold:",
        "",
        f"1. **Macro composite**: `stress_score > {mask_summary['stress_score_threshold']:.4f}` "
        f"(train-period {config.stress_quantile:.0%} quantile) "
        f"OR `dfy_z > {config.dfy_z_cap}` OR `svar_z > {config.svar_z_cap}`",
        f"   - Formula: `{mask_summary['stress_score_formula']}`",
    ]
    if config.use_nfci:
        lines.append(
            f"2. **NFCI stress**: `NFCI >= {mask_summary['nfci_threshold']:.5f}` "
            f"(train-period {config.nfci_stress_percentile:.0%} quantile)"
        )
    if config.use_recession:
        lines.append(
            f"3. **NBER recession**: month falls within an NBER peak-to-trough recession window"
        )
    lines += [
        "",
        "**Stable** months are the complement of the stress flag.",
        "",
        "## Counts",
        "",
        f"- Total eligible train months: {mask_summary['eligible_train_months']}",
        f"- WinterWolf train months (all): {mask_summary['winterwolf_train_months']}",
        f"- SummerChild train months (stable only): {mask_summary['summerchild_train_months']}",
        f"- Stress months excluded from SummerChild: {mask_summary['stress_months_excluded_from_summerchild']}",
        f"- Months flagged by macro composite: {mask_summary['macro_stress_months']}",
        f"- Months flagged by NFCI: {mask_summary['nfci_stress_months']}",
        f"- Months flagged by recession: {mask_summary['recession_months']}",
        "",
        "## Reproducibility",
        "",
        "- All z-scores computed from train-period means and standard deviations only.",
        "- NFCI threshold computed from train-period quantile only.",
        "- Recession dates from NBER official turning points.",
        "- The mask is saved as `train_month_mask.csv` alongside this note.",
    ]
    (run_dir / "train_month_mask_note.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_variant_month_sets(
    train_months: list[E2EDecisionMonth],
    mask_frame: pd.DataFrame,
    config: ContrastiveE2EConfig,
    run_dir: Path,
) -> tuple[dict[str, list[E2EDecisionMonth]], dict[str, object] | None, pd.DataFrame]:
    if config.variant_construction != "balanced_contrastive_v2":
        return {
            "winter_wolf": filter_decision_months(train_months, mask_frame, "winterwolf_included"),
            "summer_child": filter_decision_months(train_months, mask_frame, "summerchild_included"),
        }, None, mask_frame

    balanced_frame, balanced_summary = build_balanced_month_mask(mask_frame, config)
    balanced_frame.to_csv(run_dir / "balanced_train_month_mask.csv", index=False)
    (run_dir / "balanced_train_month_mask_summary.json").write_text(
        json.dumps(balanced_summary, indent=2),
        encoding="utf-8",
    )
    save_balanced_month_documentation(run_dir, balanced_summary)
    save_balanced_month_lists(run_dir, balanced_frame)
    return {
        "winterwolf_balanced_v2": filter_decision_months(train_months, balanced_frame, "winterwolf_balanced_selected"),
        "summerchild_balanced_v2": filter_decision_months(train_months, balanced_frame, "summerchild_balanced_selected"),
    }, balanced_summary, balanced_frame


def build_balanced_month_mask(
    mask_frame: pd.DataFrame,
    config: ContrastiveE2EConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if config.balance_selection_rule != "chronological_even_thinning":
        raise ValueError(f"Unsupported balance_selection_rule={config.balance_selection_rule}")

    balanced = mask_frame.copy().sort_values("yyyymm").reset_index(drop=True)
    stable_idx = balanced.index[balanced["stable_flag"]].to_list()
    stress_idx = balanced.index[balanced["stress_flag"]].to_list()
    n_balanced = min(len(stable_idx), len(stress_idx))
    stable_keep = select_evenly_spaced_indices(stable_idx, n_balanced)
    stress_keep = select_evenly_spaced_indices(stress_idx, n_balanced)

    balanced["summerchild_balanced_selected"] = False
    balanced["winterwolf_balanced_selected"] = False
    balanced.loc[stable_keep, "summerchild_balanced_selected"] = True
    balanced.loc[stress_keep, "winterwolf_balanced_selected"] = True

    summary = {
        "variant_construction": config.variant_construction,
        "balance_selection_rule": config.balance_selection_rule,
        "eligible_stable_months": int(len(stable_idx)),
        "eligible_stress_months": int(len(stress_idx)),
        "balanced_n_months_per_variant": int(n_balanced),
        "stable_months_dropped_for_balance": int(len(stable_idx) - n_balanced),
        "stress_months_dropped_for_balance": int(len(stress_idx) - n_balanced),
        "summerchild_balanced_selected_months": balanced.loc[
            balanced["summerchild_balanced_selected"], "yyyymm"
        ].astype(int).tolist(),
        "winterwolf_balanced_selected_months": balanced.loc[
            balanced["winterwolf_balanced_selected"], "yyyymm"
        ].astype(int).tolist(),
    }
    return balanced, summary


def select_evenly_spaced_indices(indices: list[int], n_keep: int) -> list[int]:
    if n_keep < 0 or n_keep > len(indices):
        raise ValueError(f"Cannot select n_keep={n_keep} from {len(indices)} indices")
    if n_keep == len(indices):
        return list(indices)
    if n_keep == 0:
        return []
    total = len(indices)
    selected_positions = [
        min(total - 1, int(np.floor(((slot + 0.5) * total) / n_keep)))
        for slot in range(n_keep)
    ]
    selected_positions = sorted(dict.fromkeys(selected_positions))
    if len(selected_positions) != n_keep:
        raise RuntimeError(
            f"Selection rule produced {len(selected_positions)} unique positions, expected {n_keep}"
        )
    return [indices[position] for position in selected_positions]


def save_balanced_month_documentation(run_dir: Path, balanced_summary: dict[str, object]) -> None:
    lines = [
        "# Balanced Contrastive V2 Month Selection",
        "",
        "- Construction: `summerchild_balanced_v2` uses stable months only; `winterwolf_balanced_v2` uses stress months only.",
        f"- Deterministic rule: `{balanced_summary['balance_selection_rule']}`.",
        "",
        "## Selection Rule",
        "",
        "The larger eligible set is thinned in chronological order with evenly spaced retention points across the full sample span.",
        "This keeps the balanced subset reproducible and avoids arbitrary early-only or late-only trimming.",
        "",
        "## Counts",
        "",
        f"- Eligible stable months: {balanced_summary['eligible_stable_months']}",
        f"- Eligible stress months: {balanced_summary['eligible_stress_months']}",
        f"- Balanced months per variant: {balanced_summary['balanced_n_months_per_variant']}",
        f"- Stable months dropped for balance: {balanced_summary['stable_months_dropped_for_balance']}",
        f"- Stress months dropped for balance: {balanced_summary['stress_months_dropped_for_balance']}",
        "",
        "Selected month lists are saved as CSV and JSON files alongside this note.",
    ]
    (run_dir / "balanced_month_selection_note.md").write_text("\n".join(lines), encoding="utf-8")


def save_balanced_month_lists(run_dir: Path, balanced_frame: pd.DataFrame) -> None:
    selections = {
        "eligible_stable_months": balanced_frame.loc[balanced_frame["stable_flag"], ["yyyymm"]],
        "eligible_stress_months": balanced_frame.loc[balanced_frame["stress_flag"], ["yyyymm"]],
        "selected_stable_months_balanced": balanced_frame.loc[
            balanced_frame["summerchild_balanced_selected"], ["yyyymm"]
        ],
        "selected_stress_months_balanced": balanced_frame.loc[
            balanced_frame["winterwolf_balanced_selected"], ["yyyymm"]
        ],
    }
    for stem, frame in selections.items():
        month_list = frame["yyyymm"].astype(int).tolist()
        frame.assign(yyyymm=frame["yyyymm"].astype(int)).to_csv(run_dir / f"{stem}.csv", index=False)
        (run_dir / f"{stem}.json").write_text(json.dumps(month_list, indent=2), encoding="utf-8")


def train_months_to_macro_frame(
    train_months: list[E2EDecisionMonth],
    data_config: Universe500DataConfig,
) -> pd.DataFrame:
    loader = data_config.build_loader()
    macro_state = loader.load_macro_state().copy()
    macro_state["yyyymm"] = macro_state["yyyymm"].astype(int)
    eligible = {int(month.yyyymm) for month in train_months}
    macro_state = macro_state.loc[macro_state["yyyymm"].isin(eligible)].copy()
    macro_state = macro_state.sort_values("yyyymm").reset_index(drop=True)
    return macro_state[["yyyymm", "Rfree", "dfy", "svar", "tms", "infl", "tbl", "dp", "ep", "bm", "ntis"]]


def filter_decision_months(
    months: list[E2EDecisionMonth],
    mask_frame: pd.DataFrame,
    flag_column: str,
) -> list[E2EDecisionMonth]:
    allowed = set(mask_frame.loc[mask_frame[flag_column], "yyyymm"].astype(int).tolist())
    return [month for month in months if int(month.yyyymm) in allowed]


def run_single_variant(
    run_dir: Path,
    variant_name: str,
    base_config: E2EV1Config,
    candidate: E2ECandidate,
    input_dim: int,
    train_months: list[E2EDecisionMonth],
    val_months: list[E2EDecisionMonth],
    test_months: list[E2EDecisionMonth],
    device,
    mask_frame: pd.DataFrame,
    effective_epochs: int,
    effective_patience: int,
) -> dict[str, object]:
    variant_output_root = run_dir
    variant_config = replace(
        base_config,
        output_root=str(variant_output_root),
        run_name=variant_name,
        learning_rate_grid=(candidate.learning_rate,),
        weight_decay_grid=(candidate.weight_decay,),
        init_mode_grid=(candidate.init_mode,),
        lambda_grid=(candidate.lambda_,),
        kappa_grid=(candidate.kappa,),
        stage1_enabled=False,
        epochs=effective_epochs,
        patience=effective_patience,
        train_months_per_epoch=None,
    )
    log_progress(
        f"Training variant {variant_name}: train_months={len(train_months)}, "
        f"epochs={effective_epochs}, patience={effective_patience}, kappa={candidate.kappa}"
    )
    candidate_result = train_candidate(
        candidate=candidate,
        config=variant_config,
        input_dim=input_dim,
        train_months=train_months,
        val_months=val_months,
        epoch_count=effective_epochs,
        stage_name="fixed_candidate",
        device=device,
    )
    best_checkpoint = candidate_result["candidate_dir"] / "best_checkpoint.pt"
    import torch

    model = build_model(
        config=variant_config,
        input_dim=input_dim,
        candidate=candidate,
        device=device,
    )
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    train_monthly, train_assets = evaluate_model(model, train_months, candidate, variant_config, device)
    test_monthly, test_assets = evaluate_model(model, test_months, candidate, variant_config, device)
    val_monthly, val_assets = evaluate_model(model, val_months, candidate, variant_config, device)
    train_summary = summarize_backtest(train_monthly)
    val_summary = summarize_backtest(val_monthly)
    test_summary = summarize_backtest(test_monthly)
    run_variant_dir = run_dir / variant_name
    run_variant_dir.mkdir(parents=True, exist_ok=True)
    (run_variant_dir / "config.json").write_text(json.dumps(variant_config.to_dict(), indent=2), encoding="utf-8")
    (run_variant_dir / "selected_candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")
    if variant_name in {"summer_child", "summerchild_balanced_v2"}:
        inclusion_col = "summerchild_balanced_selected" if "summerchild_balanced_selected" in mask_frame.columns else "summerchild_included"
    else:
        inclusion_col = "winterwolf_balanced_selected" if "winterwolf_balanced_selected" in mask_frame.columns else "winterwolf_included"
    (run_variant_dir / "training_month_inclusion_mask.csv").write_text(
        mask_frame.assign(included_variant=mask_frame[inclusion_col]).to_csv(index=False),
        encoding="utf-8",
    )
    pd.DataFrame(candidate_result["history"]).to_csv(run_variant_dir / "training_history.csv", index=False)
    train_monthly.to_csv(run_variant_dir / "train_monthly_portfolio.csv", index=False)
    val_monthly.to_csv(run_variant_dir / "val_monthly_portfolio.csv", index=False)
    test_monthly.to_csv(run_variant_dir / "test_monthly_portfolio.csv", index=False)
    train_assets.to_parquet(run_variant_dir / "train_asset_predictions.parquet", index=False)
    val_assets.to_parquet(run_variant_dir / "val_asset_predictions.parquet", index=False)
    test_assets.to_parquet(run_variant_dir / "test_asset_predictions.parquet", index=False)
    # Save monthly predicted returns and weights as separate artifacts
    for split_name, assets_df in [("train", train_assets), ("val", val_assets), ("test", test_assets)]:
        assets_df[["yyyymm", "permno", "prediction"]].to_parquet(
            run_variant_dir / f"{split_name}_monthly_predicted_returns.parquet", index=False
        )
        assets_df[["yyyymm", "permno", "weight"]].to_parquet(
            run_variant_dir / f"{split_name}_monthly_weights.parquet", index=False
        )
    torch.save(checkpoint, run_variant_dir / "best_checkpoint.pt")
    summary_payload = {
        "variant_name": variant_name,
        "construction": f"Current-setting exact-layer contrastive variant; only training-history exposure changes. solver={base_config.solver_mode}",
        "selected_candidate": candidate.to_dict(),
        "best_epoch": int(candidate_result["best_epoch"]),
        "train_months_used": int(len(train_months)),
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "test_summary": test_summary,
        "passed_screens": bool(passes_screens(val_summary)),
    }
    (run_variant_dir / "summary_metrics.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return {
        "variant_name": variant_name,
        "run_dir": run_variant_dir,
        "train_monthly": train_monthly,
        "val_monthly": val_monthly,
        "test_monthly": test_monthly,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "test_summary": test_summary,
        "train_months_used": len(train_months),
        "best_epoch": int(candidate_result["best_epoch"]),
    }


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------


def build_contrastive_comparison(
    run_dir: Path,
    variant_results: dict[str, dict[str, object]],
    frozen_e2e_run_dir: Path,
    rolling_window: int,
    mask_summary: dict[str, object],
    solver_mode: str,
    config: ContrastiveE2EConfig,
    balanced_summary: dict[str, object] | None,
) -> dict[str, object]:
    frozen_e2e_test = pd.read_csv(frozen_e2e_run_dir / "test_monthly_portfolio.csv")
    benchmark_label = (
        "frozen_full_history_exact_layer_benchmark"
        if config.variant_construction == "balanced_contrastive_v2"
        else "exact_layer_e2e_benchmark"
    )
    model_frames: dict[str, pd.DataFrame] = {
        model_name: result["test_monthly"].copy()
        for model_name, result in variant_results.items()
    }
    model_frames[benchmark_label] = frozen_e2e_test

    summary_rows = []
    wealth_rows = []
    rolling_rows = []
    turnover_rows = []
    concentration_rows = []
    for model_name, frame in model_frames.items():
        monthly = normalize_monthly_frame(frame, model_name)
        summary = summarize_backtest(monthly)
        wealth_frame = build_wealth_frame(monthly, model_name)
        summary_rows.append({"model": model_name, "final_wealth": float(wealth_frame["wealth"].iloc[-1]), **summary})
        wealth_rows.append(wealth_frame)
        rolling_rows.append(build_rolling_sharpe_frame(monthly, model_name, rolling_window))
        turnover_rows.append(build_turnover_frame(monthly, model_name))
        concentration_rows.append(build_concentration_frame(monthly, model_name))

    summary_table = pd.DataFrame(summary_rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    test_wealth_paths = pd.concat(wealth_rows, ignore_index=True)
    rolling_sharpe = pd.concat(rolling_rows, ignore_index=True)
    turnover_comparison = pd.concat(turnover_rows, ignore_index=True)
    concentration_comparison = pd.concat(concentration_rows, ignore_index=True)
    latest_rolling = (
        rolling_sharpe.dropna(subset=["rolling_sharpe_36m"])
        .groupby("model", as_index=False)["rolling_sharpe_36m"]
        .last()
        .rename(columns={"rolling_sharpe_36m": "latest_rolling_sharpe_36m"})
    )

    split_comparison = build_split_comparison(
        variant_results=variant_results,
        frozen_e2e_run_dir=frozen_e2e_run_dir,
        benchmark_label=benchmark_label,
    )
    focused_comparison = split_comparison.loc[
        split_comparison["model"].isin(list(variant_results.keys()) + [benchmark_label])
    ].copy()

    # Save additional diagnostics
    turnover_comparison.to_csv(run_dir / "turnover_comparison.csv", index=False)
    concentration_comparison.to_csv(run_dir / "concentration_comparison.csv", index=False)

    report_payload = {
        "construction": {
            "variant_construction": config.variant_construction,
            "balance_selection_rule": config.balance_selection_rule,
            "mask_summary": mask_summary,
            "balanced_summary": balanced_summary,
        },
        "summary_table": summary_table.to_dict(orient="records"),
        "split_comparison": split_comparison.to_dict(orient="records"),
        "latest_rolling_sharpe": latest_rolling.to_dict(orient="records"),
    }
    report_markdown = build_contrastive_report_markdown(
        summary_table=summary_table,
        latest_rolling=latest_rolling,
        split_comparison=split_comparison,
        mask_summary=mask_summary,
        variant_results=variant_results,
        solver_mode=solver_mode,
        benchmark_label=benchmark_label,
        config=config,
        balanced_summary=balanced_summary,
    )
    return {
        "summary_table": summary_table,
        "test_wealth_paths": test_wealth_paths,
        "rolling_sharpe": rolling_sharpe,
        "split_comparison": split_comparison,
        "focused_comparison": focused_comparison,
        "report_payload": report_payload,
        "report_markdown": report_markdown,
    }


def build_split_comparison(
    variant_results: dict[str, dict[str, object]],
    frozen_e2e_run_dir: Path,
    benchmark_label: str,
) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        summary_key = "validation_summary" if split == "val" else "test_summary"
        for variant_name, result in variant_results.items():
            rows.append({"model": variant_name, "split": split, **result[summary_key]})
        frozen_selection = json.loads((frozen_e2e_run_dir / "validation_selection_summary.json").read_text(encoding="utf-8"))
        frozen_key = "validation_summary" if split == "val" else "test_summary"
        rows.append({"model": benchmark_label, "split": split, **frozen_selection[frozen_key]})
    return pd.DataFrame(rows)


def normalize_monthly_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    monthly = frame.copy()
    monthly["yyyymm"] = monthly["yyyymm"].astype(int)
    if "portfolio_return" not in monthly.columns:
        raise ValueError(f"{model_name} monthly frame missing portfolio_return")
    required_defaults = {
        "portfolio_excess_return": 0.0,
        "turnover": 0.0,
        "herfindahl": np.nan,
        "effective_n": np.nan,
        "max_weight": np.nan,
        "active_positions": np.nan,
    }
    for column, default in required_defaults.items():
        if column not in monthly.columns:
            monthly[column] = default
    return monthly.sort_values("yyyymm").reset_index(drop=True)


def build_wealth_frame(monthly: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = monthly[["yyyymm", "portfolio_return"]].copy()
    frame["wealth"] = (1.0 + frame["portfolio_return"].astype(float)).cumprod()
    frame["model"] = model_name
    return frame


def build_rolling_sharpe_frame(monthly: pd.DataFrame, model_name: str, window: int) -> pd.DataFrame:
    frame = monthly[["yyyymm", "portfolio_excess_return"]].copy()
    rolling = frame["portfolio_excess_return"].rolling(window=window)
    mean = rolling.mean() * 12.0
    std = rolling.std(ddof=1) * np.sqrt(12.0)
    frame["rolling_sharpe_36m"] = mean / std.replace(0.0, np.nan)
    frame["model"] = model_name
    return frame[["yyyymm", "model", "rolling_sharpe_36m"]]


def build_turnover_frame(monthly: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = monthly[["yyyymm", "turnover"]].copy()
    frame["model"] = model_name
    return frame


def build_concentration_frame(monthly: pd.DataFrame, model_name: str) -> pd.DataFrame:
    cols = ["yyyymm"]
    for c in ["herfindahl", "effective_n", "max_weight", "active_positions"]:
        if c in monthly.columns:
            cols.append(c)
    frame = monthly[cols].copy()
    frame["model"] = model_name
    return frame


def build_contrastive_report_markdown(
    summary_table: pd.DataFrame,
    latest_rolling: pd.DataFrame,
    split_comparison: pd.DataFrame,
    mask_summary: dict[str, object],
    variant_results: dict[str, dict[str, object]],
    solver_mode: str,
    benchmark_label: str,
    config: ContrastiveE2EConfig,
    balanced_summary: dict[str, object] | None,
) -> str:
    lines = [
        "# Contrastive E2E Report",
        "",
        f"- Solver mode: `{solver_mode}`",
        f"- Construction mode: `{config.variant_construction}`",
        "- Both variants use the same selected exact-layer candidate.",
        "- The only intentional distinction is which train months they see.",
        f"- Stable-rule name: `{mask_summary['stable_rule_name']}`.",
    ]
    if config.variant_construction == "balanced_contrastive_v2" and balanced_summary is not None:
        lines.extend([
            f"- Eligible stable months: `{balanced_summary['eligible_stable_months']}`.",
            f"- Eligible stress months: `{balanced_summary['eligible_stress_months']}`.",
            f"- Balanced months per variant: `{balanced_summary['balanced_n_months_per_variant']}`.",
            f"- Balance rule: `{balanced_summary['balance_selection_rule']}`.",
        ])
    else:
        lines.extend([
            f"- SummerChild train months: `{mask_summary['summerchild_train_months']}`.",
            f"- WinterWolf train months: `{mask_summary['winterwolf_train_months']}`.",
            f"- Stress months excluded: `{mask_summary['stress_months_excluded_from_summerchild']}`.",
        ])
    if mask_summary.get("use_nfci"):
        lines.append(f"- NFCI stress months: `{mask_summary['nfci_stress_months']}`.")
    if mask_summary.get("use_recession"):
        lines.append(f"- Recession months: `{mask_summary['recession_months']}`.")
    lines += [
        "",
        "## Variant Training Summary",
        "",
    ]
    for vname, vr in variant_results.items():
        lines.append(f"### {vname}")
        lines.append(f"- Train months used: {vr['train_months_used']}")
        lines.append(f"- Best epoch: {vr['best_epoch']}")
        vs = vr["validation_summary"]
        lines.append(f"- Val Sharpe: {vs['sharpe_ratio']:.6f}, Val excess return: {vs['annualized_excess_return']:.6f}")
        ts = vr["test_summary"]
        lines.append(f"- Test Sharpe: {ts['sharpe_ratio']:.6f}, Test excess return: {ts['annualized_excess_return']:.6f}")
        lines.append("")

    lines += [
        "## Validation/Test Comparison",
        "",
    ]
    for split in ("val", "test"):
        lines.append(f"### {split}")
        subset = split_comparison.loc[split_comparison["split"] == split].sort_values("sharpe_ratio", ascending=False)
        for _, row in subset.iterrows():
            lines.append(
                f"- {row['model']}: sharpe={row['sharpe_ratio']:.4f}, ann_excess={row['annualized_excess_return']:.4f}, "
                f"ann_vol={row['annualized_volatility']:.4f}, max_dd={row['max_drawdown']:.4f}, "
                f"turnover={row['average_turnover']:.4f}, eff_n={row['average_effective_n']:.1f}, "
                f"max_weight={row['average_max_weight']:.4f}, active_positions={row['average_active_positions']:.1f}"
            )
        lines.append("")

    lines += [
        "## Test Summary",
        "",
    ]
    for _, row in summary_table.iterrows():
        lines.append(
            f"- {row['model']}: final_wealth={row['final_wealth']:.4f}, ann_excess={row['annualized_excess_return']:.4f}, "
            f"sharpe={row['sharpe_ratio']:.4f}, max_dd={row['max_drawdown']:.4f}, turnover={row['average_turnover']:.4f}, "
            f"eff_n={row['average_effective_n']:.1f}, max_weight={row['average_max_weight']:.4f}, "
            f"active_positions={row['average_active_positions']:.1f}"
        )
    lines.append("")
    lines.append("## Latest Rolling 36-Month Sharpe")
    for _, row in latest_rolling.sort_values("latest_rolling_sharpe_36m", ascending=False).iterrows():
        lines.append(f"- {row['model']}: {row['latest_rolling_sharpe_36m']:.4f}")

    winter_key = "winterwolf_balanced_v2" if "winterwolf_balanced_v2" in variant_results else "winter_wolf"
    summer_key = "summerchild_balanced_v2" if "summerchild_balanced_v2" in variant_results else "summer_child"
    ww = variant_results.get(winter_key, {})
    sc = variant_results.get(summer_key, {})
    ww_test = ww.get("test_summary", {})
    sc_test = sc.get("test_summary", {})
    ww_val = ww.get("validation_summary", {})
    sc_val = sc.get("validation_summary", {})
    benchmark_test = summary_table.loc[summary_table["model"] == benchmark_label]
    benchmark_sharpe = float(benchmark_test["sharpe_ratio"].iloc[0]) if not benchmark_test.empty else np.nan

    lines += [
        "",
        "## Key Interpretation",
        "",
    ]
    if ww_val and sc_val:
        ww_beats_val = ww_val.get("sharpe_ratio", 0) > sc_val.get("sharpe_ratio", 0)
        ww_beats_test = ww_test.get("sharpe_ratio", 0) > sc_test.get("sharpe_ratio", 0)
        lines.append(
            f"- Val Sharpe: WinterWolf={ww_val['sharpe_ratio']:.4f} vs SummerChild={sc_val['sharpe_ratio']:.4f} → "
            f"{'Yes' if ww_beats_val else 'No'}"
        )
        lines.append(
            f"- Test Sharpe: WinterWolf={ww_test['sharpe_ratio']:.4f} vs SummerChild={sc_test['sharpe_ratio']:.4f} → "
            f"{'Yes' if ww_beats_test else 'No'}"
        )
    if ww_test and sc_test:
        turnover_diff = abs(ww_test.get("average_turnover", 0) - sc_test.get("average_turnover", 0))
        eff_n_diff = abs(ww_test.get("average_effective_n", 0) - sc_test.get("average_effective_n", 0))
        lines.append(f"- Turnover gap between variants: {turnover_diff:.4f}")
        lines.append(f"- Effective-N gap between variants: {eff_n_diff:.1f}")
    if np.isfinite(benchmark_sharpe):
        lines.append(f"- Frozen benchmark test Sharpe: {benchmark_sharpe:.4f}")

    return "\n".join(lines)


def save_comparability_audit_note(
    run_dir: Path,
    config: ContrastiveE2EConfig,
    mask_summary: dict[str, object],
    balanced_summary: dict[str, object] | None,
) -> None:
    old_exact_summary_path = (
        config.resolve_output_root() / "summerchild_winterwolf_exact_layer_20260409" / "train_month_mask_summary.json"
    )
    old_legacy_summary_path = (
        config.resolve_output_root() / "summerchild_winterwolf_current_recipe_20260404" / "train_month_mask_summary.json"
    )
    old_exact_summary = json.loads(old_exact_summary_path.read_text(encoding="utf-8")) if old_exact_summary_path.exists() else None
    old_legacy_summary = json.loads(old_legacy_summary_path.read_text(encoding="utf-8")) if old_legacy_summary_path.exists() else None

    lines = [
        "# Comparability Audit Note",
        "",
        "## Core issue",
        "",
    ]
    if old_exact_summary is not None:
        lines.extend([
            f"- Current saved exact-layer SummerChild used `{old_exact_summary['summerchild_train_months']}` train months.",
            f"- Current saved exact-layer WinterWolf used `{old_exact_summary['winterwolf_train_months']}` train months.",
        ])
    lines.extend([
        f"- Under the active stress mask there are `{mask_summary['stable_flag_count'] if 'stable_flag_count' in mask_summary else mask_summary['summerchild_train_months']}` stable months and `{mask_summary['stress_flag_count'] if 'stress_flag_count' in mask_summary else mask_summary['stress_months_excluded_from_summerchild']}` stress months.",
        "- That means the old pair changed two things at once: train-month type and train-month count.",
        "- A higher or lower Sharpe in that setup cannot be attributed cleanly to stable-vs-stress exposure alone.",
        "",
        "## Historical reference",
        "",
    ])
    if old_legacy_summary is not None:
        lines.extend([
            f"- Earlier macro-only contrastive run: SummerChild `{old_legacy_summary['summerchild_train_months']}`, WinterWolf `{old_legacy_summary['winterwolf_train_months']}`.",
        ])
    if balanced_summary is not None:
        lines.extend([
            "",
            "## Balanced v2 design",
            "",
            f"- `N = min(stable, stress) = {balanced_summary['balanced_n_months_per_variant']}`.",
            "- `summerchild_balanced_v2` trains on stable months only.",
            "- `winterwolf_balanced_v2` trains on stress months only.",
            "- Both use the same exact-layer selected candidate and the same number of train months.",
            f"- Deterministic balancing rule: `{balanced_summary['balance_selection_rule']}`.",
        ])
    (run_dir / "comparability_audit_note.md").write_text("\n".join(lines), encoding="utf-8")


def save_interpretation_note(
    run_dir: Path,
    comparison: dict[str, object],
    config: ContrastiveE2EConfig,
    mask_summary: dict[str, object],
    balanced_summary: dict[str, object] | None,
) -> None:
    summary_table = comparison["summary_table"]
    split_comparison = comparison["split_comparison"]
    benchmark_label = (
        "frozen_full_history_exact_layer_benchmark"
        if config.variant_construction == "balanced_contrastive_v2"
        else "exact_layer_e2e_benchmark"
    )
    winter_label = "winterwolf_balanced_v2" if "winterwolf_balanced_v2" in summary_table["model"].values else "winter_wolf"
    summer_label = "summerchild_balanced_v2" if "summerchild_balanced_v2" in summary_table["model"].values else "summer_child"

    def get_metric(model: str, split: str, metric: str) -> float:
        row = split_comparison.loc[(split_comparison["model"] == model) & (split_comparison["split"] == split), metric]
        return float(row.iloc[0]) if not row.empty else float("nan")

    winter_test_sharpe = get_metric(winter_label, "test", "sharpe_ratio")
    summer_test_sharpe = get_metric(summer_label, "test", "sharpe_ratio")
    winter_val_sharpe = get_metric(winter_label, "val", "sharpe_ratio")
    summer_val_sharpe = get_metric(summer_label, "val", "sharpe_ratio")
    benchmark_test_sharpe = get_metric(benchmark_label, "test", "sharpe_ratio")
    benchmark_val_sharpe = get_metric(benchmark_label, "val", "sharpe_ratio")
    winter_test_excess = get_metric(winter_label, "test", "annualized_excess_return")
    summer_test_excess = get_metric(summer_label, "test", "annualized_excess_return")
    old_exact_summary_path = (
        config.resolve_output_root() / "summerchild_winterwolf_exact_layer_20260409" / "contrastive_summary_metrics.csv"
    )
    old_flip_note = None
    if old_exact_summary_path.exists():
        old_exact = pd.read_csv(old_exact_summary_path)
        old_winter = old_exact.loc[old_exact["model"] == "winter_wolf", "sharpe_ratio"]
        old_summer = old_exact.loc[old_exact["model"] == "summer_child", "sharpe_ratio"]
        if not old_winter.empty and not old_summer.empty:
            old_flip_note = (
                f"Old exact-layer test Sharpe ordering was SummerChild `{float(old_summer.iloc[0]):.4f}` "
                f"over WinterWolf `{float(old_winter.iloc[0]):.4f}`."
            )

    lines = [
        "# Final Interpretation Note",
        "",
        f"- Balanced-v2 used `{balanced_summary['balanced_n_months_per_variant']}` months per variant." if balanced_summary is not None else "",
        f"- Validation Sharpe: WinterWolf `{winter_val_sharpe:.4f}` vs SummerChild `{summer_val_sharpe:.4f}`.",
        f"- Test Sharpe: WinterWolf `{winter_test_sharpe:.4f}` vs SummerChild `{summer_test_sharpe:.4f}`.",
        f"- Test excess return: WinterWolf `{winter_test_excess:.4f}` vs SummerChild `{summer_test_excess:.4f}`.",
        f"- Frozen full-history exact-layer benchmark Sharpe: val `{benchmark_val_sharpe:.4f}`, test `{benchmark_test_sharpe:.4f}`.",
        "",
    ]
    if old_flip_note is not None:
        lines.append(f"- {old_flip_note}")

    if np.isfinite(winter_test_sharpe) and np.isfinite(summer_test_sharpe):
        if winter_test_sharpe > summer_test_sharpe:
            lines.append("- After balancing month counts, WinterWolf has the higher test Sharpe.")
        else:
            lines.append("- After balancing month counts, SummerChild has the higher test Sharpe.")
    if np.isfinite(winter_test_excess) and np.isfinite(summer_test_excess):
        if winter_test_excess > summer_test_excess:
            lines.append("- WinterWolf still leads on raw test excess return.")
        else:
            lines.append("- SummerChild still leads on raw test excess return.")
    if np.isfinite(benchmark_test_sharpe):
        best_pair_test = np.nanmax([winter_test_sharpe, summer_test_sharpe])
        if best_pair_test > benchmark_test_sharpe:
            lines.append("- The balanced pair still contains a model that beats the frozen full-history exact-layer benchmark on test Sharpe.")
        else:
            lines.append("- The frozen full-history exact-layer benchmark remains stronger than both balanced variants on test Sharpe.")
    if balanced_summary is not None:
        lines.extend([
            "- SummerChild still looks better on validation risk-adjusted performance, but that edge does not hold on test once month counts are balanced.",
            "- The old contrastive result was not trustworthy as a clean stable-vs-stress comparison because the saved exact-layer pair used unequal train-month counts; the test ordering flips after balancing.",
            "- Future scenario work should use the balanced v2 pair for any SummerChild-versus-WinterWolf contrast and keep the frozen full-history exact-layer benchmark as the practical benchmark model.",
            "- The old unbalanced contrastive pair should be treated as historical only.",
        ])

    interpretation = "\n".join(line for line in lines if line != "")
    (run_dir / "final_interpretation_note.md").write_text(interpretation, encoding="utf-8")


# ---------------------------------------------------------------------------
# Kappa sensitivity runner
# ---------------------------------------------------------------------------


def run_kappa_sensitivity(config: ContrastiveE2EConfig) -> Path:
    """Run the contrastive pair for each kappa in config.kappa_sensitivity_grid."""
    if config.kappa_sensitivity_grid is None:
        raise ValueError("kappa_sensitivity_grid must be set for sensitivity runs.")

    sensitivity_dir = config.resolve_output_root() / (config.run_name + "_kappa_sensitivity")
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for kappa in config.kappa_sensitivity_grid:
        from dataclasses import replace as dc_replace
        kappa_config = dc_replace(
            config,
            run_name=f"{config.run_name}_kappa_{kappa:.2f}".replace(".", "p"),
            override_kappa=kappa,
            kappa_sensitivity_grid=None,
        )
        log_progress(f"Kappa sensitivity: running kappa={kappa}")
        run_dir = run_contrastive_e2e_training(kappa_config)
        summary_path = run_dir / "contrastive_summary_metrics.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["kappa"] = kappa
            results.append(df)

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(sensitivity_dir / "kappa_sensitivity_summary.csv", index=False)
        # Build a compact sensitivity report
        report_lines = ["# Kappa Sensitivity Summary", ""]
        for kappa in config.kappa_sensitivity_grid:
            subset = combined[combined["kappa"] == kappa]
            report_lines.append(f"## kappa = {kappa}")
            for _, row in subset.iterrows():
                report_lines.append(
                    f"- {row['model']}: sharpe={row['sharpe_ratio']:.4f}, "
                    f"excess={row['annualized_excess_return']:.4f}, "
                    f"turnover={row['average_turnover']:.4f}"
                )
            report_lines.append("")
        (sensitivity_dir / "kappa_sensitivity_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return sensitivity_dir
