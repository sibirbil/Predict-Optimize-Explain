from __future__ import annotations

import copy
import itertools
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.configs.e2e import E2EV1Config
from src.data import TARGET_COLUMN
from src.modeling.e2e.data import E2EDecisionMonth, prepare_e2e_months
from src.modeling.e2e.model import E2ERegressor
from src.modeling.e2e.portfolio import robust_portfolio_weights, robust_preference
from src.modeling.pto.metrics import compute_turnover, summarize_backtest, weights_diagnostics
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class E2ECandidate:
    learning_rate: float
    weight_decay: float
    init_mode: str
    lambda_: float
    kappa: float
    omega_mode: str

    def label(self) -> str:
        return (
            f"lr={self.learning_rate:g}__wd={self.weight_decay:g}__init={self.init_mode}"
            f"__lambda={self.lambda_}__kappa={self.kappa}__omega={self.omega_mode}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "init_mode": self.init_mode,
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "omega_mode": self.omega_mode,
            "label": self.label(),
        }


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_e2e_training(config: E2EV1Config) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    month_data = prepare_e2e_months(config=config)
    input_dim = month_data["train"][0].X.shape[1]

    candidates = [
        E2ECandidate(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            init_mode=str(init_mode),
            lambda_=float(lambda_),
            kappa=float(kappa),
            omega_mode=str(config.omega_mode),
        )
        for learning_rate, weight_decay, init_mode, lambda_, kappa in itertools.product(
            config.learning_rate_grid,
            config.weight_decay_grid,
            config.init_mode_grid,
            config.lambda_grid,
            config.kappa_grid,
        )
    ]

    run_dir = config.resolve_output_root() / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    log_progress(
        "E2E run started "
        + f"run_name={config.run_name} solver_mode={config.solver_mode} device={device.type} "
        + f"train_months={len(month_data['train'])} val_months={len(month_data['val'])} test_months={len(month_data['test'])} "
        + f"candidates={len(candidates)} output_dir={run_dir}"
    )

    if config.stage1_enabled:
        stage1_val_months = select_stage1_val_months(month_data["val"], config)
        log_progress(
            f"Stage 1 starting with {len(candidates)} candidates on {len(stage1_val_months)} validation months for {config.stage1_epochs} epoch(s)"
        )
        stage1_table, _ = run_candidate_search(
            candidates=candidates,
            config=config,
            input_dim=input_dim,
            train_months=month_data["train"],
            val_months=stage1_val_months,
            stage_name="stage1",
            epoch_count=config.stage1_epochs,
            device=device,
        )
        finalists = select_stage2_finalists(stage1_table, config)
        log_progress(
            "Stage 1 complete "
            + f"finalists={len(finalists)} "
            + ", ".join(candidate.label() for candidate in finalists)
        )
        log_progress(
            f"Stage 2 starting with {len(finalists)} finalists on {len(month_data['val'])} validation months for {config.stage2_epochs} epoch(s)"
        )
        stage2_table, candidate_payloads = run_candidate_search(
            candidates=finalists,
            config=config,
            input_dim=input_dim,
            train_months=month_data["train"],
            val_months=month_data["val"],
            stage_name="stage2",
            epoch_count=config.stage2_epochs,
            device=device,
        )
        stage1_table.to_csv(run_dir / "tuning_table_stage1.csv", index=False)
        stage2_table.to_csv(run_dir / "tuning_table_stage2.csv", index=False)
        tuning_table = stage2_table
    else:
        tuning_table, candidate_payloads = run_candidate_search(
            candidates=candidates,
            config=config,
            input_dim=input_dim,
            train_months=month_data["train"],
            val_months=month_data["val"],
            stage_name="single_pass",
            epoch_count=config.epochs,
            device=device,
        )
    tuning_table.to_csv(run_dir / "tuning_table.csv", index=False)

    selected_row = tuning_table.iloc[0]
    selected_label = str(selected_row["label"])
    selected_result = candidate_payloads[selected_label]
    selected_candidate = selected_result["candidate"]
    log_progress(f"Selected candidate {selected_candidate.label()} with validation Sharpe={selected_result['validation_summary']['sharpe_ratio']:.6f}")
    selected_model = build_model(
        config=config,
        input_dim=input_dim,
        candidate=selected_candidate,
        device=device,
    )
    selected_checkpoint = torch.load(
        Path(selected_result["candidate_dir"]) / "best_checkpoint.pt",
        map_location="cpu",
    )
    selected_model.load_state_dict(selected_checkpoint["model_state_dict"])
    train_monthly, train_assets = evaluate_model(
        model=selected_model,
        decision_months=month_data["train"],
        candidate=selected_candidate,
        config=config,
        device=device,
    )
    test_monthly, test_assets = evaluate_model(
        model=selected_model,
        decision_months=month_data["test"],
        candidate=selected_candidate,
        config=config,
        device=device,
    )
    selected_result["train_monthly"] = train_monthly
    selected_result["train_assets"] = train_assets
    selected_result["train_summary"] = summarize_backtest(train_monthly)
    selected_result["test_monthly"] = test_monthly
    selected_result["test_assets"] = test_assets
    selected_result["test_summary"] = summarize_backtest(test_monthly)

    save_selected_artifacts(
        run_dir=run_dir,
        config=config,
        selected_result=selected_result,
        tuning_table=tuning_table,
        pto_baseline_config_path=config.resolve_pto_baseline_config_path(),
        previous_e2e_run_dir=config.resolve_previous_e2e_run_dir(),
    )
    log_progress(
        "E2E run finished "
        + f"run_name={config.run_name} "
        + f"val_sharpe={selected_result['validation_summary']['sharpe_ratio']:.6f} "
        + f"test_sharpe={selected_result['test_summary']['sharpe_ratio']:.6f}"
    )
    return run_dir


def run_candidate_search(
    candidates: list[E2ECandidate],
    config: E2EV1Config,
    input_dim: int,
    train_months: list[E2EDecisionMonth],
    val_months: list[E2EDecisionMonth],
    stage_name: str,
    epoch_count: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    tuning_rows: list[dict[str, object]] = []
    candidate_payloads: dict[str, dict[str, object]] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        log_progress(
            f"{stage_name}: candidate {candidate_index}/{len(candidates)} starting {candidate.label()}"
        )
        candidate_result = train_candidate(
            candidate=candidate,
            config=config,
            input_dim=input_dim,
            train_months=train_months,
            val_months=val_months,
            epoch_count=epoch_count,
            stage_name=stage_name,
            device=device,
        )
        log_progress(
            f"{stage_name}: candidate {candidate_index}/{len(candidates)} finished "
            + f"best_epoch={candidate_result['best_epoch']} "
            + f"val_sharpe={candidate_result['validation_summary']['sharpe_ratio']:.6f} "
            + f"passed_screens={candidate_result['passed_screens']}"
        )
        tuning_rows.append(
            {
                **candidate.to_dict(),
                **candidate_result["validation_summary"],
                "best_epoch": candidate_result["best_epoch"],
                "passed_screens": candidate_result["passed_screens"],
                "stage": stage_name,
                "candidate_dir": candidate_result["candidate_dir"].name,
            }
        )
        candidate_payloads[candidate.label()] = candidate_result
    tuning_table = pd.DataFrame(tuning_rows).sort_values(
        ["passed_screens", "sharpe_ratio", "annualized_volatility", "average_herfindahl", "average_turnover"],
        ascending=[False, False, True, True, True],
    )
    return tuning_table, candidate_payloads


def train_candidate(
    candidate: E2ECandidate,
    config: E2EV1Config,
    input_dim: int,
    train_months: list[E2EDecisionMonth],
    val_months: list[E2EDecisionMonth],
    epoch_count: int,
    stage_name: str,
    device: torch.device,
) -> dict[str, object]:
    model = build_model(config=config, input_dim=input_dim, candidate=candidate, device=device)
    optimizer = Adam(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )

    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_sharpe = -float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epoch_count + 1):
        log_progress(
            f"{stage_name}: {candidate.label()} epoch {epoch}/{epoch_count} started"
        )
        train_preference = train_epoch(
            model=model,
            decision_months=train_months,
            optimizer=optimizer,
            candidate=candidate,
            config=config,
            device=device,
            stage_name=stage_name,
            epoch=epoch,
            epoch_count=epoch_count,
        )
        val_monthly, _ = evaluate_model(
            model=model,
            decision_months=val_months,
            candidate=candidate,
            config=config,
            device=device,
        )
        val_summary = summarize_backtest(val_monthly)
        history.append(
            {
                "epoch": epoch,
                "train_average_preference": train_preference,
                "val_sharpe_ratio": val_summary["sharpe_ratio"],
                "val_annualized_excess_return": val_summary["annualized_excess_return"],
                "val_annualized_volatility": val_summary["annualized_volatility"],
                "val_average_turnover": val_summary["average_turnover"],
            }
        )
        current_sharpe = float(val_summary["sharpe_ratio"])
        log_progress(
            f"{stage_name}: {candidate.label()} epoch {epoch}/{epoch_count} "
            + f"train_preference={train_preference:.6f} "
            + f"val_sharpe={current_sharpe:.6f} "
            + f"val_excess={val_summary['annualized_excess_return']:.6f} "
            + f"val_vol={val_summary['annualized_volatility']:.6f} "
            + f"val_turnover={val_summary['average_turnover']:.6f}"
        )
        if current_sharpe > best_val_sharpe + 1e-12:
            best_val_sharpe = current_sharpe
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            log_progress(
                f"{stage_name}: {candidate.label()} new best checkpoint at epoch {epoch} "
                + f"val_sharpe={current_sharpe:.6f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                log_progress(
                    f"{stage_name}: {candidate.label()} early stopping at epoch {epoch} "
                    + f"best_epoch={best_epoch} best_val_sharpe={best_val_sharpe:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("E2E candidate did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    val_monthly, val_assets = evaluate_model(
        model=model,
        decision_months=val_months,
        candidate=candidate,
        config=config,
        device=device,
    )

    validation_summary = summarize_backtest(val_monthly)
    candidate_dir = (
        config.resolve_output_root()
        / config.run_name
        / "candidates"
        / stage_name
        / candidate.label()
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "candidate": candidate.to_dict(),
            "best_epoch": best_epoch,
        },
        candidate_dir / "best_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(candidate_dir / "training_history.csv", index=False)
    val_monthly.to_csv(candidate_dir / "val_monthly_portfolio.csv", index=False)
    val_assets.to_parquet(candidate_dir / "val_asset_predictions.parquet", index=False)
    (candidate_dir / "candidate_config.json").write_text(
        json.dumps(candidate.to_dict(), indent=2),
        encoding="utf-8",
    )

    return {
        "candidate": candidate,
        "candidate_dir": candidate_dir,
        "best_epoch": best_epoch,
        "history": history,
        "val_monthly": val_monthly,
        "val_assets": val_assets,
        "validation_summary": validation_summary,
        "passed_screens": passes_screens(validation_summary),
    }


def build_model(
    config: E2EV1Config,
    input_dim: int,
    candidate: E2ECandidate,
    device: torch.device,
) -> nn.Module:
    model = E2ERegressor(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    ).to(device)
    if candidate.init_mode in {"transferred", "pretrained"}:
        state_dict = torch.load(config.resolve_pretrained_dir() / "state_dict.pt", map_location="cpu")
        state_dict = remap_pretrained_state_dict_for_e2e(state_dict)
        model.load_state_dict(state_dict)
    return model


def remap_pretrained_state_dict_for_e2e(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "layer1.0.weight" in state_dict or "output.weight" in state_dict:
        return state_dict

    # 21April/local FNN checkpoints save a flat sequential `network.*` state dict.
    network_to_e2e = {
        "network.0.weight": "layer1.0.weight",
        "network.0.bias": "layer1.0.bias",
        "network.1.weight": "layer1.1.weight",
        "network.1.bias": "layer1.1.bias",
        "network.1.running_mean": "layer1.1.running_mean",
        "network.1.running_var": "layer1.1.running_var",
        "network.1.num_batches_tracked": "layer1.1.num_batches_tracked",
        "network.4.weight": "layer2.0.weight",
        "network.4.bias": "layer2.0.bias",
        "network.5.weight": "layer2.1.weight",
        "network.5.bias": "layer2.1.bias",
        "network.5.running_mean": "layer2.1.running_mean",
        "network.5.running_var": "layer2.1.running_var",
        "network.5.num_batches_tracked": "layer2.1.num_batches_tracked",
        "network.8.weight": "layer3.0.weight",
        "network.8.bias": "layer3.0.bias",
        "network.9.weight": "layer3.1.weight",
        "network.9.bias": "layer3.1.bias",
        "network.9.running_mean": "layer3.1.running_mean",
        "network.9.running_var": "layer3.1.running_var",
        "network.9.num_batches_tracked": "layer3.1.num_batches_tracked",
        "network.12.weight": "output.weight",
        "network.12.bias": "output.bias",
    }
    if not any(key.startswith("network.") for key in state_dict):
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for source_key, target_key in network_to_e2e.items():
        if source_key in state_dict:
            remapped[target_key] = state_dict[source_key]
    return remapped if remapped else state_dict


def train_epoch(
    model: nn.Module,
    decision_months: list[E2EDecisionMonth],
    optimizer: Adam,
    candidate: E2ECandidate,
    config: E2EV1Config,
    device: torch.device,
    stage_name: str,
    epoch: int,
    epoch_count: int,
) -> float:
    model.train()
    shuffled = list(decision_months)
    random.shuffle(shuffled)
    if config.train_months_per_epoch is not None:
        shuffled = shuffled[: min(config.train_months_per_epoch, len(shuffled))]
    total_months = len(shuffled)
    heartbeat_every = max(1, min(25, total_months // 10 if total_months > 0 else 1))
    epoch_start_time = perf_counter()
    losses: list[float] = []
    for month_index, decision in enumerate(shuffled, start=1):
        features = torch.from_numpy(decision.X).to(device)
        realized_excess = torch.from_numpy(
            decision.metadata["excess_ret"].to_numpy(dtype=np.float32, copy=True)
        ).to(device)
        sigma = torch.from_numpy(decision.covariance).to(device)

        optimizer.zero_grad()
        predicted_returns = model(features)
        weights = robust_portfolio_weights(
            mu=predicted_returns,
            sigma=sigma,
            lambda_=candidate.lambda_,
            kappa=candidate.kappa,
            omega_mode=candidate.omega_mode,
            solver_mode=config.solver_mode,
            steps=config.mirror_descent_steps,
            step_size=config.mirror_descent_step_size,
            fixed_n_assets=config.exact_fixed_n_assets,
        )
        objective = robust_preference(
            weights=weights,
            realized_excess_returns=realized_excess,
            sigma=sigma,
            lambda_=candidate.lambda_,
            kappa=candidate.kappa,
            omega_mode=candidate.omega_mode,
        )
        loss = -objective
        loss.backward()
        if config.max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=float(config.max_grad_norm))
        optimizer.step()
        losses.append(float(objective.detach().cpu().item()))
        if month_index == 1 or month_index % heartbeat_every == 0 or month_index == total_months:
            elapsed_seconds = perf_counter() - epoch_start_time
            avg_seconds = elapsed_seconds / month_index
            remaining_seconds = max(total_months - month_index, 0) * avg_seconds
            log_progress(
                f"{stage_name}: {candidate.label()} epoch {epoch}/{epoch_count} "
                + f"month {month_index}/{total_months} "
                + f"yyyymm={decision.yyyymm} "
                + f"elapsed={elapsed_seconds / 60.0:.1f}m "
                + f"eta={remaining_seconds / 60.0:.1f}m"
            )
    elapsed_seconds = perf_counter() - epoch_start_time
    log_progress(
        f"{stage_name}: {candidate.label()} epoch {epoch}/{epoch_count} "
        + f"training pass complete in {elapsed_seconds / 60.0:.1f}m"
    )
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    decision_months: list[E2EDecisionMonth],
    candidate: E2ECandidate,
    config: E2EV1Config,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    monthly_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    previous_returns: pd.Series | None = None

    for decision in decision_months:
        features = torch.from_numpy(decision.X).to(device)
        sigma = torch.from_numpy(decision.covariance).to(device)
        predicted_returns = model(features)
        weights = robust_portfolio_weights(
            mu=predicted_returns,
            sigma=sigma,
            lambda_=candidate.lambda_,
            kappa=candidate.kappa,
            omega_mode=candidate.omega_mode,
            solver_mode=config.solver_mode,
            steps=config.mirror_descent_steps,
            step_size=config.mirror_descent_step_size,
            fixed_n_assets=config.exact_fixed_n_assets,
        ).detach().cpu().numpy()
        predictions = predicted_returns.detach().cpu().numpy()
        frame = decision.metadata.reset_index(drop=True).copy()
        permnos = frame["permno"].astype(int).tolist()
        current_weights = pd.Series(weights, index=permnos, dtype=float)
        current_total_returns = pd.Series(frame["ret_tplus1"].to_numpy(dtype=float), index=permnos)
        current_excess_returns = pd.Series(frame["excess_ret"].to_numpy(dtype=float), index=permnos)
        current_rf = pd.Series(frame["Rfree"].to_numpy(dtype=float), index=permnos)

        turnover = compute_turnover(
            previous_weights=previous_weights,
            previous_returns=previous_returns,
            current_weights=current_weights,
        )
        diagnostics = weights_diagnostics(weights)
        monthly_rows.append(
            {
                "split": decision.split,
                "yyyymm": decision.yyyymm,
                "n_assets": float(len(frame)),
                "portfolio_return": float(np.dot(weights, current_total_returns.to_numpy(dtype=float))),
                "portfolio_excess_return": float(np.dot(weights, current_excess_returns.to_numpy(dtype=float))),
                "portfolio_risk_free": float(np.dot(weights, current_rf.to_numpy(dtype=float))),
                "turnover": turnover,
                "herfindahl": diagnostics["herfindahl"],
                "effective_n": diagnostics["effective_n"],
                "max_weight": diagnostics["max_weight"],
                "active_positions": diagnostics["active_positions"],
            }
        )
        asset_frame = frame.copy()
        asset_frame["prediction"] = predictions
        asset_frame["weight"] = weights
        asset_frame["residual"] = asset_frame[TARGET_COLUMN] - asset_frame["prediction"]
        asset_rows.append(asset_frame)

        previous_weights = current_weights
        previous_returns = current_total_returns

    return pd.DataFrame(monthly_rows), pd.concat(asset_rows, ignore_index=True)


def save_selected_artifacts(
    run_dir: Path,
    config: E2EV1Config,
    selected_result: dict[str, object],
    tuning_table: pd.DataFrame,
    pto_baseline_config_path: Path,
    previous_e2e_run_dir: Path | None,
) -> None:
    candidate = selected_result["candidate"]
    train_monthly = selected_result["train_monthly"]
    val_monthly = selected_result["val_monthly"]
    test_monthly = selected_result["test_monthly"]
    train_assets = selected_result["train_assets"]
    val_assets = selected_result["val_assets"]
    test_assets = selected_result["test_assets"]

    pd.DataFrame(selected_result["history"]).to_csv(run_dir / "training_history.csv", index=False)
    train_monthly.to_csv(run_dir / "train_monthly_portfolio.csv", index=False)
    val_monthly.to_csv(run_dir / "val_monthly_portfolio.csv", index=False)
    test_monthly.to_csv(run_dir / "test_monthly_portfolio.csv", index=False)
    train_assets.to_parquet(run_dir / "train_asset_predictions.parquet", index=False)
    val_assets.to_parquet(run_dir / "val_asset_predictions.parquet", index=False)
    test_assets.to_parquet(run_dir / "test_asset_predictions.parquet", index=False)
    train_assets[["yyyymm", "permno", "prediction"]].to_parquet(run_dir / "train_monthly_predicted_returns.parquet", index=False)
    val_assets[["yyyymm", "permno", "prediction"]].to_parquet(run_dir / "val_monthly_predicted_returns.parquet", index=False)
    test_assets[["yyyymm", "permno", "prediction"]].to_parquet(run_dir / "test_monthly_predicted_returns.parquet", index=False)
    train_assets[["yyyymm", "permno", "weight"]].to_parquet(run_dir / "train_monthly_weights.parquet", index=False)
    val_assets[["yyyymm", "permno", "weight"]].to_parquet(run_dir / "val_monthly_weights.parquet", index=False)
    test_assets[["yyyymm", "permno", "weight"]].to_parquet(run_dir / "test_monthly_weights.parquet", index=False)

    checkpoint_path = Path(selected_result["candidate_dir"]) / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")

    selection_summary = {
        "selected_candidate": candidate.to_dict(),
        "best_epoch": int(selected_result["best_epoch"]),
        "solver_mode": config.solver_mode,
        "train_summary": selected_result["train_summary"],
        "validation_summary": selected_result["validation_summary"],
        "test_summary": selected_result["test_summary"],
        "warm_start_help_summary": summarize_warm_start_help(tuning_table),
    }
    (run_dir / "validation_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "final_test_summary.json").write_text(
        json.dumps(selected_result["test_summary"], indent=2),
        encoding="utf-8",
    )

    if previous_e2e_run_dir is not None:
        comparison = build_pto_comparison(
            pto_baseline_config_path=pto_baseline_config_path,
            previous_e2e_run_dir=previous_e2e_run_dir,
            e2e_validation_summary=selected_result["validation_summary"],
            e2e_test_summary=selected_result["test_summary"],
            selected_label=candidate.label(),
            current_model_name=config.current_model_name,
            previous_model_name=config.previous_model_name,
        )
        comparison.to_csv(run_dir / "pto_vs_e2e_comparison.csv", index=False)
        (run_dir / "pto_vs_e2e_comparison.md").write_text(
            build_comparison_markdown(comparison=comparison, selection_summary=selection_summary),
            encoding="utf-8",
        )


def build_pto_comparison(
    pto_baseline_config_path: Path,
    previous_e2e_run_dir: Path,
    e2e_validation_summary: dict[str, float],
    e2e_test_summary: dict[str, float],
    selected_label: str,
    current_model_name: str,
    previous_model_name: str,
) -> pd.DataFrame:
    with pto_baseline_config_path.open("r", encoding="utf-8") as handle:
        baseline_config = json.load(handle)
    pto_run_dir = resolve_repo_path("batuhan/artifacts/pto") / str(baseline_config["source_run_name"])
    with (pto_run_dir / "validation_selection_summary.json").open("r", encoding="utf-8") as handle:
        pto_summary = json.load(handle)
    with (previous_e2e_run_dir / "validation_selection_summary.json").open("r", encoding="utf-8") as handle:
        previous_e2e_summary = json.load(handle)
    rows = [
        {
            "model": "frozen_robust_pto_baseline",
            "split": "val",
            **pto_summary["validation_summary"],
        },
        {
            "model": previous_model_name,
            "split": "val",
            "label": str(previous_e2e_summary["selected_candidate"]["label"]),
            **previous_e2e_summary["validation_summary"],
        },
        {
            "model": current_model_name,
            "split": "val",
            "label": selected_label,
            **e2e_validation_summary,
        },
        {
            "model": "frozen_robust_pto_baseline",
            "split": "test",
            **pto_summary["test_summary"],
        },
        {
            "model": previous_model_name,
            "split": "test",
            "label": str(previous_e2e_summary["selected_candidate"]["label"]),
            **previous_e2e_summary["test_summary"],
        },
        {
            "model": current_model_name,
            "split": "test",
            "label": selected_label,
            **e2e_test_summary,
        },
    ]
    return pd.DataFrame(rows)


def build_comparison_markdown(
    comparison: pd.DataFrame,
    selection_summary: dict[str, object],
) -> str:
    warm = selection_summary["warm_start_help_summary"]
    lines = [
        "# PTO vs E2E Comparison",
        "",
        f"- selected E2E candidate: `{selection_summary['selected_candidate']['label']}`",
        f"- pretrained initialization helped on validation: `{warm['pretrained_init_helped_on_validation']}`",
        f"- best pretrained-init val Sharpe: `{warm['best_pretrained_val_sharpe']:.6f}`",
        f"- best scratch val Sharpe: `{warm['best_scratch_val_sharpe']:.6f}`",
        "",
    ]
    for split in ("val", "test"):
        lines.append(f"## {split.upper()}")
        for _, row in comparison.loc[comparison["split"] == split].iterrows():
            lines.append(
                "- "
                + (
                    f"{row['model']}: ann_return={row['annualized_return']:.6f}, "
                    f"ann_excess={row['annualized_excess_return']:.6f}, "
                    f"ann_vol={row['annualized_volatility']:.6f}, "
                    f"sharpe={row['sharpe_ratio']:.6f}, "
                    f"max_dd={row['max_drawdown']:.6f}, "
                    f"avg_turnover={row['average_turnover']:.6f}, "
                    f"avg_herfindahl={row['average_herfindahl']:.6f}, "
                    f"avg_effective_n={row['average_effective_n']:.2f}, "
                    f"avg_max_weight={row['average_max_weight']:.6f}, "
                    f"avg_active_positions={row['average_active_positions']:.2f}"
                )
            )
        lines.append("")
    return "\n".join(lines)


def select_stage1_val_months(
    val_months: list[E2EDecisionMonth],
    config: E2EV1Config,
) -> list[E2EDecisionMonth]:
    if config.stage1_val_mode == "stride":
        return val_months[:: config.stage1_val_stride]
    if config.stage1_val_mode == "first_n":
        return val_months[: config.stage1_val_n_months]
    raise ValueError(f"Unknown stage1_val_mode={config.stage1_val_mode}")


def select_stage2_finalists(stage1_table: pd.DataFrame, config: E2EV1Config) -> list[E2ECandidate]:
    screened = stage1_table.loc[stage1_table["passed_screens"] == True].copy()  # noqa: E712
    if screened.empty:
        screened = stage1_table.copy()
    finalists = screened.sort_values(
        ["sharpe_ratio", "annualized_volatility", "average_herfindahl", "average_turnover"],
        ascending=[False, True, True, True],
    ).head(config.stage2_n_finalists)
    return [
        E2ECandidate(
            learning_rate=float(row["learning_rate"]),
            weight_decay=float(row["weight_decay"]),
            init_mode=str(row["init_mode"]),
            lambda_=float(row["lambda"]),
            kappa=float(row["kappa"]),
            omega_mode=str(row["omega_mode"]),
        )
        for _, row in finalists.iterrows()
    ]


def summarize_warm_start_help(tuning_table: pd.DataFrame) -> dict[str, float | bool]:
    pretrained = tuning_table.loc[
        tuning_table["init_mode"].isin(["transferred", "pretrained"]),
        "sharpe_ratio",
    ]
    scratch = tuning_table.loc[tuning_table["init_mode"] == "scratch", "sharpe_ratio"]
    best_pretrained = float(pretrained.max()) if not pretrained.empty else -float("inf")
    best_scratch = float(scratch.max()) if not scratch.empty else -float("inf")
    return {
        "best_pretrained_val_sharpe": best_pretrained,
        "best_scratch_val_sharpe": best_scratch,
        "pretrained_init_helped_on_validation": bool(best_pretrained > best_scratch + 1e-12),
    }


def passes_screens(summary: dict[str, float]) -> bool:
    return bool(
        summary["average_max_weight"] <= 0.20
        and summary["average_herfindahl"] <= 0.20
        and summary["average_effective_n"] >= 10.0
        and summary["average_active_positions"] >= 10.0
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
