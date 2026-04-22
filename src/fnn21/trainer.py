from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset

from .config import FNNTrainConfig
from .data import SplitArrays, load_feature_names, load_split_arrays
from .evaluation import regression_metrics, save_predictions, summarize_prediction_dir
from .model import FeedForwardRegressor


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_fnn_training(config: FNNTrainConfig) -> Path:
    set_seed(config.seed)
    dataset_root = config.resolve_dataset_root()
    output_root = config.resolve_output_root()
    device = resolve_device(config.device)

    feature_names = list(load_feature_names(dataset_root))
    train_split = load_split_arrays(dataset_root, "train", config.max_train_rows)
    val_split = load_split_arrays(dataset_root, "val", config.max_val_rows)

    model = FeedForwardRegressor(
        input_dim=train_split.X.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    ).to(device)

    train_loader = build_dataloader(train_split, config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_split, config.eval_batch_size, shuffle=False)
    optimizer = build_optimizer(model, config)
    loss_fn = nn.MSELoss()

    run_dir = output_root / config.run_name
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_r2 = -float("inf")
    best_epoch = 0
    patience_counter = 0

    log_progress(
        "FNN training started "
        f"run_name={config.run_name} device={device.type} "
        f"dataset_root={dataset_root} rows(train/val)=({len(train_split.y)}/{len(val_split.y)}) "
        f"features={train_split.X.shape[1]}"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_predictions, val_metrics = predict_and_score(model, val_loader, val_split.y, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "val_r2_zero": val_metrics["r2_zero"],
                "val_correlation": val_metrics["correlation"],
            }
        )
        log_progress(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_r2={val_metrics['r2']:.6f} "
            f"val_corr={val_metrics['correlation']:.6f}"
        )

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            log_progress(f"new best epoch={epoch} val_r2={best_val_r2:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                log_progress(f"early stopping at epoch={epoch} best_epoch={best_epoch} best_val_r2={best_val_r2:.6f}")
                break

        _ = val_predictions

    if best_state is None:
        raise RuntimeError("FNN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    split_metrics: dict[str, dict[str, float]] = {}
    row_counts: dict[str, int] = {}
    for split_name, row_limit in (
        ("train", config.max_train_rows),
        ("val", config.max_val_rows),
        ("test", config.max_test_rows),
    ):
        split = load_split_arrays(dataset_root, split_name, row_limit)
        loader = build_dataloader(split, config.eval_batch_size, shuffle=False)
        predictions, metrics = predict_and_score(model, loader, split.y, device)
        split_metrics[split_name] = metrics
        row_counts[split_name] = len(split.y)
        save_predictions(predictions_dir / f"{split_name}_predictions.parquet", split.metadata, split.y, predictions)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": train_split.X.shape[1],
        "feature_names": feature_names,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_val_r2": best_val_r2,
    }
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")
    # Foundation E2E transfer code expects a plain state_dict.pt artifact.
    torch.save(model.state_dict(), run_dir / "state_dict.pt")

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **config.to_dict(),
                "resolved_dataset_root": str(dataset_root),
                "resolved_output_root": str(output_root),
            },
            handle,
            indent=2,
        )

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    evaluation_summary = summarize_prediction_dir(predictions_dir, run_dir, quantiles=10)
    summary = {
        "best_epoch": best_epoch,
        "best_val_r2": best_val_r2,
        "device": device.type,
        "row_counts": row_counts,
        "metrics": split_metrics,
        "evaluation": evaluation_summary,
    }
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log_progress(
        f"FNN training completed run_name={config.run_name} "
        f"best_epoch={best_epoch} val_r2={best_val_r2:.6f} "
        f"test_r2={split_metrics['test']['r2']:.6f}"
    )
    return run_dir


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for features, target in data_loader:
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_and_score(
    model: nn.Module,
    data_loader: DataLoader,
    y_true: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    batches: list[np.ndarray] = []
    for features, _ in data_loader:
        features = features.to(device)
        predictions = model(features).detach().cpu().numpy()
        batches.append(predictions)
    y_pred = np.concatenate(batches, axis=0)
    return y_pred, regression_metrics(y_true=y_true, y_pred=y_pred)


def build_dataloader(split: SplitArrays, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(split.X), torch.from_numpy(split.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def build_optimizer(model: nn.Module, config: FNNTrainConfig):
    name = config.optimizer_name.lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")
