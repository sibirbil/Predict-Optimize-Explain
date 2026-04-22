from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.configs.data import Universe500DataConfig
from src.configs.model import FNNTrainConfig
from src.data import TARGET_COLUMN
from src.modeling.fnn.model import FeedForwardRegressor


@dataclass(frozen=True)
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def run_fnn_training(
    config: FNNTrainConfig,
    data_config: Universe500DataConfig | None = None,
) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_config = data_config or config.load_data_config()
    loader = data_config.build_loader()
    feature_names = list(loader.load_feature_names())

    train_split = load_split_arrays(loader, "train", config.max_train_rows)
    val_split = load_split_arrays(loader, "val", config.max_val_rows)

    model = FeedForwardRegressor(
        input_dim=train_split.X.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    ).to(device)

    train_loader = build_dataloader(
        X=train_split.X,
        y=train_split.y,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        X=val_split.X,
        y=val_split.y,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    optimizer = build_optimizer(model=model, config=config)
    loss_fn = nn.MSELoss()

    run_dir = config.resolve_output_root() / config.run_name
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_r2 = -float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_predictions, val_metrics = predict_and_score(
            model=model,
            data_loader=val_loader,
            y_true=val_split.y,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "val_correlation": val_metrics["correlation"],
            }
        )

        # -- Updated Progress Logging (incorporating R-squared) --
        log_msg = (
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"Val R2_Zero: {val_metrics['r2_zero']:.4f} | "
            f"Val Corr: {val_metrics['correlation']:.4f}"
        )
        print(log_msg, flush=True)

        # Selection based on maximizing R2 (which is equivalent to minimizing MSE)
        # We use standard R2 here as it is the most common selection metric.
        if val_metrics["r2"] > best_val_r2: 
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  --> New best model found at epoch {epoch} (Val R2: {best_val_r2:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  --> Early stopping triggered at epoch {epoch}", flush=True)
                break

        _ = val_predictions

    if best_state is None:
        raise RuntimeError("FNN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    split_metrics: dict[str, dict[str, float]] = {}
    split_row_counts = {
        "train": len(train_split.y),
        "val": len(val_split.y),
    }

    for split_name, row_limit in (
        ("train", config.max_train_rows),
        ("val", config.max_val_rows),
        ("test", config.max_test_rows),
    ):
        split_arrays = load_split_arrays(loader, split_name, row_limit)
        data_loader = build_dataloader(
            X=split_arrays.X,
            y=split_arrays.y,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        predictions, metrics = predict_and_score(
            model=model,
            data_loader=data_loader,
            y_true=split_arrays.y,
            device=device,
        )
        split_metrics[split_name] = metrics
        split_row_counts[split_name] = len(split_arrays.y)
        save_predictions(
            path=predictions_dir / f"{split_name}_predictions.parquet",
            metadata=split_arrays.metadata,
            y_true=split_arrays.y,
            predictions=predictions,
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": train_split.X.shape[1],
        "feature_names": feature_names,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_val_r2": best_val_r2,
    }
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **config.to_dict(),
                "resolved_dataset_root": str(data_config.resolve_dataset_root()),
                "resolved_output_root": str(config.resolve_output_root()),
            },
            handle,
            indent=2,
        )

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_r2": best_val_r2,
        "device": device.type,
        "row_counts": split_row_counts,
        "metrics": split_metrics,
    }
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

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
    prediction_batches: list[np.ndarray] = []
    for features, _ in data_loader:
        features = features.to(device)
        predictions = model(features).detach().cpu().numpy()
        prediction_batches.append(predictions)

    y_pred = np.concatenate(prediction_batches, axis=0)
    return y_pred, regression_metrics(y_true=y_true, y_pred=y_pred)


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_split_arrays(loader, split: str, max_rows: int | None) -> SplitArrays:
    split_data = loader.load_split(split)  # type: ignore[arg-type]
    X = split_data.X
    y = split_data.y
    metadata = split_data.metadata

    if max_rows is not None:
        X = X.iloc[:max_rows].copy()
        y = y.iloc[:max_rows].copy()
        metadata = metadata.iloc[:max_rows].copy()

    return SplitArrays(
        X=X.to_numpy(dtype=np.float32, copy=True),
        y=y[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True),
        metadata=metadata.reset_index(drop=True),
    )


def save_predictions(
    path: Path,
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> None:
    prediction_frame = metadata.copy()
    prediction_frame[TARGET_COLUMN] = y_true
    prediction_frame["prediction"] = predictions
    prediction_frame["residual"] = y_true - predictions
    prediction_frame.to_parquet(path, index=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(np.square(residuals)))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum(np.square(y_true - y_mean)))
    ss_zero = float(np.sum(np.square(y_true)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_zero = 1.0 - ss_res / ss_zero if ss_zero > 0 else 0.0
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    if np.isnan(correlation):
        correlation = 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "r2_zero": float(r2_zero),
        "correlation": correlation,
    }


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
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")
