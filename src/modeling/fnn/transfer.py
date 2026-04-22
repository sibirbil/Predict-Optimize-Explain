from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.configs.data import DEFAULT_DATA_CONFIG_PATH, Universe500DataConfig
from src.data import TARGET_COLUMN
from src.modeling.benchmarks.runner import build_comparison_table
from src.modeling.fnn.evaluation import MonthlyEvaluationConfig, evaluate_prediction_run
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class PretrainedTransferConfig:
    run_name: str = "pretrained_7000_transfer_on_universe500"
    pretrained_dir: str = "models/fnn_v1"
    output_root: str = "batuhan/artifacts/fnn_transfer"
    data_config_path: str = str(DEFAULT_DATA_CONFIG_PATH)
    eval_batch_size: int = 4096
    device: str = "auto"
    quantiles: int = 10

    def load_data_config(self) -> Universe500DataConfig:
        return Universe500DataConfig.from_json(self.data_config_path)

    def resolve_output_root(self) -> Path:
        return resolve_repo_path(self.output_root)

    def resolve_pretrained_dir(self) -> Path:
        return resolve_repo_path(self.pretrained_dir)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureCompatibilityReport:
    old_feature_count: int
    current_feature_count: int
    feature_count_match: bool
    feature_set_match: bool
    feature_order_match: bool
    can_reorder_current_to_old: bool
    missing_old_columns: tuple[str, ...]
    extra_current_columns: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return self.can_reorder_current_to_old and len(self.missing_old_columns) == 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LegacyArtifactFNN(nn.Module):
    """Legacy notebook-style FNN with module names matching the saved state_dict."""

    def __init__(self, input_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(inputs)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        return self.output(hidden).squeeze(-1)


def audit_feature_compatibility(
    old_feature_names: list[str],
    current_feature_names: list[str],
) -> FeatureCompatibilityReport:
    old_set = set(old_feature_names)
    current_set = set(current_feature_names)
    missing_old = tuple(column for column in old_feature_names if column not in current_set)
    extra_current = tuple(column for column in current_feature_names if column not in old_set)
    can_reorder = len(missing_old) == 0
    return FeatureCompatibilityReport(
        old_feature_count=len(old_feature_names),
        current_feature_count=len(current_feature_names),
        feature_count_match=len(old_feature_names) == len(current_feature_names),
        feature_set_match=old_set == current_set,
        feature_order_match=old_feature_names == current_feature_names,
        can_reorder_current_to_old=can_reorder,
        missing_old_columns=missing_old,
        extra_current_columns=extra_current,
    )


def run_pretrained_transfer(config: PretrainedTransferConfig | None = None) -> Path:
    config = config or PretrainedTransferConfig()
    data_config = config.load_data_config()
    loader = data_config.build_loader()
    pretrained_dir = config.resolve_pretrained_dir()

    old_feature_names = json.loads((pretrained_dir / "feature_columns.json").read_text())
    model_config = json.loads((pretrained_dir / "model_config.json").read_text())
    current_feature_names = list(loader.load_feature_names())
    compatibility = audit_feature_compatibility(old_feature_names, current_feature_names)
    if not compatibility.is_compatible:
        raise ValueError(
            "Old artifact feature contract is incompatible with current universe_500 features."
        )

    device = resolve_device(config.device)
    model = LegacyArtifactFNN(
        input_dim=int(model_config["input_dim"]),
        dropout_rate=float(model_config.get("dropout_rate", 0.5)),
    ).to(device)
    state_dict = torch.load(pretrained_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    reorder_indices = build_reorder_indices(
        old_feature_names=old_feature_names,
        current_feature_names=current_feature_names,
    )

    run_dir = config.resolve_output_root() / config.run_name
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        X, y_true, metadata = load_split_arrays(loader, split)
        reordered_X = X[:, reorder_indices]
        predictions = predict_numpy(
            model=model,
            X=reordered_X,
            batch_size=config.eval_batch_size,
            device=device,
        )
        save_predictions(
            path=predictions_dir / f"{split}_predictions.parquet",
            metadata=metadata,
            y_true=y_true,
            predictions=predictions,
        )

    monthly_config = MonthlyEvaluationConfig(quantiles=config.quantiles)
    evaluate_prediction_run(
        run_name=config.run_name,
        output_root=config.output_root,
        monthly_config=monthly_config,
    )

    comparison = build_comparison_table(
        rows=[
            ("zero_shot_pretrained_backbone", config.output_root, config.run_name),
            ("zero_forecast_archived", "batuhan/archive/artifacts/benchmarks", "zero_forecast_full_suite_20260404"),
            (
                "train_mean_forecast_archived",
                "batuhan/archive/artifacts/benchmarks",
                "train_mean_forecast_full_suite_20260404",
            ),
            ("current_clean_fnn_archived", "batuhan/archive/artifacts/fnn", "full_fnn_baseline_20260404"),
            ("legacy_style_fnn_archived", "batuhan/archive/artifacts/fnn", "full_fnn_legacy_style_20260404"),
            ("elastic_net_archived", "batuhan/archive/artifacts/benchmarks", "elastic_net_full_suite_20260404"),
        ]
    )
    comparison.to_csv(run_dir / "transfer_comparison.csv", index=False)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **config.to_dict(),
                "resolved_pretrained_dir": str(pretrained_dir),
                "resolved_output_root": str(config.resolve_output_root()),
                "resolved_data_root": str(loader.root),
            },
            handle,
            indent=2,
        )

    with (run_dir / "compatibility_report.json").open("w", encoding="utf-8") as handle:
        json.dump(compatibility.to_dict(), handle, indent=2)

    with (run_dir / "source_model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)

    return run_dir


def build_reorder_indices(
    old_feature_names: list[str],
    current_feature_names: list[str],
) -> list[int]:
    column_to_index = {column: idx for idx, column in enumerate(current_feature_names)}
    return [column_to_index[column] for column in old_feature_names]


def predict_numpy(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            batch = torch.from_numpy(X[start:stop]).to(device)
            prediction = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            outputs.append(prediction)
    return np.concatenate(outputs, axis=0) if outputs else np.empty(shape=(0,), dtype=np.float32)


def load_split_arrays(loader, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_data = loader.load_split(split)  # type: ignore[arg-type]
    X = split_data.X.to_numpy(dtype=np.float32, copy=True)
    y = split_data.y[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
    metadata = split_data.metadata.reset_index(drop=True)
    return X, y, metadata


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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)
