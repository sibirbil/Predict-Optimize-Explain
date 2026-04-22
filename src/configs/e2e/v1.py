from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.configs.data import DEFAULT_DATA_CONFIG_PATH, Universe500DataConfig
from src.utils.paths import resolve_repo_path

DEFAULT_E2E_CONFIG_PATH = Path("src/configs/e2e/standardized_benchmark.json")


@dataclass(frozen=True)
class E2EV1Config:
    run_name: str = "e2e_v1"
    output_root: str = "batuhan/artifacts/e2e"
    data_config_path: str = str(DEFAULT_DATA_CONFIG_PATH)
    pto_baseline_config_path: str = "src/configs/pto/robust_v1_selected_baseline.json"
    previous_e2e_run_name: str = "e2e_v1_full_20260404"
    previous_e2e_output_root: str = "batuhan/artifacts/e2e"
    pretrained_dir: str = "models/fnn_v1"
    lookback_months: int = 60
    ewma_lambda: float = 0.94
    covariance_shrinkage: float = 0.10
    covariance_ridge: float = 1e-6
    min_assets: int = 20
    hidden_dims: tuple[int, ...] = (32, 16, 8)
    dropout: float = 0.5
    batch_norm: bool = True
    epochs: int = 12
    patience: int = 3
    train_months_per_epoch: int | None = 60
    learning_rate_grid: tuple[float, ...] = (1e-4, 3e-4)
    weight_decay_grid: tuple[float, ...] = (1e-5, 1e-4)
    init_mode_grid: tuple[str, ...] = ("scratch", "pretrained")
    lambda_grid: tuple[float, ...] = (5.0,)
    kappa_grid: tuple[float, ...] = (0.25,)
    omega_mode: str = "diag_sigma"
    solver_mode: str = "mirror_descent"
    exact_fixed_n_assets: int | None = None
    mirror_descent_steps: int = 10
    mirror_descent_step_size: float = 1.0
    max_grad_norm: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    max_train_months: int | None = None
    max_val_months: int | None = None
    max_test_months: int | None = None
    stage1_enabled: bool = False
    stage1_epochs: int = 1
    stage1_val_mode: str = "stride"
    stage1_val_stride: int = 2
    stage1_val_n_months: int = 24
    stage2_epochs: int = 2
    stage2_n_finalists: int = 8
    current_model_name: str = "selected_e2e"
    previous_model_name: str = "previous_e2e_benchmark"

    @classmethod
    def from_json(cls, path: str | Path) -> "E2EV1Config":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for field_name in ("hidden_dims", "learning_rate_grid", "weight_decay_grid", "init_mode_grid", "lambda_grid", "kappa_grid"):
            if field_name in payload:
                if field_name == "init_mode_grid":
                    payload[field_name] = [
                        "pretrained" if str(mode) == "transferred" else str(mode)
                        for mode in payload[field_name]
                    ]
                payload[field_name] = tuple(payload[field_name])
        return cls(**payload)

    def load_data_config(self) -> Universe500DataConfig:
        return Universe500DataConfig.from_json(self.data_config_path)

    def resolve_output_root(self) -> Path:
        return resolve_repo_path(self.output_root)

    def resolve_pretrained_dir(self) -> Path:
        return resolve_repo_path(self.pretrained_dir)

    def resolve_pto_baseline_config_path(self) -> Path:
        return resolve_repo_path(self.pto_baseline_config_path)

    def resolve_previous_e2e_run_dir(self) -> Path | None:
        if self.previous_e2e_run_name is None:
            return None
        return resolve_repo_path(self.previous_e2e_output_root) / self.previous_e2e_run_name

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
