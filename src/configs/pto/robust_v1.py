from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.configs.pto.predictive_backbone import (
    DEFAULT_PTO_BACKBONE_CONFIG_PATH,
    PredictiveBackboneConfig,
)
from src.utils.paths import resolve_repo_path

DEFAULT_PTO_CONFIG_PATH = Path("src/configs/pto/standardized_pto.json")


@dataclass(frozen=True)
class PTOV1Config:
    run_name: str = "pto_robust_v1"
    output_root: str = "batuhan/artifacts/pto"
    predictive_backbone_config_path: str = str(DEFAULT_PTO_BACKBONE_CONFIG_PATH)
    lookback_months: int = 60
    ewma_lambda: float = 0.94
    covariance_shrinkage: float = 0.10
    covariance_ridge: float = 1e-6
    min_assets: int = 20
    lambda_grid: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    kappa_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
    omega_grid: tuple[str, ...] = ("identity", "diag_sigma")
    max_avg_turnover: float = 1.0
    max_avg_weight: float = 0.20
    max_avg_herfindahl: float = 0.20
    min_avg_effective_n: float = 10.0
    min_avg_active_positions: float = 10.0
    stage1_mode: str = "first_n"
    stage1_n_months: int = 24
    stage1_stride: int = 2
    stage2_n_finalists: int = 8

    @classmethod
    def from_json(cls, path: str | Path) -> "PTOV1Config":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["lambda_grid"] = tuple(float(value) for value in payload["lambda_grid"])
        payload["kappa_grid"] = tuple(float(value) for value in payload["kappa_grid"])
        payload["omega_grid"] = tuple(str(value) for value in payload["omega_grid"])
        return cls(**payload)

    def load_backbone_config(self) -> PredictiveBackboneConfig:
        return PredictiveBackboneConfig.from_json(self.predictive_backbone_config_path)

    def resolve_output_root(self) -> Path:
        return resolve_repo_path(self.output_root)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
