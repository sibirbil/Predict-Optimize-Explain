from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.configs.data import DEFAULT_DATA_CONFIG_PATH, Universe500DataConfig
from src.utils.paths import resolve_repo_path

DEFAULT_PTO_BACKBONE_CONFIG_PATH = Path(
    "src/configs/pto/standardized_pto_backbone.json"
)


@dataclass(frozen=True)
class PredictiveBackboneConfig:
    source_type: str = "saved_predictions"
    source_label: str = "zero_shot_pretrained_backbone"
    run_name: str = "pretrained_7000_transfer_on_universe500"
    output_root: str = "batuhan/artifacts/fnn_transfer"
    data_config_path: str = str(DEFAULT_DATA_CONFIG_PATH)

    @classmethod
    def from_json(cls, path: str | Path) -> "PredictiveBackboneConfig":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)

    def load_data_config(self) -> Universe500DataConfig:
        return Universe500DataConfig.from_json(self.data_config_path)

    def resolve_run_dir(self) -> Path:
        return resolve_repo_path(self.output_root) / self.run_name

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
