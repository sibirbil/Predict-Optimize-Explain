from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.configs.data import DEFAULT_DATA_CONFIG_PATH, Universe500DataConfig
from src.utils.paths import resolve_repo_path

DEFAULT_FNN_CONFIG_PATH = Path("src/configs/model/fnn_baseline.json")


@dataclass(frozen=True)
class FNNTrainConfig:
    run_name: str = "baseline"
    output_root: str = "batuhan/artifacts/fnn"
    data_config_path: str = str(DEFAULT_DATA_CONFIG_PATH)
    variant_name: str = "clean_baseline"
    hidden_dims: tuple[int, ...] = (32, 16, 8)
    dropout: float = 0.1
    batch_norm: bool = True
    batch_size: int = 1024
    eval_batch_size: int = 2048
    epochs: int = 10
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    seed: int = 42
    device: str = "auto"
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "FNNTrainConfig":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        
        # Filter for only valid dataclass fields
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_payload = {k: v for k, v in payload.items() if k in valid_fields}
        
        if "hidden_dims" in filtered_payload:
            filtered_payload["hidden_dims"] = tuple(filtered_payload["hidden_dims"])
            
        return cls(**filtered_payload)

    def load_data_config(self) -> Universe500DataConfig:
        return Universe500DataConfig.from_json(self.data_config_path)

    def resolve_output_root(self) -> Path:
        return resolve_repo_path(self.output_root)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
