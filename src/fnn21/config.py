from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import DATASET_ROOT, FNN_ARTIFACTS_ROOT, REPO_ROOT


@dataclass(frozen=True)
class FNNTrainConfig:
    run_name: str = "baseline"
    dataset_root: str = "21April/universe_500_v21April_stdz_macro"
    output_root: str = "21April/artifacts/fnn"
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.1
    batch_norm: bool = True
    batch_size: int = 1024
    eval_batch_size: int = 4096
    epochs: int = 20
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    seed: int = 42
    device: str = "auto"
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "FNNTrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        valid_fields = {field.name for field in dataclasses.fields(cls)}
        filtered_payload = {key: value for key, value in payload.items() if key in valid_fields}
        if "hidden_dims" in filtered_payload:
            filtered_payload["hidden_dims"] = tuple(int(value) for value in filtered_payload["hidden_dims"])
        return cls(**filtered_payload)

    def resolve_dataset_root(self) -> Path:
        path = Path(self.dataset_root)
        return path if path.is_absolute() else REPO_ROOT / path

    def resolve_output_root(self) -> Path:
        path = Path(self.output_root)
        return path if path.is_absolute() else REPO_ROOT / path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_TRAIN_CONFIG = FNNTrainConfig(
    dataset_root=str(DATASET_ROOT.relative_to(REPO_ROOT)),
    output_root=str(FNN_ARTIFACTS_ROOT.relative_to(REPO_ROOT)),
)
