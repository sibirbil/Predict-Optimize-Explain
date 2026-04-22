from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.universe_500 import Universe500Loader
from src.utils.paths import resolve_repo_path

DEFAULT_DATA_CONFIG_PATH = Path("src/configs/data/universe_500.json")


@dataclass(frozen=True)
class Universe500DataConfig:
    dataset_root: str = "batuhan/universe_500"
    standardize: bool = False  # opt-in: must be explicitly set to True in standardized configs

    @classmethod
    def from_json(cls, path: str | Path) -> "Universe500DataConfig":
        with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)

    def resolve_dataset_root(self) -> Path:
        return resolve_repo_path(self.dataset_root)

    def build_loader(self) -> Universe500Loader:
        if self.standardize:
            from src.data.universe_500 import StandardizedUniverse500Loader
            return StandardizedUniverse500Loader(root=self.resolve_dataset_root())
        return Universe500Loader(root=self.resolve_dataset_root())

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)
