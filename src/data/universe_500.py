from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]
REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
    "yyyymm",
    "permno",
    "ret_tplus1",
    "Rfree",
    "excess_ret",
)
TARGET_COLUMN = "target"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "universe_500"


@dataclass(frozen=True)
class DatasetSplit:
    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class Universe500Bundle:
    root: Path
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    feature_names: tuple[str, ...]
    firm_feature_names: tuple[str, ...]
    macro_predictors: tuple[str, ...]
    macro_state: pd.DataFrame


class Universe500Loader:
    """Canonical split-aware loader for batuhan/universe_500 artifacts."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_DATASET_ROOT

    def load_bundle(self) -> Universe500Bundle:
        return Universe500Bundle(
            root=self.root,
            train=self.load_split("train"),
            val=self.load_split("val"),
            test=self.load_split("test"),
            feature_names=self.load_feature_names(),
            firm_feature_names=self.load_firm_feature_names(),
            macro_predictors=self.load_macro_predictors(),
            macro_state=self.load_macro_state(),
        )

    def load_split(self, split: SplitName) -> DatasetSplit:
        X = pd.read_parquet(self._split_path("X", split))
        y = pd.read_parquet(self._split_path("y", split))
        metadata = pd.read_parquet(self._split_path("metadata", split))
        dataset_split = DatasetSplit(X=X, y=y, metadata=metadata)
        self._validate_split(split, dataset_split)
        return dataset_split

    def load_feature_names(self) -> tuple[str, ...]:
        feature_names = self._load_string_array("feature_names.npy")
        if not feature_names:
            raise ValueError("feature_names.npy is empty.")
        return feature_names

    def load_firm_feature_names(self) -> tuple[str, ...]:
        firm_feature_names = self._load_string_array("firm_feature_names.npy")
        if not firm_feature_names:
            raise ValueError("firm_feature_names.npy is empty.")
        return firm_feature_names

    def load_macro_predictors(self) -> tuple[str, ...]:
        macro_predictors = self._load_string_array("macro_predictors.npy")
        if not macro_predictors:
            raise ValueError("macro_predictors.npy is empty.")
        return macro_predictors

    def load_macro_state(self) -> pd.DataFrame:
        macro_state = pd.read_parquet(self.root / "macro_final.parquet")
        required_columns = {"yyyymm", "Rfree", *self.load_macro_predictors()}
        missing_columns = sorted(required_columns.difference(macro_state.columns))
        if missing_columns:
            raise ValueError(
                "macro_final.parquet is missing required columns: "
                + ", ".join(missing_columns)
            )
        if macro_state.empty:
            raise ValueError("macro_final.parquet is empty.")
        return macro_state

    def _split_path(self, prefix: str, split: SplitName) -> Path:
        return self.root / f"{prefix}_{split}.parquet"

    def _load_string_array(self, filename: str) -> tuple[str, ...]:
        values = np.load(self.root / filename, allow_pickle=True)
        return tuple(str(value) for value in values.tolist())

    def _validate_split(self, split: SplitName, dataset_split: DatasetSplit) -> None:
        X, y, metadata = dataset_split.X, dataset_split.y, dataset_split.metadata

        if X.empty or y.empty or metadata.empty:
            raise ValueError(f"{split} split contains an empty artifact.")

        if not (len(X) == len(y) == len(metadata)):
            raise ValueError(
                f"{split} split row mismatch: X={len(X)}, y={len(y)}, metadata={len(metadata)}"
            )

        if not X.index.equals(y.index) or not X.index.equals(metadata.index):
            raise ValueError(f"{split} split index alignment check failed.")

        missing_columns = sorted(set(REQUIRED_METADATA_COLUMNS).difference(metadata.columns))
        if missing_columns:
            raise ValueError(
                f"{split} metadata is missing required columns: {', '.join(missing_columns)}"
            )

        if list(y.columns) != [TARGET_COLUMN]:
            raise ValueError(
                f"{split} target columns must be ['{TARGET_COLUMN}'], found {list(y.columns)}"
            )

        feature_names = self.load_feature_names()
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"{split} feature count mismatch: X has {X.shape[1]} columns but "
                f"feature_names.npy has {len(feature_names)} entries."
            )

        if tuple(X.columns.tolist()) != feature_names:
            raise ValueError(f"{split} feature names do not match feature_names.npy.")


class StandardizedUniverse500Loader(Universe500Loader):
    """Loads Universe500 and immediately scales the X features using the saved training scaler."""

    def __init__(self, root: Path | str | None = None, scaler_dir: Path | str | None = None) -> None:
        super().__init__(root)
        from src.utils.paths import resolve_repo_path
        self._scaler_dir = Path(scaler_dir) if scaler_dir is not None else resolve_repo_path("batuhan/artifacts/scaler")
        self._scaler = None

    @property
    def scaler(self):
        from src.data.scaler import FeatureScaler
        if self._scaler is None:
            self._scaler = FeatureScaler.load(self._scaler_dir)
        return self._scaler

    def load_split(self, split: SplitName) -> DatasetSplit:
        dataset_split = super().load_split(split)
        scaled_X = self.scaler.transform(dataset_split.X)
        return DatasetSplit(X=scaled_X, y=dataset_split.y, metadata=dataset_split.metadata)


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "REQUIRED_METADATA_COLUMNS",
    "DatasetSplit",
    "SplitName",
    "Universe500Bundle",
    "Universe500Loader",
    "StandardizedUniverse500Loader",
]
