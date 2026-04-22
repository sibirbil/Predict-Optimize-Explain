from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATASET_ROOT


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame
    X_raw_backup: pd.DataFrame | None = None


@dataclass(frozen=True)
class Universe500Bundle:
    root: Path
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    metadata_all: pd.DataFrame
    macro_final: pd.DataFrame
    macro_fred_supplement: pd.DataFrame | None
    feature_names: tuple[str, ...]
    firm_feature_names: tuple[str, ...]
    macro_predictors: tuple[str, ...]
    selected_permnos: tuple[int, ...]


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_string_array(path: Path) -> tuple[str, ...]:
    values = np.load(path, allow_pickle=True)
    return tuple(str(value) for value in values.tolist())


def _load_int_array(path: Path) -> tuple[int, ...]:
    values = np.load(path, allow_pickle=True)
    return tuple(int(value) for value in values.tolist())


def _load_split(root: Path, name: str) -> DatasetSplit:
    return DatasetSplit(
        name=name,
        X=_load_parquet(root / f"X_{name}.parquet"),
        y=_load_parquet(root / f"y_{name}.parquet"),
        metadata=_load_parquet(root / f"metadata_{name}.parquet"),
        X_raw_backup=_load_optional_parquet(root / f"X_{name}_raw_backup.parquet"),
    )


def load_universe_500(root: Path = DATASET_ROOT) -> Universe500Bundle:
    return Universe500Bundle(
        root=root,
        train=_load_split(root, "train"),
        val=_load_split(root, "val"),
        test=_load_split(root, "test"),
        metadata_all=_load_parquet(root / "metadata.parquet"),
        macro_final=_load_parquet(root / "macro_final.parquet"),
        macro_fred_supplement=_load_optional_parquet(root / "macro_fred_supplement.parquet"),
        feature_names=_load_string_array(root / "feature_names.npy"),
        firm_feature_names=_load_string_array(root / "firm_feature_names.npy"),
        macro_predictors=_load_string_array(root / "macro_predictors.npy"),
        selected_permnos=_load_int_array(root / "selected_500_permnos.npy"),
    )

