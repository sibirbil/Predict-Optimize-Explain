from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "target"


@dataclass(frozen=True)
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def load_feature_names(dataset_root: Path) -> tuple[str, ...]:
    values = np.load(dataset_root / "feature_names.npy", allow_pickle=True)
    return tuple(str(value) for value in values.tolist())


def load_split_arrays(dataset_root: Path, split: str, max_rows: int | None = None) -> SplitArrays:
    X = pd.read_parquet(dataset_root / f"X_{split}.parquet")
    y = pd.read_parquet(dataset_root / f"y_{split}.parquet")
    metadata = pd.read_parquet(dataset_root / f"metadata_{split}.parquet")

    if max_rows is not None:
        X = X.iloc[:max_rows].copy()
        y = y.iloc[:max_rows].copy()
        metadata = metadata.iloc[:max_rows].copy()

    return SplitArrays(
        X=X.to_numpy(dtype=np.float32, copy=True),
        y=y[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True),
        metadata=metadata.reset_index(drop=True),
    )
