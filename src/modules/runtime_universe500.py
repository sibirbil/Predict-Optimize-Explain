from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

from src.modules.runtime_paths import DATA_DIR, SCALER_DIR
from src.utils.dates import shift_yyyymm
from src.utils.helper_functions import make_psd_np


def _load_string_array(path) -> tuple[str, ...]:
    return tuple(str(item) for item in np.load(path, allow_pickle=True).tolist())


class FeatureScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray, feature_names: tuple[str, ...]) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.where(np.asarray(std, dtype=np.float32) < 1e-8, np.float32(1e-8), np.asarray(std, dtype=np.float32))
        self.feature_names = tuple(feature_names)

    @classmethod
    def load(cls, scaler_dir) -> "FeatureScaler":
        feature_names = _load_string_array(scaler_dir / "feature_names.npy")
        if (scaler_dir / "feature_std_stabilized.npy").exists():
            std = np.load(scaler_dir / "feature_std_stabilized.npy")
        else:
            std = np.load(scaler_dir / "feature_std.npy")
        return cls(
            mean=np.load(scaler_dir / "feature_mean.npy"),
            std=std,
            feature_names=feature_names,
        )

    def transform_tensor(self, features: torch.Tensor, feature_names: tuple[str, ...]) -> torch.Tensor:
        if tuple(feature_names) != self.feature_names:
            raise ValueError("Feature order mismatch between runtime scaler and runtime_universe500 feature_names.")
        mean_t = torch.from_numpy(self.mean).to(features.device)
        std_t = torch.from_numpy(self.std).to(features.device)
        return (features - mean_t) / std_t


def _parse_interaction_feature(name: str, macro_predictors: Iterable[str]) -> tuple[str, str] | None:
    for macro_name in macro_predictors:
        suffix = f"_x_{macro_name}"
        if name.endswith(suffix):
            return name[: -len(suffix)], macro_name
    return None


@dataclass
class ScaledInteractionBuilder:
    base_frame: pd.DataFrame
    feature_names: tuple[str, ...]
    macro_predictors: tuple[str, ...]
    scaler: FeatureScaler

    def __post_init__(self) -> None:
        self.macro_to_index = {name: idx for idx, name in enumerate(self.macro_predictors)}
        self.base_columns = tuple(
            name for name in self.feature_names if _parse_interaction_feature(name, self.macro_predictors) is None
        )
        self.base_tensors = {
            name: torch.tensor(
                self.base_frame[name].to_numpy(dtype=np.float32, copy=True),
                dtype=torch.float32,
            )
            for name in self.base_columns
        }

    def build(self, macro_state: torch.Tensor) -> torch.Tensor:
        columns: list[torch.Tensor] = []
        for feature_name in self.feature_names:
            parsed = _parse_interaction_feature(feature_name, self.macro_predictors)
            if parsed is None:
                columns.append(self.base_tensors[feature_name].to(device=macro_state.device))
                continue
            base_name, macro_name = parsed
            base = self.base_tensors[base_name].to(device=macro_state.device)
            columns.append(base * macro_state[self.macro_to_index[macro_name]])
        features = torch.stack(columns, dim=1)
        return self.scaler.transform_tensor(features, feature_names=self.feature_names)


def _ewma_covariance(
    returns_matrix: np.ndarray,
    ewma_lambda: float = 0.94,
    shrinkage: float = 0.10,
    ridge: float = 1e-6,
) -> np.ndarray:
    n_months = returns_matrix.shape[0]
    exponents = np.arange(n_months - 1, -1, -1)
    weights = (1.0 - ewma_lambda) * (ewma_lambda ** exponents)
    weights = weights / weights.sum()
    mean = np.sum(returns_matrix * weights[:, None], axis=0)
    centered = returns_matrix - mean[None, :]
    sigma = centered.T @ (centered * weights[:, None])
    sigma = 0.5 * (sigma + sigma.T)
    diag = np.diag(np.diag(sigma))
    sigma = (1.0 - shrinkage) * sigma + shrinkage * diag
    sigma = sigma + ridge * np.eye(sigma.shape[0])
    sigma = make_psd_np(sigma)
    return 0.5 * (sigma + sigma.T)


def load_runtime_data() -> Dict[str, object]:
    metadata = pd.read_parquet(DATA_DIR / "metadata.parquet")
    metadata_test = pd.read_parquet(DATA_DIR / "metadata_test.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    macro_final = pd.read_parquet(DATA_DIR / "macro_final.parquet")

    metadata["yyyymm"] = metadata["yyyymm"].astype(int)
    metadata_test["yyyymm"] = metadata_test["yyyymm"].astype(int)
    macro_final["yyyymm"] = macro_final["yyyymm"].astype(int)

    return {
        "metadata": metadata,
        "metadata_test": metadata_test,
        "X_test": X_test,
        "macro_final": macro_final,
        "feature_names": _load_string_array(DATA_DIR / "feature_names.npy"),
        "firm_feature_names": _load_string_array(DATA_DIR / "firm_feature_names.npy"),
        "macro_predictors": _load_string_array(DATA_DIR / "macro_predictors.npy"),
    }


data = load_runtime_data()
_SCALER = FeatureScaler.load(SCALER_DIR)


def construct_C2(
    data: Dict[str, object],
    date: int,
    K: int,
):
    lookback = 60
    metadata = data["metadata"].copy()
    metadata_test = data["metadata_test"].copy()
    X_test = data["X_test"]

    metadata_test["_orig_index"] = metadata_test.index
    grouped = metadata[
        metadata["yyyymm"].between(shift_yyyymm(int(date), -lookback), shift_yyyymm(int(date), -1))
    ].groupby("permno")["excess_ret"]
    valid_today = metadata_test.loc[metadata_test["yyyymm"] == int(date), "permno"]
    permnos = (
        grouped.mean()
        .where(grouped.count() == lookback)
        .loc[lambda s: s.index.isin(valid_today)]
        .sort_values(ascending=False, na_position="last")
        .head(int(K))
        .index
        .tolist()
    )

    history = metadata[
        (metadata["yyyymm"] >= shift_yyyymm(int(date), -lookback))
        & (metadata["yyyymm"] <= shift_yyyymm(int(date), -1))
        & (metadata["permno"].isin(permnos))
    ].copy()
    history_wide = history.pivot(index="yyyymm", columns="permno", values="excess_ret").sort_index()
    history_wide = history_wide[permnos].dropna(axis=1, how="any")
    used_assets = [int(p) for p in history_wide.columns.tolist()]
    if len(used_assets) != int(K):
        raise ValueError(
            f"Runtime Universe500 construct_C2 kept {len(used_assets)} assets, expected {K}. "
            "The selected month does not satisfy the full 60-month lookback filter for all requested names."
        )

    sigma = _ewma_covariance(history_wide.to_numpy(dtype=float))

    today = metadata_test[metadata_test["yyyymm"] == int(date)].drop_duplicates(subset=["permno"], keep="first").set_index("permno")
    selected_today = today.loc[used_assets]
    dd = X_test.loc[selected_today["_orig_index"]].copy()

    C_t = ScaledInteractionBuilder(
        base_frame=dd[list(data["firm_feature_names"])].copy(),
        feature_names=tuple(data["feature_names"]),
        macro_predictors=tuple(data["macro_predictors"]),
        scaler=_SCALER,
    )
    rets_t = torch.tensor(selected_today["excess_ret"].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
    Sigma = torch.tensor(sigma.astype(np.float32, copy=False), dtype=torch.float32)
    return Sigma, C_t, rets_t, used_assets


__all__ = ["ScaledInteractionBuilder", "construct_C2", "data"]
