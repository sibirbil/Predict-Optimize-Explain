from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.macro_scaler import MacroScaler


def _parse_interaction_feature(
    name: str,
    macro_predictors: tuple[str, ...],
) -> tuple[str, str] | None:
    for macro_name in macro_predictors:
        suffix = f"_x_{macro_name}"
        if name.endswith(suffix):
            return name[: -len(suffix)], macro_name
    return None


def _is_interaction_feature(name: str, macro_predictors: tuple[str, ...]) -> bool:
    return _parse_interaction_feature(name, macro_predictors) is not None


class FeatureScenarioBuilder:
    """Build scenario features from a base firm frame and a macro state.

    If a FeatureScaler is provided (explicitly, not by path), it is applied
    after feature construction. The caller is responsible for loading the
    correct scaler that was bundled with the model artifact being evaluated.
    """

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        macro_predictors: tuple[str, ...],
        scaler: "object | None" = None,  # FeatureScaler | None — must be passed explicitly
        macro_scaler: MacroScaler | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.macro_predictors = macro_predictors
        self.macro_to_index = {name: idx for idx, name in enumerate(macro_predictors)}
        self.base_columns = [name for name in feature_names if not _is_interaction_feature(name, macro_predictors)]
        self.base_tensors = {
            name: torch.tensor(
                feature_frame[name].to_numpy(dtype=np.float32, copy=True),
                dtype=torch.float32,
            )
            for name in self.base_columns
        }
        self.scaler = scaler  # None = raw features; caller must wire in the model's own scaler
        self.macro_scaler = macro_scaler

    def build(self, macro_state: torch.Tensor) -> torch.Tensor:
        if self.macro_scaler is not None:
            macro_state = self.macro_scaler.transform_tensor(macro_state)

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

        if self.scaler is not None:
            # Pass feature_names so the scaler can hard-verify column order
            features = self.scaler.transform_tensor(features, feature_names=self.feature_names)

        return features


__all__ = ["FeatureScenarioBuilder"]
