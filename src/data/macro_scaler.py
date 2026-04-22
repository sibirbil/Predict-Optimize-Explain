from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

MACRO_COLUMNS = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl")


class MacroScaler:
    """Standardizes the 9 raw macro columns using train-period mean/std.

    Stored as torch tensors (no .npy). Used identically in the training pipeline
    (src/data/builders.py) and scenario generation (runtime_components.py).

    Critical invariant: the same MacroScaler artifact that is used during training
    must be used at scenario time. The chain is:
      raw macro → MacroScaler.transform_tensor() → FeatureScenarioBuilder.build()
                → FeatureScaler.transform_tensor() → model
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.float()
        self.std = torch.clamp(std.float(), min=1e-8)

    @classmethod
    def fit(cls, macro_df: pd.DataFrame) -> "MacroScaler":
        """Fit on train-period rows. macro_df must contain all MACRO_COLUMNS."""
        cols = macro_df[list(MACRO_COLUMNS)].to_numpy(dtype="float32")
        mean = torch.tensor(cols.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(cols.std(axis=0), dtype=torch.float32)
        return cls(mean=mean, std=std)

    def transform_tensor(self, m: torch.Tensor) -> torch.Tensor:
        """Standardize a macro state tensor of shape [9]. Returns float32 on same device."""
        return (m.float() - self.mean.to(m.device)) / self.std.to(m.device)

    def transform_df(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize macro columns in a DataFrame. Returns a copy with same index/columns."""
        cols = list(MACRO_COLUMNS)
        arr = torch.tensor(macro_df[cols].to_numpy(dtype="float32"))  # (N, 9)
        scaled = (arr - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)
        out = macro_df.copy()
        out[cols] = scaled.numpy()
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, path / "macro_scaler.pt")

    @classmethod
    def load(cls, path: Path) -> "MacroScaler":
        d = torch.load(Path(path) / "macro_scaler.pt", map_location="cpu")
        return cls(mean=d["mean"], std=d["std"])
