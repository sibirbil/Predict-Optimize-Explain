from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch


class FeatureScaler:
    """Per-column standardizer for the fully constructed interaction matrices.

    Design rules (enforced):
    - Fit only on train split. Call FeatureScaler.fit(X_train).
    - Zero-variance columns are clamped to std=1e-8 (not dropped).
      The post-clamp std values are what get saved and used in transforms.
    - Feature names are saved alongside mean/std and verified at transform time.
      Any column-order mismatch between training and inference raises a hard error.
    - The scaler artifact travels with the model bundle (not in a global path).
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        # Clamp to avoid division by zero; save the stabilized values that are
        # ACTUALLY used in transformation so audits inspect real behaviour.
        raw_std = np.asarray(std, dtype=np.float32)
        self.std = np.where(raw_std < 1e-8, np.float32(1e-8), raw_std)
        self.feature_names: tuple[str, ...] | None = feature_names

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "FeatureScaler":
        """Fit on a DataFrame. Column names are saved for order verification."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureScaler.fit() requires a pd.DataFrame with named columns.")
        feature_names = tuple(str(c) for c in X.columns)
        mean = np.nanmean(X.to_numpy(), axis=0)
        std = np.nanstd(X.to_numpy(), axis=0)
        return cls(mean=mean, std=std, feature_names=feature_names)

    def _verify_column_order(self, columns: list[str] | tuple[str, ...]) -> None:
        """Hard check: column order must exactly match what was fit on."""
        if self.feature_names is None:
            return  # loaded from legacy artifact without names — skip check
        cols = tuple(str(c) for c in columns)
        if cols != self.feature_names:
            n_fit = len(self.feature_names)
            n_now = len(cols)
            if n_fit != n_now:
                raise ValueError(
                    f"FeatureScaler column count mismatch: fit on {n_fit} columns, "
                    f"got {n_now} columns at transform time."
                )
            first_bad = next(
                (i for i, (a, b) in enumerate(zip(self.feature_names, cols)) if a != b), None
            )
            raise ValueError(
                f"FeatureScaler column order mismatch at index {first_bad}: "
                f"fit on {self.feature_names[first_bad]!r}, "
                f"got {cols[first_bad]!r} at transform time. "
                "Ensure the feature construction pipeline produces columns in the same order "
                "as training."
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame, verifying column order against training."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureScaler.transform() requires a pd.DataFrame.")
        self._verify_column_order(list(X.columns))
        X_scaled = (X.to_numpy(dtype=np.float32) - self.mean) / self.std
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def transform_tensor(
        self,
        X: torch.Tensor,
        feature_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """Transform a float32 tensor (n_assets, n_features).

        If feature_names are provided they are verified against what was fit on.
        """
        if feature_names is not None:
            self._verify_column_order(feature_names)
        mean_t = torch.from_numpy(self.mean).to(X.device)
        std_t = torch.from_numpy(self.std).to(X.device)
        return (X - mean_t) / std_t

    def save(self, path: Path) -> None:
        """Save scaler artifacts. The stabilized std (post-clamp) is what gets saved."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "feature_mean.npy", self.mean)
        np.save(path / "feature_std_stabilized.npy", self.std)  # post-clamp values
        if self.feature_names is not None:
            np.save(path / "feature_names.npy", np.array(self.feature_names, dtype=object))

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """Load a saved FeatureScaler. Accepts both old (feature_std.npy) and new artifacts."""
        path = Path(path)
        if not (path / "feature_mean.npy").exists():
            raise FileNotFoundError(f"FeatureScaler artifact missing at: {path}")
        mean = np.load(path / "feature_mean.npy")
        # Support both old name ('feature_std.npy') and new name ('feature_std_stabilized.npy')
        if (path / "feature_std_stabilized.npy").exists():
            std = np.load(path / "feature_std_stabilized.npy")
        elif (path / "feature_std.npy").exists():
            std = np.load(path / "feature_std.npy")
        else:
            raise FileNotFoundError(f"FeatureScaler std artifact missing at: {path}")
        feature_names: tuple[str, ...] | None = None
        if (path / "feature_names.npy").exists():
            feature_names = tuple(str(n) for n in np.load(path / "feature_names.npy", allow_pickle=True).tolist())
        # Pass already-clamped values directly; constructor will clamp again but effect is idempotent
        return cls(mean=mean, std=std, feature_names=feature_names)
