"""Nearest-neighbor lookup from generated macro states to historical yyyymm.

Three distance metrics are computed in parallel so the manuscript narrative
can contrast them:

- ``nn_var1_*``: Mahalanobis using the VAR(1) innovation covariance inverse
  (``Sigma_inv`` shipped under ``VAR1Regularizer/artifacts/var1_standardized``).
  This is the same metric driving the regularizer and the chi-square tail
  diagnostic, so distances here are comparable to ``mah_dist`` / ``mah_chi2_*``.
- ``nn_hist_*``: Mahalanobis using the inverse covariance of the standardized
  historical panel (full sample, unconditional). Captures macro cross-
  correlations that plain Euclidean ignores.
- ``nn_eucl_*``: plain L2 in z-score space. Simplest reading; no economic
  distance structure beyond per-variable rescaling.

Inputs are expected in **standardized** units (``z = (raw - macro_mean) /
macro_std``). The same MacroScaler used by the live pipeline is the canonical
source of those constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class HistoricalNNIndex:
    """Pre-built index over the historical macro panel.

    Holds the z-scored historical macro vectors plus the precomputed inverse
    covariances for the three metrics. Cheap to keep around and call once per
    saved frame.
    """

    def __init__(
        self,
        historical_std: pd.DataFrame,
        macro_columns: Sequence[str],
        sigma_inv_var1: np.ndarray,
        cov_ridge: float = 1e-8,
    ) -> None:
        if "yyyymm" not in historical_std.columns:
            raise ValueError("historical_std must include a 'yyyymm' column")
        for col in macro_columns:
            if col not in historical_std.columns:
                raise ValueError(f"historical_std missing macro column {col!r}")

        self.macro_columns = list(macro_columns)
        self.yyyymm = historical_std["yyyymm"].astype(int).to_numpy()
        self.hist_z = historical_std[self.macro_columns].to_numpy(dtype=np.float64)

        self.VI_var1 = np.asarray(sigma_inv_var1, dtype=np.float64)
        if self.VI_var1.shape != (len(self.macro_columns), len(self.macro_columns)):
            raise ValueError(
                f"sigma_inv_var1 has shape {self.VI_var1.shape}, "
                f"expected ({len(self.macro_columns)}, {len(self.macro_columns)})"
            )

        cov_hist = np.cov(self.hist_z.T)
        cov_hist = cov_hist + cov_ridge * np.eye(cov_hist.shape[0])
        self.VI_hist = np.linalg.inv(cov_hist)

    def attach(self, states_std: np.ndarray | pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Return a per-row frame of top-k yyyymm and distances under each metric.

        Output columns: ``nn_{var1,hist,eucl}_yyyymm_{1..k}`` and
        ``nn_{var1,hist,eucl}_dist_{1..k}``, sorted ascending by distance.
        """
        if isinstance(states_std, pd.DataFrame):
            states = states_std[self.macro_columns].to_numpy(dtype=np.float64)
        else:
            states = np.asarray(states_std, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        if states.shape[1] != len(self.macro_columns):
            raise ValueError(
                f"states_std has {states.shape[1]} columns, expected {len(self.macro_columns)}"
            )
        if k < 1 or k > self.hist_z.shape[0]:
            raise ValueError(f"k={k} out of bounds [1, {self.hist_z.shape[0]}]")

        d_var1 = cdist(states, self.hist_z, metric="mahalanobis", VI=self.VI_var1)
        d_hist = cdist(states, self.hist_z, metric="mahalanobis", VI=self.VI_hist)
        d_eucl = cdist(states, self.hist_z, metric="euclidean")

        out: dict[str, np.ndarray] = {}
        row_idx = np.arange(d_var1.shape[0])[:, None]
        for tag, d in (("var1", d_var1), ("hist", d_hist), ("eucl", d_eucl)):
            top_unsorted = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(d[row_idx, top_unsorted], axis=1)
            top = top_unsorted[row_idx, order]
            yy = self.yyyymm[top]
            dd = d[row_idx, top]
            for i in range(k):
                out[f"nn_{tag}_yyyymm_{i + 1}"] = yy[:, i].astype(int)
                out[f"nn_{tag}_dist_{i + 1}"] = dd[:, i]
        return pd.DataFrame(out)


def build_index_from_runtime(
    macro_columns: Sequence[str] | None = None,
    macro_panel_path: Path | str | None = None,
    scaler_dir: Path | str | None = None,
    sigma_inv_path: Path | str | None = None,
) -> HistoricalNNIndex:
    """Build the default index using the runtime artifacts.

    Resolves paths relative to the repo root (two parents above this file).
    """
    from src.data.macro_scaler import MacroScaler

    root = Path(__file__).resolve().parents[2]
    macro_panel_path = Path(macro_panel_path) if macro_panel_path is not None else root / "runtime_universe500" / "data" / "macro_final.parquet"
    scaler_dir = Path(scaler_dir) if scaler_dir is not None else root / "runtime_universe500" / "scaler"
    sigma_inv_path = Path(sigma_inv_path) if sigma_inv_path is not None else root / "VAR1Regularizer" / "artifacts" / "var1_standardized" / "innovation_covariance_inv.npy"

    if macro_columns is None:
        from src.modules.runtime_regime import MACRO_ORDER
        macro_columns = list(MACRO_ORDER)
    macro_columns = list(macro_columns)

    macro_final = pd.read_parquet(macro_panel_path)
    macro_final["yyyymm"] = macro_final["yyyymm"].astype(int)
    raw = macro_final[macro_columns].to_numpy(dtype=np.float64)

    scaler = MacroScaler.load(scaler_dir)
    mean = scaler.mean.detach().cpu().numpy().astype(np.float64)
    std = scaler.std.detach().cpu().numpy().astype(np.float64)
    if mean.shape != (len(macro_columns),):
        raise ValueError(f"scaler mean shape {mean.shape} mismatches {len(macro_columns)} macro columns")

    z = (raw - mean[None, :]) / std[None, :]
    historical_std = pd.DataFrame(z, columns=macro_columns)
    historical_std.insert(0, "yyyymm", macro_final["yyyymm"].to_numpy())

    sigma_inv = np.load(sigma_inv_path)
    return HistoricalNNIndex(
        historical_std=historical_std,
        macro_columns=macro_columns,
        sigma_inv_var1=sigma_inv,
    )


def attach_nearest_neighbors(
    states_std: np.ndarray | pd.DataFrame,
    historical_std: pd.DataFrame,
    macro_columns: Sequence[str],
    sigma_inv_var1: np.ndarray,
    k: int = 3,
) -> pd.DataFrame:
    """One-shot convenience for callers that already hold the inputs."""
    index = HistoricalNNIndex(
        historical_std=historical_std,
        macro_columns=macro_columns,
        sigma_inv_var1=sigma_inv_var1,
    )
    return index.attach(states_std, k=k)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    index = build_index_from_runtime()
    n_hist = index.hist_z.shape[0]
    nn = index.attach(index.hist_z, k=1)

    self_var1 = (nn["nn_var1_yyyymm_1"].to_numpy() == index.yyyymm).all()
    self_hist = (nn["nn_hist_yyyymm_1"].to_numpy() == index.yyyymm).all()
    self_eucl = (nn["nn_eucl_yyyymm_1"].to_numpy() == index.yyyymm).all()
    zero_var1 = bool((nn["nn_var1_dist_1"].abs() < 1e-6).all())
    zero_hist = bool((nn["nn_hist_dist_1"].abs() < 1e-6).all())
    zero_eucl = bool((nn["nn_eucl_dist_1"].abs() < 1e-6).all())

    assert self_var1, "VAR(1) Mahalanobis self-NN failed"
    assert self_hist, "Historical Mahalanobis self-NN failed"
    assert self_eucl, "Euclidean self-NN failed"
    assert zero_var1 and zero_hist and zero_eucl, "Self-distance not ~0 under some metric"
    print(f"OK: self-NN sanity passed across {n_hist} historical months under all three metrics.")
