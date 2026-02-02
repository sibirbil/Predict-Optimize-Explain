"""
E2E Month Cache Dataset Module.

PyTorch Dataset for loading pre-computed monthly cache.
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.utils.data

try:
    from ..utils.io import load_json
except ImportError:
    from utils.io import load_json


class MonthCacheDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for PAO month-level cache.

    Each sample represents one month with:
    - X: Feature matrix (n_assets, n_features)
    - y: Realized returns (n_assets,)
    - Sigma_factor: Cholesky-like factor for risk (n_assets, n_assets)
    - sigma_vol: Volatility vector (n_assets,)
    - fnn_preds: FNN predictions (n_assets,)
    - yyyymm: Month identifier
    - assets: Asset identifiers

    Usage:
        cache_dir = Path('outputs/pao/cache/topk_50/train')
        ds = MonthCacheDataset(cache_dir)
        sample = ds[0]  # Get first month
        X = sample['X']  # torch.Tensor of shape (50, n_features)
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize dataset from cache directory.

        Args:
            cache_dir: Path to cache directory with manifest.json and .npz files

        Raises:
            RuntimeError: If manifest.json missing or no .npz files found
        """
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"

        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest.json in {cache_dir}")

        manifest = load_json(manifest_path)
        self.topk = int(manifest["topk"])
        self.feature_dim = int(manifest["feature_dim"])

        # Find all month files
        self.files = sorted([p for p in self.cache_dir.glob("*.npz") if p.is_file()])
        if not self.files:
            raise RuntimeError(f"No .npz files found in {cache_dir}")

    def __len__(self) -> int:
        """Return number of months in cache."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load month data from cache.

        Args:
            idx: Month index (0 to len-1)

        Returns:
            Dictionary with:
            - yyyymm: Month identifier (int)
            - assets: Asset identifiers (numpy array)
            - X: Features (torch.Tensor, float32, shape (n_assets, n_features))
            - y: Realized returns (torch.Tensor, float32, shape (n_assets,))
            - Sigma_factor: Risk factor matrix (torch.Tensor, float32, shape (n_assets, n_assets))
            - sigma_vol: Volatility vector (torch.Tensor, float32, shape (n_assets,))
            - fnn_preds: FNN predictions (torch.Tensor, float32, shape (n_assets,))
        """
        d = np.load(self.files[idx], allow_pickle=True)
        return {
            "yyyymm": int(d["yyyymm"]),
            "assets": d["assets"],
            "X": torch.tensor(d["X"], dtype=torch.float32),
            "y": torch.tensor(d["y"], dtype=torch.float32),
            "Sigma_factor": torch.tensor(d["Sigma_factor"], dtype=torch.float32),
            "sigma_vol": torch.tensor(d["sigma_vol"], dtype=torch.float32),
            "fnn_preds": torch.tensor(d["fnn_preds"], dtype=torch.float32),
        }
