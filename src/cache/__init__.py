"""
E2E Cache System.

Month-level caching for PAO training with pre-computed:
- Universe selection (top-k assets)
- FNN predictions
- EWMA covariance matrices
- Risk factors
"""
from .builder import (
    expected_cache_manifest,
    manifest_matches,
    get_universe_sizes,
    build_month_cache,
)
from .dataset import MonthCacheDataset

__all__ = [
    "expected_cache_manifest",
    "manifest_matches",
    "get_universe_sizes",
    "build_month_cache",
    "MonthCacheDataset",
]
