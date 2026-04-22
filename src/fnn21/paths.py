from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APRIL_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = APRIL_ROOT / "universe_500_v21April_stdz_macro"
ARTIFACTS_ROOT = APRIL_ROOT / "artifacts"
FNN_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "fnn"
