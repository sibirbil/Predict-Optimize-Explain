from __future__ import annotations

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = WORKSPACE_ROOT / "universe_500_v21April_stdz_macro"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
