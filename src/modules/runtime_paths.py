from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "runtime_universe500"
DATA_DIR = RUNTIME_ROOT / "data"
SCALER_DIR = RUNTIME_ROOT / "scaler"
MODELS_DIR = RUNTIME_ROOT / "models"
REGIME_DIR = RUNTIME_ROOT / "regime"


def runtime_path(*parts: str) -> Path:
    return RUNTIME_ROOT.joinpath(*parts)


__all__ = [
    "REPO_ROOT",
    "RUNTIME_ROOT",
    "DATA_DIR",
    "SCALER_DIR",
    "MODELS_DIR",
    "REGIME_DIR",
    "runtime_path",
]
