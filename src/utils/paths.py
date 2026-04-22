from __future__ import annotations

from pathlib import Path


def batuhan_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return batuhan_root().parent


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate

