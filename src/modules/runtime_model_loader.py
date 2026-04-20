from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


def _normalize_omega_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "diag_sigma":
        return "diagSigma"
    if normalized == "identity":
        return "identity"
    return str(value)


class ExactLayerPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float, batch_norm: bool) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(h) for h in hidden_dims]]
        blocks = []
        for idx in range(len(hidden_dims)):
            layers = [nn.Linear(dims[idx], dims[idx + 1])]
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[idx + 1]))
            layers.extend([nn.ReLU(), nn.Dropout(float(dropout))])
            blocks.append(nn.Sequential(*layers))
        self.layer1, self.layer2, self.layer3 = blocks
        self.output = nn.Linear(int(hidden_dims[-1]), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x).squeeze(-1)


class RuntimeExactLayerModel:
    """Small duck-typed wrapper for src.modules.probe_eval."""

    def __init__(
        self,
        predictor: nn.Module,
        lambd: float,
        kappa: float,
        omega_mode: str,
        mu_transform: str = "raw",
        mu_scale: float = 1.0,
        mu_cap: float = 1.0,
    ) -> None:
        self.predictor = predictor
        self.lambd = float(lambd)
        self.kappa = float(kappa)
        self.omega_mode = _normalize_omega_mode(omega_mode)
        self.mu_transform = str(mu_transform)
        self.mu_scale = float(mu_scale)
        self.mu_cap = float(mu_cap)

    def eval(self) -> "RuntimeExactLayerModel":
        self.predictor.eval()
        return self

    def _transform_mu(self, mu_raw: torch.Tensor) -> torch.Tensor:
        if self.mu_transform == "raw":
            return mu_raw
        if self.mu_transform == "zscore":
            mu_mean = mu_raw.mean()
            mu_std = mu_raw.std(unbiased=True) + 1e-12
            return self.mu_scale * ((mu_raw - mu_mean) / mu_std)
        if self.mu_transform == "tanh_cap":
            return self.mu_cap * torch.tanh(mu_raw)
        raise ValueError(f"Unknown mu_transform={self.mu_transform}")


def _load_selected_candidate(run_dir: Path) -> Dict[str, Any]:
    candidate_path = run_dir / "selected_candidate.json"
    if candidate_path.exists():
        return json.loads(candidate_path.read_text(encoding="utf-8"))

    validation_path = run_dir / "validation_selection_summary.json"
    if validation_path.exists():
        return json.loads(validation_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"Missing selected candidate metadata in {run_dir}. "
        "Expected selected_candidate.json or validation_selection_summary.json."
    )


def load_exact_e2e_model_from_run(
    run_dir: Union[str, Path],
    map_location: str = "cpu",
) -> Tuple[RuntimeExactLayerModel, Dict[str, Any]]:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing: {checkpoint_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint

    predictor = ExactLayerPredictor(
        input_dim=int(checkpoint.get("input_dim", 1400) if isinstance(checkpoint, dict) else 1400),
        hidden_dims=tuple(int(h) for h in config["hidden_dims"]),
        dropout=float(config["dropout"]),
        batch_norm=bool(config.get("batch_norm", True)),
    )
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()

    selected_candidate = _load_selected_candidate(run_dir)
    model = RuntimeExactLayerModel(
        predictor=predictor,
        lambd=float(selected_candidate["lambda"]),
        kappa=float(selected_candidate["kappa"]),
        omega_mode=str(selected_candidate["omega_mode"]),
        # Locked exact-layer artifacts do not carry the old PAO mu=zscore contract.
        # Runtime inference uses raw model outputs before the robust optimizer.
        mu_transform="raw",
        mu_scale=1.0,
        mu_cap=1.0,
    )
    merged_config = dict(config)
    merged_config.update(
        {
            "lambda": float(selected_candidate["lambda"]),
            "kappa": float(selected_candidate["kappa"]),
            "selected_candidate": selected_candidate,
            "run_dir": str(run_dir),
        }
    )
    return model, merged_config


__all__ = ["RuntimeExactLayerModel", "load_exact_e2e_model_from_run"]
