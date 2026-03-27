"""
pao_model_defs.py

Model definitions and loaders for the E2E Predict-and-Optimize pipeline.

Contents:
  - AssetPricingFNN + load_fnn_from_dir (for selection, optional)
  - DifferentiableMVOLayer / DifferentiableRobustMVOLayer (cvxpylayers; solver forced)
  - PAOScoreNetwork + PAOPortfolioModel
  - load_pao_model_from_run (config.json + best_state.pt)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

# ---- Optional deps: CVXPY + CVXPYLayers
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _HAVE_CVXPYLAYERS = True
    _CVXPY_IMPORT_ERROR = None
except Exception as e:
    cp = None  # type: ignore
    CvxpyLayer = None  # type: ignore
    _HAVE_CVXPYLAYERS = False
    _CVXPY_IMPORT_ERROR = e


# =============================================================================
# 1) PTO-style FNN definition + loader (optional)
# =============================================================================
class AssetPricingFNN(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(int(input_dim), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x).squeeze()


def load_fnn_from_dir(
    load_dir: Union[str, Path],
    map_location: str = "cpu"
) -> Tuple[AssetPricingFNN, List[str], Dict[str, Any]]:
    load_dir = Path(load_dir)
    cfg_path = load_dir / "model_config.json"
    cols_path = load_dir / "feature_columns.json"
    state_path = load_dir / "state_dict.pt"

    for p in (cfg_path, cols_path, state_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    model_cfg = json.loads(cfg_path.read_text())
    feature_cols = json.loads(cols_path.read_text())

    state = torch.load(state_path, map_location=map_location)
    model = AssetPricingFNN(
        input_dim=int(model_cfg["input_dim"]),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.5)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, feature_cols, model_cfg


# =============================================================================
# 2) Differentiable optimization layers (cvxpylayers)
# =============================================================================
class DifferentiableMVOLayer(nn.Module):
    """
    max mu'w - (lambd/2) w'Σw  s.t. sum(w)=1, w>=0
    risk implemented as ||U w||^2 where U^T U = Σ
    """
    def __init__(self, n_assets: int, lambd: float):
        super().__init__()
        if not _HAVE_CVXPYLAYERS:
            raise ImportError(f"cvxpy/cvxpylayers not available: {_CVXPY_IMPORT_ERROR}")

        n = int(n_assets)
        w = cp.Variable(n, nonneg=True)
        mu = cp.Parameter(n)
        U = cp.Parameter((n, n))

        risk = cp.sum_squares(U @ w)
        obj = cp.Minimize(-mu @ w + (float(lambd) / 2.0) * risk)
        cons = [cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, parameters=[mu, U], variables=[w])
        self.n_assets = n

    def forward(self, mu: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        solver_try = []
        if hasattr(cp, "ECOS") and ("ECOS" in cp.installed_solvers()):
            solver_try.append({"solver": cp.ECOS, "ignore_dpp": True})
        if hasattr(cp, "SCS") and ("SCS" in cp.installed_solvers()):
            solver_try.append({"solver": cp.SCS, "max_iters": 5000, "eps": 1e-4, "verbose": False, "ignore_dpp": True})
        if not solver_try:
            solver_try.append({"max_iters": 5000, "eps": 1e-4, "verbose": False, "ignore_dpp": True})

        last_err = None
        for sargs in solver_try:
            try:
                (w,) = self.layer(mu.double(), U.double(), solver_args=sargs)
                w = w.float()
                w = torch.clamp(w, min=0.0)
                return w / (w.sum() + 1e-12)
            except Exception as e:
                last_err = e

        print(f"[MVO solver fail] {last_err}")
        return torch.ones(self.n_assets, device=mu.device) / self.n_assets


class DifferentiableRobustMVOLayer(nn.Module):
    """
    max mu'w - kappa||A w||_2 - (lambd/2) w'Σw  s.t. sum(w)=1, w>=0
    """
    def __init__(self, n_assets: int, lambd: float, kappa: float):
        super().__init__()
        if not _HAVE_CVXPYLAYERS:
            raise ImportError(f"cvxpy/cvxpylayers not available: {_CVXPY_IMPORT_ERROR}")

        n = int(n_assets)
        w = cp.Variable(n, nonneg=True)
        mu = cp.Parameter(n)
        U = cp.Parameter((n, n))
        A = cp.Parameter((n, n))

        risk = cp.sum_squares(U @ w)
        obj = cp.Minimize(-mu @ w + float(kappa) * cp.norm(A @ w, 2) + (float(lambd) / 2.0) * risk)
        cons = [cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, parameters=[mu, U, A], variables=[w])
        self.n_assets = n

    def forward(self, mu: torch.Tensor, U: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        solver_try = []
        if hasattr(cp, "ECOS") and ("ECOS" in cp.installed_solvers()):
            solver_try.append({"solver": cp.ECOS, "ignore_dpp": True})
        if hasattr(cp, "SCS") and ("SCS" in cp.installed_solvers()):
            solver_try.append({"solver": cp.SCS, "max_iters": 8000, "eps": 1e-4, "verbose": False, "ignore_dpp": True})
        if not solver_try:
            solver_try.append({"max_iters": 8000, "eps": 1e-4, "verbose": False, "ignore_dpp": True})

        last_err = None
        for sargs in solver_try:
            try:
                (w,) = self.layer(mu.double(), U.double(), A.double(), solver_args=sargs)
                w = w.float()
                w = torch.clamp(w, min=0.0)
                return w / (w.sum() + 1e-12)
            except Exception as e:
                last_err = e

        print(f"[Robust MVO solver fail] {last_err}")
        return torch.ones(self.n_assets, device=mu.device) / self.n_assets


# =============================================================================
# 3) Predictor + E2E portfolio model
# =============================================================================
class PAOScoreNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            h_i = int(h)
            layers += [
                nn.Linear(prev, h_i),
                nn.BatchNorm1d(h_i),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            ]
            prev = h_i
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PAOPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        lambd: float,
        kappa: float,
        omega_mode: str,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        mu_transform: str,
        mu_scale: float,
        mu_cap: float,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.lambd = float(lambd)
        self.kappa = float(kappa)
        self.omega_mode = str(omega_mode)

        self.mu_transform = str(mu_transform)
        self.mu_scale = float(mu_scale)
        self.mu_cap = float(mu_cap)

        self.predictor = PAOScoreNetwork(int(input_dim), tuple(hidden_dims), float(dropout))

        if self.kappa > 0:
            self.opt = DifferentiableRobustMVOLayer(self.n_assets, self.lambd, self.kappa)
        else:
            self.opt = DifferentiableMVOLayer(self.n_assets, self.lambd)

    def _build_A(self, sigma_vol: torch.Tensor) -> torch.Tensor:
        if self.omega_mode == "identity":
            return torch.eye(self.n_assets, dtype=torch.float32, device=sigma_vol.device)
        if self.omega_mode == "diagSigma":
            return torch.diag(sigma_vol)
        raise ValueError(f"Unknown omega_mode={self.omega_mode}")

    def _transform_mu(self, mu_raw: torch.Tensor) -> torch.Tensor:
        if self.mu_transform == "raw":
            return mu_raw
        if self.mu_transform == "zscore":
            m = mu_raw.mean()
            s = mu_raw.std(unbiased=True) + 1e-12
            return self.mu_scale * ((mu_raw - m) / s)
        if self.mu_transform == "tanh_cap":
            return self.mu_cap * torch.tanh(mu_raw)
        raise ValueError(f"Unknown mu_transform={self.mu_transform}")

    def forward(self, X: torch.Tensor, Sigma_factor: torch.Tensor, sigma_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_raw = self.predictor(X)
        mu = self._transform_mu(mu_raw)

        if self.kappa > 0:
            A = self._build_A(sigma_vol)
            w = self.opt(mu, Sigma_factor, A)
        else:
            w = self.opt(mu, Sigma_factor)

        return w, mu_raw


# =============================================================================
# 4) Run loader
# =============================================================================
def load_pao_model_from_run(
    run_dir: Union[str, Path],
    map_location: str = "cpu"
) -> Tuple[PAOPortfolioModel, Dict[str, Any]]:
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    state_path = run_dir / "best_state.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing: {state_path}")

    cfg = json.loads(cfg_path.read_text())

    hidden_dims = tuple(int(x) for x in cfg["hidden_dims"])
    dropout = float(cfg["dropout"])

    model = PAOPortfolioModel(
        input_dim=int(cfg["input_dim"]),
        n_assets=int(cfg["topk"]),
        lambd=float(cfg["gamma"]),
        kappa=float(cfg["kappa"]),
        omega_mode=str(cfg["omega_mode"]),
        hidden_dims=hidden_dims,
        dropout=dropout,
        mu_transform=str(cfg["mu_transform"]),
        mu_scale=float(cfg["mu_scale"]),
        mu_cap=float(cfg["mu_cap"]),
    )

    state = torch.load(state_path, map_location=map_location)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


__all__ = [
    "AssetPricingFNN",
    "load_fnn_from_dir",
    "DifferentiableMVOLayer",
    "DifferentiableRobustMVOLayer",
    "PAOScoreNetwork",
    "PAOPortfolioModel",
    "load_pao_model_from_run",
]
