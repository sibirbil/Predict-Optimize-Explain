import argparse
import datetime
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from end2endportfolio.src import langevin
from src.data.macro_scaler import MacroScaler
from src.modules.probe_eval import (
    A,
    Sigma_inv,
    c,
    AllocationPipeline,
    G_decision_fragility_function,
    G_policy_disagreement_function,
    G_training_paradigm_gap_function,
    evaluate_policy_decision,
    mahalonobis_reg,
)
from src.modules.runtime_components import FeatureScenarioBuilder
from src.modules.runtime_model_loader import (
    ExactLayerPredictor,
    RuntimeExactLayerModel,
    load_exact_e2e_model_from_run,
)
from src.modules.runtime_regime import MACRO_ORDER, regime_classifier
from src.modules.nn_historical import build_index_from_runtime as _build_nn_index
from src.utils.plotting import (
    macro_density_grid,
    macro_pair_grid,
    macro_shift_bar,
    scenario5_story_figure,
)


warnings.filterwarnings(
    "ignore",
    message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.",
    category=UserWarning,
)


RUNTIME = ROOT / "runtime_universe500"
DATA = RUNTIME / "data"
MODEL_ROOT = RUNTIME / "models"

ASSET_SIZE = 50
DEFAULT_DATE = 202004
DEFAULT_PROBE = "training_gap"
DEFAULT_MODEL = "locked_e2e"
DEFAULT_MODEL_A = "locked_e2e"
DEFAULT_MODEL_B = "standardized_pto"
DEFAULT_N_SEEDS = 20
DEFAULT_N_STEPS = 500
DEFAULT_BASE_SEED = 20260424
DEFAULT_L2REG = 0.3
DEFAULT_ETA = 0.09
DEFAULT_BETA = 10.0
DEFAULT_RANDOM_START_SCALE = 0.005
DIAGNOSTIC_BURN_IN_FRACTION = 0.50
DIAGNOSTIC_THIN = 5
DECISION_METRICS = ("return", "volatility", "sharpe", "entropy", "hhi", "effective_n", "max_weight", "top10_weight")

BOX_BARRIER_MARGIN_FRACTION = 0.05
BOX_BARRIER_SHARPNESS = 25.0
BOX_BARRIER_WEIGHT = 25.0
STRONG_SOFT_MULTIPLIER = 5.0

MODEL_ALIASES = {
    "locked_e2e": "locked_e2e",
    "standardized_pto": "standardized_pto",
    "summer_child": "summer_child",
    "winter_wolf": "winter_wolf",
}


def _load_runtime_tables():
    meta_all = pd.read_parquet(DATA / "metadata.parquet")
    meta_test = pd.read_parquet(DATA / "metadata_test.parquet")
    x_test = pd.read_parquet(DATA / "X_test.parquet")
    meta_all["yyyymm"] = meta_all["yyyymm"].astype(int)
    meta_test["yyyymm"] = meta_test["yyyymm"].astype(int)
    feature_names = tuple(np.load(DATA / "feature_names.npy", allow_pickle=True).tolist())
    firm_feature_names = tuple(np.load(DATA / "firm_feature_names.npy", allow_pickle=True).tolist())
    macro_predictors = tuple(np.load(DATA / "macro_predictors.npy", allow_pickle=True).tolist())
    macro_final = pd.read_parquet(DATA / "macro_final.parquet")
    macro_final["yyyymm"] = macro_final["yyyymm"].astype(int)
    return meta_all, meta_test, x_test, feature_names, firm_feature_names, macro_predictors, macro_final


def _ewma_covariance(
    returns_matrix: np.ndarray,
    ewma_lambda: float = 0.94,
    shrinkage: float = 0.10,
    ridge: float = 1e-6,
) -> np.ndarray:
    from src.utils.helper_functions import make_psd_np

    n = returns_matrix.shape[0]
    exp = np.arange(n - 1, -1, -1)
    w = (1.0 - ewma_lambda) * (ewma_lambda ** exp)
    w /= w.sum()
    mu = np.sum(returns_matrix * w[:, None], axis=0)
    centered = returns_matrix - mu
    sigma = centered.T @ (centered * w[:, None])
    sigma = 0.5 * (sigma + sigma.T)
    diag = np.diag(np.diag(sigma))
    sigma = (1.0 - shrinkage) * sigma + shrinkage * diag + ridge * np.eye(sigma.shape[0])
    sigma = make_psd_np(sigma)
    return 0.5 * (sigma + sigma.T)


def construct_C2_21a(
    date: int,
    meta_all: pd.DataFrame,
    meta_test: pd.DataFrame,
    x_test: pd.DataFrame,
    feature_names: tuple[str, ...],
    firm_feature_names: tuple[str, ...],
    macro_predictors: tuple[str, ...],
    K: Optional[int] = ASSET_SIZE,
):
    from src.utils.dates import shift_yyyymm

    lookback = 60
    hist = meta_all[
        meta_all["yyyymm"].between(shift_yyyymm(date, -lookback), shift_yyyymm(date, -1))
    ].groupby("permno")["excess_ret"]
    valid_today = meta_test.loc[meta_test["yyyymm"] == date, "permno"]
    candidates = (
        hist.mean()
        .where(hist.count() == lookback)
        .dropna()
        .loc[lambda s: s.index.isin(valid_today)]
        .sort_values(ascending=False, na_position="last")
    )
    if K is not None:
        candidates = candidates.head(K)
    permnos = candidates.index.tolist()
    if not permnos:
        raise ValueError(f"construct_C2_21a: no valid assets found for {date}.")
    if K is not None and len(permnos) != K:
        raise ValueError(f"construct_C2_21a: got {len(permnos)} assets for {date}, expected {K}.")

    hist_wide = (
        meta_all[
            (meta_all["yyyymm"] >= shift_yyyymm(date, -lookback))
            & (meta_all["yyyymm"] <= shift_yyyymm(date, -1))
            & (meta_all["permno"].isin(permnos))
        ]
        .pivot(index="yyyymm", columns="permno", values="excess_ret")
        .sort_index()[permnos]
        .dropna(axis=1, how="any")
    )
    sigma = _ewma_covariance(hist_wide.to_numpy(dtype=float))
    Sigma = torch.tensor(sigma.astype(np.float32), dtype=torch.float32)

    today_rows = meta_test[(meta_test["yyyymm"] == date) & (meta_test["permno"].isin(permnos))]
    today_rows = today_rows.drop_duplicates("permno")
    perm_to_row = {int(row["permno"]): idx for idx, row in today_rows.iterrows()}
    orig_indices = [perm_to_row[p] for p in permnos]
    feature_frame = x_test.loc[orig_indices, list(firm_feature_names)].copy()
    feature_frame.index = range(len(feature_frame))

    C_t = FeatureScenarioBuilder(
        feature_frame=feature_frame,
        feature_names=feature_names,
        macro_predictors=macro_predictors,
        scaler=None,
        macro_scaler=None,
    )
    rets_t = torch.tensor(
        today_rows.set_index("permno").loc[permnos, "excess_ret"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    return Sigma, C_t, rets_t, permnos


def load_standardized_pto_model(run_dir: Path) -> tuple[RuntimeExactLayerModel, dict]:
    backbone_dir = MODEL_ROOT / "standardized_backbone"
    backbone_cfg = json.loads((backbone_dir / "config.json").read_text(encoding="utf-8"))
    robust_cfg = json.loads((run_dir / "selected_robust_config.json").read_text(encoding="utf-8"))
    state_dict = torch.load(backbone_dir / "state_dict.pt", map_location="cpu")
    input_dim = int(state_dict["layer1.0.weight"].shape[1])
    predictor = ExactLayerPredictor(
        input_dim=input_dim,
        hidden_dims=tuple(int(h) for h in backbone_cfg["hidden_dims"]),
        dropout=float(backbone_cfg["dropout"]),
        batch_norm=bool(backbone_cfg.get("batch_norm", True)),
    )
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()
    model = RuntimeExactLayerModel(
        predictor=predictor,
        lambd=float(robust_cfg["lambda"]),
        kappa=float(robust_cfg["kappa"]),
        omega_mode=str(robust_cfg["omega_mode"]),
        mu_transform="raw",
    )
    config = {
        **json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "lambda": float(robust_cfg["lambda"]),
        "kappa": float(robust_cfg["kappa"]),
        "omega_mode": str(robust_cfg["omega_mode"]),
        "selected_candidate": robust_cfg,
        "run_dir": str(run_dir),
        "backbone_run_dir": str(backbone_dir),
    }
    return model, config


def load_runtime_model(name: str) -> tuple[RuntimeExactLayerModel, dict]:
    canonical = MODEL_ALIASES.get(name, name)
    run_dir = MODEL_ROOT / canonical
    if canonical == "standardized_pto":
        return load_standardized_pto_model(run_dir)
    return load_exact_e2e_model_from_run(run_dir)


def combine_reg_fns(*reg_fns: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    active = [fn for fn in reg_fns if fn is not None]
    if not active:
        raise ValueError("combine_reg_fns requires at least one regularizer.")

    def _combined(m: torch.Tensor) -> torch.Tensor:
        total = active[0](m)
        for fn in active[1:]:
            total = total + fn(m)
        return total

    return _combined


def build_box_barrier_reg_fn(
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin_fraction: float = BOX_BARRIER_MARGIN_FRACTION,
    sharpness: float = BOX_BARRIER_SHARPNESS,
) -> Callable[[torch.Tensor], torch.Tensor]:
    span = torch.clamp(upper - lower, min=1e-6)
    margin = margin_fraction * span

    def _barrier(m: torch.Tensor) -> torch.Tensor:
        lower_pen = F.softplus(sharpness * ((lower + margin) - m)) / sharpness
        upper_pen = F.softplus(sharpness * (m - (upper - margin))) / sharpness
        scaled = (lower_pen / span).square() + (upper_pen / span).square()
        return scaled.sum()

    return _barrier


def build_base_reg_fn(mode: str, anchor: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == "var1":
        return mahalonobis_reg(A, c, Sigma_inv, anchor)
    if mode == "l2":
        def _l2(m: torch.Tensor) -> torch.Tensor:
            diff = m - anchor
            return diff @ diff

        return _l2
    raise ValueError(f"Unknown reg_mode={mode!r}")


def build_constraint_setup(
    reg_mode: str,
    constraint_mode: str,
    anchor: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, object]:
    base_reg_fn = build_base_reg_fn(reg_mode, anchor)
    clip_low = None
    clip_high = None
    if constraint_mode == "clip":
        reg_fn = base_reg_fn
        clip_low, clip_high = lower, upper
        description = f"{reg_mode} + hard box clip"
    elif constraint_mode == "none":
        reg_fn = base_reg_fn
        description = f"{reg_mode} + no box constraint"
    elif constraint_mode == "box_barrier":
        barrier_reg = build_box_barrier_reg_fn(lower, upper)

        def _weighted_barrier(m: torch.Tensor) -> torch.Tensor:
            return BOX_BARRIER_WEIGHT * barrier_reg(m)

        reg_fn = combine_reg_fns(base_reg_fn, _weighted_barrier)
        description = (
            f"{reg_mode} + smooth box barrier"
            f" (w={BOX_BARRIER_WEIGHT:g}, margin={BOX_BARRIER_MARGIN_FRACTION:.2f})"
        )
    elif constraint_mode == "strong_soft":
        def _stronger_reg(m: torch.Tensor) -> torch.Tensor:
            return STRONG_SOFT_MULTIPLIER * base_reg_fn(m)

        reg_fn = _stronger_reg
        description = f"{reg_mode} + strong soft regularization (x{STRONG_SOFT_MULTIPLIER:g})"
    else:
        raise ValueError(f"Unknown constraint_mode={constraint_mode!r}")
    return {
        "reg_fn": reg_fn,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "description": description,
    }


def l2_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    diff = x.detach().float() - anchor.detach().float()
    return float((diff @ diff).item())


def mahalanobis_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    diff = x.detach().float() - (A @ anchor.detach().float() + c)
    return float((diff @ (Sigma_inv @ diff)).item())


def anchor_mahalanobis_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    diff = x.detach().float() - anchor.detach().float()
    return float((diff @ (Sigma_inv @ diff)).item())


def mahalanobis_chi2_summary(mah2: float) -> dict[str, float]:
    statistic = max(float(mah2), 0.0)
    df = len(MACRO_ORDER)
    return {
        "mah_chi2_percentile": float(chi2.cdf(statistic, df=df)),
        "mah_chi2_tail": float(chi2.sf(statistic, df=df)),
    }


def box_violation_squared(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    x_detached = x.detach().float()
    lower_gap = torch.relu(lower.detach().float() - x_detached)
    upper_gap = torch.relu(x_detached - upper.detach().float())
    return float((lower_gap.square() + upper_gap.square()).sum().item())


def touches_box(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, atol: float = 1e-6) -> bool:
    x_detached = x.detach()
    on_lower = torch.any(torch.isclose(x_detached, lower.detach(), atol=atol, rtol=0.0))
    on_upper = torch.any(torch.isclose(x_detached, upper.detach(), atol=atol, rtol=0.0))
    return bool((on_lower or on_upper).item())


def effective_sample_size(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return float(n)
    centered = values - values.mean()
    variance = np.dot(centered, centered) / n
    if not np.isfinite(variance) or variance <= 1e-12:
        return float(n)
    rho_sum = 0.0
    max_lag = min(n - 1, max(1, n // 2))
    for lag in range(1, max_lag + 1):
        rho = np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance)
        if not np.isfinite(rho) or rho <= 0:
            break
        rho_sum += rho
    ess = n / (1.0 + 2.0 * rho_sum)
    return float(max(1.0, min(float(n), ess)))


def summarize_ess(traj: torch.Tensor, accepted_flags: list[bool]) -> dict[str, float]:
    values = traj.detach().cpu().numpy()
    raw_ess = np.asarray([effective_sample_size(values[:, idx]) for idx in range(values.shape[1])], dtype=np.float64)
    state_change_count = 1 + int(sum(bool(flag) for flag in accepted_flags))
    ess_values = np.minimum(raw_ess, float(state_change_count))
    return {
        "accepted_moves": float(sum(bool(flag) for flag in accepted_flags)),
        "state_change_count": float(state_change_count),
        "raw_ess_min": float(raw_ess.min()),
        "raw_ess_mean": float(raw_ess.mean()),
        "raw_ess_max": float(raw_ess.max()),
        "ess_min": float(ess_values.min()),
        "ess_mean": float(ess_values.mean()),
        "ess_max": float(ess_values.max()),
    }


class Scenario5Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.date = int(args.date)
        self.macro_scaler = MacroScaler.load(RUNTIME / "scaler")
        self.macro_mean = self.macro_scaler.mean
        self.macro_std = self.macro_scaler.std
        (
            self.meta_all,
            self.meta_test,
            self.x_test,
            self.feature_names,
            self.firm_feature_names,
            self.macro_predictors,
            self.macro_df,
        ) = _load_runtime_tables()
        self.macro_columns = list(MACRO_ORDER)
        self.Sigma, self.C_t, self.rets_t, self.permnos = construct_C2_21a(
            self.date,
            self.meta_all,
            self.meta_test,
            self.x_test,
            self.feature_names,
            self.firm_feature_names,
            self.macro_predictors,
            K=ASSET_SIZE,
        )
        raw_values = self.macro_df.loc[self.macro_df["yyyymm"] == self.date, self.macro_columns]
        if raw_values.empty:
            raise ValueError(f"No macro row found for date={self.date}.")
        self.m0_raw = torch.tensor(raw_values.to_numpy(dtype=np.float32, copy=True)[0], dtype=torch.float32)
        self.m0 = (self.m0_raw - self.macro_mean) / self.macro_std

        macro_arr = torch.tensor(self.macro_df[self.macro_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        macro_std_hist = macro_arr.std(dim=0)
        macro_std_panel = ((macro_arr - self.macro_mean) / self.macro_std).cpu().numpy()
        empirical_cov_inv = np.linalg.pinv(np.cov(macro_std_panel, rowvar=False))
        self.empirical_macro_cov_inv = torch.tensor(empirical_cov_inv.astype(np.float32), dtype=torch.float32)
        self.lower = (macro_arr.min(dim=0).values - self.macro_mean) / self.macro_std - macro_std_hist / self.macro_std
        self.upper = (macro_arr.max(dim=0).values - self.macro_mean) / self.macro_std + macro_std_hist / self.macro_std

        self.model_names = self._required_models()
        self.models: dict[str, RuntimeExactLayerModel] = {}
        self.model_configs: dict[str, dict] = {}
        self.policies: dict[str, AllocationPipeline] = {}
        for name in self.model_names:
            model, cfg = load_runtime_model(name)
            self.models[name] = model
            self.model_configs[name] = cfg
            self.policies[name] = AllocationPipeline(model, self.Sigma)

        self.constraint_setup = build_constraint_setup(
            reg_mode=args.reg_mode,
            constraint_mode=args.constraint_mode,
            anchor=self.m0,
            lower=self.lower,
            upper=self.upper,
        )
        self.G, self.gradG = self._build_objective()
        self.anchor_metrics, self.anchor_weights = self._evaluate_models(self.m0)

    def _required_models(self) -> list[str]:
        if self.args.probe == "decision_fragility":
            return [self.args.model]
        return [self.args.model_a, self.args.model_b]

    def _build_objective(self):
        common = {
            "C_t": self.C_t,
            "rets_t": self.rets_t,
            "Sigma_t": self.Sigma,
            "objective": self.args.objective,
            "anchor": self.m0,
            "l2reg": self.args.l2reg,
            "reg_fn": self.constraint_setup["reg_fn"],
        }
        if self.args.probe == "decision_fragility":
            return G_decision_fragility_function(self.policies[self.args.model], **common)
        if self.args.probe == "training_gap":
            return G_training_paradigm_gap_function(
                self.policies[self.args.model_a],
                self.policies[self.args.model_b],
                **common,
            )
        if self.args.probe == "pair_disagreement":
            return G_policy_disagreement_function(
                self.policies[self.args.model_a],
                self.policies[self.args.model_b],
                **common,
            )
        raise ValueError(f"Unknown probe={self.args.probe!r}")

    def unstd(self, state: torch.Tensor) -> np.ndarray:
        return (state.detach().float() * self.macro_std + self.macro_mean).cpu().numpy()

    def save_trajectory_tensors(self, results: dict[str, object], out_dir: Path, ts: str) -> None:
        trajs = results.get("m_trajs", [])
        if not trajs:
            return

        output_format = str(self.args.trajectory_format).lower()
        if output_format not in {"pt", "npy", "both"}:
            raise ValueError(f"Unknown trajectory_format={output_format!r}; use pt, npy, or both.")

        thin = max(1, int(self.args.trajectory_thin))
        burn_frac = float(self.args.trajectory_burn_in_frac)
        n_steps = int(trajs[0].shape[0])
        burn_in = int(np.floor(burn_frac * n_steps))
        burn_in = min(max(burn_in, 0), max(n_steps - 1, 0))

        standardized = torch.stack(
            [traj.detach().cpu().to(dtype=torch.float32) for traj in trajs],
            dim=0,
        )
        mean = self.macro_mean.detach().cpu().to(dtype=torch.float32).reshape(1, 1, -1)
        std = self.macro_std.detach().cpu().to(dtype=torch.float32).reshape(1, 1, -1)
        raw = standardized * std + mean
        standardized_post = standardized[:, burn_in::thin, :].contiguous()
        raw_post = raw[:, burn_in::thin, :].contiguous()

        files: dict[str, str] = {}

        def _write_tensor(name: str, tensor: torch.Tensor, units: str, postburnin: bool) -> None:
            stem = f"{name}_{ts}"
            payload = {
                "tensor": tensor,
                "shape": list(tensor.shape),
                "axis_order": ["seed", "step", "macro_variable"],
                "macro_columns": list(self.macro_columns),
                "date": int(self.date),
                "scenario": self.output_scenario_key(),
                "probe": self.args.probe,
                "objective": self.args.objective,
                "models": list(self.model_names),
                "units": units,
                "postburnin": bool(postburnin),
                "burn_in_steps": int(burn_in) if postburnin else 0,
                "trajectory_thin": int(thin) if postburnin else 1,
                "step_indexing": "zero-based inside tensor; original MALA step is tensor_step + 1",
            }
            if output_format in {"pt", "both"}:
                path = out_dir / f"{stem}.pt"
                torch.save(payload, path)
                files[f"{name}_pt"] = path.name
            if output_format in {"npy", "both"}:
                path = out_dir / f"{stem}.npy"
                np.save(path, tensor.numpy())
                files[f"{name}_npy"] = path.name

        _write_tensor("trajectories_standardized_3d", standardized, "standardized_macro_z_score", False)
        _write_tensor("trajectories_unstandardized_3d", raw, "unstandardized_raw_macro_units", False)
        _write_tensor("trajectories_raw_3d", raw, "unstandardized_raw_macro_units", False)
        _write_tensor("trajectories_postburnin_standardized_3d", standardized_post, "standardized_macro_z_score", True)
        _write_tensor("trajectories_postburnin_unstandardized_3d", raw_post, "unstandardized_raw_macro_units", True)
        _write_tensor("trajectories_postburnin_raw_3d", raw_post, "unstandardized_raw_macro_units", True)

        metadata = {
            "description": "Full MALA macro trajectories saved as 3D tensors shaped [seed, step, macro_variable].",
            "scenario": self.output_scenario_key(),
            "date": int(self.date),
            "probe": self.args.probe,
            "objective": self.args.objective,
            "models": list(self.model_names),
            "n_seeds_completed": int(standardized.shape[0]),
            "n_steps_per_seed": int(standardized.shape[1]),
            "n_macro_variables": int(standardized.shape[2]),
            "macro_columns": list(self.macro_columns),
            "full_shape": list(standardized.shape),
            "postburnin_shape": list(standardized_post.shape),
            "burn_in_fraction": burn_frac,
            "burn_in_steps": int(burn_in),
            "postburnin_original_start_step": int(burn_in + 1),
            "trajectory_thin": int(thin),
            "format": output_format,
            "files": files,
        }
        (out_dir / f"trajectory_tensor_metadata_{ts}.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def regime_summary(self, state: torch.Tensor) -> dict[str, object]:
        result = regime_classifier.classify(self.unstd(state))
        return {"label": result["label"], "probabilities": result["probabilities"]}

    def _evaluate_models(self, state: torch.Tensor) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
        metrics = {}
        weights = {}
        for name, policy in self.policies.items():
            metric, weight = evaluate_policy_decision(state, self.C_t, self.rets_t, self.Sigma, policy)
            metrics[name] = metric
            weights[name] = weight.detach()
        return metrics, weights

    def decision_record(self, state: torch.Tensor) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        metrics, weights = self._evaluate_models(state)
        record: dict[str, object] = {}
        for name, vals in metrics.items():
            for metric_name, value in vals.items():
                record[f"{name}_{metric_name}"] = value

        if self.args.probe == "decision_fragility":
            name = self.args.model
            anchor_w = self.anchor_weights[name]
            final_w = weights[name]
            allocation_shift = float(torch.sum(torch.abs(final_w - anchor_w)).item())
            for metric_name in DECISION_METRICS:
                if metric_name in metrics[name] and metric_name in self.anchor_metrics[name]:
                    anchor_value = self.anchor_metrics[name][metric_name]
                    final_value = metrics[name][metric_name]
                    record[f"anchor_{metric_name}"] = anchor_value
                    record[f"final_{metric_name}"] = final_value
                    record[f"delta_{metric_name}"] = final_value - anchor_value
            if self.args.objective == "hhi_min":
                target = self.anchor_metrics[name]["hhi"] - metrics[name]["hhi"]
            elif self.args.objective == "entropy_max":
                target = metrics[name]["entropy"] - self.anchor_metrics[name]["entropy"]
            elif self.args.objective == "concentration_hhi":
                target = metrics[name]["hhi"] - self.anchor_metrics[name]["hhi"]
            elif self.args.objective == "entropy_min":
                target = self.anchor_metrics[name]["entropy"] - metrics[name]["entropy"]
            else:
                target = allocation_shift
            record.update(
                {
                    "target_metric": target,
                    "target_metric_name": self.args.objective,
                    "allocation_l1_from_anchor": allocation_shift,
                    "delta_entropy": metrics[name]["entropy"] - self.anchor_metrics[name]["entropy"],
                    "delta_hhi": metrics[name]["hhi"] - self.anchor_metrics[name]["hhi"],
                    "delta_effective_n": metrics[name]["effective_n"] - self.anchor_metrics[name]["effective_n"],
                    "delta_max_weight": metrics[name]["max_weight"] - self.anchor_metrics[name]["max_weight"],
                    "delta_top10_weight": metrics[name]["top10_weight"] - self.anchor_metrics[name]["top10_weight"],
                    "diversification_score": (
                        metrics[name]["entropy"] - self.anchor_metrics[name]["entropy"]
                        + self.anchor_metrics[name]["hhi"] - metrics[name]["hhi"]
                    ),
                    "model": name,
                }
            )
        else:
            a_name = self.args.model_a
            b_name = self.args.model_b
            w_a = weights[a_name]
            w_b = weights[b_name]
            allocation_gap = float(torch.sum(torch.abs(w_a - w_b)).item())
            return_gap = metrics[a_name]["return"] - metrics[b_name]["return"]
            sharpe_gap = metrics[a_name]["sharpe"] - metrics[b_name]["sharpe"]
            anchor_return_gap = self.anchor_metrics[a_name]["return"] - self.anchor_metrics[b_name]["return"]
            anchor_sharpe_gap = self.anchor_metrics[a_name]["sharpe"] - self.anchor_metrics[b_name]["sharpe"]
            if self.args.objective == "a_beats_b":
                target = 100.0 * return_gap
            elif self.args.objective == "b_beats_a":
                target = -100.0 * return_gap
            elif self.args.objective == "return_gap_abs":
                target = abs(100.0 * return_gap)
            elif self.args.objective == "sharpe_gap_abs":
                target = abs(sharpe_gap)
            elif self.args.objective == "allocation_l1_close":
                target = allocation_gap
            else:
                target = allocation_gap
            record.update(
                {
                    "target_metric": target,
                    "target_metric_name": self.args.objective,
                    "allocation_l1_gap": allocation_gap,
                    "return_gap_a_minus_b": return_gap,
                    "sharpe_gap_a_minus_b": sharpe_gap,
                    "anchor_return_gap_a_minus_b": anchor_return_gap,
                    "anchor_sharpe_gap_a_minus_b": anchor_sharpe_gap,
                    "return_gap_improvement_for_b": anchor_return_gap - return_gap,
                    "sharpe_gap_improvement_for_b": anchor_sharpe_gap - sharpe_gap,
                    "b_return_matches_or_beats_a": bool(return_gap <= 0.0),
                    "winner_by_return": a_name if return_gap >= 0 else b_name,
                    "model_a": a_name,
                    "model_b": b_name,
                }
            )
        return record, weights

    def state_record(self, state: torch.Tensor, seed: int, step: int, source: str) -> dict[str, object]:
        raw = self.unstd(state)
        regime = regime_classifier.classify(raw)
        probabilities = regime["probabilities"]
        mah2 = mahalanobis_squared(state, self.m0)
        anchor_mah2 = anchor_mahalanobis_squared(state, self.m0)
        diff = state.detach().float() - self.m0.detach().float()
        anchor_empirical_mah2 = float((diff @ (self.empirical_macro_cov_inv @ diff)).item())
        record: dict[str, object] = {
            "seed": seed,
            "step": step,
            "source": source,
            "regime": regime["label"],
            "l2_dist": l2_squared(state, self.m0) ** 0.5,
            "mah2": mah2,
            "mah_dist": mah2 ** 0.5,
            **mahalanobis_chi2_summary(mah2),
            "anchor_mah2": anchor_mah2,
            "anchor_mah_dist": anchor_mah2 ** 0.5,
            **{
                f"anchor_{key}": value
                for key, value in mahalanobis_chi2_summary(anchor_mah2).items()
            },
            "anchor_empirical_mah2": anchor_empirical_mah2,
            "anchor_empirical_mah_dist": anchor_empirical_mah2 ** 0.5,
            **{
                f"anchor_empirical_{key}": value
                for key, value in mahalanobis_chi2_summary(anchor_empirical_mah2).items()
            },
            "box_violation": box_violation_squared(state, self.lower, self.upper) ** 0.5,
        }
        for name, value in zip(self.macro_columns, raw):
            record[name] = float(value)
        standardized = state.detach().float().cpu().numpy()
        for name, value in zip(self.macro_columns, standardized):
            record[f"{name}_std"] = float(value)
        for name, value in probabilities.items():
            record[f"prob_{name}"] = float(value)
        return record

    def run_mala_chain(self, start: torch.Tensor, n_steps: int):
        clip_low = self.constraint_setup["clip_low"]
        clip_high = self.constraint_setup["clip_high"]
        x = start.detach().clone().to(dtype=self.m0.dtype)
        traj = []
        accepted: list[bool] = []
        accept_probs: list[float] = []
        log_accept_ratios: list[float] = []
        boundary_touches = 0
        for _ in range(n_steps):
            hyps = (self.G, self.gradG, self.args.eta, self.args.beta, clip_low, clip_high)
            x = x.detach().requires_grad_(True)
            x, step_info = langevin.torch_MALA_step(x, hyps, return_info=True)
            traj.append(x.detach())
            accepted.append(step_info["accepted"])
            accept_probs.append(step_info["accept_prob"])
            log_accept_ratios.append(step_info["log_accept_ratio"])
            if clip_low is not None and clip_high is not None and touches_box(x.detach(), clip_low, clip_high):
                boundary_touches += 1
        info = {
            "accepted": accepted,
            "accept_probs": accept_probs,
            "log_accept_ratios": log_accept_ratios,
            "accept_rate": float(np.mean(accepted)) if accepted else 0.0,
            "mean_accept_prob": float(np.mean(accept_probs)) if accept_probs else 0.0,
            "boundary_touch_rate": float(boundary_touches / n_steps) if n_steps > 0 else 0.0,
            "final_box_violation": box_violation_squared(x.detach(), self.lower, self.upper) ** 0.5,
        }
        return x.detach(), torch.stack(traj), info

    def diagnostic_steps(self, n_steps: int) -> list[int]:
        burn_in = int(np.floor(DIAGNOSTIC_BURN_IN_FRACTION * n_steps))
        burn_in = min(max(burn_in, 0), max(n_steps - 1, 0))
        thin = max(1, int(DIAGNOSTIC_THIN))
        steps = list(range(burn_in, n_steps, thin))
        return steps if steps else [max(n_steps - 1, 0)]

    def run(self) -> dict[str, object]:
        print_header(self)
        m_trajs = []
        m_lasts = []
        seed_summaries = []
        final_weight_rows = []
        for seed_idx in range(1, self.args.n_seeds + 1):
            chain_seed = None if self.args.base_seed < 0 else int(self.args.base_seed) + seed_idx - 1
            if chain_seed is not None:
                torch.manual_seed(chain_seed)
            generator = None
            if chain_seed is not None:
                generator = torch.Generator(device=self.m0.device)
                generator.manual_seed(chain_seed)
            random_draw = torch.rand((9,), dtype=self.m0.dtype, device=self.m0.device, generator=generator)
            m_start = self.m0 + (self.upper - self.lower) * self.args.random_start_scale * random_draw
            print(f"[seed {seed_idx:02d}] running {self.args.probe} trajectory")
            start_time = time.time()
            try:
                m_last, m_traj, chain_info = self.run_mala_chain(m_start, self.args.n_steps)
            except KeyboardInterrupt:
                print(f"[seed {seed_idx:02d}] interrupted; saving completed seeds only")
                break
            m_trajs.append(m_traj)
            m_lasts.append(m_last)
            ess = summarize_ess(m_traj, chain_info["accepted"])
            final_decision, final_weights = self.decision_record(m_last)
            for model_name, weights in final_weights.items():
                anchor_weights = self.anchor_weights[model_name]
                for permno, weight, anchor_weight in zip(self.permnos, weights.detach().cpu().numpy(), anchor_weights.detach().cpu().numpy()):
                    final_weight_rows.append(
                        {
                            "seed": seed_idx,
                            "model": model_name,
                            "permno": int(permno),
                            "weight": float(weight),
                            "anchor_weight": float(anchor_weight),
                            "delta_weight": float(weight - anchor_weight),
                        }
                    )

            start_mah2 = mahalanobis_squared(m_start, self.m0)
            final_mah2 = mahalanobis_squared(m_last, self.m0)
            start_anchor_mah2 = anchor_mahalanobis_squared(m_start, self.m0)
            final_anchor_mah2 = anchor_mahalanobis_squared(m_last, self.m0)
            start_diff = m_start.detach().float() - self.m0.detach().float()
            final_diff = m_last.detach().float() - self.m0.detach().float()
            start_anchor_empirical_mah2 = float((start_diff @ (self.empirical_macro_cov_inv @ start_diff)).item())
            final_anchor_empirical_mah2 = float((final_diff @ (self.empirical_macro_cov_inv @ final_diff)).item())
            final_mah = mahalanobis_chi2_summary(final_mah2)
            final_anchor_mah = mahalanobis_chi2_summary(final_anchor_mah2)
            final_anchor_empirical_mah = mahalanobis_chi2_summary(final_anchor_empirical_mah2)
            start_l2 = l2_squared(m_start, self.m0)
            final_l2 = l2_squared(m_last, self.m0)
            summary = {
                "seed": seed_idx,
                "start_regime": self.regime_summary(m_start),
                "final_regime": self.regime_summary(m_last),
                "start_l2_dist": start_l2 ** 0.5,
                "final_l2_dist": final_l2 ** 0.5,
                "start_mah_dist": start_mah2 ** 0.5,
                "final_mah_dist": final_mah2 ** 0.5,
                "final_mah_chi2_tail": final_mah["mah_chi2_tail"],
                "start_anchor_mah_dist": start_anchor_mah2 ** 0.5,
                "final_anchor_mah_dist": final_anchor_mah2 ** 0.5,
                "final_anchor_mah_chi2_tail": final_anchor_mah["mah_chi2_tail"],
                "start_anchor_empirical_mah_dist": start_anchor_empirical_mah2 ** 0.5,
                "final_anchor_empirical_mah_dist": final_anchor_empirical_mah2 ** 0.5,
                "final_anchor_empirical_mah_chi2_tail": final_anchor_empirical_mah["mah_chi2_tail"],
                "accept_rate": chain_info["accept_rate"],
                "mean_accept_prob": chain_info["mean_accept_prob"],
                "boundary_touch_rate": chain_info["boundary_touch_rate"],
                "final_box_violation": chain_info["final_box_violation"],
                "chain_info": chain_info,
                "elapsed_seconds": time.time() - start_time,
                **ess,
                **final_decision,
            }
            seed_summaries.append(summary)
            print_seed_summary(self, summary)
        print_run_summary(seed_summaries)
        results = {
            "m_trajs": m_trajs,
            "m_lasts": m_lasts,
            "seed_summaries": seed_summaries,
            "final_weight_rows": final_weight_rows,
        }
        self.save_results(results)
        return results

    def build_chain_state_frame(self, results: dict[str, object]) -> pd.DataFrame:
        rows = []
        for seed_idx, traj in enumerate(results["m_trajs"], start=1):
            for step in self.diagnostic_steps(traj.shape[0]):
                record = self.state_record(traj[step], seed=seed_idx, step=step + 1, source="post_burnin_thinned")
                decision, _ = self.decision_record(traj[step])
                record.update(decision)
                rows.append(record)
        return pd.DataFrame(rows)

    def build_final_state_frame(self, results: dict[str, object]) -> pd.DataFrame:
        rows = []
        for seed_idx, state in enumerate(results["m_lasts"], start=1):
            record = self.state_record(state, seed=seed_idx, step=-1, source="final")
            summary = results["seed_summaries"][seed_idx - 1]
            for key, value in summary.items():
                if key in {"start_regime", "final_regime", "chain_info"}:
                    continue
                if isinstance(value, (int, float, str)):
                    record[key] = value
            rows.append(record)
        return pd.DataFrame(rows)

    def anchor_weight_frame(self) -> pd.DataFrame:
        rows = []
        for model_name, weights in self.anchor_weights.items():
            metrics = self.anchor_metrics[model_name]
            for permno, weight in zip(self.permnos, weights.detach().cpu().numpy()):
                rows.append(
                    {
                        "model": model_name,
                        "permno": int(permno),
                        "anchor_weight": float(weight),
                        **{f"anchor_{key}": value for key, value in metrics.items()},
                    }
                )
        return pd.DataFrame(rows)

    def save_results(self, results: dict[str, object]) -> None:
        out_root = ROOT / "scenario_outputs" / self.output_scenario_key() / "runs"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = out_root / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for row in results["seed_summaries"]:
            flat = {}
            for key, value in row.items():
                if key == "start_regime":
                    flat["start_regime"] = value["label"]
                elif key == "final_regime":
                    flat["final_regime"] = value["label"]
                elif key != "chain_info" and isinstance(value, (int, float, str)):
                    flat[key] = value
            summary_rows.append(flat)
        seed_summary = pd.DataFrame(summary_rows)
        final_frame = self.build_final_state_frame(results)
        sample_frame = self.build_chain_state_frame(results)
        anchor_weights = self.anchor_weight_frame()
        final_weights = pd.DataFrame(results["final_weight_rows"])
        hist_raw = self.macro_df[["yyyymm", *self.macro_columns]].copy()

        nn_index = _build_nn_index(macro_columns=list(self.macro_columns))
        std_cols = [f"{c}_std" for c in self.macro_columns]
        if not final_frame.empty and all(c in final_frame.columns for c in std_cols):
            nn_final = nn_index.attach(final_frame[std_cols].to_numpy(dtype=np.float64), k=3)
            nn_final.index = final_frame.index
            final_frame = pd.concat([final_frame, nn_final], axis=1)
        if not sample_frame.empty and all(c in sample_frame.columns for c in std_cols):
            nn_sample = nn_index.attach(sample_frame[std_cols].to_numpy(dtype=np.float64), k=3)
            nn_sample.index = sample_frame.index
            sample_frame = pd.concat([sample_frame, nn_sample], axis=1)
        if not seed_summary.empty and "seed" in seed_summary.columns and not final_frame.empty:
            nn_pull_cols = ["seed"] + [f"nn_{m}_yyyymm_1" for m in ("var1", "hist", "eucl")] + [f"nn_{m}_dist_1" for m in ("var1", "hist", "eucl")]
            if all(c in final_frame.columns for c in nn_pull_cols):
                nn_pull = final_frame[nn_pull_cols].rename(columns={
                    **{f"nn_{m}_yyyymm_1": f"final_nn_{m}_yyyymm_1" for m in ("var1", "hist", "eucl")},
                    **{f"nn_{m}_dist_1": f"final_nn_{m}_dist_1" for m in ("var1", "hist", "eucl")},
                })
                seed_summary = seed_summary.merge(nn_pull, on="seed", how="left")

        seed_summary.to_csv(out_dir / f"seed_summary_{ts}.csv", index=False)
        final_frame.to_csv(out_dir / f"final_state_diagnostics_{ts}.csv", index=False)
        sample_frame.to_csv(out_dir / f"generated_macro_sample_{ts}.csv", index=False)
        hist_raw.to_csv(out_dir / f"historical_macro_panel_{ts}.csv", index=False)
        anchor_weights.to_csv(out_dir / f"anchor_weights_{ts}.csv", index=False)
        final_weights.to_csv(out_dir / f"final_weights_{ts}.csv", index=False)

        if results["m_lasts"]:
            lasts_np = np.stack([self.unstd(m) for m in results["m_lasts"]])
            pd.DataFrame(lasts_np, columns=self.macro_columns).to_csv(out_dir / f"final_states_{ts}.csv", index=False)

        if self.args.save_trajectory_tensors:
            self.save_trajectory_tensors(results, out_dir, ts)

        anchor_regime = self.regime_summary(self.m0)
        true_label = regime_classifier.historical_label(self.date) or "unknown"
        if not sample_frame.empty and not final_frame.empty:
            anchor_raw = self.unstd(self.m0)
            fig = macro_density_grid(
                historical_df=hist_raw,
                generated_df=sample_frame,
                anchor=anchor_raw,
                columns=self.macro_columns,
                save_path=out_dir / f"macro_density_grid_scenario5_{ts}.pdf",
                title=None,
            )
            plt.close(fig)
            fig = macro_pair_grid(
                historical_df=hist_raw,
                generated_df=sample_frame,
                anchor=anchor_raw,
                columns=self.macro_columns,
                save_path=out_dir / f"macro_pair_grid_scenario5_{ts}.pdf",
                title="Scenario 5 Generated Macro Scenarios vs Historical Macro Panel",
            )
            plt.close(fig)
            generated_std = sample_frame[[f"{col}_std" for col in self.macro_columns]].copy()
            generated_std.columns = self.macro_columns
            fig = macro_shift_bar(
                generated_std_df=generated_std,
                anchor_std=self.m0,
                columns=self.macro_columns,
                save_path=out_dir / f"macro_shift_bar_scenario5_{ts}.pdf",
                title="Scenario 5 Generated Macro Shift from Anchor",
            )
            plt.close(fig)
            fig = scenario5_story_figure(
                historical_df=hist_raw,
                generated_df=sample_frame,
                final_diagnostics_df=final_frame,
                anchor_raw=anchor_raw,
                anchor_regime_probs=anchor_regime["probabilities"],
                true_anchor_label=true_label,
                target_metric_col="target_metric",
                target_metric_label=self.target_metric_label(),
                columns=self.macro_columns,
                save_path=out_dir / f"scenario5_story_figure_{ts}.pdf",
                title=self.story_title(),
                subtitle=self.story_subtitle(),
                anchor_label=f"{self.date} anchor",
            )
            plt.close(fig)

        config = {
            "DATE": self.date,
            "ASSET_SIZE": ASSET_SIZE,
            "PROBE": self.args.probe,
            "OBJECTIVE": self.args.objective,
            "MODELS": self.model_names,
            "ETA": self.args.eta,
            "BETA": self.args.beta,
            "N_STEPS": self.args.n_steps,
            "N_SEEDS": self.args.n_seeds,
            "BASE_SEED": self.args.base_seed,
            "L2REG": self.args.l2reg,
            "REG_MODE": self.args.reg_mode,
            "CONSTRAINT_MODE": self.args.constraint_mode,
            "RANDOM_START_SCALE": self.args.random_start_scale,
            "SAVE_TRAJECTORY_TENSORS": bool(self.args.save_trajectory_tensors),
            "TRAJECTORY_FORMAT": self.args.trajectory_format,
            "TRAJECTORY_BURN_IN_FRACTION": self.args.trajectory_burn_in_frac,
            "TRAJECTORY_THIN": self.args.trajectory_thin,
            "CONSTRAINT_DESCRIPTION": self.constraint_setup["description"],
            "ANCHOR_REGIME": anchor_regime,
            "ANCHOR_TRUE_LABEL": true_label,
            "ANCHOR_METRICS": self.anchor_metrics,
            "MODEL_CONFIGS": self.model_configs,
            "OUTPUT_SCENARIO_KEY": self.output_scenario_key(),
        }
        (out_dir / f"config_{ts}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.write_story_markdown(out_dir, ts, final_frame, sample_frame)
        print(f"\nResults saved -> {out_dir.relative_to(ROOT)}")

    def output_scenario_key(self) -> str:
        if (
            self.args.probe == "decision_fragility"
            and self.args.model == "locked_e2e"
            and self.args.objective in {"entropy_max", "hhi_min"}
        ):
            return f"scenario_e2e_diversify_{self.date}"
        if (
            self.args.probe == "training_gap"
            and self.args.model_a == "locked_e2e"
            and self.args.model_b == "standardized_pto"
            and self.args.objective == "b_beats_a"
        ):
            return f"scenario_pto_catchup_{self.date}"
        return f"scenario5_{self.date}"

    def target_metric_label(self) -> str:
        if self.args.probe == "decision_fragility" and self.args.objective == "entropy_max":
            return "Entropy increase from anchor"
        if self.args.probe == "decision_fragility" and self.args.objective == "hhi_min":
            return "HHI decrease from anchor"
        if self.args.probe == "decision_fragility":
            return "Allocation L1 distance from anchor"
        if self.args.objective == "allocation_l1_gap":
            return "Allocation L1 gap between policies"
        if self.args.objective == "allocation_l1_close":
            return "Allocation L1 gap between policies (minimized)"
        if self.args.objective == "b_beats_a":
            return "PTO return advantage, percentage points"
        if self.args.objective in {"a_beats_b", "return_gap_abs"}:
            return "Return target, percentage points"
        if self.args.objective == "sharpe_gap_abs":
            return "Absolute Sharpe gap"
        return self.args.objective

    def story_title(self) -> str:
        if (
            self.args.probe == "decision_fragility"
            and self.args.model == "locked_e2e"
            and self.args.objective in {"entropy_max", "hhi_min"}
        ):
            return "Scenario: Locked E2E Diversifies Under Nearby Macro States"
        if (
            self.args.probe == "training_gap"
            and self.args.model_a == "locked_e2e"
            and self.args.model_b == "standardized_pto"
            and self.args.objective == "b_beats_a"
        ):
            return "Scenario: PTO Catches Locked E2E Under Nearby Macro States"
        if self.args.probe == "decision_fragility":
            return f"Scenario 5: Fragility of {self.args.model} Allocations"
        if self.args.probe == "training_gap":
            return f"Scenario 5: E2E vs PTO Decision Gap"
        return f"Scenario 5: {self.args.model_a} vs {self.args.model_b} Policy Disagreement"

    def story_subtitle(self) -> str:
        if (
            self.args.probe == "decision_fragility"
            and self.args.model == "locked_e2e"
            and self.args.objective in {"entropy_max", "hhi_min"}
        ):
            return "Generated states ask the locked E2E policy to reduce concentration while staying near the anchor macro state."
        if (
            self.args.probe == "training_gap"
            and self.args.model_a == "locked_e2e"
            and self.args.model_b == "standardized_pto"
            and self.args.objective == "b_beats_a"
        ):
            return "Generated states ask whether standardized PTO can close or reverse an initial realized-return gap against locked E2E."
        if self.args.probe == "decision_fragility":
            return "Generated states find nearby macro conditions where one policy materially reallocates the same asset basket."
        if self.args.probe == "training_gap":
            return "Generated states expose where training paradigms imply materially different robust portfolios."
        return "Generated states stress-test disagreement between two policies beyond realized return reversals."

    def write_story_markdown(
        self,
        out_dir: Path,
        ts: str,
        final_frame: pd.DataFrame,
        sample_frame: pd.DataFrame,
    ) -> None:
        regime_counts = sample_frame["regime"].value_counts(normalize=True).mul(100.0) if not sample_frame.empty else pd.Series(dtype=float)
        anchor_regime = self.regime_summary(self.m0)
        anchor_label = str(anchor_regime["label"])
        final_non_anchor_share = float((final_frame["regime"] != anchor_label).mean() * 100.0) if not final_frame.empty else float("nan")
        transition_counts = final_frame["regime"].value_counts() if not final_frame.empty else pd.Series(dtype=int)
        std_cols = [f"{col}_std" for col in self.macro_columns]
        macro_shift_lines = []
        if not sample_frame.empty and all(col in sample_frame.columns for col in std_cols):
            med_shift = sample_frame[std_cols].median().to_numpy(dtype=float) - self.m0.detach().cpu().numpy()
            q25_shift = sample_frame[std_cols].quantile(0.25).to_numpy(dtype=float) - self.m0.detach().cpu().numpy()
            q75_shift = sample_frame[std_cols].quantile(0.75).to_numpy(dtype=float) - self.m0.detach().cpu().numpy()
            ranked = sorted(
                zip(self.macro_columns, med_shift, q25_shift, q75_shift),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            macro_shift_lines = [
                f"- `{name}`: median shift `{median:+.3f}` z, IQR [`{q25:+.3f}`, `{q75:+.3f}`]"
                for name, median, q25, q75 in ranked
            ]

        decision_lines = []
        nn_lines = []
        caution_lines = []
        if not final_frame.empty:
            decision_lines.extend(
                [
                    f"- Final mean target metric: `{final_frame['target_metric'].mean():.4f}`",
                    f"- Final median target metric: `{final_frame['target_metric'].median():.4f}`",
                    f"- Final median L2 distance: `{final_frame['l2_dist'].median():.3f}`",
                    f"- Final median VAR(1)-forecast Mahalanobis distance: `{final_frame['mah_dist'].median():.3f}`",
                    f"- Mean acceptance rate: `{final_frame['accept_rate'].mean():.1%}`",
                    f"- Mean box violation: `{final_frame['box_violation'].mean():.4f}`",
                ]
            )
            if "mah_chi2_tail" in final_frame.columns:
                decision_lines.append(f"- Final median VAR(1)-forecast Mahalanobis chi2 tail: `{final_frame['mah_chi2_tail'].median():.3g}`")
            if "anchor_mah_dist" in final_frame.columns:
                decision_lines.append(f"- Final median anchor VAR-innovation Mahalanobis distance: `{final_frame['anchor_mah_dist'].median():.3f}`")
            if "anchor_mah_chi2_tail" in final_frame.columns:
                decision_lines.append(f"- Final median anchor VAR-innovation Mahalanobis chi2 tail: `{final_frame['anchor_mah_chi2_tail'].median():.3g}`")
            if "anchor_empirical_mah_dist" in final_frame.columns:
                decision_lines.append(f"- Final median anchor empirical Mahalanobis distance: `{final_frame['anchor_empirical_mah_dist'].median():.3f}`")
            if "anchor_empirical_mah_chi2_tail" in final_frame.columns:
                decision_lines.append(f"- Final median anchor empirical Mahalanobis chi2 tail: `{final_frame['anchor_empirical_mah_chi2_tail'].median():.3g}`")
            if "allocation_l1_gap" in final_frame.columns:
                decision_lines.append(f"- Final median allocation L1 gap: `{final_frame['allocation_l1_gap'].median():.4f}`")
            if "return_gap_a_minus_b" in final_frame.columns:
                decision_lines.append(f"- Final median return gap A-B: `{final_frame['return_gap_a_minus_b'].median():+.3%}`")
            if "winner_by_return" in final_frame.columns:
                winner_counts = final_frame["winner_by_return"].value_counts()
                decision_lines.append(
                    "- Winner by realized return: "
                    + ", ".join(f"`{name}` {int(count)}/{len(final_frame)}" for name, count in winner_counts.items())
                )
            if self.args.probe == "decision_fragility" and self.args.objective in {"entropy_max", "hhi_min"}:
                name = self.args.model
                decision_lines.extend(
                    [
                        f"- Anchor `{name}` entropy: `{self.anchor_metrics[name]['entropy']:.4f}`",
                        f"- Final median entropy: `{final_frame[f'{name}_entropy'].median():.4f}`",
                        f"- Median entropy change: `{final_frame['delta_entropy'].median():+.4f}`",
                        f"- Anchor `{name}` HHI: `{self.anchor_metrics[name]['hhi']:.4f}`",
                        f"- Final median HHI: `{final_frame[f'{name}_hhi'].median():.4f}`",
                        f"- Median HHI change: `{final_frame['delta_hhi'].median():+.4f}`",
                        f"- Anchor effective N: `{self.anchor_metrics[name]['effective_n']:.2f}`",
                        f"- Final median effective N: `{final_frame[f'{name}_effective_n'].median():.2f}`",
                        f"- Median max-weight change: `{final_frame['delta_max_weight'].median():+.3%}`",
                        f"- Median top-10 weight change: `{final_frame['delta_top10_weight'].median():+.3%}`",
                    ]
                )
            if self.args.probe == "training_gap" and "anchor_return_gap_a_minus_b" in final_frame.columns:
                decision_lines.extend(
                    [
                        f"- Anchor return gap A-B: `{final_frame['anchor_return_gap_a_minus_b'].median():+.3%}`",
                        f"- Final median return gap A-B: `{final_frame['return_gap_a_minus_b'].median():+.3%}`",
                        f"- Median gap improvement for B: `{final_frame['return_gap_improvement_for_b'].median():+.3%}`",
                    ]
                )
                if "b_return_matches_or_beats_a" in final_frame.columns:
                    decision_lines.append(
                        f"- B matches/beats A by return in `{100.0 * final_frame['b_return_matches_or_beats_a'].astype(bool).mean():.1f}%` of final seeds"
                    )
            for metric, label in (
                ("var1", "VAR(1)-innovation Mahalanobis NN"),
                ("hist", "historical-covariance Mahalanobis NN"),
                ("eucl", "Euclidean z-score NN"),
            ):
                yy_col = f"nn_{metric}_yyyymm_1"
                dist_col = f"nn_{metric}_dist_1"
                if yy_col in final_frame.columns and dist_col in final_frame.columns:
                    modes = final_frame[yy_col].dropna().astype(int).mode()
                    modal = int(modes.iloc[0]) if not modes.empty else -1
                    nn_lines.append(
                        f"- {label}: modal top-1 month `{modal}`, median distance `{final_frame[dist_col].median():.3f}`"
                    )

            var_tail = None
            if "anchor_mah_chi2_tail" in final_frame.columns:
                var_tail = float(final_frame["anchor_mah_chi2_tail"].median())
            elif "mah_chi2_tail" in final_frame.columns:
                var_tail = float(final_frame["mah_chi2_tail"].median())
            empirical_tail = None
            if "anchor_empirical_mah_chi2_tail" in final_frame.columns:
                empirical_tail = float(final_frame["anchor_empirical_mah_chi2_tail"].median())
            if var_tail is not None and empirical_tail is not None and var_tail < 0.05 <= empirical_tail:
                caution_lines = [
                    "",
                    "## Manuscript Caution",
                    "",
                    (
                        f"- Anchor VAR-innovation tail is `{var_tail:.3g}` while empirical-anchor tail is "
                        f"`{empirical_tail:.3g}`. Describe this run as empirically local, not VAR(1)-innovation plausible."
                    ),
                ]

        lines = [
            "# Scenario 5 Diagnostics",
            "",
            f"- Anchor month: `{self.date}`",
            f"- Probe: `{self.args.probe}`",
            f"- Objective: `{self.args.objective}`",
            f"- Models: `{', '.join(self.model_names)}`",
            f"- Hyperparameters: `L2REG={self.args.l2reg}`, `BETA={self.args.beta}`, `ETA={self.args.eta}`, `N_STEPS={self.args.n_steps}`, `N_SEEDS={self.args.n_seeds}`",
            f"- Anchor classifier regime: `{anchor_label}`",
            f"- Final non-anchor regime share: `{final_non_anchor_share:.1f}%`",
            *decision_lines,
            "",
            "## Final Regime Outcomes",
            "",
            *(f"- `{anchor_label}` to `{name}`: `{int(count)}` final seeds" for name, count in transition_counts.items()),
            "",
            "## Generated Regime Occupancy",
            "",
            *(f"- `{name}`: `{value:.1f}%`" for name, value in regime_counts.items()),
            "",
            "## Historical Analog Diagnostics",
            "",
            *nn_lines,
            *caution_lines,
            "",
            "## Median Macro Shifts",
            "",
            *macro_shift_lines,
            "",
        ]
        (out_dir / f"diagnostic_storyline_{ts}.md").write_text("\n".join(lines), encoding="utf-8")


def format_regime(regime_info: dict[str, object]) -> str:
    probs = regime_info["probabilities"]
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name}={value:.2%}" for name, value in ordered[:2])
    return f"{regime_info['label']} ({top})"


def print_header(runtime: Scenario5Runtime) -> None:
    anchor_regime = runtime.regime_summary(runtime.m0)
    eval_summary = regime_classifier.eval_summary["test"]
    is_minimization_probe = runtime.args.objective == "allocation_l1_close"
    behavior_phrase = "becomes small" if is_minimization_probe else "becomes large"
    optimization_verb = "minimize" if is_minimization_probe else "maximize"
    print("=" * 88)
    print("Scenario 5 | Decision-level Universe500 runtime")
    print(
        f"Question: find plausible macro states near {runtime.date} where the target "
        f"`{runtime.args.probe}/{runtime.args.objective}` {behavior_phrase} on the same "
        f"{len(runtime.permnos)}-asset Universe500 basket."
    )
    reg_desc = "VAR1-Mahalanobis²(m, A@anchor+c)" if runtime.args.reg_mode == "var1" else "L2²(m, anchor)"
    print(
        f"Objective: {optimization_verb} {runtime.target_metric_label()} + {runtime.args.l2reg:.4g} * {reg_desc} "
        f"[constraint={runtime.args.constraint_mode!r}]"
    )
    print(
        f"Chain: MALA, steps={runtime.args.n_steps}, seeds={runtime.args.n_seeds}, "
        f"beta={runtime.args.beta:.2f}, eta={runtime.args.eta:.6f}, "
        f"random_start_scale={runtime.args.random_start_scale:.4f}"
    )
    print(f"Models: {', '.join(runtime.model_names)}")
    for name, metrics in runtime.anchor_metrics.items():
        print(
            f"Anchor {name}: return={metrics['return']:.2%}, Sharpe={metrics['sharpe']:.2f}, "
            f"entropy={metrics['entropy']:.2f}, HHI={metrics['hhi']:.3f}, max_weight={metrics['max_weight']:.2%}"
        )
    true_label = regime_classifier.historical_label(runtime.date) or "unknown"
    print(f"Anchor classifier regime: {format_regime(anchor_regime)} | true hard label: {true_label}")
    print(
        f"Regime classifier held-out test: accuracy={eval_summary['accuracy']:.3f}, "
        f"macro_f1={eval_summary['macro_f1']:.3f}, months={eval_summary['n_months']}"
    )
    print("=" * 88)


def print_seed_summary(runtime: Scenario5Runtime, summary: dict[str, object]) -> None:
    print(
        f"[seed {summary['seed']:02d}] accept_rate={summary['accept_rate']:.1%}, "
        f"mean_accept_prob={summary['mean_accept_prob']:.1%}, "
        f"box_violation={summary['final_box_violation']:.4f}, "
        f"ESS cap(mean)={summary['ess_mean']:.1f}"
    )
    print(
        f"[seed {summary['seed']:02d}] start regime={format_regime(summary['start_regime'])}, "
        f"final regime={format_regime(summary['final_regime'])}"
    )
    print(
        f"[seed {summary['seed']:02d}] target_metric={summary['target_metric']:.4f}, "
        f"L2={summary['final_l2_dist']:.3f}, Mah={summary['final_mah_dist']:.3f}, "
        f"anchorVARMah={summary['final_anchor_mah_dist']:.3f}, "
        f"anchorEmpMah={summary['final_anchor_empirical_mah_dist']:.3f}, "
        f"empTail={summary['final_anchor_empirical_mah_chi2_tail']:.3g}"
    )
    if runtime.args.probe != "decision_fragility":
        print(
            f"[seed {summary['seed']:02d}] allocation_gap={summary.get('allocation_l1_gap', float('nan')):.4f}, "
            f"return_gap(a-b)={summary.get('return_gap_a_minus_b', float('nan')):.4%}, "
            f"winner={summary.get('winner_by_return', 'n/a')}"
        )
    print("-" * 88)


def print_run_summary(seed_summaries: list[dict[str, object]]) -> None:
    if not seed_summaries:
        return
    frame = pd.DataFrame(
        [
            {
                "accept_rate": row.get("accept_rate", np.nan),
                "target_metric": row.get("target_metric", np.nan),
                "final_l2_dist": row.get("final_l2_dist", np.nan),
                "final_mah_dist": row.get("final_mah_dist", np.nan),
                "final_anchor_mah_dist": row.get("final_anchor_mah_dist", np.nan),
                "final_anchor_empirical_mah_dist": row.get("final_anchor_empirical_mah_dist", np.nan),
                "final_anchor_empirical_mah_chi2_tail": row.get("final_anchor_empirical_mah_chi2_tail", np.nan),
                "final_box_violation": row.get("final_box_violation", np.nan),
            }
            for row in seed_summaries
        ]
    )
    print("Run summary across completed seeds")
    print(
        f"median_accept={frame['accept_rate'].median():.1%}, "
        f"median_target={frame['target_metric'].median():.4f}, "
        f"median_L2={frame['final_l2_dist'].median():.3f}, "
        f"median_VARforecastMah={frame['final_mah_dist'].median():.3f}, "
        f"median_anchorVARMah={frame['final_anchor_mah_dist'].median():.3f}, "
        f"median_anchorEmpMah={frame['final_anchor_empirical_mah_dist'].median():.3f}, "
        f"median_anchorEmpTail={frame['final_anchor_empirical_mah_chi2_tail'].median():.3g}, "
        f"mean_box_violation={frame['final_box_violation'].mean():.4f}"
    )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario 5 decision-level MALA probes.")
    parser.add_argument("--date", type=int, default=DEFAULT_DATE)
    parser.add_argument("--probe", choices=["decision_fragility", "training_gap", "pair_disagreement"], default=DEFAULT_PROBE)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--objective", default=None)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--reg-mode", choices=["l2", "var1"], default="l2")
    parser.add_argument("--constraint-mode", choices=["clip", "none", "box_barrier", "strong_soft"], default="box_barrier")
    parser.add_argument("--l2reg", type=float, default=DEFAULT_L2REG)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--random-start-scale", type=float, default=DEFAULT_RANDOM_START_SCALE)
    parser.add_argument(
        "--save-trajectory-tensors",
        action="store_true",
        help="Save full and post-burn-in 3D trajectory tensors shaped [seed, step, macro_variable].",
    )
    parser.add_argument(
        "--trajectory-format",
        choices=["pt", "npy", "both"],
        default="both",
        help="Serialization format for trajectory tensors.",
    )
    parser.add_argument(
        "--trajectory-burn-in-frac",
        type=float,
        default=DIAGNOSTIC_BURN_IN_FRACTION,
        help="Fraction of each chain discarded in the post-burn-in trajectory tensor.",
    )
    parser.add_argument(
        "--trajectory-thin",
        type=int,
        default=1,
        help="Thinning interval for the post-burn-in trajectory tensor; full tensors are always unthinned.",
    )
    args = parser.parse_args()
    supplied = set(sys.argv[1:])
    if args.objective is None:
        if args.probe == "decision_fragility":
            args.objective = "allocation_l1_from_anchor"
        else:
            args.objective = "allocation_l1_gap"
    if args.probe == "pair_disagreement" and "--model-a" not in supplied and "--model-b" not in supplied:
        args.model_a = "summer_child"
        args.model_b = "winter_wolf"
    return args


def main() -> None:
    args = parse_args()
    runtime = Scenario5Runtime(args)
    runtime.run()


if __name__ == "__main__":
    main()
