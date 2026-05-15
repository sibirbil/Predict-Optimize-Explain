import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Callable, Dict, List, Optional
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
import torch.nn.functional as F

from end2endportfolio.src import langevin
from src.modules.probe_eval import (
    A,
    Sigma_inv,
    c,
    AllocationPipeline,
    G_contrast_function,
    evaluate,
    mahalonobis_reg,
)
from src.modules.runtime_model_loader import load_exact_e2e_model_from_run
from src.modules.runtime_regime import MACRO_ORDER, regime_classifier
from src.modules.nn_historical import build_index_from_runtime as _build_nn_index
from src.data.macro_scaler import MacroScaler
from src.modules.runtime_components import FeatureScenarioBuilder

_RUNTIME = ROOT / "runtime_universe500"
_macro_scaler = MacroScaler.load(_RUNTIME / "scaler")
macro_mean: torch.Tensor = _macro_scaler.mean  # shape [9], float32
macro_std:  torch.Tensor = _macro_scaler.std   # shape [9], float32

# ── Universe500 runtime data (firm features + returns) ────────────────────────
_DATA = _RUNTIME / "data"
_meta_all  = pd.read_parquet(_DATA / "metadata.parquet")
_meta_test = pd.read_parquet(_DATA / "metadata_test.parquet")
_X_test    = pd.read_parquet(_DATA / "X_test.parquet")
_meta_all["yyyymm"]  = _meta_all["yyyymm"].astype(int)
_meta_test["yyyymm"] = _meta_test["yyyymm"].astype(int)
_feature_names      = tuple(np.load(_DATA / "feature_names.npy",      allow_pickle=True).tolist())
_firm_feature_names = tuple(np.load(_DATA / "firm_feature_names.npy", allow_pickle=True).tolist())
_macro_predictors   = tuple(np.load(_DATA / "macro_predictors.npy",   allow_pickle=True).tolist())
_macro_final_21a    = pd.read_parquet(_DATA / "macro_final.parquet")
_macro_final_21a["yyyymm"] = _macro_final_21a["yyyymm"].astype(int)


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
    c_ = returns_matrix - mu
    sigma = c_.T @ (c_ * w[:, None])
    sigma = 0.5 * (sigma + sigma.T)
    diag = np.diag(np.diag(sigma))
    sigma = (1.0 - shrinkage) * sigma + shrinkage * diag + ridge * np.eye(sigma.shape[0])
    sigma = make_psd_np(sigma)
    return 0.5 * (sigma + sigma.T)


def construct_C2_21a(date: int, K: Optional[int] = None):
    from src.utils.dates import shift_yyyymm
    lookback = 60
    hist = _meta_all[
        _meta_all["yyyymm"].between(shift_yyyymm(date, -lookback), shift_yyyymm(date, -1))
    ].groupby("permno")["excess_ret"]
    valid_today = _meta_test.loc[_meta_test["yyyymm"] == date, "permno"]
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

    # EWMA covariance from return history
    hist_wide = (
        _meta_all[
            (_meta_all["yyyymm"] >= shift_yyyymm(date, -lookback))
            & (_meta_all["yyyymm"] <= shift_yyyymm(date, -1))
            & (_meta_all["permno"].isin(permnos))
        ]
        .pivot(index="yyyymm", columns="permno", values="excess_ret")
        .sort_index()[permnos]
        .dropna(axis=1, how="any")
    )
    sigma = _ewma_covariance(hist_wide.to_numpy(dtype=float))
    Sigma = torch.tensor(sigma.astype(np.float32), dtype=torch.float32)

    # Firm features at date from X_test (no FeatureScaler — 21April models don't use one)
    today_rows = _meta_test[(_meta_test["yyyymm"] == date) & (_meta_test["permno"].isin(permnos))]
    today_rows = today_rows.drop_duplicates("permno")
    # Map permno → original integer row index shared with X_test
    perm_to_row = {int(row["permno"]): idx for idx, row in today_rows.iterrows()}
    orig_indices = [perm_to_row[p] for p in permnos]

    feature_frame = _X_test.loc[orig_indices, list(_firm_feature_names)].copy()
    feature_frame.index = range(len(feature_frame))  # reset for FeatureScenarioBuilder

    C_t = FeatureScenarioBuilder(
        feature_frame=feature_frame,
        feature_names=_feature_names,
        macro_predictors=_macro_predictors,
        scaler=None,       # 21April models trained without FeatureScaler
        macro_scaler=None, # m is already standardized when passed from the chain
    )

    rets_t = torch.tensor(
        today_rows.set_index("permno").loc[permnos, "excess_ret"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    return Sigma, C_t, rets_t, permnos

warnings.filterwarnings(
    "ignore",
    message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.",
    category=UserWarning,
)


def print_traj(m_traj: torch.Tensor):
    # un-standardize so printed values are in raw macro units
    raw = np.stack([_unstd(m) for m in m_traj])
    df = pd.DataFrame(raw, columns=MACRO_ORDER)
    print(df.describe())
    return df


def _early_cli_int(name: str, default: int) -> int:
    prefix = f"{name}="
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == name and idx + 1 < len(sys.argv):
            return int(sys.argv[idx + 1])
        if arg.startswith(prefix):
            return int(arg.split("=", 1)[1])
    return default


ASSET_SIZE: Optional[int] = 50  # None = all assets with full history; int = top-K by past return

DATE = _early_cli_int("--date", 202004)

LAMBDA = 10.0

KAPPA = 0.5 

GAMMA = 0.5 # controls the strength of the contrast in the G function; higher gamma → stronger contrast, more aggressive moves to increase the return gap (for "distinct_return" contrast)

L2REG = 0.3  # 2026-04-24 sweep winner: high SC-WW gap, 100% SC wins, no box violations.

# Standard MALA parameterization for target pi(m) ∝ exp(-BETA * G(m)):
#   m' = m - ETA * BETA * gradG(m) + sqrt(2 * ETA) * xi
# ETA=0.001 with L2REG=0.3 and box_barrier gave acc=74%, L2_med=1.36,
# gap_mean=+3.32%, SCwin=100%, outBox=0 in the 2026-04-24 sweep.
ETA = 0.001

BETA = 10.0
N_STEPS = 500
N_SEEDS = 20
BASE_SEED = 20260424
RANDOM_START_SCALE = 0.005
DIAGNOSTIC_BURN_IN_FRACTION = 0.50
DIAGNOSTIC_THIN = 5

# ── Regularizer mode (switch to change scenario story) ────────────────────────
# "var1" : penalize distance from VAR(1) one-step prediction A@anchor+c
#           Story: "find plausible next-month states from the anchor that maximise SC-WW gap"
# "l2"   : penalize squared L2 distance from the anchor itself
#           Story: "find nearby variants of the anchor month that flip the SC-WW outcome"
REG_MODE = "l2"
CONSTRAINT_MODE = "box_barrier"  # clip | none | box_barrier | strong_soft

# ── Contrast function (switch to change scenario story) ────────────────────────
# "distinct_return" : maximize |r_SC - r_WW|  (symmetric, flat near anchor → poor acceptance)
# "sc_beats_ww"     : maximize r_SC - r_WW    (linear, non-zero gradient everywhere → good acceptance)
# "ww_beats_sc"     : maximize r_WW - r_SC    (linear)
CONTRAST_FUNCTION = "sc_beats_ww"

# Constraint/regularization knobs used when CONSTRAINT_MODE changes.
BOX_BARRIER_MARGIN_FRACTION = 0.05
BOX_BARRIER_SHARPNESS = 25.0
BOX_BARRIER_WEIGHT = 25.0
STRONG_SOFT_MULTIPLIER = 5.0

_CONTRAST_DESC = {
    "distinct_return": lambda: f"exp(-{GAMMA}*(100*(r_SC-r_WW))²)/{GAMMA}",
    "sc_beats_ww":     lambda: "-100*(r_SC - r_WW)",
    "ww_beats_sc":     lambda: "100*(r_SC - r_WW)",
    "distinct_Sharpe": lambda: f"exp(-{GAMMA}*(10*(s_SC-s_WW))²)/{GAMMA}",
}


# SC trained on stable months, WW trained on stress months
run_dir_summer_child = _RUNTIME / "models/summer_child"
run_dir_winter_wolf  = _RUNTIME / "models/winter_wolf"
summer_model, summer_cfg = load_exact_e2e_model_from_run(run_dir_summer_child)
winter_model, winter_cfg = load_exact_e2e_model_from_run(run_dir_winter_wolf)

Sigma, C_t, rets_t, permnos = construct_C2_21a(DATE, ASSET_SIZE)

summer_pi = AllocationPipeline(summer_model, Sigma)
winter_pi = AllocationPipeline(winter_model, Sigma)

macro_df: pd.DataFrame = _macro_final_21a
macro_columns = list(MACRO_ORDER)
# Standardize anchor: MALA operates in standardized macro space (mirror configured_runner.py)
m0_raw = torch.tensor(
    macro_df.loc[macro_df["yyyymm"] == DATE, macro_columns].to_numpy(dtype=np.float32, copy=True)[0],
    dtype=torch.float32,
)
m0 = (m0_raw - macro_mean) / macro_std  # standardized anchor

# Bounds also in standardized space
macro_arr = torch.tensor(macro_df[macro_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
macro_std_hist = macro_arr.std(dim=0)
macro_min_std  = (macro_arr.min(dim=0).values - macro_mean) / macro_std
macro_max_std  = (macro_arr.max(dim=0).values - macro_mean) / macro_std
a = macro_min_std - macro_std_hist / macro_std
b = macro_max_std + macro_std_hist / macro_std

# m is already standardized — do NOT pass macro_mean/macro_std (would double-standardize)
results_summer, w_summer = evaluate(m0, C_t, rets_t, Sigma, summer_pi)
results_winter, w_winter = evaluate(m0, C_t, rets_t, Sigma, winter_pi)

return_summer = results_summer[0].item()
sharpe_summer = results_summer[2].item()
return_winter = results_winter[0].item()
sharpe_winter = results_winter[2].item()


def build_base_reg_fn(mode: str, anchor: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == "var1":
        return mahalonobis_reg(A, c, Sigma_inv, anchor)
    if mode == "l2":
        def _l2(m: torch.Tensor) -> torch.Tensor:
            diff = m - anchor  # penalize distance from anchor in standardized space
            return diff @ diff
        return _l2
    raise ValueError(f"Unknown REG_MODE: {mode!r}")


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


def build_constraint_setup(
    reg_mode: str,
    constraint_mode: str,
    anchor: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> Dict[str, object]:
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
        raise ValueError(f"Unknown CONSTRAINT_MODE: {constraint_mode!r}")

    return {
        "reg_fn": reg_fn,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "constraint_mode": constraint_mode,
        "description": description,
    }


def build_objective_setup(
    contrast: str,
    reg_mode: str,
    l2reg: float,
    constraint_mode: str = CONSTRAINT_MODE,
) -> Dict[str, object]:
    constraint_setup = build_constraint_setup(
        reg_mode=reg_mode,
        constraint_mode=constraint_mode,
        anchor=m0,
        lower=a,
        upper=b,
    )
    G, gradG = G_contrast_function(
        summer_pi,
        winter_pi,
        C_t,
        rets_t,
        contrast,
        anchor=m0,
        l2reg=l2reg,
        reg_fn=constraint_setup["reg_fn"],
        gamma=GAMMA,
    )
    return {
        **constraint_setup,
        "G": G,
        "gradG": gradG,
        "l2reg": float(l2reg),
        "contrast": contrast,
        "reg_mode": reg_mode,
    }


OBJECTIVE_SETUP = build_objective_setup(
    contrast=CONTRAST_FUNCTION,
    reg_mode=REG_MODE,
    l2reg=L2REG,
    constraint_mode=CONSTRAINT_MODE,
)
G = OBJECTIVE_SETUP["G"]
gradG = OBJECTIVE_SETUP["gradG"]


def configure_objective(
    *,
    contrast_function: Optional[str] = None,
    reg_mode: Optional[str] = None,
    constraint_mode: Optional[str] = None,
    l2reg: Optional[float] = None,
    eta: Optional[float] = None,
    beta: Optional[float] = None,
    random_start_scale: Optional[float] = None,
) -> None:
    """Update MALA hyperparameters and rebuild the objective before a run."""
    global CONTRAST_FUNCTION, REG_MODE, CONSTRAINT_MODE, L2REG, ETA, BETA, RANDOM_START_SCALE, OBJECTIVE_SETUP, G, gradG

    if contrast_function is not None:
        CONTRAST_FUNCTION = contrast_function
    if reg_mode is not None:
        REG_MODE = reg_mode
    if constraint_mode is not None:
        CONSTRAINT_MODE = constraint_mode
    if l2reg is not None:
        L2REG = float(l2reg)
    if eta is not None:
        ETA = float(eta)
    if beta is not None:
        BETA = float(beta)
    if random_start_scale is not None:
        RANDOM_START_SCALE = float(random_start_scale)

    OBJECTIVE_SETUP = build_objective_setup(
        contrast=CONTRAST_FUNCTION,
        reg_mode=REG_MODE,
        l2reg=L2REG,
        constraint_mode=CONSTRAINT_MODE,
    )
    G = OBJECTIVE_SETUP["G"]
    gradG = OBJECTIVE_SETUP["gradG"]

def mahalanobis_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    """Always reports VAR(1) Mahalanobis for interpretability, regardless of REG_MODE."""
    diff = x.detach().float() - (A @ anchor.detach().float() + c)
    return float((diff @ (Sigma_inv @ diff)).item())


def mahalanobis_chi2_summary(mah2: float) -> Dict[str, float]:
    """Interpret VAR(1) Mahalanobis² as a chi-square statistic in macro dimension."""
    statistic = max(float(mah2), 0.0)
    df = len(MACRO_ORDER)
    return {
        "mah_chi2_percentile": float(chi2.cdf(statistic, df=df)),
        "mah_chi2_tail": float(chi2.sf(statistic, df=df)),
    }


def l2_squared(x: torch.Tensor, anchor: torch.Tensor) -> float:
    diff = x.detach().float() - anchor.detach().float()
    return float((diff @ diff).item())


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


def regime_summary(macro_state: torch.Tensor) -> Dict[str, object]:
    # regime_classifier expects raw macro — un-standardize first
    raw = _unstd(macro_state)
    result = regime_classifier.classify(raw)
    return {
        "label": result["label"],
        "probabilities": result["probabilities"],
    }


def format_regime(regime_info: Dict[str, object]) -> str:
    probs = regime_info["probabilities"]
    assert isinstance(probs, dict)
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name}={value:.2%}" for name, value in ordered[:2])
    return f"{regime_info['label']} ({top})"


def evaluate_pair(macro_state: torch.Tensor) -> Dict[str, float]:
    # macro_state is standardized — no macro_mean/macro_std needed
    summer_results, _ = evaluate(macro_state, C_t, rets_t, Sigma, summer_pi)
    winter_results, _ = evaluate(macro_state, C_t, rets_t, Sigma, winter_pi)
    summer_return = float(summer_results[0].item())
    winter_return = float(winter_results[0].item())
    summer_sharpe = float(summer_results[2].item())
    winter_sharpe = float(winter_results[2].item())
    winner = "summer_child" if summer_return >= winter_return else "winter_wolf"
    return {
        "summer_return": summer_return,
        "winter_return": winter_return,
        "summer_sharpe": summer_sharpe,
        "winter_sharpe": winter_sharpe,
        "return_gap": summer_return - winter_return,
        "sharpe_gap": summer_sharpe - winter_sharpe,
        "winner": winner,
    }


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


def summarize_ess(traj: torch.Tensor, accepted_flags: List[bool]) -> Dict[str, float]:
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


def _unstd(t: torch.Tensor) -> np.ndarray:
    return (t.detach().float() * macro_std + macro_mean).cpu().numpy()


def _save_trajectory_tensors(results: dict, out_dir: Path, ts: str) -> None:
    import json

    trajs = results.get("m_trajs_contrast", [])
    if not trajs:
        return

    output_format = str(results.get("trajectory_format", "both")).lower()
    if output_format not in {"pt", "npy", "both"}:
        raise ValueError(f"Unknown trajectory_format={output_format!r}; use pt, npy, or both.")

    thin = max(1, int(results.get("trajectory_thin", 1)))
    burn_frac = float(results.get("trajectory_burn_in_fraction", DIAGNOSTIC_BURN_IN_FRACTION))
    n_steps = int(trajs[0].shape[0])
    burn_in = int(np.floor(burn_frac * n_steps))
    burn_in = min(max(burn_in, 0), max(n_steps - 1, 0))

    standardized = torch.stack(
        [traj.detach().cpu().to(dtype=torch.float32) for traj in trajs],
        dim=0,
    )
    mean = macro_mean.detach().cpu().to(dtype=torch.float32).reshape(1, 1, -1)
    std = macro_std.detach().cpu().to(dtype=torch.float32).reshape(1, 1, -1)
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
            "macro_columns": list(macro_columns),
            "date": int(DATE),
            "scenario": "scenario4",
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
        "scenario": "scenario4",
        "date": int(DATE),
        "n_seeds_completed": int(standardized.shape[0]),
        "n_steps_per_seed": int(standardized.shape[1]),
        "n_macro_variables": int(standardized.shape[2]),
        "macro_columns": list(macro_columns),
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


def macro_shift_frame(start: torch.Tensor, final: torch.Tensor) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "macro": macro_columns,
            "anchor": _unstd(m0),
            "start":  _unstd(start),
            "final":  _unstd(final),
        }
    )
    frame["shift_vs_anchor"] = frame["final"] - frame["anchor"]
    frame["shift_vs_start"]  = frame["final"] - frame["start"]
    return frame


def _historical_raw_frame() -> pd.DataFrame:
    return macro_df[["yyyymm", *macro_columns]].copy()


def _raw_to_std_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    values = torch.tensor(frame[macro_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    standardized = ((values - macro_mean) / macro_std).cpu().numpy()
    frame.loc[:, macro_columns] = standardized
    return frame


def _state_record(
    state: torch.Tensor,
    seed: int,
    step: int,
    source: str,
    accepted: Optional[bool] = None,
    accept_prob: Optional[float] = None,
    log_accept_ratio: Optional[float] = None,
) -> Dict[str, object]:
    raw = _unstd(state)
    regime = regime_classifier.classify(raw)
    probabilities = regime["probabilities"]
    assert isinstance(probabilities, dict)
    mah2 = mahalanobis_squared(state, m0)
    record: Dict[str, object] = {
        "seed": seed,
        "step": step,
        "source": source,
        "regime": regime["label"],
        "l2_dist": l2_squared(state, m0) ** 0.5,
        "mah2": mah2,
        "mah_dist": mah2 ** 0.5,
        **mahalanobis_chi2_summary(mah2),
        "box_violation": box_violation_squared(state, a, b) ** 0.5,
    }
    if accepted is not None:
        record["accepted"] = bool(accepted)
    if accept_prob is not None:
        record["accept_prob"] = float(accept_prob)
    if log_accept_ratio is not None:
        record["log_accept_ratio"] = float(log_accept_ratio)
    for name, value in zip(macro_columns, raw):
        record[name] = float(value)
    standardized = state.detach().float().cpu().numpy()
    for name, value in zip(macro_columns, standardized):
        record[f"{name}_std"] = float(value)
    for name, value in probabilities.items():
        record[f"prob_{name}"] = float(value)
    return record


def _diagnostic_steps(n_steps: int) -> List[int]:
    burn_in = int(np.floor(DIAGNOSTIC_BURN_IN_FRACTION * n_steps))
    burn_in = min(max(burn_in, 0), max(n_steps - 1, 0))
    thin = max(1, int(DIAGNOSTIC_THIN))
    steps = list(range(burn_in, n_steps, thin))
    return steps if steps else [max(n_steps - 1, 0)]


def build_chain_state_frame(results: dict) -> pd.DataFrame:
    rows = []
    for seed_idx, traj in enumerate(results["m_trajs_contrast"], start=1):
        summary = results["seed_summaries"][seed_idx - 1]
        chain_info = summary.get("chain_info", {})
        accepted = chain_info.get("accepted", [])
        accept_probs = chain_info.get("accept_probs", [])
        log_accept_ratios = chain_info.get("log_accept_ratios", [])
        for step in _diagnostic_steps(traj.shape[0]):
            rows.append(
                _state_record(
                    traj[step],
                    seed=seed_idx,
                    step=step + 1,
                    source="post_burnin_thinned",
                    accepted=accepted[step] if step < len(accepted) else None,
                    accept_prob=accept_probs[step] if step < len(accept_probs) else None,
                    log_accept_ratio=log_accept_ratios[step] if step < len(log_accept_ratios) else None,
                )
            )
    return pd.DataFrame(rows)


def build_final_state_frame(results: dict) -> pd.DataFrame:
    rows = []
    for seed_idx, state in enumerate(results["m_lasts_contrast"], start=1):
        record = _state_record(state, seed=seed_idx, step=-1, source="final")
        summary = results["seed_summaries"][seed_idx - 1]
        record.update(
            {
                "winner": summary["winner"],
                "return_gap": summary["return_gap"],
                "sharpe_gap": summary["sharpe_gap"],
                "summer_return": summary["summer_return"],
                "winter_return": summary["winter_return"],
                "summer_sharpe": summary["summer_sharpe"],
                "winter_sharpe": summary["winter_sharpe"],
                "accept_rate": summary["accept_rate"],
                "ess_mean": summary["ess_mean"],
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def build_regime_transition_frame(results: dict) -> pd.DataFrame:
    rows = []
    for summary in results["seed_summaries"]:
        rows.append(
            {
                "seed": summary["seed"],
                "start_regime": summary["start_regime"]["label"],
                "final_regime": summary["final_regime"]["label"],
                "return_gap": summary["return_gap"],
                "sharpe_gap": summary["sharpe_gap"],
                "l2_dist": summary["final_l2_dist"],
                "mah2": summary["final_mah2"],
                "mah_dist": summary["final_mah_dist"],
                "mah_chi2_percentile": summary["final_mah_chi2_percentile"],
                "mah_chi2_tail": summary["final_mah_chi2_tail"],
            }
        )
    return pd.DataFrame(rows)


def _plot_density_panel(hist_raw: pd.DataFrame, sample_raw: pd.DataFrame, output_dir: Path, ts: str) -> None:
    anchor_raw = _unstd(m0)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.8))
    hist_color = "#5b6770"
    gen_color = "#0b7a75"
    median_color = "#c44e52"
    for ax, col, anchor_value in zip(axes.ravel(), macro_columns, anchor_raw):
        hist_values = hist_raw[col].to_numpy(dtype=float)
        gen_values = sample_raw[col].to_numpy(dtype=float)
        lo = np.nanmin([np.nanmin(hist_values), np.nanmin(gen_values), anchor_value])
        hi = np.nanmax([np.nanmax(hist_values), np.nanmax(gen_values), anchor_value])
        if np.isclose(lo, hi):
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 34)
        ax.hist(hist_values, bins=bins, density=True, color=hist_color, alpha=0.24, label="Historical")
        ax.hist(gen_values, bins=bins, density=True, histtype="step", linewidth=2.0, color=gen_color, label="Generated")
        ax.axvline(anchor_value, color="#111111", linewidth=1.8, linestyle="--", label="Anchor")
        ax.axvline(np.nanmedian(gen_values), color=median_color, linewidth=1.8, label="Generated median")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, linewidth=0.6)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Scenario 4 Generated Macro Distribution vs Historical Macro Panel", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    fig.savefig(output_dir / f"macro_distribution_shift_{ts}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"macro_distribution_shift_{ts}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_standardized_shift(sample_frame: pd.DataFrame, output_dir: Path, ts: str) -> None:
    rows = []
    for col in macro_columns:
        values = sample_frame[f"{col}_std"].to_numpy(dtype=float) - float(m0[macro_columns.index(col)].item())
        rows.append(
            {
                "macro": col,
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanquantile(values, 0.25)),
                "q75": float(np.nanquantile(values, 0.75)),
            }
        )
    shift = pd.DataFrame(rows).sort_values("median")
    y = np.arange(len(shift))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    colors = ["#b55d4c" if value < 0 else "#287c71" for value in shift["median"]]
    ax.barh(y, shift["median"], color=colors, alpha=0.88)
    ax.errorbar(
        shift["median"],
        y,
        xerr=[shift["median"] - shift["q25"], shift["q75"] - shift["median"]],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0.0, color="#111111", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(shift["macro"])
    ax.set_xlabel("Generated minus anchor, standardized macro units")
    ax.set_title("Macro Shift Required for SummerChild to Beat WinterWolf", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / f"standardized_macro_shift_{ts}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"standardized_macro_shift_{ts}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_regime_diagnostics(sample_frame: pd.DataFrame, transition_frame: pd.DataFrame, output_dir: Path, ts: str) -> None:
    classes = list(regime_classifier.classes)
    anchor_probs = regime_classifier.classify(_unstd(m0))["probabilities"]
    assert isinstance(anchor_probs, dict)
    generated_probs = {name: float(sample_frame[f"prob_{name}"].mean()) for name in classes if f"prob_{name}" in sample_frame}
    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    width = 0.38
    axes[0].bar(x - width / 2, [anchor_probs.get(name, 0.0) for name in classes], width=width, color="#4c566a", label="Anchor")
    axes[0].bar(x + width / 2, [generated_probs.get(name, 0.0) for name in classes], width=width, color="#d08770", label="Generated")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=25, ha="right")
    axes[0].set_ylabel("Mean classifier probability")
    axes[0].set_title("Regime Probability Shift", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    matrix = pd.crosstab(transition_frame["start_regime"], transition_frame["final_regime"])
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    im = axes[1].imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu")
    axes[1].set_xticks(np.arange(len(classes)))
    axes[1].set_xticklabels(classes, rotation=35, ha="right")
    axes[1].set_yticks(np.arange(len(classes)))
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("Final regime")
    axes[1].set_ylabel("Start regime")
    axes[1].set_title("Seed-Level Regime Transitions", fontsize=12, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1].text(j, i, int(matrix.iat[i, j]), ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"regime_diagnostics_{ts}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"regime_diagnostics_{ts}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_performance_diagnostics(final_frame: pd.DataFrame, output_dir: Path, ts: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    scatter = axes[0].scatter(
        final_frame["l2_dist"],
        100.0 * final_frame["return_gap"],
        c=final_frame["accept_rate"],
        cmap="viridis",
        s=58,
        edgecolor="#222222",
        linewidth=0.4,
    )
    axes[0].axhline(0.0, color="#111111", linewidth=1.0)
    axes[0].set_xlabel("Final L2 distance from anchor")
    axes[0].set_ylabel("Final return gap, SC minus WW (%)")
    axes[0].set_title("Target Achievement Within Plausible Distance", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.22)
    fig.colorbar(scatter, ax=axes[0], label="Chain acceptance rate")

    axes[1].hist(final_frame["accept_rate"], bins=np.linspace(0, 1, 16), color="#4c78a8", alpha=0.78)
    axes[1].axvspan(0.50, 0.80, color="#59a14f", alpha=0.14, label="Healthy high-acceptance band")
    axes[1].set_xlabel("Acceptance rate")
    axes[1].set_ylabel("Number of chains")
    axes[1].set_title("MALA Mixing Diagnostic", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / f"mala_performance_diagnostics_{ts}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"mala_performance_diagnostics_{ts}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_trace_diagnostics(results: dict, output_dir: Path, ts: str) -> None:
    selected = ["dp", "ep", "bm", "dfy"]
    fig, axes = plt.subplots(len(selected), 1, figsize=(11.5, 8.5), sharex=True)
    for ax, col in zip(axes, selected):
        idx = macro_columns.index(col)
        for seed_idx, traj in enumerate(results["m_trajs_contrast"][: min(8, len(results["m_trajs_contrast"]))], start=1):
            values = traj[:, idx].detach().cpu().numpy()
            ax.plot(np.arange(1, len(values) + 1), values, linewidth=0.8, alpha=0.72, label=f"seed {seed_idx}" if col == selected[0] else None)
        ax.axhline(float(m0[idx].item()), color="#111111", linewidth=1.0, linestyle="--")
        ax.set_ylabel(f"{col} z")
        ax.grid(alpha=0.20)
    axes[0].set_title("MALA Trace Diagnostics in Standardized Macro Space", fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("MALA step")
    axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"mala_trace_diagnostics_{ts}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"mala_trace_diagnostics_{ts}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_diagnostic_markdown(
    output_dir: Path,
    ts: str,
    sample_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    n_steps: int,
    n_seeds: int,
) -> None:
    regime_counts = sample_frame["regime"].value_counts(normalize=True).mul(100.0)
    shift_lines = []
    for col in macro_columns:
        shift = sample_frame[f"{col}_std"].to_numpy(dtype=float) - float(m0[macro_columns.index(col)].item())
        shift_lines.append(f"- `{col}`: median shift `{np.nanmedian(shift):+.3f}` z, IQR [`{np.nanquantile(shift, 0.25):+.3f}`, `{np.nanquantile(shift, 0.75):+.3f}`]")
    transition_counts = (
        transition_frame.groupby(["start_regime", "final_regime"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["start_regime", "final_regime"])
    )
    transition_lines = [
        f"- `{row.start_regime}` to `{row.final_regime}`: `{int(row.count)}`"
        for row in transition_counts.itertuples(index=False)
    ]
    var1_lines = []
    if "mah_chi2_tail" in final_frame.columns:
        var1_lines = [
            "",
            "## VAR(1)-Forecast Distance",
            "",
            f"- Final median VAR(1)-forecast Mahalanobis distance: `{final_frame['mah_dist'].median():.3f}`",
            f"- Final median VAR(1)-forecast chi-square tail probability: `{final_frame['mah_chi2_tail'].median():.2g}`",
            f"- Share inside 95% chi-square contour: `{100.0 * (final_frame['mah_chi2_percentile'] <= 0.95).mean():.1f}%`",
        ]
    nn_lines = []
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
    caution_lines = []
    if "mah_chi2_tail" in final_frame.columns:
        var_tail = float(final_frame["mah_chi2_tail"].median())
        if var_tail < 0.05:
            caution_lines = [
                "",
                "## Manuscript Caution",
                "",
                (
                    f"- VAR(1)-forecast tail is `{var_tail:.3g}`. Do not describe this run as "
                    "VAR(1)-innovation plausible without a separate confirmed plausibility statistic."
                ),
            ]
    lines = [
        "# Scenario 4 Diagnostics",
        "",
        f"- Anchor month: `{DATE}`",
        f"- Objective: `{CONTRAST_FUNCTION}` with `{REG_MODE}` regularization and `{CONSTRAINT_MODE}` constraints",
        f"- Hyperparameters: `L2REG={L2REG}`, `BETA={BETA}`, `ETA={ETA}`, `N_STEPS={n_steps}`, `N_SEEDS={n_seeds}`",
        f"- Generated diagnostic sample size: `{len(sample_frame)}` post-burn-in thinned states",
        f"- Final SC win rate: `{100.0 * (final_frame['winner'] == 'summer_child').mean():.1f}%`",
        f"- Final mean SC-WW return gap: `{100.0 * final_frame['return_gap'].mean():+.3f}%`",
        f"- Final median L2 distance: `{final_frame['l2_dist'].median():.3f}`",
        f"- Final median VAR(1)-forecast Mahalanobis distance: `{final_frame['mah_dist'].median():.3f}`",
        f"- Mean box violation: `{final_frame['box_violation'].mean():.4f}`",
        *var1_lines,
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
        *shift_lines,
        "",
        "## Regime Transition Counts",
        "",
        *transition_lines,
        "",
    ]
    (output_dir / f"diagnostic_storyline_{ts}.md").write_text("\n".join(lines), encoding="utf-8")


def run_mala_chain(
    start: torch.Tensor,
    n_steps: int,
):
    clip_low = OBJECTIVE_SETUP["clip_low"]
    clip_high = OBJECTIVE_SETUP["clip_high"]
    x = start.detach().clone().to(dtype=m0.dtype)
    traj = []
    accepted: List[bool] = []
    accept_probs: List[float] = []
    log_accept_ratios: List[float] = []
    boundary_touches = 0

    for step in range(n_steps):
        hyps = (G, gradG, ETA, BETA, clip_low, clip_high)
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
        "n_steps": int(n_steps),
        "eta": float(ETA),
        "boundary_touch_rate": float(boundary_touches / n_steps) if n_steps > 0 else 0.0,
        "constraint_mode": OBJECTIVE_SETUP["constraint_mode"],
        "final_box_violation": box_violation_squared(x.detach(), a, b) ** 0.5,
    }
    return x.detach(), torch.stack(traj), info


def print_run_header(n_seeds: int, n_steps: int, base_seed: Optional[int] = BASE_SEED) -> None:
    anchor_regime = regime_summary(m0)
    eval_summary = regime_classifier.eval_summary["test"]
    print("=" * 88)
    print("Scenario 4 | Universe500 root runtime")
    print(
        f"Question: find plausible macro states near {DATE} where SummerChild and WinterWolf "
        f"produce distinct realized returns on the same {len(permnos)}-asset Universe500 basket."
    )
    reg_desc = (
        f"VAR1-Mahalanobis²(m, A@anchor+c)"
        if REG_MODE == "var1"
        else f"L2²(m, anchor)"
    )
    contrast_desc = _CONTRAST_DESC.get(CONTRAST_FUNCTION, lambda: CONTRAST_FUNCTION)()
    print(
        f"Objective: {contrast_desc} + {L2REG:.4g} * {reg_desc}  "
        f"[contrast={CONTRAST_FUNCTION!r}, REG_MODE={REG_MODE!r}, CONSTRAINT_MODE={CONSTRAINT_MODE!r}]"
    )
    print(
        f"Chain: MALA, steps={n_steps}, seeds={n_seeds}, beta={BETA:.2f}, "
        f"eta(fixed)={ETA:.6f} (constant-step, no decay), random_start_scale={RANDOM_START_SCALE:.4f}"
    )
    start_seed_desc = "unseeded" if base_seed is None else f"base_seed={base_seed} (chain i uses base_seed+i)"
    print(f"Random starts: {start_seed_desc}")
    print(f"Constraint handling: {OBJECTIVE_SETUP['description']}")
    print(
        f"Models: SC={run_dir_summer_child.name} (lambda={summer_cfg['lambda']}, "
        f"kappa={summer_cfg['kappa']}), WW={run_dir_winter_wolf.name} "
        f"(lambda={winter_cfg['lambda']}, kappa={winter_cfg['kappa']})"
    )
    print(
        f"Anchor month {DATE}: SC return={return_summer:.2%}, SC Sharpe={sharpe_summer:.2f}, "
        f"WW return={return_winter:.2%}, WW Sharpe={sharpe_winter:.2f}"
    )
    true_label = regime_classifier.historical_label(DATE)
    true_label_str = true_label if true_label is not None else "unknown"
    print(f"Anchor classifier regime: {format_regime(anchor_regime)}  |  true hard label: {true_label_str}")
    print(
        f"Regime classifier held-out test: accuracy={eval_summary['accuracy']:.3f}, "
        f"macro_f1={eval_summary['macro_f1']:.3f}, months={eval_summary['n_months']}"
    )
    print("=" * 88)


def print_seed_summary(seed_idx: int, summary: Dict[str, object]) -> None:
    print(
        f"[seed {seed_idx:02d}] accept_rate={summary['accept_rate']:.1%}, "
        f"mean_accept_prob={summary['mean_accept_prob']:.1%}, "
        f"eta(fixed)={summary['eta']:.6f}, "
        f"boundary_touch_rate={summary['boundary_touch_rate']:.1%}, "
        f"box_violation={summary['final_box_violation']:.4f}, "
        f"accepted_moves={int(summary['accepted_moves'])}, "
        f"state_changes={int(summary['state_change_count'])}, "
        f"ESS cap(min/mean/max)={summary['ess_min']:.1f}/{summary['ess_mean']:.1f}/{summary['ess_max']:.1f}"
    )
    print(
        f"[seed {seed_idx:02d}] start regime={format_regime(summary['start_regime'])}, "
        f"final regime={format_regime(summary['final_regime'])}"
    )
    if REG_MODE == "var1":
        print(
            f"[seed {seed_idx:02d}] VAR1-Mah: start={summary['start_mah_dist']:.3f} "
            f"(sq={summary['start_mah2']:.3f}, tail={summary['start_mah_chi2_tail']:.2g}), "
            f"final={summary['final_mah_dist']:.3f} "
            f"(sq={summary['final_mah2']:.3f}, tail={summary['final_mah_chi2_tail']:.2g})"
        )
    else:
        print(
            f"[seed {seed_idx:02d}] L2(std): start={summary['start_l2_dist']:.4f}, "
            f"final={summary['final_l2_dist']:.4f}"
        )
    print(
        f"[seed {seed_idx:02d}] final winner={summary['winner']}, "
        f"return_gap(SC-WW)={summary['return_gap']:.4%}, "
        f"sharpe_gap(SC-WW)={summary['sharpe_gap']:.4f}"
    )
    frame = summary["macro_shift_frame"]
    assert isinstance(frame, pd.DataFrame)
    print(frame.to_string(index=False, float_format=lambda value: f"{value: .4f}"))
    print("-" * 88)


def main(
    n_seeds: int = N_SEEDS,
    lasts: Optional[List[torch.Tensor]] = None,
    n_steps: int = N_STEPS,
    base_seed: Optional[int] = BASE_SEED,
    save_trajectory_tensors: bool = False,
    trajectory_format: str = "both",
    trajectory_burn_in_fraction: float = DIAGNOSTIC_BURN_IN_FRACTION,
    trajectory_thin: int = 1,
):
    m_trajs_contrast = []
    m_lasts_contrast = []
    seed_summaries = []
    if lasts is not None:
        n_seeds = len(lasts)

    print_run_header(n_seeds=n_seeds, n_steps=n_steps, base_seed=base_seed)

    for i in range(n_seeds):
        chain_seed = None if base_seed is None else int(base_seed) + i
        if chain_seed is not None:
            torch.manual_seed(chain_seed)
        if lasts is None:
            generator = None
            if chain_seed is not None:
                generator = torch.Generator(device=m0.device)
                generator.manual_seed(chain_seed)
            random_draw = torch.rand(
                (9,),
                dtype=m0.dtype,
                device=m0.device,
                generator=generator,
            )
            m_start = m0 + (b - a) * RANDOM_START_SCALE * random_draw
        else:
            m_start = lasts[i].detach().clone().to(dtype=m0.dtype)
        print(f"[seed {i + 1:02d}] running contrastive trajectory from start state")
        seed_start_time = time.time()
        try:
            m_last, m_traj, chain_info = run_mala_chain(m_start, n_steps)
        except KeyboardInterrupt:
            elapsed = time.time() - seed_start_time
            print(f"[seed {i + 1:02d}] interrupted after {elapsed:.1f}s; returning completed seeds only")
            break
        m_lasts_contrast.append(m_last)
        m_trajs_contrast.append(m_traj)

        ess_summary = summarize_ess(m_traj, chain_info["accepted"])
        start_mah2 = mahalanobis_squared(m_start, m0)
        final_mah2 = mahalanobis_squared(m_last, m0)
        start_mah_chi2 = mahalanobis_chi2_summary(start_mah2)
        final_mah_chi2 = mahalanobis_chi2_summary(final_mah2)
        start_l2 = l2_squared(m_start, m0)
        final_l2 = l2_squared(m_last, m0)
        final_eval = evaluate_pair(m_last)
        summary = {
            "seed": i + 1,
            "start_regime": regime_summary(m_start),
            "final_regime": regime_summary(m_last),
            "start_mah2": start_mah2,
            "final_mah2": final_mah2,
            "start_mah_dist": start_mah2 ** 0.5,
            "final_mah_dist": final_mah2 ** 0.5,
            "start_mah_chi2_percentile": start_mah_chi2["mah_chi2_percentile"],
            "start_mah_chi2_tail": start_mah_chi2["mah_chi2_tail"],
            "final_mah_chi2_percentile": final_mah_chi2["mah_chi2_percentile"],
            "final_mah_chi2_tail": final_mah_chi2["mah_chi2_tail"],
            "start_l2": start_l2,
            "final_l2": final_l2,
            "start_l2_dist": start_l2 ** 0.5,
            "final_l2_dist": final_l2 ** 0.5,
            "accept_rate": chain_info["accept_rate"],
            "mean_accept_prob": chain_info["mean_accept_prob"],
            "eta": chain_info["eta"],
            "boundary_touch_rate": chain_info["boundary_touch_rate"],
            "final_box_violation": chain_info["final_box_violation"],
            "chain_info": chain_info,
            "macro_shift_frame": macro_shift_frame(m_start, m_last),
            **ess_summary,
            **final_eval,
        }
        seed_summaries.append(summary)
        summary["elapsed_seconds"] = time.time() - seed_start_time
        print_seed_summary(i + 1, summary)

    results = {
        "date": DATE,
        "gamma": LAMBDA,
        "kappa": KAPPA,
        "n_assets": ASSET_SIZE,
        "beta": BETA,
        "constraint_mode": CONSTRAINT_MODE,
        "base_seed": None if base_seed is None else int(base_seed),
        "n_steps": int(n_steps),
        "n_seeds": int(n_seeds),
        "m_trajs_contrast": m_trajs_contrast,
        "m_lasts_contrast": m_lasts_contrast,
        "seed_summaries": seed_summaries,
        "save_trajectory_tensors": bool(save_trajectory_tensors),
        "trajectory_format": trajectory_format,
        "trajectory_burn_in_fraction": float(trajectory_burn_in_fraction),
        "trajectory_thin": int(trajectory_thin),
    }
    _save_results(results)
    return results


def _save_results(results: dict) -> None:
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "scenario_outputs" / f"scenario4_{DATE}" / "runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed-level summary CSV
    rows = []
    for s in results["seed_summaries"]:
        frame = s["macro_shift_frame"]
        shift = {f"shift_{row['macro']}": row["shift_vs_anchor"] for _, row in frame.iterrows()}
        rows.append({
            "seed": s["seed"],
            "winner": s["winner"],
            "return_gap": s["return_gap"],
            "sharpe_gap": s["sharpe_gap"],
            "summer_return": s["summer_return"],
            "winter_return": s["winter_return"],
            "summer_sharpe": s["summer_sharpe"],
                "winter_sharpe": s["winter_sharpe"],
                "final_regime": s["final_regime"]["label"],
                "final_mah2": s["final_mah2"],
                "final_mah_dist": s["final_mah_dist"],
                "final_mah_chi2_percentile": s["final_mah_chi2_percentile"],
                "final_mah_chi2_tail": s["final_mah_chi2_tail"],
                "accept_rate": s["accept_rate"],
                "boundary_touch_rate": s["boundary_touch_rate"],
            "final_box_violation": s["final_box_violation"],
            "ess_mean": s["ess_mean"],
            "elapsed_s": s.get("elapsed_seconds", float("nan")),
            **shift,
        })
    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / f"seed_summary_{ts}.csv"
    summary_df.to_csv(csv_path, index=False)

    final_frame = build_final_state_frame(results)
    sample_frame = build_chain_state_frame(results)
    transition_frame = build_regime_transition_frame(results)
    hist_raw = _historical_raw_frame()

    nn_index = _build_nn_index(macro_columns=macro_columns)
    std_cols = [f"{c}_std" for c in macro_columns]
    if not final_frame.empty:
        final_nn = nn_index.attach(final_frame[std_cols].to_numpy(dtype=np.float64), k=3)
        final_nn.index = final_frame.index
        final_frame = pd.concat([final_frame, final_nn], axis=1)
    if not sample_frame.empty:
        chain_nn = nn_index.attach(sample_frame[std_cols].to_numpy(dtype=np.float64), k=3)
        chain_nn.index = sample_frame.index
        sample_frame = pd.concat([sample_frame, chain_nn], axis=1)
    if not final_frame.empty:
        nn_pull = final_frame[["seed"] + [
            f"nn_{m}_yyyymm_1" for m in ("var1", "hist", "eucl")
        ] + [f"nn_{m}_dist_1" for m in ("var1", "hist", "eucl")]].copy()
        nn_pull = nn_pull.rename(columns={
            **{f"nn_{m}_yyyymm_1": f"final_nn_{m}_yyyymm_1" for m in ("var1", "hist", "eucl")},
            **{f"nn_{m}_dist_1": f"final_nn_{m}_dist_1" for m in ("var1", "hist", "eucl")},
        })
        summary_df = summary_df.merge(nn_pull, on="seed", how="left")
        summary_df.to_csv(csv_path, index=False)

    final_frame.to_csv(out_dir / f"final_state_diagnostics_{ts}.csv", index=False)
    sample_frame.to_csv(out_dir / f"generated_macro_sample_postburnin_{ts}.csv", index=False)
    transition_frame.to_csv(out_dir / f"regime_transitions_{ts}.csv", index=False)
    hist_raw.to_csv(out_dir / f"historical_macro_panel_{ts}.csv", index=False)

    # Final macro states in raw (un-standardized) units, retained for backward compatibility.
    if results["m_lasts_contrast"]:
        lasts_np = np.stack([_unstd(m) for m in results["m_lasts_contrast"]])
        lasts_df = pd.DataFrame(lasts_np, columns=macro_columns)
        lasts_df.to_csv(out_dir / f"final_states_{ts}.csv", index=False)

    if results.get("save_trajectory_tensors"):
        _save_trajectory_tensors(results, out_dir, ts)

    if not sample_frame.empty and not final_frame.empty:
        _plot_density_panel(hist_raw=hist_raw, sample_raw=sample_frame, output_dir=out_dir, ts=ts)
        _plot_standardized_shift(sample_frame=sample_frame, output_dir=out_dir, ts=ts)
        _plot_regime_diagnostics(sample_frame=sample_frame, transition_frame=transition_frame, output_dir=out_dir, ts=ts)
        _plot_performance_diagnostics(final_frame=final_frame, output_dir=out_dir, ts=ts)
        _plot_trace_diagnostics(results=results, output_dir=out_dir, ts=ts)
        _write_diagnostic_markdown(
            output_dir=out_dir,
            ts=ts,
            sample_frame=sample_frame,
            final_frame=final_frame,
            transition_frame=transition_frame,
            n_steps=int(results.get("n_steps", N_STEPS)),
            n_seeds=int(results.get("n_seeds", len(final_frame))),
        )

    # Run config JSON
    config = {
        "DATE": DATE, "ASSET_SIZE": ASSET_SIZE, "ETA": ETA, "BETA": BETA,
        "N_STEPS": results.get("n_steps", N_STEPS), "N_SEEDS": results.get("n_seeds", N_SEEDS),
        "BASE_SEED": results.get("base_seed", BASE_SEED),
        "L2REG": L2REG, "GAMMA": GAMMA,
        "REG_MODE": REG_MODE, "CONSTRAINT_MODE": CONSTRAINT_MODE,
        "CONTRAST_FUNCTION": CONTRAST_FUNCTION,
        "RANDOM_START_SCALE": RANDOM_START_SCALE,
        "SAVE_TRAJECTORY_TENSORS": bool(results.get("save_trajectory_tensors", False)),
        "TRAJECTORY_FORMAT": results.get("trajectory_format", "both"),
        "TRAJECTORY_BURN_IN_FRACTION": results.get("trajectory_burn_in_fraction", DIAGNOSTIC_BURN_IN_FRACTION),
        "TRAJECTORY_THIN": results.get("trajectory_thin", 1),
        "DIAGNOSTIC_BURN_IN_FRACTION": DIAGNOSTIC_BURN_IN_FRACTION,
        "DIAGNOSTIC_THIN": DIAGNOSTIC_THIN,
        "LAMBDA": LAMBDA, "KAPPA": KAPPA,
        "BOX_BARRIER_WEIGHT": BOX_BARRIER_WEIGHT,
        "BOX_BARRIER_MARGIN_FRACTION": BOX_BARRIER_MARGIN_FRACTION,
        "STRONG_SOFT_MULTIPLIER": STRONG_SOFT_MULTIPLIER,
        "ANCHOR_SUMMER_RETURN": return_summer,
        "ANCHOR_WINTER_RETURN": return_winter,
        "ANCHOR_RETURN_GAP": return_summer - return_winter,
        "ANCHOR_SUMMER_SHARPE": sharpe_summer,
        "ANCHOR_WINTER_SHARPE": sharpe_winter,
        "ANCHOR_REGIME": regime_summary(m0),
        "ANCHOR_TRUE_LABEL": regime_classifier.historical_label(DATE),
    }
    (out_dir / f"config_{ts}.json").write_text(json.dumps(config, indent=2))

    print(f"\nResults saved → {out_dir.relative_to(ROOT)} ({ts})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Scenario 4 MALA probing with paper-grade macro and regime diagnostics."
    )
    parser.add_argument("--date", type=int, default=DATE, help="Anchor yyyymm. Parsed before runtime setup.")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS, help="Number of independent MALA chains.")
    parser.add_argument("--n-steps", type=int, default=N_STEPS, help="MALA steps per chain.")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="Base RNG seed for chain starts; chain i uses base_seed+i. Use -1 for unseeded starts.",
    )
    parser.add_argument(
        "--reg-mode",
        choices=["l2", "var1"],
        default=REG_MODE,
        help="Regularizer: L2 distance from anchor or VAR(1) Mahalanobis distance from A@anchor+c.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["clip", "none", "box_barrier", "strong_soft"],
        default=CONSTRAINT_MODE,
        help="Constraint handling for the historical macro box.",
    )
    parser.add_argument(
        "--contrast-function",
        choices=["sc_beats_ww", "ww_beats_sc", "distinct_return", "distinct_Sharpe"],
        default=CONTRAST_FUNCTION,
        help="Target contrast for scenario generation.",
    )
    parser.add_argument("--l2reg", type=float, default=L2REG, help="Regularization weight.")
    parser.add_argument("--eta", type=float, default=ETA, help="Fixed MALA step size.")
    parser.add_argument("--beta", type=float, default=BETA, help="MALA inverse-temperature multiplier.")
    parser.add_argument(
        "--random-start-scale",
        type=float,
        default=RANDOM_START_SCALE,
        help="Perturbation scale around the 202004 anchor. Use 0 to start every chain exactly at 202004.",
    )
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

    configure_objective(
        contrast_function=args.contrast_function,
        reg_mode=args.reg_mode,
        constraint_mode=args.constraint_mode,
        l2reg=args.l2reg,
        eta=args.eta,
        beta=args.beta,
        random_start_scale=args.random_start_scale,
    )
    base_seed = None if args.base_seed < 0 else int(args.base_seed)
    main(
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        base_seed=base_seed,
        save_trajectory_tensors=args.save_trajectory_tensors,
        trajectory_format=args.trajectory_format,
        trajectory_burn_in_fraction=args.trajectory_burn_in_frac,
        trajectory_thin=args.trajectory_thin,
    )
