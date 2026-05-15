"""Hyperparameter sweep for scenario4.py.

Reuses the heavy objects already built by scenario4 (models, Sigma, C_t, rets_t,
anchor m0, bounds a/b) and rebuilds only G, gradG and the loop parameters per
config. The sweep compares multiple box-handling / regularization strategies:
hard clip, no clip, smooth box barrier in G, and stronger soft regularization.

Usage:
    python scripts/scenario4_sweep.py

Edit GRID / CONSTRAINT_MODES below to change configs.
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCENARIO4_PATH = ROOT / "scripts" / "scenario4.py"
_spec = importlib.util.spec_from_file_location("scenario4_mod", _SCENARIO4_PATH)
s4 = importlib.util.module_from_spec(_spec)
sys.modules["scenario4_mod"] = s4
_spec.loader.exec_module(s4)
from end2endportfolio.src import langevin


def run_config(
    l2reg,
    beta,
    eta,
    n_steps,
    n_seeds,
    random_start_scale,
    contrast="sc_beats_ww",
    reg_mode="l2",
    constraint_mode="clip",
):
    """Run n_seeds chains with the given hyperparameters. Returns summary dict."""
    m0 = s4.m0
    a = s4.a
    b = s4.b

    objective = s4.build_objective_setup(
        contrast=contrast,
        reg_mode=reg_mode,
        l2reg=l2reg,
        constraint_mode=constraint_mode,
    )
    G = objective["G"]
    gradG = objective["gradG"]
    clip_low = objective["clip_low"]
    clip_high = objective["clip_high"]

    accept_rates = []
    final_L2 = []
    final_mah = []
    return_gaps = []
    box_violations = []
    boundary_touch_rates = []
    sc_wins = 0

    t0 = time.time()
    for seed_idx in range(n_seeds):
        m_start = m0 + (b - a) * random_start_scale * torch.rand((9,), dtype=m0.dtype, device=m0.device)
        x = m_start.detach().clone().to(dtype=m0.dtype)
        accepted = []
        boundary_touches = 0
        for step in range(n_steps):
            hyps = (G, gradG, eta, beta, clip_low, clip_high)
            x = x.detach().requires_grad_(True)
            x, info = langevin.torch_MALA_step(x, hyps, return_info=True)
            accepted.append(info["accepted"])
            if clip_low is not None and clip_high is not None and s4.touches_box(x.detach(), clip_low, clip_high):
                boundary_touches += 1
        m_last = x.detach()

        accept_rates.append(float(np.mean(accepted)) if accepted else 0.0)
        final_L2.append(s4.l2_squared(m_last, m0) ** 0.5)
        final_mah.append(s4.mahalanobis_squared(m_last, m0) ** 0.5)
        box_violations.append(s4.box_violation_squared(m_last, a, b) ** 0.5)
        boundary_touch_rates.append(float(boundary_touches / n_steps) if n_steps > 0 else 0.0)
        ev = s4.evaluate_pair(m_last)
        return_gaps.append(ev["return_gap"])
        if ev["winner"] == "summer_child":
            sc_wins += 1

    elapsed = time.time() - t0
    return {
        "reg_mode": reg_mode,
        "constraint_mode": constraint_mode,
        "constraint_desc": str(objective["description"]),
        "l2reg": l2reg,
        "beta": beta,
        "eta": eta,
        "n_steps": n_steps,
        "n_seeds": n_seeds,
        "accept_rate_mean": float(np.mean(accept_rates)),
        "accept_rate_std": float(np.std(accept_rates)),
        "boundary_touch_rate_mean": float(np.mean(boundary_touch_rates)),
        "box_violation_mean": float(np.mean(box_violations)),
        "box_violation_max": float(np.max(box_violations)),
        "L2_mean": float(np.mean(final_L2)),
        "L2_med": float(np.median(final_L2)),
        "mah_mean": float(np.mean(final_mah)),
        "mah_med": float(np.median(final_mah)),
        "gap_mean": float(np.mean(return_gaps)),
        "gap_best": float(np.max(return_gaps)),
        "gap_worst": float(np.min(return_gaps)),
        "sc_win_pct": 100.0 * sc_wins / n_seeds,
        "elapsed_s": elapsed,
    }


# Per-reg-mode grids — L2 and VAR1 need different l2reg scales because VAR1
# regularizer uses Σ⁻¹ weighting (eigenvalues [0.69..554], trace/d≈110 for the
# 21April standardized fit) while L2 is identity-weighted. To balance the same
# 100×-rescaled signal gradient (|∇G|≈1.1 per coord at anchor), VAR1 L2REG must
# be ~100× smaller than L2 L2REG. ETA sweet spot is near 0.002 for both
# (determined by signal gradient, not reg choice).
GRIDS_BY_REG_MODE = {
    # Pilot run (2026-04-24) winner: L2REG=0.5, ETA=0.002, acc=54%, L2med=1.48, gap=+3.16%, SCwin=100%.
    "l2": [
        dict(l2reg=0.3, beta=10.0, eta=0.001),
        dict(l2reg=0.3, beta=10.0, eta=0.002),
        dict(l2reg=0.5, beta=10.0, eta=0.001),
        dict(l2reg=0.5, beta=10.0, eta=0.002),
        dict(l2reg=0.5, beta=10.0, eta=0.003),
        dict(l2reg=1.0, beta=10.0, eta=0.002),
        dict(l2reg=1.0, beta=10.0, eta=0.004),
    ],
    # VAR1 exploration: Σ⁻¹ is highly anisotropic — eigenvalues [0.69..554], the
    # tight directions are exactly the persistent macros (dp/ep) where the signal
    # lives per the cross-model macro audit. Single-scalar L2REG tradeoff:
    # too small → chain blows out in loose dirs (svar, infl); too large →
    # frozen in tight dirs (dp, ep, bm) where we need motion.
    # Balance target: ~0.1-0.3 innovation-σ of motion in tight dirs → L2REG ≈ 0.03-0.3.
    # ETA likely small because tight-direction precision dominates MALA stability.
    "var1": [
        dict(l2reg=0.01,  beta=10.0, eta=0.001),
        dict(l2reg=0.01,  beta=10.0, eta=0.002),
        dict(l2reg=0.03,  beta=10.0, eta=0.001),
        dict(l2reg=0.03,  beta=10.0, eta=0.002),
        dict(l2reg=0.1,   beta=10.0, eta=0.0005),
        dict(l2reg=0.1,   beta=10.0, eta=0.001),
        dict(l2reg=0.1,   beta=10.0, eta=0.002),
        dict(l2reg=0.3,   beta=10.0, eta=0.0005),
        dict(l2reg=0.3,   beta=10.0, eta=0.001),
    ],
}

CONSTRAINT_MODES = [
    "box_barrier",
    "none",
]
REG_MODES = [
    "l2",
    "var1",
]

# Follow-up defaults: narrower search, more reliable estimates.
N_SEEDS = 8
N_STEPS = 100
RANDOM_START_SCALE = 0.005


def _print_mode_section(reg_mode: str, mode: str, rows):
    print(f"\nreg_mode={reg_mode}, constraint_mode={mode}")
    print(f"strategy: {rows[0]['constraint_desc']}")
    header = (
        f"{'l2reg':>7} {'beta':>5} {'eta':>7} "
        f"{'acc%':>6} {'edge%':>6} {'outBox':>7} {'L2med':>6} "
        f"{'gap_mean':>9} {'gap_best':>9} {'SCwin%':>7} {'time':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['l2reg']:>7.4g} {r['beta']:>5.1f} {r['eta']:>7.4g} "
            f"{100*r['accept_rate_mean']:>5.1f}% {100*r['boundary_touch_rate_mean']:>5.1f}% "
            f"{r['box_violation_mean']:>7.3f} {r['L2_med']:>6.3f} "
            f"{100*r['gap_mean']:>+8.3f}% {100*r['gap_best']:>+8.3f}% "
            f"{r['sc_win_pct']:>6.1f}% {r['elapsed_s']:>5.1f}s"
        )
    print("=" * len(header))

    plausible = [r for r in rows if r["L2_med"] <= 2.0 and r["box_violation_mean"] <= 0.05]
    if plausible:
        best_plausible = max(
            plausible,
            key=lambda r: (r["gap_mean"], r["sc_win_pct"], -r["L2_med"], -r["box_violation_mean"]),
        )
        print(
            "recommended_plausible: "
            f"L2REG={best_plausible['l2reg']}, BETA={best_plausible['beta']}, ETA={best_plausible['eta']} "
            f"-> acc={100*best_plausible['accept_rate_mean']:.0f}%, "
            f"SC_win={best_plausible['sc_win_pct']:.0f}%, "
            f"gap_mean={100*best_plausible['gap_mean']:+.3f}%, "
            f"L2_med={best_plausible['L2_med']:.2f}, "
            f"outBox={best_plausible['box_violation_mean']:.3f}"
        )
    else:
        print("recommended_plausible: none matched L2_med<=2 and outBox<=0.05")

    # MALA-band: accept rate 0.5-0.6 (healthy mixing) with non-trivial motion
    in_band = [
        r for r in rows
        if 0.50 <= r["accept_rate_mean"] <= 0.60
        and r["L2_med"] >= 0.3
        and r["box_violation_mean"] <= 0.05
    ]
    if in_band:
        best_band = max(in_band, key=lambda r: (r["gap_mean"], r["sc_win_pct"], -r["L2_med"]))
        print(
            "recommended_accept_band: "
            f"L2REG={best_band['l2reg']}, BETA={best_band['beta']}, ETA={best_band['eta']} "
            f"-> acc={100*best_band['accept_rate_mean']:.1f}%, "
            f"SC_win={best_band['sc_win_pct']:.0f}%, "
            f"gap_mean={100*best_band['gap_mean']:+.3f}%, "
            f"L2_med={best_band['L2_med']:.2f}"
        )
    else:
        near_band = sorted(rows, key=lambda r: abs(r["accept_rate_mean"] - 0.55))[:3]
        print("recommended_accept_band: none in 50-60% band. Closest 3:")
        for r in near_band:
            print(
                f"  acc={100*r['accept_rate_mean']:.1f}% at "
                f"L2REG={r['l2reg']}, ETA={r['eta']}, L2_med={r['L2_med']:.2f}"
            )

    best_balanced = max(
        rows,
        key=lambda r: (
            r["gap_mean"],
            r["sc_win_pct"],
            -r["box_violation_mean"],
            -r["L2_med"],
        ),
    )
    print(
        "recommended_balanced: "
        f"L2REG={best_balanced['l2reg']}, BETA={best_balanced['beta']}, ETA={best_balanced['eta']} "
        f"-> acc={100*best_balanced['accept_rate_mean']:.0f}%, "
        f"SC_win={best_balanced['sc_win_pct']:.0f}%, "
        f"gap_mean={100*best_balanced['gap_mean']:+.3f}%, "
        f"L2_med={best_balanced['L2_med']:.2f}, "
        f"outBox={best_balanced['box_violation_mean']:.3f}"
    )

    best_raw = max(rows, key=lambda r: (r["sc_win_pct"], r["gap_mean"], -r["L2_med"]))
    print(
        "best_raw: "
        f"L2REG={best_raw['l2reg']}, BETA={best_raw['beta']}, ETA={best_raw['eta']} "
        f"-> acc={100*best_raw['accept_rate_mean']:.0f}%, SC_win={best_raw['sc_win_pct']:.0f}%, "
        f"gap_mean={100*best_raw['gap_mean']:+.3f}%, L2_med={best_raw['L2_med']:.2f}, "
        f"outBox={best_raw['box_violation_mean']:.3f}"
    )
    return best_raw


def _format_eta(cfg: dict) -> str:
    return f"{cfg['eta']:.4g}"


def main():
    total_runs = sum(len(GRIDS_BY_REG_MODE[rm]) for rm in REG_MODES) * len(CONSTRAINT_MODES)
    print(
        f"scenario4 hyperparameter sweep — {total_runs} runs "
        f"({len(REG_MODES)} reg modes × {len(CONSTRAINT_MODES)} constraint modes, per-reg grids) "
        f"× {N_SEEDS} seeds × {N_STEPS} steps"
    )
    print(f"anchor 202004: SC={s4.return_summer:.2%}, WW={s4.return_winter:.2%}, gap={s4.return_summer - s4.return_winter:+.2%}")
    print(f"objective=sc_beats_ww, reg_modes={','.join(REG_MODES)}")
    print("selection=positive gap_mean, then SCwin%, then lower outBox and L2_med")

    best_by_mode = []
    run_idx = 0
    sweep_start = time.time()
    for reg_mode in REG_MODES:
        grid = GRIDS_BY_REG_MODE[reg_mode]
        for mode in CONSTRAINT_MODES:
            mode_rows = []
            mode_start = time.time()
            print(f"\nstarting block reg_mode={reg_mode}, constraint_mode={mode}", flush=True)
            for cfg_idx, cfg in enumerate(grid, start=1):
                run_idx += 1
                cfg_start = time.time()
                print(
                    f"[{run_idx:02d}/{total_runs:02d}] "
                    f"reg_mode={reg_mode}, constraint_mode={mode}, "
                    f"cfg={cfg_idx}/{len(grid)} "
                    f"(l2reg={cfg['l2reg']:.4g}, beta={cfg['beta']:.1f}, eta={_format_eta(cfg)})",
                    flush=True,
                )
                mode_rows.append(
                    run_config(
                        l2reg=cfg["l2reg"],
                        beta=cfg["beta"],
                        eta=cfg["eta"],
                        n_steps=N_STEPS,
                        n_seeds=N_SEEDS,
                        random_start_scale=RANDOM_START_SCALE,
                        contrast="sc_beats_ww",
                        reg_mode=reg_mode,
                        constraint_mode=mode,
                    )
                )
                last = mode_rows[-1]
                cfg_elapsed = time.time() - cfg_start
                total_elapsed = time.time() - sweep_start
                print(
                    f"  completed in {cfg_elapsed:.1f}s | "
                    f"acc={100*last['accept_rate_mean']:.1f}% | "
                    f"L2med={last['L2_med']:.3f} | "
                    f"gap_mean={100*last['gap_mean']:+.3f}% | "
                    f"SCwin={last['sc_win_pct']:.1f}% | "
                    f"elapsed_total={total_elapsed/60:.1f}m",
                    flush=True,
                )
            print(
                f"finished block reg_mode={reg_mode}, constraint_mode={mode} "
                f"in {(time.time() - mode_start)/60:.1f}m",
                flush=True,
            )
            best_by_mode.append(_print_mode_section(reg_mode, mode, mode_rows))

    overall_best = max(
        best_by_mode,
        key=lambda r: (r["sc_win_pct"], r["gap_mean"], -r["box_violation_mean"], -r["L2_med"]),
    )
    print(
        "\noverall_best_raw_across_modes: "
        f"reg_mode={overall_best['reg_mode']}, "
        f"constraint_mode={overall_best['constraint_mode']}, "
        f"L2REG={overall_best['l2reg']}, BETA={overall_best['beta']}, ETA={overall_best['eta']} "
        f"-> acc={100*overall_best['accept_rate_mean']:.0f}%, "
        f"SC_win={overall_best['sc_win_pct']:.0f}%, "
        f"gap_mean={100*overall_best['gap_mean']:+.3f}%, "
        f"L2_med={overall_best['L2_med']:.2f}, "
        f"outBox={overall_best['box_violation_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
