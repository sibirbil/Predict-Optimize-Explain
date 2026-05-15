"""Clean manuscript figures for the accepted Scenario 4 20k run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.macro_scaler import MacroScaler  # noqa: E402
from src.utils.plotting import set_publication_style  # noqa: E402


RUN_DIR = ROOT / "scenario_outputs" / "scenario4_202004" / "runs" / "20260509_072049"
OUT_DIR = ROOT / "submission_plots" / "scenario4_clean_20k"
ANCHOR = 202004
MACRO_COLS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl"]
REGIME_ORDER = ["expansion", "contraction", "financial_stress"]
REGIME_LABELS = {
    "expansion": "Expansion",
    "contraction": "Contraction",
    "financial_stress": "Financial stress",
}
REGIME_PROB_COLS = {
    "expansion": "prob_expansion",
    "contraction": "prob_contraction",
    "financial_stress": "prob_financial_stress",
}

COLORS = {
    "history": "#b8bec6",
    "generated": "#1f7a74",
    "final": "#111111",
    "anchor": "#c44e52",
    "grid": "#d8dde3",
    "text": "#111111",
    "positive": "#287c71",
    "negative": "#b55d4c",
    "bar_anchor": "#9aa3ad",
    "bar_generated": "#1f7a74",
}


def apply_style() -> None:
    set_publication_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 8.6,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def newest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file in {run_dir} matching {pattern}")
    return matches[-1]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def historical_std_frame(historical: pd.DataFrame) -> pd.DataFrame:
    scaler = MacroScaler.load(ROOT / "runtime_universe500" / "scaler")
    values = historical[MACRO_COLS].to_numpy(dtype=np.float32)
    std = (values - scaler.mean.cpu().numpy()) / scaler.std.cpu().numpy()
    out = historical[["yyyymm"]].copy()
    for idx, col in enumerate(MACRO_COLS):
        out[f"{col}_std"] = std[:, idx]
    return out


def anchor_std(historical: pd.DataFrame, anchor: int) -> pd.Series:
    hist_std = historical_std_frame(historical)
    rows = hist_std[hist_std["yyyymm"].astype(int).eq(anchor)]
    if rows.empty:
        raise ValueError(f"Anchor {anchor} not found in historical panel")
    return rows.iloc[0]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return pdf, png


def load_inputs(run_dir: Path) -> dict[str, object]:
    paths = {
        "config": newest(run_dir, "config_*.json"),
        "final": newest(run_dir, "final_state_diagnostics_*.csv"),
        "seed": newest(run_dir, "seed_summary_*.csv"),
        "sample": newest(run_dir, "generated_macro_sample_postburnin_*.csv"),
        "historical": newest(run_dir, "historical_macro_panel_*.csv"),
        "transitions": newest(run_dir, "regime_transitions_*.csv"),
        "metadata": newest(run_dir, "trajectory_tensor_metadata_*.json"),
    }
    return {
        "paths": paths,
        "config": json.loads(paths["config"].read_text(encoding="utf-8")),
        "metadata": json.loads(paths["metadata"].read_text(encoding="utf-8")),
        "final": pd.read_csv(paths["final"]),
        "seed": pd.read_csv(paths["seed"]),
        "sample": pd.read_csv(paths["sample"]),
        "historical": pd.read_csv(paths["historical"]),
        "transitions": pd.read_csv(paths["transitions"]),
    }


def validate_run(data: dict[str, object]) -> None:
    config = data["config"]
    metadata = data["metadata"]
    final = data["final"]
    sample = data["sample"]

    assert int(config["DATE"]) == ANCHOR, f"Expected DATE={ANCHOR}, found {config.get('DATE')}"
    assert int(config["N_SEEDS"]) == 4, f"Expected N_SEEDS=4, found {config.get('N_SEEDS')}"
    assert int(config["N_STEPS"]) == 20000, f"Expected N_STEPS=20000, found {config.get('N_STEPS')}"
    assert config["CONTRAST_FUNCTION"] == "sc_beats_ww", config.get("CONTRAST_FUNCTION")
    assert metadata["scenario"] == "scenario4", metadata.get("scenario")
    assert int(metadata["date"]) == ANCHOR, metadata.get("date")
    assert int(metadata["n_seeds_completed"]) == 4, metadata.get("n_seeds_completed")
    assert int(metadata["n_steps_per_seed"]) == 20000, metadata.get("n_steps_per_seed")
    assert list(metadata["full_shape"]) == [4, 20000, 9], metadata.get("full_shape")
    assert list(metadata["postburnin_shape"]) == [4, 10000, 9], metadata.get("postburnin_shape")
    assert len(final) == 4, f"Expected 4 final states, found {len(final)}"
    assert final["winner"].eq("summer_child").all(), final["winner"].tolist()
    assert len(sample) == 8000, f"Expected 8000 generated post-burn-in rows, found {len(sample)}"


def std_matrices(data: dict[str, object]) -> dict[str, object]:
    historical = data["historical"]
    sample = data["sample"]
    final = data["final"]
    hist_std = historical_std_frame(historical)
    cols = [f"{col}_std" for col in MACRO_COLS]
    anchor = anchor_std(historical, ANCHOR)
    return {
        "cols": cols,
        "hist_std": hist_std,
        "hist_x": hist_std[cols].to_numpy(dtype=float),
        "sample_x": sample[cols].to_numpy(dtype=float),
        "final_x": final[cols].to_numpy(dtype=float),
        "anchor_x": anchor[cols].to_numpy(dtype=float),
    }


def plot_macro_location(data: dict[str, object], out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    mats = std_matrices(data)
    hist_x = mats["hist_x"]
    sample_x = mats["sample_x"]
    final_x = mats["final_x"]
    anchor_x = mats["anchor_x"]

    rng = np.random.default_rng(20260509)
    tsne_sample_size = min(450, len(sample_x))
    tsne_sample_idx = np.sort(rng.choice(len(sample_x), size=tsne_sample_size, replace=False))
    pca_sample_size = min(2200, len(sample_x))
    pca_sample_idx = np.sort(rng.choice(len(sample_x), size=pca_sample_size, replace=False))

    pca = PCA(n_components=2, random_state=0).fit(hist_x)
    hist_p = pca.transform(hist_x)
    sample_p = pca.transform(sample_x[pca_sample_idx])
    final_p = pca.transform(final_x)
    anchor_p = pca.transform(anchor_x.reshape(1, -1))

    combined = np.vstack([hist_x, sample_x[tsne_sample_idx], final_x, anchor_x.reshape(1, -1)])
    labels = (
        ["history"] * len(hist_x)
        + ["generated"] * len(tsne_sample_idx)
        + ["final"] * len(final_x)
        + ["anchor"]
    )
    perplexity = min(40, max(5, (len(combined) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=20260509,
        perplexity=perplexity,
        init="random",
        learning_rate=200.0,
        method="exact",
        n_iter=700,
    )
    emb = tsne.fit_transform(combined)
    labels_arr = np.asarray(labels)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)

    for ax, coords, sample_coords, final_coords, anchor_coords in [
        (axes[0], hist_p, sample_p, final_p, anchor_p),
        (
            axes[1],
            emb[labels_arr == "history"],
            emb[labels_arr == "generated"],
            emb[labels_arr == "final"],
            emb[labels_arr == "anchor"],
        ),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], s=9, color=COLORS["history"], alpha=0.42, linewidths=0, rasterized=True)
        ax.scatter(
            sample_coords[:, 0],
            sample_coords[:, 1],
            s=8,
            color=COLORS["generated"],
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            final_coords[:, 0],
            final_coords[:, 1],
            s=52,
            marker="o",
            color=COLORS["final"],
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        ax.scatter(
            anchor_coords[:, 0],
            anchor_coords[:, 1],
            s=76,
            marker="X",
            color=COLORS["anchor"],
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )

    axes[0].set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    handles = [
        mlines.Line2D([], [], color=COLORS["history"], marker="o", linestyle="", markersize=4, label="Historical"),
        mlines.Line2D([], [], color=COLORS["generated"], marker="o", linestyle="", markersize=4, label="Generated"),
        mlines.Line2D([], [], color=COLORS["final"], marker="o", linestyle="", markersize=5, label="Final states"),
        mlines.Line2D([], [], color=COLORS["anchor"], marker="X", linestyle="", markersize=6, label="Anchor"),
    ]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    paths = save_figure(fig, out_dir, "s4_macro_location_pca_tsne")
    info = {
        "pca_sample_size": int(pca_sample_size),
        "tsne_sample_size": int(tsne_sample_size),
        "tsne_random_state": 20260509,
        "tsne_perplexity": int(perplexity),
        "tsne_method": "exact",
        "tsne_n_iter": 700,
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }
    return (*paths, info)


def plot_regime_probability_shift(data: dict[str, object], out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    config = data["config"]
    sample = data["sample"]
    anchor_probs = config["ANCHOR_REGIME"]["probabilities"]

    anchor_vals = np.array([anchor_probs[regime] for regime in REGIME_ORDER], dtype=float)
    med = np.array([sample[REGIME_PROB_COLS[regime]].median() for regime in REGIME_ORDER], dtype=float)
    q25 = np.array([sample[REGIME_PROB_COLS[regime]].quantile(0.25) for regime in REGIME_ORDER], dtype=float)
    q75 = np.array([sample[REGIME_PROB_COLS[regime]].quantile(0.75) for regime in REGIME_ORDER], dtype=float)
    change = 100.0 * (med - anchor_vals)

    x = np.arange(len(REGIME_ORDER))
    width = 0.34
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    axes[0].bar(x - width / 2, 100.0 * anchor_vals, width=width, color=COLORS["bar_anchor"], label="Anchor")
    axes[0].bar(x + width / 2, 100.0 * med, width=width, color=COLORS["bar_generated"], label="Generated median")
    axes[0].errorbar(
        x + width / 2,
        100.0 * med,
        yerr=np.vstack([100.0 * (med - q25), 100.0 * (q75 - med)]),
        fmt="none",
        ecolor=COLORS["text"],
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
    )
    axes[0].set_xticks(x, [REGIME_LABELS[r] for r in REGIME_ORDER], rotation=18, ha="right")
    axes[0].set_ylabel("Probability (%)")
    axes[0].legend(loc="upper left")

    colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in change]
    axes[1].bar(x, change, color=colors, width=0.52)
    axes[1].axhline(0, color=COLORS["text"], linewidth=0.8)
    axes[1].set_xticks(x, [REGIME_LABELS[r] for r in REGIME_ORDER], rotation=18, ha="right")
    axes[1].set_ylabel("Change (pp)")
    paths = save_figure(fig, out_dir, "s4_regime_probability_shift")
    info = {
        "anchor_probabilities": {r: float(anchor_vals[i]) for i, r in enumerate(REGIME_ORDER)},
        "generated_median_probabilities": {r: float(med[i]) for i, r in enumerate(REGIME_ORDER)},
        "generated_iqr_probabilities": {
            r: [float(q25[i]), float(q75[i])] for i, r in enumerate(REGIME_ORDER)
        },
        "change_pp": {r: float(change[i]) for i, r in enumerate(REGIME_ORDER)},
    }
    return (*paths, info)


def density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 3 or float(np.std(clean)) <= 1e-10:
        counts, edges = np.histogram(clean, bins=min(12, max(2, clean.size)), range=(grid.min(), grid.max()), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, counts, left=0.0, right=0.0)
    try:
        return gaussian_kde(clean)(grid)
    except Exception:
        counts, edges = np.histogram(clean, bins=24, range=(grid.min(), grid.max()), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, counts, left=0.0, right=0.0)


def plot_macro_density_scaled(data: dict[str, object], out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    mats = std_matrices(data)
    hist_std = mats["hist_std"]
    sample = data["sample"]
    final = data["final"]
    anchor = anchor_std(data["historical"], ANCHOR)

    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 6.8), constrained_layout=True)
    grid = np.linspace(-4.0, 4.0, 320)
    scaling: dict[str, float] = {}

    for idx, (ax, col) in enumerate(zip(axes.ravel(), MACRO_COLS)):
        std_col = f"{col}_std"
        hist_y = density_curve(hist_std[std_col].to_numpy(dtype=float), grid)
        gen_y = density_curve(sample[std_col].to_numpy(dtype=float), grid)
        scale = max(float(np.nanmax(hist_y)), float(np.nanmax(gen_y)), 1e-12)
        hist_y = hist_y / scale
        gen_y = gen_y / scale
        scaling[col] = scale
        ax.fill_between(grid, hist_y, color=COLORS["history"], alpha=0.45, linewidth=0)
        ax.plot(grid, hist_y, color="#8e969f", linewidth=0.9)
        ax.plot(grid, gen_y, color=COLORS["generated"], linewidth=1.25)
        ax.fill_between(grid, gen_y, color=COLORS["generated"], alpha=0.16, linewidth=0)
        ax.axvline(float(anchor[std_col]), color=COLORS["anchor"], linewidth=1.1)
        for value in final[std_col].to_numpy(dtype=float):
            ax.plot([value, value], [0, 0.12], color=COLORS["final"], linewidth=0.8, solid_capstyle="butt")
        ax.set_title(col)
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Standardized value" if idx >= 6 else "")
        ax.set_ylabel("Scaled density" if idx % 3 == 0 else "")

    handles = [
        mlines.Line2D([], [], color="#8e969f", linewidth=1.2, label="Historical"),
        mlines.Line2D([], [], color=COLORS["generated"], linewidth=1.4, label="Generated"),
        mlines.Line2D([], [], color=COLORS["anchor"], linewidth=1.2, label="Anchor"),
        mlines.Line2D([], [], color=COLORS["final"], linewidth=1.2, label="Final states"),
    ]
    axes.ravel()[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5), borderaxespad=0.0)
    paths = save_figure(fig, out_dir, "s4_macro_density_scaled")
    info = {
        "x_range": [-4.0, 4.0],
        "grid_points": 320,
        "scaling_rule": "For each macro variable, historical and generated density curves are divided by the maximum density across the two curves.",
        "density_fallback": "If KDE fails or variance is near zero, histogram density is interpolated on the same grid.",
        "scale_constants": scaling,
    }
    return (*paths, info)


def plot_return_gap_minimal(data: dict[str, object], out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    config = data["config"]
    final = data["final"]
    anchor_gap_pp = 100.0 * float(config["ANCHOR_RETURN_GAP"])
    gaps_pp = 100.0 * final["return_gap"].to_numpy(dtype=float)
    median_gap_pp = float(np.median(gaps_pp))

    apply_style()
    fig, ax = plt.subplots(figsize=(4.7, 2.55), constrained_layout=True)
    ax.axhline(0, color=COLORS["text"], linewidth=0.8)
    ax.scatter([0], [anchor_gap_pp], marker="X", s=70, color=COLORS["anchor"], edgecolors="white", linewidths=0.7, label="Anchor", zorder=4)
    rng = np.random.default_rng(20260509)
    jitter = rng.uniform(-0.055, 0.055, size=len(gaps_pp))
    ax.scatter(np.ones_like(gaps_pp) + jitter, gaps_pp, s=52, color=COLORS["final"], edgecolors="white", linewidths=0.7, label="Final states", zorder=4)
    ax.scatter([1.22], [median_gap_pp], marker="D", s=62, color=COLORS["generated"], edgecolors="white", linewidths=0.7, label="Median", zorder=4)
    ax.set_xlim(-0.35, 1.55)
    ax.set_xticks([0, 1], ["Anchor", "Generated"])
    ax.set_ylabel("SC - WW return gap (pp)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    paths = save_figure(fig, out_dir, "s4_return_gap_minimal")
    info = {
        "anchor_gap_pp": anchor_gap_pp,
        "final_gaps_pp": [float(v) for v in gaps_pp],
        "median_final_gap_pp": median_gap_pp,
    }
    return (*paths, info)


def summarize(data: dict[str, object]) -> dict[str, object]:
    config = data["config"]
    final = data["final"]
    seed = data["seed"]
    sample = data["sample"]
    transitions = data["transitions"]
    return {
        "run_path": rel(RUN_DIR),
        "date": int(config["DATE"]),
        "n_seeds": int(config["N_SEEDS"]),
        "n_steps": int(config["N_STEPS"]),
        "contrast_function": config["CONTRAST_FUNCTION"],
        "eta": float(config["ETA"]),
        "beta": float(config["BETA"]),
        "l2reg": float(config["L2REG"]),
        "reg_mode": config["REG_MODE"],
        "constraint_mode": config["CONSTRAINT_MODE"],
        "anchor_return_gap": float(config["ANCHOR_RETURN_GAP"]),
        "target_achievement": f"{int(final['winner'].eq('summer_child').sum())}/{len(final)}",
        "median_return_gap": float(final["return_gap"].median()),
        "mean_return_gap": float(final["return_gap"].mean()),
        "acceptance_mean": float(seed["accept_rate"].mean()),
        "acceptance_median": float(seed["accept_rate"].median()),
        "acceptance_min": float(seed["accept_rate"].min()),
        "acceptance_max": float(seed["accept_rate"].max()),
        "ess_mean": float(seed["ess_mean"].mean()),
        "ess_median": float(seed["ess_mean"].median()),
        "ess_min": float(seed["ess_mean"].min()),
        "ess_max": float(seed["ess_mean"].max()),
        "final_regime_counts": final["regime"].value_counts().sort_index().to_dict(),
        "generated_regime_counts": sample["regime"].value_counts().sort_index().to_dict(),
        "transition_counts": transitions[["start_regime", "final_regime"]].value_counts().to_dict(),
        "var_median_mah_dist": float(final["mah_dist"].median()),
        "var_median_chi2_tail": float(final["mah_chi2_tail"].median()),
    }


def write_manifest(out_dir: Path, records: list[dict[str, str]]) -> None:
    with (out_dir / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["filename", "source_run", "source_files", "figure_type", "manuscript_use", "notes"],
        )
        writer.writeheader()
        writer.writerows(records)


def write_notes(
    out_dir: Path,
    data: dict[str, object],
    summary: dict[str, object],
    pca_tsne_info: dict[str, object],
    regime_info: dict[str, object],
    density_info: dict[str, object],
    return_info: dict[str, object],
) -> None:
    paths = data["paths"]
    metadata = data["metadata"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.4f}%"

    lines = [
        "# Scenario 4 Clean 20k Figure Notes",
        "",
        f"Run path: `{summary['run_path']}`.",
        "",
        "## Config",
        "",
        f"- Scenario: `scenario4`; anchor: `{summary['date']}`; objective: `{summary['contrast_function']}`.",
        f"- Seeds/chains: `{summary['n_seeds']}`; MALA steps per chain: `{summary['n_steps']}`.",
        f"- Full trajectory shape: `{metadata['full_shape']}`; post-burn-in shape: `{metadata['postburnin_shape']}`.",
        f"- MALA settings: `eta={summary['eta']}`, `beta={summary['beta']}`, `l2reg={summary['l2reg']}`, `reg_mode={summary['reg_mode']}`, `constraint_mode={summary['constraint_mode']}`.",
        "",
        "## Source Files",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{rel(path)}`")

    lines.extend(
        [
            "",
            "## Target Achievement",
            "",
            f"- Anchor SC-WW return gap: `{100.0 * summary['anchor_return_gap']:+.4f}` percentage points.",
            f"- Final target achievement: `{summary['target_achievement']}` final states have `winner=summer_child`.",
            f"- Final median SC-WW return gap: `{100.0 * summary['median_return_gap']:+.4f}` percentage points.",
            f"- Final mean SC-WW return gap: `{100.0 * summary['mean_return_gap']:+.4f}` percentage points.",
            "",
            "## MALA Diagnostics",
            "",
            f"- Acceptance rate: mean `{summary['acceptance_mean']:.6f}`, median `{summary['acceptance_median']:.6f}`, range `[{summary['acceptance_min']:.6f}, {summary['acceptance_max']:.6f}]`.",
            f"- ESS mean column: mean `{summary['ess_mean']:.6f}`, median `{summary['ess_median']:.6f}`, range `[{summary['ess_min']:.6f}, {summary['ess_max']:.6f}]`.",
            "",
            "## Regime Diagnostics",
            "",
            f"- Final hard-label counts: `{summary['final_regime_counts']}`.",
            f"- Generated post-burn-in hard-label counts: `{summary['generated_regime_counts']}`.",
            f"- Regime transition counts: `{summary['transition_counts']}`.",
            f"- Anchor probabilities: `{regime_info['anchor_probabilities']}`.",
            f"- Generated median probabilities: `{regime_info['generated_median_probabilities']}`.",
            f"- Generated probability IQRs: `{regime_info['generated_iqr_probabilities']}`.",
            f"- Generated-minus-anchor probability changes in pp: `{regime_info['change_pp']}`.",
            "",
            "## PCA/t-SNE Rules",
            "",
            f"- PCA is fit on historical standardized macro states only. Generated states, final states, and the anchor are projected into that fitted PCA basis.",
            f"- PCA sample for generated post-burn-in plotting uses `{pca_tsne_info['pca_sample_size']}` rows sampled without replacement using RNG seed `20260509`.",
            f"- t-SNE is fit on a combined plotting sample of all historical states, `{pca_tsne_info['tsne_sample_size']}` generated post-burn-in states sampled without replacement, the four final states, and the anchor.",
            f"- t-SNE uses `random_state={pca_tsne_info['tsne_random_state']}`, `perplexity={pca_tsne_info['tsne_perplexity']}`, `method={pca_tsne_info['tsne_method']}`, and `n_iter={pca_tsne_info['tsne_n_iter']}`.",
            f"- PCA explained variance ratios: `{pca_tsne_info['pca_explained_variance']}`.",
            "",
            "## Density Rules",
            "",
            f"- Standardized macro variables: `{MACRO_COLS}`.",
            f"- X-axis range for all density panels: `{density_info['x_range']}`.",
            f"- Density scaling rule: {density_info['scaling_rule']}",
            f"- KDE fallback rule: {density_info['density_fallback']}",
            "",
            "## Return-Gap Figure",
            "",
            f"- Anchor gap: `{return_info['anchor_gap_pp']:+.4f}` pp.",
            f"- Final gaps by seed: `{return_info['final_gaps_pp']}` pp.",
            f"- Median final gap: `{return_info['median_final_gap_pp']:+.4f}` pp.",
            "",
            "## Plausibility Warning",
            "",
            f"- Final median VAR(1)-forecast Mahalanobis distance: `{summary['var_median_mah_dist']:.6f}`.",
            f"- Final median VAR(1) chi-square tail probability: `{summary['var_median_chi2_tail']:.6e}`.",
            "- Describe these figures as stress-escape counterfactuals, not as VAR-plausible one-step forecasts.",
        ]
    )
    (out_dir / "figure_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-return-gap", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    data = load_inputs(run_dir)
    validate_run(data)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_files = "; ".join(rel(path) for path in data["paths"].values())
    source_run = rel(run_dir)
    records: list[dict[str, str]] = []

    pca_pdf, pca_png, pca_info = plot_macro_location(data, out_dir)
    records.append(
        {
            "filename": f"{rel(pca_pdf)}; {rel(pca_png)}",
            "source_run": source_run,
            "source_files": source_files,
            "figure_type": "macro_location_pca_tsne",
            "manuscript_use": "Main text candidate",
            "notes": "PCA fit on historical standardized macro states; t-SNE fit on combined plotting sample.",
        }
    )

    reg_pdf, reg_png, regime_info = plot_regime_probability_shift(data, out_dir)
    records.append(
        {
            "filename": f"{rel(reg_pdf)}; {rel(reg_png)}",
            "source_run": source_run,
            "source_files": source_files,
            "figure_type": "regime_probability_shift",
            "manuscript_use": "Main text candidate",
            "notes": "Anchor probabilities from config; generated medians and IQRs from post-burn-in sample probabilities.",
        }
    )

    den_pdf, den_png, density_info = plot_macro_density_scaled(data, out_dir)
    records.append(
        {
            "filename": f"{rel(den_pdf)}; {rel(den_png)}",
            "source_run": source_run,
            "source_files": source_files,
            "figure_type": "macro_density_scaled",
            "manuscript_use": "Main text or appendix candidate",
            "notes": "Densities are plotted in standardized macro units and scaled within variable.",
        }
    )

    if args.skip_return_gap:
        return_info = {"anchor_gap_pp": float("nan"), "final_gaps_pp": [], "median_final_gap_pp": float("nan")}
    else:
        ret_pdf, ret_png, return_info = plot_return_gap_minimal(data, out_dir)
        records.append(
            {
                "filename": f"{rel(ret_pdf)}; {rel(ret_png)}",
                "source_run": source_run,
                "source_files": source_files,
                "figure_type": "return_gap_minimal",
                "manuscript_use": "Appendix or compact main-text target figure",
                "notes": "Anchor SC-WW gap and four final seed SC-WW gaps.",
            }
        )

    write_manifest(out_dir, records)
    write_notes(out_dir, data, summarize(data), pca_info, regime_info, density_info, return_info)
    print(f"Wrote {len(records)} figures to {rel(out_dir)}")


if __name__ == "__main__":
    main()
