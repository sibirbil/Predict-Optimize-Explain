from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

try:
    import torch
except ImportError:  # pragma: no cover - plotting helpers can work without torch.
    torch = None


PAPER_COLORS = {
    "historical": "#7a838a",
    "generated": "#157f78",
    "anchor": "#c44e52",
    "grid": "#d8dadd",
    "text": "#111111",
    "positive": "#287c71",
    "negative": "#b55d4c",
}


def set_publication_style() -> None:
    """Apply a quiet matplotlib style suitable for paper figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_COLORS["text"],
            "axes.labelcolor": PAPER_COLORS["text"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def _as_1d_array(values) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().reshape(-1).astype(float)
    return np.asarray(values, dtype=float).reshape(-1)


def _resolve_columns(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    columns: Optional[Sequence[str]],
) -> list[str]:
    if columns is not None:
        selected = list(columns)
    else:
        selected = [col for col in historical_df.columns if col in generated_df.columns]
    if not selected:
        raise ValueError("No shared columns found for plotting.")
    missing_hist = [col for col in selected if col not in historical_df.columns]
    missing_gen = [col for col in selected if col not in generated_df.columns]
    if missing_hist or missing_gen:
        raise ValueError(f"Missing columns. historical={missing_hist}, generated={missing_gen}")
    return selected


def _limits_for_columns(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    columns: Sequence[str],
    lower_limits=None,
    upper_limits=None,
    pad_fraction: float = 0.05,
) -> dict[str, tuple[float, float]]:
    if lower_limits is not None and upper_limits is not None:
        lower = _as_1d_array(lower_limits)
        upper = _as_1d_array(upper_limits)
        if len(lower) != len(columns) or len(upper) != len(columns):
            raise ValueError("lower_limits and upper_limits must match the number of plotted columns.")
        spans = np.maximum(upper - lower, 1e-12)
        return {
            col: (float(lower[i] - pad_fraction * spans[i]), float(upper[i] + pad_fraction * spans[i]))
            for i, col in enumerate(columns)
        }

    limits: dict[str, tuple[float, float]] = {}
    for col in columns:
        values = pd.concat([historical_df[col], generated_df[col]], axis=0).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            limits[col] = (-1.0, 1.0)
            continue
        lo = float(np.nanquantile(values, 0.005))
        hi = float(np.nanquantile(values, 0.995))
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        span = hi - lo
        limits[col] = (lo - pad_fraction * span, hi + pad_fraction * span)
    return limits


def _sample_frame(frame: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(max_points, random_state=random_state)


def _density_curve(values: pd.Series | np.ndarray, grid: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 3 or float(np.nanstd(clean)) <= 1e-12:
        counts, edges = np.histogram(clean, bins=min(10, max(2, clean.size)), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        return np.interp(grid, centers, counts, left=0.0, right=0.0)
    kde = gaussian_kde(clean)
    return kde(grid)


def _plot_shaded_density(
    ax: plt.Axes,
    values: pd.Series | np.ndarray,
    grid: np.ndarray,
    color: str,
    label: str,
    alpha: float,
    linewidth: float = 1.8,
) -> np.ndarray:
    density = _density_curve(values, grid)
    ax.fill_between(grid, density, 0.0, color=color, alpha=alpha, linewidth=0.0, label=label)
    ax.plot(grid, density, color=color, linewidth=linewidth)
    return density


def macro_pair_grid(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    anchor,
    columns: Optional[Sequence[str]] = None,
    lower_limits=None,
    upper_limits=None,
    save_path: Optional[str | Path] = None,
    title: str = "Generated Macro Scenarios vs Historical Macro Panel",
    historical_label: str = "Historical",
    generated_label: str = "Generated",
    max_points: int = 2500,
    random_state: int = 7,
) -> plt.Figure:
    """Create a publication-oriented macro pair grid.

    Diagonal cells show marginal shaded densities. Lower-triangle cells
    compare historical and generated scatter. Upper-triangle cells report
    historical and generated Pearson correlations, which is often more useful
    for the paper than empty panels.
    """
    set_publication_style()
    cols = _resolve_columns(historical_df, generated_df, columns)
    anchor_values = _as_1d_array(anchor)
    if len(anchor_values) < len(cols):
        raise ValueError("anchor must have at least as many values as plotted columns.")
    anchor_values = anchor_values[: len(cols)]

    hist = _sample_frame(historical_df[cols].dropna(), max_points=max_points, random_state=random_state)
    gen = _sample_frame(generated_df[cols].dropna(), max_points=max_points, random_state=random_state)
    limits = _limits_for_columns(hist, gen, cols, lower_limits=lower_limits, upper_limits=upper_limits)

    n = len(cols)
    cell = 1.25 if n >= 7 else 1.55
    fig, axes = plt.subplots(n, n, figsize=(cell * n, cell * n), squeeze=False)
    hist_corr = hist.corr(numeric_only=True)
    gen_corr = gen.corr(numeric_only=True)

    for row, y_col in enumerate(cols):
        for col_idx, x_col in enumerate(cols):
            ax = axes[row, col_idx]
            if row == col_idx:
                grid = np.linspace(*limits[x_col], 180)
                _plot_shaded_density(
                    ax,
                    hist[x_col],
                    grid,
                    PAPER_COLORS["historical"],
                    historical_label if row == 0 else "_nolegend_",
                    alpha=0.20,
                    linewidth=1.1,
                )
                _plot_shaded_density(
                    ax,
                    gen[x_col],
                    grid,
                    PAPER_COLORS["generated"],
                    generated_label if row == 0 else "_nolegend_",
                    alpha=0.18,
                    linewidth=1.6,
                )
                ax.axvline(anchor_values[col_idx], color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2)
                ax.set_xlim(limits[x_col])
                ax.set_yticks([])
                ax.set_title(x_col, pad=4)
            elif row > col_idx:
                ax.scatter(
                    hist[x_col],
                    hist[y_col],
                    s=5,
                    color=PAPER_COLORS["historical"],
                    alpha=0.18,
                    edgecolors="none",
                    rasterized=True,
                )
                ax.scatter(
                    gen[x_col],
                    gen[y_col],
                    s=8,
                    color=PAPER_COLORS["generated"],
                    alpha=0.42,
                    edgecolors="none",
                    rasterized=True,
                )
                ax.scatter(
                    [anchor_values[col_idx]],
                    [anchor_values[row]],
                    s=28,
                    color=PAPER_COLORS["anchor"],
                    marker="x",
                    linewidth=1.5,
                    zorder=5,
                )
                ax.set_xlim(limits[x_col])
                ax.set_ylim(limits[y_col])
            else:
                h = hist_corr.loc[y_col, x_col]
                g = gen_corr.loc[y_col, x_col]
                delta = g - h
                ax.axis("off")
                ax.text(
                    0.5,
                    0.52,
                    f"Delta rho\n{delta:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=PAPER_COLORS["positive"] if delta >= 0 else PAPER_COLORS["negative"],
                    transform=ax.transAxes,
                )

            if row < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x_col)
                ax.tick_params(axis="x", rotation=35)
            if col_idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(y_col)
            ax.grid(color=PAPER_COLORS["grid"], alpha=0.45, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_COLORS["historical"], alpha=0.35, markersize=6, label=historical_label),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_COLORS["generated"], alpha=0.7, markersize=6, label=generated_label),
        plt.Line2D([0], [0], marker="x", color=PAPER_COLORS["anchor"], markersize=7, linestyle="none", label="Anchor"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, y=1.005, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def macro_density_grid(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    anchor,
    columns: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    show_generated_mean: bool = True,
) -> plt.Figure:
    """Create a compact shaded-density grid for all macro variables."""
    set_publication_style()
    cols = _resolve_columns(historical_df, generated_df, columns)
    anchor_values = _as_1d_array(anchor)[: len(cols)]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 2.65 * n_rows), squeeze=False)
    limits = _limits_for_columns(historical_df, generated_df, cols)

    for idx, col in enumerate(cols):
        ax = axes[idx // n_cols, idx % n_cols]
        grid = np.linspace(*limits[col], 220)
        _plot_shaded_density(
            ax,
            historical_df[col],
            grid,
            PAPER_COLORS["historical"],
            "Historical",
            alpha=0.22,
            linewidth=1.2,
        )
        _plot_shaded_density(
            ax,
            generated_df[col],
            grid,
            PAPER_COLORS["generated"],
            "Generated",
            alpha=0.20,
            linewidth=1.8,
        )
        ax.axvline(anchor_values[idx], color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5, label="Anchor")
        if show_generated_mean:
            ax.axvline(
                float(np.nanmean(generated_df[col].to_numpy(dtype=float))),
                color=PAPER_COLORS["generated"],
                linestyle=":",
                linewidth=1.7,
                label="Generated mean",
            )
        ax.set_title(col)
        ax.set_yticks([])
        ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.45, linewidth=0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune=None))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    for idx in range(len(cols), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if title:
        fig.suptitle(title, y=0.985, fontsize=15, fontweight="bold")
        top = 0.92
    else:
        top = 0.96
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, top))
    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def macro_story_figure(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    anchor_raw,
    generated_std_df: pd.DataFrame,
    anchor_std,
    columns: Optional[Sequence[str]] = None,
    focus_columns: Sequence[str] = ("bm", "dp", "tms", "dfy"),
    save_path: Optional[str | Path] = None,
    title: str = "Scenario 4: SummerChild Wins Outside Acute Financial Stress",
    subtitle: str = (
        "Generated states leave the April 2020 financial-stress anchor mainly through "
        "higher valuation ratios, especially book-to-market."
    ),
) -> plt.Figure:
    """Create a less crowded, paper-facing story figure.

    The left panel ranks standardized shifts. The right panels show shaded
    historical/generated densities for the key economic channels.
    """
    set_publication_style()
    cols = _resolve_columns(historical_df, generated_df, columns)
    focus = [col for col in focus_columns if col in cols]
    if not focus:
        raise ValueError("None of focus_columns are available in the supplied data.")

    anchor_raw_values = dict(zip(cols, _as_1d_array(anchor_raw)[: len(cols)]))
    anchor_std_values = _as_1d_array(anchor_std)[: len(cols)]
    std_anchor_map = dict(zip(cols, anchor_std_values))
    rows = []
    for col in cols:
        shift = generated_std_df[col].to_numpy(dtype=float) - std_anchor_map[col]
        rows.append(
            {
                "macro": col,
                "median": float(np.nanmedian(shift)),
                "q25": float(np.nanquantile(shift, 0.25)),
                "q75": float(np.nanquantile(shift, 0.75)),
            }
        )
    shift_frame = pd.DataFrame(rows).sort_values("median")

    fig = plt.figure(figsize=(13.5, 7.8))
    grid_spec = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.38, 1.0, 1.0],
        left=0.06,
        right=0.985,
        bottom=0.11,
        top=0.79,
        wspace=0.34,
        hspace=0.55,
    )
    shift_ax = fig.add_subplot(grid_spec[:, 0])
    y = np.arange(len(shift_frame))
    colors = [
        PAPER_COLORS["positive"] if value >= 0 else PAPER_COLORS["negative"]
        for value in shift_frame["median"]
    ]
    shift_ax.barh(y, shift_frame["median"], color=colors, alpha=0.88)
    shift_ax.errorbar(
        shift_frame["median"],
        y,
        xerr=[shift_frame["median"] - shift_frame["q25"], shift_frame["q75"] - shift_frame["median"]],
        fmt="none",
        ecolor=PAPER_COLORS["text"],
        elinewidth=1.0,
        capsize=3,
    )
    shift_ax.axvline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    shift_ax.set_yticks(y)
    shift_ax.set_yticklabels(shift_frame["macro"])
    shift_ax.set_xlabel("Generated minus anchor, standardized units")
    shift_ax.set_title("Required macro shift", fontsize=12, fontweight="bold", pad=8)
    shift_ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)

    limits = _limits_for_columns(historical_df, generated_df, focus, pad_fraction=0.08)
    density_axes = []
    for idx, col in enumerate(focus[:4]):
        ax = fig.add_subplot(grid_spec[idx // 2, 1 + idx % 2])
        density_axes.append(ax)
        x_grid = np.linspace(*limits[col], 240)
        _plot_shaded_density(
            ax,
            historical_df[col],
            x_grid,
            PAPER_COLORS["historical"],
            "Historical" if idx == 0 else "_nolegend_",
            alpha=0.22,
            linewidth=1.2,
        )
        _plot_shaded_density(
            ax,
            generated_df[col],
            x_grid,
            PAPER_COLORS["generated"],
            "Generated" if idx == 0 else "_nolegend_",
            alpha=0.20,
            linewidth=1.9,
        )
        ax.axvline(anchor_raw_values[col], color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5, label="Anchor" if idx == 0 else "_nolegend_")
        shift_row = shift_frame.loc[shift_frame["macro"] == col].iloc[0]
        ax.set_title(f"{col}: median shift {shift_row['median']:+.2f} z", fontsize=11, fontweight="bold", pad=8)
        ax.set_yticks([])
        ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.45, linewidth=0.5)

    handles, labels = density_axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        fontsize=10,
        color=PAPER_COLORS["text"],
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.63, 0.875))
    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def _fit_pca_projection(reference: pd.DataFrame, values: pd.DataFrame, columns: Sequence[str]):
    mean = reference[list(columns)].mean(axis=0)
    std = reference[list(columns)].std(axis=0).replace(0.0, 1.0)
    ref_z = ((reference[list(columns)] - mean) / std).to_numpy(dtype=float)
    val_z = ((values[list(columns)] - mean) / std).to_numpy(dtype=float)
    _, singular_values, vh = np.linalg.svd(ref_z - ref_z.mean(axis=0, keepdims=True), full_matrices=False)
    components = vh[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return val_z @ components, components, explained[:2], mean, std


def macro_pca_story_figure(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    final_diagnostics_df: pd.DataFrame,
    anchor_raw,
    anchor_returns: dict[str, float],
    anchor_regime_probs: dict[str, float],
    true_anchor_label: str,
    columns: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Scenario 4: Macro Counterfactuals Reverse the Model Ranking",
    subtitle: Optional[str] = None,
    anchor_label: str = "Anchor",
    target: str = "sc_beats_ww",
    max_historical_points: int = 900,
    max_generated_points: int = 1000,
    random_state: int = 7,
) -> plt.Figure:
    """Create a paper-ready summary: PCA geography, returns, regimes, plausibility."""
    set_publication_style()
    if target not in {"sc_beats_ww", "ww_beats_sc"}:
        raise ValueError("target must be 'sc_beats_ww' or 'ww_beats_sc'.")
    target_model = "SummerChild" if target == "sc_beats_ww" else "WinterWolf"
    target_win_label = "SC wins" if target == "sc_beats_ww" else "WW wins"
    target_margin_label = "SC-WW" if target == "sc_beats_ww" else "WW-SC"
    cols = _resolve_columns(historical_df, generated_df, columns)
    anchor_values = _as_1d_array(anchor_raw)[: len(cols)]
    anchor_frame = pd.DataFrame([anchor_values], columns=cols)
    generated_mean_frame = pd.DataFrame([generated_df[cols].mean(axis=0)], columns=cols)

    hist_plot = _sample_frame(historical_df[cols].dropna(), max_historical_points, random_state)
    gen_plot = _sample_frame(generated_df[cols].dropna(), max_generated_points, random_state)
    all_for_projection = pd.concat([hist_plot, gen_plot, anchor_frame, generated_mean_frame], ignore_index=True)
    projected, _, explained, _, _ = _fit_pca_projection(historical_df, all_for_projection, cols)
    hist_xy = projected[: len(hist_plot)]
    gen_xy = projected[len(hist_plot) : len(hist_plot) + len(gen_plot)]
    anchor_xy = projected[len(hist_plot) + len(gen_plot)]
    generated_mean_xy = projected[len(hist_plot) + len(gen_plot) + 1]

    classes = [name.replace("prob_", "") for name in generated_df.columns if name.startswith("prob_")]
    classes = sorted(classes)
    generated_regime_probs = {
        name: float(generated_df[f"prob_{name}"].mean())
        for name in classes
        if f"prob_{name}" in generated_df.columns
    }

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.27,
        hspace=0.36,
    )

    pca_ax = fig.add_subplot(gs[0, 0])
    pca_ax.scatter(
        hist_xy[:, 0],
        hist_xy[:, 1],
        s=12,
        color=PAPER_COLORS["historical"],
        alpha=0.24,
        edgecolors="none",
        rasterized=True,
        label="Historical months",
    )
    pca_ax.scatter(
        gen_xy[:, 0],
        gen_xy[:, 1],
        s=16,
        color=PAPER_COLORS["generated"],
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
        label="Generated scenarios",
    )
    pca_ax.scatter(anchor_xy[0], anchor_xy[1], s=110, color=PAPER_COLORS["anchor"], marker="X", linewidth=0.8, edgecolors="white", label=anchor_label, zorder=5)
    pca_ax.scatter(generated_mean_xy[0], generated_mean_xy[1], s=95, color=PAPER_COLORS["generated"], marker="D", linewidth=0.8, edgecolors="white", label="Generated mean", zorder=5)
    pca_ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% historical variance)")
    pca_ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% historical variance)")
    pca_ax.set_title("Macro-state geography", fontsize=12, fontweight="bold")
    pca_ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)
    pca_ax.legend(frameon=False, loc="best", fontsize=8)

    return_ax = fig.add_subplot(gs[0, 1])
    labels = ["SummerChild", "WinterWolf"]
    anchor_return_values = [
        float(anchor_returns["summer_child"]),
        float(anchor_returns["winter_wolf"]),
    ]
    generated_return_values = [
        float(final_diagnostics_df["summer_return"].mean()),
        float(final_diagnostics_df["winter_return"].mean()),
    ]
    generated_return_std = [
        float(final_diagnostics_df["summer_return"].std(ddof=1)),
        float(final_diagnostics_df["winter_return"].std(ddof=1)),
    ]
    x = np.arange(len(labels))
    width = 0.34
    return_ax.bar(x - width / 2, 100 * np.asarray(anchor_return_values), width, color="#cfd3d6", label="Anchor")
    return_ax.bar(
        x + width / 2,
        100 * np.asarray(generated_return_values),
        width,
        yerr=100 * np.asarray(generated_return_std),
        color=PAPER_COLORS["generated"],
        alpha=0.88,
        label="Generated final states",
        capsize=4,
    )
    for xpos, value in zip(x - width / 2, anchor_return_values):
        return_ax.text(xpos, 100 * value + 0.35, f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, generated_return_values):
        return_ax.text(xpos, 100 * value + 0.35, f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=8)
    anchor_gap = anchor_return_values[0] - anchor_return_values[1]
    generated_gap = float(final_diagnostics_df["return_gap"].mean())
    displayed_anchor_gap = anchor_gap if target == "sc_beats_ww" else -anchor_gap
    displayed_generated_gap = generated_gap if target == "sc_beats_ww" else -generated_gap
    return_ax.text(
        0.5,
        0.93,
        f"Anchor gap {target_margin_label}: {100 * displayed_anchor_gap:+.2f} pp\nGenerated gap {target_margin_label}: {100 * displayed_generated_gap:+.2f} pp",
        ha="center",
        va="top",
        transform=return_ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    return_ax.axhline(0.0, color=PAPER_COLORS["text"], linewidth=0.8)
    return_ax.set_xticks(x)
    return_ax.set_xticklabels(labels)
    return_ax.set_ylabel("Realized excess return (%)")
    return_ax.set_title(f"Generated states make {target_model} the winner", fontsize=12, fontweight="bold")
    return_ax.legend(frameon=False, loc="lower right", fontsize=8)
    return_ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    regime_ax = fig.add_subplot(gs[1, 0])
    regime_x = np.arange(len(classes))
    anchor_probs = [float(anchor_regime_probs.get(name, 0.0)) for name in classes]
    gen_probs = [float(generated_regime_probs.get(name, 0.0)) for name in classes]
    regime_ax.bar(regime_x - width / 2, 100 * np.asarray(anchor_probs), width, color="#cfd3d6", label="Anchor classifier")
    regime_ax.bar(regime_x + width / 2, 100 * np.asarray(gen_probs), width, color=PAPER_COLORS["generated"], alpha=0.88, label="Generated mean")
    if true_anchor_label in classes:
        idx = classes.index(true_anchor_label)
        ymax = max(100 * max(anchor_probs + gen_probs), 1.0)
        regime_ax.annotate(
            "true anchor label",
            xy=(idx - width / 2, 100 * anchor_probs[idx]),
            xytext=(idx - 0.25, ymax + 7.0),
            arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.2},
            color=PAPER_COLORS["anchor"],
            fontsize=8,
            ha="center",
        )
        regime_ax.set_ylim(0, ymax + 14.0)
    regime_ax.set_xticks(regime_x)
    regime_ax.set_xticklabels([name.replace("_", "\n") for name in classes])
    regime_ax.set_ylabel("Classifier probability (%)")
    regime_ax.set_title("Regime probabilities shift under generated states", fontsize=12, fontweight="bold")
    regime_ax.legend(frameon=False, loc="upper left", fontsize=8)
    regime_ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    plaus_ax = fig.add_subplot(gs[1, 1])
    regime_palette = {
        "expansion": "#4c78a8",
        "contraction": "#f28e2b",
        "financial_stress": "#c44e52",
    }
    for regime, part in final_diagnostics_df.groupby("regime"):
        y_values = 100 * part["return_gap"]
        if target == "ww_beats_sc":
            y_values = -y_values
        plaus_ax.scatter(
            part["l2_dist"],
            y_values,
            s=52,
            color=regime_palette.get(regime, PAPER_COLORS["generated"]),
            alpha=0.82,
            edgecolors="white",
            linewidth=0.5,
            label=regime,
        )
    plaus_ax.axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    plaus_ax.axvline(2.0, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label="L2 = 2")
    target_win_rate = (
        final_diagnostics_df["winner"].eq("summer_child").mean()
        if target == "sc_beats_ww"
        else final_diagnostics_df["winner"].eq("winter_wolf").mean()
    )
    plaus_ax.text(
        0.04,
        0.94,
        f"{target_win_label}: {100 * target_win_rate:.0f}%\n"
        f"median L2: {final_diagnostics_df['l2_dist'].median():.2f}\n"
        f"box violation: {final_diagnostics_df['box_violation'].mean():.3f}",
        transform=plaus_ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    plaus_ax.set_xlabel("Final L2 distance from anchor")
    plaus_ax.set_ylabel(f"Final {target_margin_label} return gap (%)")
    plaus_ax.set_title("Target achievement and plausibility", fontsize=12, fontweight="bold")
    plaus_ax.legend(frameon=False, loc="lower right", fontsize=8)
    plaus_ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    if subtitle is None:
        if target == "sc_beats_ww":
            subtitle = "PCA is fit on the historical macro panel; generated scenarios produce a positive SC-WW return gap."
        else:
            subtitle = "PCA is fit on the historical macro panel; generated scenarios produce a positive WW-SC return gap."
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        fontsize=10,
        color=PAPER_COLORS["text"],
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def macro_shift_bar(
    generated_std_df: pd.DataFrame,
    anchor_std,
    columns: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Generated Macro Shift from Anchor",
) -> plt.Figure:
    """Plot median generated minus anchor shifts in standardized units."""
    set_publication_style()
    cols = list(columns) if columns is not None else list(generated_std_df.columns)
    anchor_values = _as_1d_array(anchor_std)[: len(cols)]
    rows = []
    for idx, col in enumerate(cols):
        shift = generated_std_df[col].to_numpy(dtype=float) - anchor_values[idx]
        rows.append(
            {
                "macro": col,
                "median": float(np.nanmedian(shift)),
                "q25": float(np.nanquantile(shift, 0.25)),
                "q75": float(np.nanquantile(shift, 0.75)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("median")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.52 * len(frame))))
    colors = [PAPER_COLORS["positive"] if value >= 0 else PAPER_COLORS["negative"] for value in frame["median"]]
    ax.barh(y, frame["median"], color=colors, alpha=0.88)
    ax.errorbar(
        frame["median"],
        y,
        xerr=[frame["median"] - frame["q25"], frame["q75"] - frame["median"]],
        fmt="none",
        ecolor=PAPER_COLORS["text"],
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["macro"])
    ax.set_xlabel("Generated minus anchor, standardized macro units")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=PAPER_COLORS["grid"], alpha=0.55, linewidth=0.6)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def scenario5_story_figure(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    final_diagnostics_df: pd.DataFrame,
    anchor_raw,
    anchor_regime_probs: dict[str, float],
    true_anchor_label: str,
    target_metric_col: str,
    target_metric_label: str,
    columns: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Scenario 5: Decision-Level Counterfactual Probe",
    subtitle: Optional[str] = None,
    anchor_label: str = "Anchor",
    max_historical_points: int = 900,
    max_generated_points: int = 1000,
    random_state: int = 7,
) -> plt.Figure:
    """Paper-ready summary for Scenario 5 decision-level probes."""
    set_publication_style()
    if target_metric_col not in final_diagnostics_df.columns:
        raise ValueError(f"target_metric_col={target_metric_col!r} is missing from diagnostics.")
    cols = _resolve_columns(historical_df, generated_df, columns)
    anchor_values = _as_1d_array(anchor_raw)[: len(cols)]
    anchor_frame = pd.DataFrame([anchor_values], columns=cols)
    generated_mean_frame = pd.DataFrame([generated_df[cols].mean(axis=0)], columns=cols)

    hist_plot = _sample_frame(historical_df[cols].dropna(), max_historical_points, random_state)
    gen_plot = _sample_frame(generated_df[cols].dropna(), max_generated_points, random_state)
    all_for_projection = pd.concat([hist_plot, gen_plot, anchor_frame, generated_mean_frame], ignore_index=True)
    projected, _, explained, _, _ = _fit_pca_projection(historical_df, all_for_projection, cols)
    hist_xy = projected[: len(hist_plot)]
    gen_xy = projected[len(hist_plot) : len(hist_plot) + len(gen_plot)]
    anchor_xy = projected[len(hist_plot) + len(gen_plot)]
    generated_mean_xy = projected[len(hist_plot) + len(gen_plot) + 1]

    classes = [name.replace("prob_", "") for name in generated_df.columns if name.startswith("prob_")]
    classes = sorted(classes)
    generated_regime_probs = {
        name: float(generated_df[f"prob_{name}"].mean())
        for name in classes
        if f"prob_{name}" in generated_df.columns
    }

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.28,
        hspace=0.36,
    )

    pca_ax = fig.add_subplot(gs[0, 0])
    pca_ax.scatter(
        hist_xy[:, 0],
        hist_xy[:, 1],
        s=12,
        color=PAPER_COLORS["historical"],
        alpha=0.24,
        edgecolors="none",
        rasterized=True,
        label="Historical months",
    )
    pca_ax.scatter(
        gen_xy[:, 0],
        gen_xy[:, 1],
        s=16,
        color=PAPER_COLORS["generated"],
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
        label="Generated scenarios",
    )
    pca_ax.scatter(anchor_xy[0], anchor_xy[1], s=110, color=PAPER_COLORS["anchor"], marker="X", linewidth=0.8, edgecolors="white", label=anchor_label, zorder=5)
    pca_ax.scatter(generated_mean_xy[0], generated_mean_xy[1], s=95, color=PAPER_COLORS["generated"], marker="D", linewidth=0.8, edgecolors="white", label="Generated mean", zorder=5)
    pca_ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% historical variance)")
    pca_ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% historical variance)")
    pca_ax.set_title("Macro-state geography", fontsize=12, fontweight="bold")
    pca_ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)
    pca_ax.legend(frameon=False, loc="best", fontsize=8)

    target_ax = fig.add_subplot(gs[0, 1])
    values = final_diagnostics_df[target_metric_col].to_numpy(dtype=float)
    target_ax.hist(values, bins=min(12, max(4, len(values))), color=PAPER_COLORS["generated"], alpha=0.82)
    target_ax.axvline(float(np.nanmedian(values)), color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.5, label="Median")
    target_ax.text(
        0.04,
        0.94,
        f"mean: {np.nanmean(values):.3f}\nmedian: {np.nanmedian(values):.3f}\nq25-q75: {np.nanquantile(values, 0.25):.3f}-{np.nanquantile(values, 0.75):.3f}",
        transform=target_ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    target_ax.set_xlabel(target_metric_label)
    target_ax.set_ylabel("Number of chains")
    target_ax.set_title("Decision target achieved", fontsize=12, fontweight="bold")
    target_ax.legend(frameon=False, loc="upper right", fontsize=8)
    target_ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    regime_ax = fig.add_subplot(gs[1, 0])
    width = 0.36
    regime_x = np.arange(len(classes))
    anchor_probs = [float(anchor_regime_probs.get(name, 0.0)) for name in classes]
    gen_probs = [float(generated_regime_probs.get(name, 0.0)) for name in classes]
    regime_ax.bar(regime_x - width / 2, 100 * np.asarray(anchor_probs), width, color="#cfd3d6", label="Anchor classifier")
    regime_ax.bar(regime_x + width / 2, 100 * np.asarray(gen_probs), width, color=PAPER_COLORS["generated"], alpha=0.88, label="Generated mean")
    if true_anchor_label in classes:
        idx = classes.index(true_anchor_label)
        ymax = max(100 * max(anchor_probs + gen_probs), 1.0)
        regime_ax.annotate(
            "true anchor label",
            xy=(idx - width / 2, 100 * anchor_probs[idx]),
            xytext=(idx - 0.25, ymax + 7.0),
            arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["anchor"], "lw": 1.2},
            color=PAPER_COLORS["anchor"],
            fontsize=8,
            ha="center",
        )
        regime_ax.set_ylim(0, ymax + 14.0)
    regime_ax.set_xticks(regime_x)
    regime_ax.set_xticklabels([name.replace("_", "\n") for name in classes])
    regime_ax.set_ylabel("Classifier probability (%)")
    regime_ax.set_title("Regime probabilities", fontsize=12, fontweight="bold")
    regime_ax.legend(frameon=False, loc="upper left", fontsize=8)
    regime_ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    plaus_ax = fig.add_subplot(gs[1, 1])
    regime_palette = {
        "expansion": "#4c78a8",
        "contraction": "#f28e2b",
        "financial_stress": "#c44e52",
    }
    for regime, part in final_diagnostics_df.groupby("regime"):
        plaus_ax.scatter(
            part["l2_dist"],
            part[target_metric_col],
            s=52,
            color=regime_palette.get(regime, PAPER_COLORS["generated"]),
            alpha=0.82,
            edgecolors="white",
            linewidth=0.5,
            label=regime,
        )
    plaus_ax.axvline(2.0, color=PAPER_COLORS["anchor"], linestyle="--", linewidth=1.2, label="L2 = 2")
    plaus_ax.text(
        0.04,
        0.94,
        f"median L2: {final_diagnostics_df['l2_dist'].median():.2f}\n"
        f"median Mah: {final_diagnostics_df['mah_dist'].median():.2f}\n"
        f"box violation: {final_diagnostics_df['box_violation'].mean():.3f}",
        transform=plaus_ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"], "alpha": 0.95},
    )
    plaus_ax.set_xlabel("Final L2 distance from anchor")
    plaus_ax.set_ylabel(target_metric_label)
    plaus_ax.set_title("Target achievement and plausibility", fontsize=12, fontweight="bold")
    plaus_ax.legend(frameon=False, loc="lower right", fontsize=8)
    plaus_ax.grid(color=PAPER_COLORS["grid"], alpha=0.50, linewidth=0.6)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    if subtitle is None:
        subtitle = "PCA is fit on the historical macro panel; generated states maximize a decision-level target under macro plausibility."
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10, color=PAPER_COLORS["text"])
    if save_path is not None:
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
    return fig


def correlation_latex(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    caption: str = "Correlation matrix",
    label: str = "tab:correlation_matrix",
) -> str:
    cols = list(columns) if columns is not None else list(frame.columns)
    latex = frame[cols].corr().round(2).to_latex(
        caption=caption,
        float_format="%.2f",
        label=label,
        position="h!",
        column_format="l" + "c" * len(cols),
        escape=False,
    )
    return latex


def pairplot(
    historical_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    m0,
    lower_limits=None,
    upper_limits=None,
    save_address: Optional[str] = None,
):
    """Backward-compatible wrapper around :func:`macro_pair_grid`."""
    fig = macro_pair_grid(
        historical_df=historical_df,
        generated_df=generated_df,
        anchor=m0,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        save_path=save_address,
    )
    return fig


def plot_3x3_kde_grid(historical_df: pd.DataFrame, generated_df: pd.DataFrame, start_point):
    """Backward-compatible wrapper for the old KDE grid helper."""
    return macro_density_grid(historical_df, generated_df, start_point)
