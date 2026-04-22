from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .universe_500 import Universe500Bundle, load_universe_500


@dataclass(frozen=True)
class MacroStandardizationStats:
    train_end_yyyymm: int
    macro_predictors: tuple[str, ...]
    means: dict[str, float]
    stds: dict[str, float]


def fit_train_macro_standardization(bundle: Universe500Bundle) -> MacroStandardizationStats:
    train_end_yyyymm = int(bundle.train.metadata["yyyymm"].max())
    macro_cols = list(bundle.macro_predictors)
    macro = bundle.macro_final[["yyyymm", *macro_cols]].copy()
    train_macro = macro.loc[macro["yyyymm"] <= train_end_yyyymm, macro_cols]

    means = {col: float(train_macro[col].mean()) for col in macro_cols}
    stds = {}
    for col in macro_cols:
        std = float(train_macro[col].std())
        stds[col] = 1.0 if std == 0.0 else std

    return MacroStandardizationStats(
        train_end_yyyymm=train_end_yyyymm,
        macro_predictors=tuple(macro_cols),
        means=means,
        stds=stds,
    )


def build_standardized_macro_frame(
    bundle: Universe500Bundle,
    stats: MacroStandardizationStats,
) -> pd.DataFrame:
    macro_cols = list(stats.macro_predictors)
    macro = bundle.macro_final[["yyyymm", *macro_cols]].copy()
    for col in macro_cols:
        macro[col] = (macro[col] - stats.means[col]) / stats.stds[col]
    return macro


def build_split_features(
    split_X: pd.DataFrame,
    split_metadata: pd.DataFrame,
    macro_standardized: pd.DataFrame,
    firm_feature_names: tuple[str, ...],
    macro_predictors: tuple[str, ...],
) -> pd.DataFrame:
    firm_cols = list(firm_feature_names)
    macro_cols = list(macro_predictors)

    base = split_X[firm_cols].copy()
    macro_by_row = split_metadata[["yyyymm"]].merge(
        macro_standardized,
        on="yyyymm",
        how="left",
        validate="many_to_one",
    )
    if macro_by_row[macro_cols].isna().any().any():
        missing_months = sorted(
            macro_by_row.loc[macro_by_row[macro_cols].isna().any(axis=1), "yyyymm"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ValueError(f"Missing standardized macro rows for months: {missing_months[:10]}")

    pieces = [base]
    for macro_col in macro_cols:
        interaction = base.mul(macro_by_row[macro_col].to_numpy(), axis=0)
        interaction.columns = [f"{col}_x_{macro_col}" for col in firm_cols]
        pieces.append(interaction)

    out = pd.concat(pieces, axis=1)
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def save_dataset(
    source_root: Path,
    output_root: Path,
    bundle: Universe500Bundle,
    macro_standardized: pd.DataFrame,
    stats: MacroStandardizationStats,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": bundle.train,
        "val": bundle.val,
        "test": bundle.test,
    }

    for name, split in split_map.items():
        X_new = build_split_features(
            split_X=split.X,
            split_metadata=split.metadata,
            macro_standardized=macro_standardized,
            firm_feature_names=bundle.firm_feature_names,
            macro_predictors=bundle.macro_predictors,
        )
        expected_columns = list(bundle.feature_names)
        if list(X_new.columns) != expected_columns:
            raise ValueError(f"Feature order mismatch in split={name}.")
        X_new.to_parquet(output_root / f"X_{name}.parquet", index=False)
        split.y.to_parquet(output_root / f"y_{name}.parquet", index=False)
        split.metadata.to_parquet(output_root / f"metadata_{name}.parquet", index=False)

    bundle.metadata_all.to_parquet(output_root / "metadata.parquet", index=False)
    bundle.macro_final.to_parquet(output_root / "macro_final.parquet", index=False)
    macro_standardized.to_parquet(output_root / "macro_standardized_train_only.parquet", index=False)

    copy_names = [
        "feature_names.npy",
        "firm_feature_names.npy",
        "macro_predictors.npy",
        "selected_500_permnos.npy",
        "selected_500_permnos_ordered.csv",
        "universe_500_roster.csv",
        "universe_500_roster.parquet",
        "build_summary.csv",
        "extra_diversity_summary.csv",
        "missing_rate_summary.csv",
        "msenames_snapshot_202412.parquet",
    ]
    for name in copy_names:
        _copy_if_exists(source_root / name, output_root / name)

    payload = {
        "source_dataset_root": str(source_root),
        "construction": "firm_normalized[-1,1] + macro_standardized(train_only) + interactions",
        "train_end_yyyymm": stats.train_end_yyyymm,
        "macro_predictors": list(stats.macro_predictors),
        "macro_means": stats.means,
        "macro_stds": stats.stds,
        "saved_macro_standardized_file": "macro_standardized_train_only.parquet",
    }
    (output_root / "macro_standardization_stats.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_standardized_macro_dataset(
    source_root: Path,
    output_root: Path,
) -> Path:
    bundle = load_universe_500(source_root)
    stats = fit_train_macro_standardization(bundle)
    macro_standardized = build_standardized_macro_frame(bundle, stats)
    save_dataset(
        source_root=source_root,
        output_root=output_root,
        bundle=bundle,
        macro_standardized=macro_standardized,
        stats=stats,
    )
    return output_root
