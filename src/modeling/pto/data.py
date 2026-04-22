from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.configs.pto import PredictiveBackboneConfig
from src.data import Universe500Loader
from src.modeling.pto.covariance import ewma_covariance
from src.modeling.pto.predictive_backbone import load_predictive_backbone
from src.utils.dates import shift_yyyymm


@dataclass(frozen=True)
class DecisionMonth:
    split: str
    yyyymm: int
    frame: pd.DataFrame
    covariance: np.ndarray


def build_return_panel(loader: Universe500Loader) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        metadata = loader.load_split(split).metadata.copy()
        metadata["yyyymm"] = metadata["yyyymm"].astype(int)
        frames.append(metadata[["yyyymm", "permno", "ret_tplus1", "excess_ret", "Rfree"]])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    return panel


def prepare_decision_months(
    backbone_config: PredictiveBackboneConfig,
    lookback_months: int,
    ewma_lambda: float,
    shrinkage: float,
    ridge: float,
    min_assets: int,
) -> dict[str, list[DecisionMonth]]:
    frames = load_predictive_backbone(backbone_config)
    loader = backbone_config.load_data_config().build_loader()
    return_panel = build_return_panel(loader)

    outputs: dict[str, list[DecisionMonth]] = {"val": [], "test": []}
    for split in ("val", "test"):
        split_frame = frames[split].frame.copy()
        split_frame["yyyymm"] = split_frame["yyyymm"].astype(int)
        for yyyymm, month_frame in split_frame.groupby("yyyymm", sort=True):
            prepared = prepare_single_month(
                split=split,
                yyyymm=int(yyyymm),
                prediction_frame=month_frame.reset_index(drop=True),
                return_panel=return_panel,
                lookback_months=lookback_months,
                ewma_lambda=ewma_lambda,
                shrinkage=shrinkage,
                ridge=ridge,
                min_assets=min_assets,
            )
            if prepared is not None:
                outputs[split].append(prepared)
    return outputs


def prepare_single_month(
    split: str,
    yyyymm: int,
    prediction_frame: pd.DataFrame,
    return_panel: pd.DataFrame,
    lookback_months: int,
    ewma_lambda: float,
    shrinkage: float,
    ridge: float,
    min_assets: int,
) -> DecisionMonth | None:
    start_month = shift_yyyymm(yyyymm, -lookback_months)
    end_month = shift_yyyymm(yyyymm, -1)
    active_permnos = prediction_frame["permno"].astype(int).tolist()

    history = return_panel[
        (return_panel["yyyymm"] >= start_month)
        & (return_panel["yyyymm"] <= end_month)
        & (return_panel["permno"].isin(active_permnos))
    ].copy()
    if history.empty:
        return None

    history_wide = history.pivot(index="yyyymm", columns="permno", values="excess_ret").sort_index()
    if len(history_wide) < lookback_months:
        return None

    complete_permnos = history_wide.columns[history_wide.notna().all(axis=0)].astype(int)
    filtered = prediction_frame[prediction_frame["permno"].isin(complete_permnos)].copy()
    filtered = filtered.sort_values("permno").reset_index(drop=True)
    if len(filtered) < min_assets:
        return None

    final_permnos = filtered["permno"].astype(int).tolist()
    returns_matrix = history_wide[final_permnos].to_numpy(dtype=float)
    covariance = ewma_covariance(
        returns_matrix=returns_matrix,
        ewma_lambda=ewma_lambda,
        shrinkage=shrinkage,
        ridge=ridge,
    )
    if covariance is None:
        return None

    filtered["permno"] = filtered["permno"].astype(int)
    return DecisionMonth(
        split=split,
        yyyymm=yyyymm,
        frame=filtered,
        covariance=covariance,
    )
