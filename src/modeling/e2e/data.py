from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.configs.data import Universe500DataConfig
from src.configs.e2e import E2EV1Config
from src.data import TARGET_COLUMN
from src.modeling.pto.data import build_return_panel, prepare_single_month


@dataclass(frozen=True)
class E2EDecisionMonth:
    split: str
    yyyymm: int
    X: np.ndarray
    metadata: pd.DataFrame
    covariance: np.ndarray


def prepare_e2e_months(
    config: E2EV1Config,
    data_config: Universe500DataConfig | None = None,
) -> dict[str, list[E2EDecisionMonth]]:
    data_config = data_config or config.load_data_config()
    loader = data_config.build_loader()
    return_panel = build_return_panel(loader)
    feature_names = list(loader.load_feature_names())

    month_limits = {
        "train": config.max_train_months,
        "val": config.max_val_months,
        "test": config.max_test_months,
    }

    outputs: dict[str, list[E2EDecisionMonth]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        split_bundle = loader.load_split(split)
        combined = pd.concat(
            [
                split_bundle.metadata.reset_index(drop=True),
                split_bundle.y.reset_index(drop=True),
                split_bundle.X.reset_index(drop=True),
            ],
            axis=1,
        )
        combined["yyyymm"] = combined["yyyymm"].astype(int)

        prepared_months: list[E2EDecisionMonth] = []
        for yyyymm, month_frame in combined.groupby("yyyymm", sort=True):
            prepared = prepare_single_month(
                split=split,
                yyyymm=int(yyyymm),
                prediction_frame=month_frame.reset_index(drop=True),
                return_panel=return_panel,
                lookback_months=config.lookback_months,
                ewma_lambda=config.ewma_lambda,
                shrinkage=config.covariance_shrinkage,
                ridge=config.covariance_ridge,
                min_assets=config.min_assets,
            )
            if prepared is None:
                continue
            frame = prepared.frame.reset_index(drop=True)
            metadata = frame[["yyyymm", "permno", "ret_tplus1", "Rfree", "excess_ret", TARGET_COLUMN]].copy()
            prepared_months.append(
                E2EDecisionMonth(
                    split=split,
                    yyyymm=int(yyyymm),
                    X=frame[feature_names].to_numpy(dtype=np.float32, copy=True),
                    metadata=metadata,
                    covariance=prepared.covariance.astype(np.float32, copy=False),
                )
            )
        limit = month_limits[split]
        outputs[split] = prepared_months[:limit] if limit is not None else prepared_months
    return outputs
