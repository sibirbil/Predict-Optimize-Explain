from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.configs.pto import PredictiveBackboneConfig
from src.data import TARGET_COLUMN
from src.modeling.fnn.evaluation import load_prediction_artifact


@dataclass(frozen=True)
class PredictiveSplitFrame:
    split: str
    frame: pd.DataFrame


def load_predictive_backbone(
    config: PredictiveBackboneConfig | None = None,
) -> dict[str, PredictiveSplitFrame]:
    config = config or PredictiveBackboneConfig()
    loader = config.load_data_config().build_loader()
    run_dir = config.resolve_run_dir()
    predictions_dir = run_dir / "predictions"

    outputs: dict[str, PredictiveSplitFrame] = {}
    for split in ("train", "val", "test"):
        prediction_frame = load_prediction_artifact(predictions_dir / f"{split}_predictions.parquet")
        canonical_split = loader.load_split(split)  # type: ignore[arg-type]
        validate_prediction_alignment(
            split=split,
            prediction_frame=prediction_frame,
            metadata=canonical_split.metadata.reset_index(drop=True),
            y=canonical_split.y.reset_index(drop=True),
        )
        outputs[split] = PredictiveSplitFrame(split=split, frame=prediction_frame.reset_index(drop=True))
    return outputs


def validate_prediction_alignment(
    split: str,
    prediction_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    y: pd.DataFrame,
) -> None:
    if len(prediction_frame) != len(metadata) or len(prediction_frame) != len(y):
        raise ValueError(
            f"{split} predictive backbone row mismatch: "
            f"predictions={len(prediction_frame)}, metadata={len(metadata)}, y={len(y)}"
        )

    if not prediction_frame["yyyymm"].astype(int).reset_index(drop=True).equals(
        metadata["yyyymm"].astype(int).reset_index(drop=True)
    ):
        raise ValueError(f"{split} predictive backbone yyyymm alignment failed.")

    if not prediction_frame["permno"].astype(int).reset_index(drop=True).equals(
        metadata["permno"].astype(int).reset_index(drop=True)
    ):
        raise ValueError(f"{split} predictive backbone permno alignment failed.")

    if not np.allclose(
        prediction_frame[TARGET_COLUMN].to_numpy(dtype=float),
        y[TARGET_COLUMN].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(f"{split} predictive backbone target alignment failed.")
