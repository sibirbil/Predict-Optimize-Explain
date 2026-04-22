from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.data import DEFAULT_DATA_CONFIG_PATH, Universe500DataConfig
from src.data import TARGET_COLUMN
from src.modeling.fnn.evaluation import evaluate_prediction_run, load_prediction_artifact
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    output_root: str = "batuhan/artifacts/benchmarks"
    data_config_path: str = str(DEFAULT_DATA_CONFIG_PATH)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    elastic_net_alphas: tuple[float, ...] = (1e-3,)
    elastic_net_l1_ratios: tuple[float, ...] = (0.5,)
    quantiles: int = 10

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "data_config_path": self.data_config_path,
            "ridge_alphas": list(self.ridge_alphas),
            "elastic_net_alphas": list(self.elastic_net_alphas),
            "elastic_net_l1_ratios": list(self.elastic_net_l1_ratios),
            "quantiles": self.quantiles,
        }


def run_benchmark_suite(
    suite_name: str,
    config: BenchmarkSuiteConfig | None = None,
) -> Path:
    config = config or BenchmarkSuiteConfig()
    data_config = Universe500DataConfig.from_json(config.data_config_path)
    loader = data_config.build_loader()

    train = load_split_arrays(loader, "train", None)
    val = load_split_arrays(loader, "val", None)
    test = load_split_arrays(loader, "test", None)

    run_names = [
        run_single_benchmark(
            benchmark_name="zero_forecast",
            suite_name=suite_name,
            config=config,
            train=train,
            val=val,
            test=test,
        ),
        run_single_benchmark(
            benchmark_name="train_mean_forecast",
            suite_name=suite_name,
            config=config,
            train=train,
            val=val,
            test=test,
        ),
        run_single_benchmark(
            benchmark_name="linear_regression",
            suite_name=suite_name,
            config=config,
            train=train,
            val=val,
            test=test,
        ),
        run_single_benchmark(
            benchmark_name="ridge",
            suite_name=suite_name,
            config=config,
            train=train,
            val=val,
            test=test,
        ),
        run_single_benchmark(
            benchmark_name="elastic_net",
            suite_name=suite_name,
            config=config,
            train=train,
            val=val,
            test=test,
        ),
    ]

    comparison = build_comparison_table(
        rows=[
            ("current_clean_fnn", "batuhan/artifacts/fnn", "full_fnn_baseline_20260404"),
            ("legacy_style_fnn", "batuhan/artifacts/fnn", "full_fnn_legacy_style_20260404"),
            *[
                (run_name.removesuffix(f"_{suite_name}"), config.output_root, run_name)
                for run_name in run_names
            ],
        ]
    )
    comparison_dir = resolve_repo_path(config.output_root) / suite_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_dir / "model_comparison.csv", index=False)
    return comparison_dir


def run_single_benchmark(
    benchmark_name: str,
    suite_name: str,
    config: BenchmarkSuiteConfig,
    train,
    val,
    test,
) -> str:
    run_name = f"{benchmark_name}_{suite_name}"
    run_dir = resolve_repo_path(config.output_root) / run_name
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train_mean = float(train.y.mean())

    if benchmark_name == "zero_forecast":
        model_info = {"model_name": benchmark_name}
        train_pred = np.zeros_like(train.y)
        val_pred = np.zeros_like(val.y)
        test_pred = np.zeros_like(test.y)
    elif benchmark_name == "train_mean_forecast":
        model_info = {"model_name": benchmark_name, "train_mean": train_mean}
        train_pred = np.full_like(train.y, train_mean)
        val_pred = np.full_like(val.y, train_mean)
        test_pred = np.full_like(test.y, train_mean)
    elif benchmark_name == "linear_regression":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        estimator.fit(train.X, train.y)
        model_info = {"model_name": benchmark_name}
        train_pred = estimator.predict(train.X).astype(np.float32)
        val_pred = estimator.predict(val.X).astype(np.float32)
        test_pred = estimator.predict(test.X).astype(np.float32)
    elif benchmark_name == "ridge":
        estimator, best_info = fit_best_ridge(train.X, train.y, val.X, val.y, config.ridge_alphas)
        model_info = {"model_name": benchmark_name, **best_info}
        train_pred = estimator.predict(train.X).astype(np.float32)
        val_pred = estimator.predict(val.X).astype(np.float32)
        test_pred = estimator.predict(test.X).astype(np.float32)
    elif benchmark_name == "elastic_net":
        estimator, best_info = fit_best_elastic_net(
            train.X,
            train.y,
            val.X,
            val.y,
            config.elastic_net_alphas,
            config.elastic_net_l1_ratios,
        )
        model_info = {"model_name": benchmark_name, **best_info}
        train_pred = estimator.predict(train.X).astype(np.float32)
        val_pred = estimator.predict(val.X).astype(np.float32)
        test_pred = estimator.predict(test.X).astype(np.float32)
    else:
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

    save_predictions(predictions_dir / "train_predictions.parquet", train.metadata, train.y, train_pred)
    save_predictions(predictions_dir / "val_predictions.parquet", val.metadata, val.y, val_pred)
    save_predictions(predictions_dir / "test_predictions.parquet", test.metadata, test.y, test_pred)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite_name": suite_name,
                "run_name": run_name,
                "data_config_path": config.data_config_path,
                "resolved_dataset_root": str(Universe500DataConfig.from_json(config.data_config_path).resolve_dataset_root()),
                "benchmark_config": config.to_dict(),
                "model_info": model_info,
            },
            handle,
            indent=2,
        )

    evaluate_prediction_run(
        run_name=run_name,
        output_root=config.output_root,
    )

    return run_name


def load_split_arrays(loader, split: str, max_rows: int | None) -> SplitArrays:
    split_data = loader.load_split(split)  # type: ignore[arg-type]
    X = split_data.X
    y = split_data.y
    metadata = split_data.metadata
    if max_rows is not None:
        X = X.iloc[:max_rows].copy()
        y = y.iloc[:max_rows].copy()
        metadata = metadata.iloc[:max_rows].copy()
    return SplitArrays(
        X=X.to_numpy(dtype=np.float32, copy=True),
        y=y[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True),
        metadata=metadata.reset_index(drop=True),
    )


def save_predictions(
    path: Path,
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> None:
    prediction_frame = metadata.copy()
    prediction_frame[TARGET_COLUMN] = y_true
    prediction_frame["prediction"] = predictions
    prediction_frame["residual"] = y_true - predictions
    prediction_frame.to_parquet(path, index=False)


def fit_best_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: tuple[float, ...],
):
    best_estimator = None
    best_score = -np.inf
    best_alpha = None
    for alpha in alphas:
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        estimator.fit(X_train, y_train)
        score = r2_oos_zero(y_val, estimator.predict(X_val))
        if score > best_score:
            best_score = score
            best_estimator = estimator
            best_alpha = alpha
    if best_estimator is None or best_alpha is None:
        raise RuntimeError("Failed to fit ridge benchmark.")
    return best_estimator, {"selected_alpha": best_alpha, "val_r2_oos_zero": best_score}


def fit_best_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: tuple[float, ...],
    l1_ratios: tuple[float, ...],
):
    best_estimator = None
    best_score = -np.inf
    best_alpha = None
    best_l1_ratio = None
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            estimator = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=1000,
                            random_state=42,
                            selection="random",
                        ),
                    ),
                ]
            )
            estimator.fit(X_train, y_train)
            score = r2_oos_zero(y_val, estimator.predict(X_val))
            if score > best_score:
                best_score = score
                best_estimator = estimator
                best_alpha = alpha
                best_l1_ratio = l1_ratio
    if best_estimator is None or best_alpha is None or best_l1_ratio is None:
        raise RuntimeError("Failed to fit elastic net benchmark.")
    return best_estimator, {
        "selected_alpha": best_alpha,
        "selected_l1_ratio": best_l1_ratio,
        "val_r2_oos_zero": best_score,
    }


def r2_oos_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = y_true - y_pred
    sse_model = float(np.sum(np.square(residual)))
    sse_zero = float(np.sum(np.square(y_true)))
    if sse_zero <= 0:
        return 0.0
    return 1.0 - sse_model / sse_zero


def build_comparison_table(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    comparison_rows: list[dict[str, float | str]] = []
    for display_name, output_root, run_name in rows:
        metrics = json.loads(
            (resolve_repo_path(output_root) / run_name / "metrics_summary.json").read_text()
        )
        comparison_rows.append(
            {
                "model": display_name,
                "run_name": run_name,
                "train_r2_oos_zero": metrics["split_metrics"]["train"]["r2_oos_zero_benchmark"],
                "val_r2_oos_zero": metrics["split_metrics"]["val"]["r2_oos_zero_benchmark"],
                "test_r2_oos_zero": metrics["split_metrics"]["test"]["r2_oos_zero_benchmark"],
                "train_sign_accuracy": metrics["split_metrics"]["train"]["sign_accuracy"],
                "val_sign_accuracy": metrics["split_metrics"]["val"]["sign_accuracy"],
                "test_sign_accuracy": metrics["split_metrics"]["test"]["sign_accuracy"],
                "val_mean_monthly_ic": metrics["monthly_metrics"]["val"]["mean_monthly_ic"],
                "test_mean_monthly_ic": metrics["monthly_metrics"]["test"]["mean_monthly_ic"],
                "val_mean_monthly_rank_ic": metrics["monthly_metrics"]["val"]["mean_monthly_rank_ic"],
                "test_mean_monthly_rank_ic": metrics["monthly_metrics"]["test"]["mean_monthly_rank_ic"],
            }
        )
    return pd.DataFrame(comparison_rows)
