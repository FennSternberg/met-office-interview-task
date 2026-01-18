from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Tuple
import math
import pandas as pd

from .base import MinTempPredictor


PredFn = Callable[[pd.DataFrame], Sequence[float]]


@dataclass(frozen=True)
class PredictionResult:
    method: str
    y_true: Tuple[float, ...]
    y_pred: Tuple[float, ...]
    errors: Tuple[float, ...]
    mae: float
    rmse: float
    bias: float

def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> PredictionResult:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be same length, got {len(y_true)} vs {len(y_pred)}")

    t = tuple(float(v) for v in y_true)
    p = tuple(float(v) for v in y_pred)
    errs = tuple(pred - obs for pred, obs in zip(p, t))

    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    bias = sum(errs) / len(errs)

    return PredictionResult(
        method="",
        y_true=t,
        y_pred=p,
        errors=errs,
        mae=mae,
        rmse=rmse,
        bias=bias,
    )


def evaluate_predictor(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
    predict_fn: PredFn | MinTempPredictor,
    *,
    actual_key: str = "observed_min_temp_c",
    method_name: str | None = None,
) -> PredictionResult:
    """Compute metrics for a single prediction function given rows that include the actual Tmin."""

    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    y_true = df[actual_key].astype(float).tolist()

    if isinstance(predict_fn, MinTempPredictor) or hasattr(predict_fn, "predict"):
        y_pred = df.apply(lambda row: predict_fn.predict(row), axis=1)
    elif callable(predict_fn):
        y_pred = predict_fn(df)
    else:
        raise TypeError("predict_fn must be callable or expose predict(row)")

    resolved_name = method_name
    if resolved_name is None:
        resolved_name = getattr(predict_fn, "name", None) or getattr(predict_fn, "__name__", "predict_fn")

    metrics = compute_metrics(y_true, y_pred)
    return PredictionResult(
        method=resolved_name,
        y_true=metrics.y_true,
        y_pred=metrics.y_pred,
        errors=metrics.errors,
        mae=metrics.mae,
        rmse=metrics.rmse,
        bias=metrics.bias,
    )


def evaluate_predictors(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
    methods: Mapping[str, PredFn | MinTempPredictor],
    *,
    actual_key: str = "observed_min_temp_c",
) -> Tuple[PredictionResult, ...]:
    """Evaluate multiple methods"""
    results = []
    for name, fn in methods.items():
        results.append(evaluate_predictor(rows, fn, actual_key=actual_key, method_name=name))
    return tuple(results)


__all__ = [
    "PredictionResult",
    "compute_metrics",
    "evaluate_predictor",
    "evaluate_predictors",
]
