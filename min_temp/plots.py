from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .metrics import PredictionResult


def plot_error_bars(
    results: Iterable[PredictionResult],
    *,
    save_path: Optional[str] = None,
):
    """Bar plot of MAE and RMSE for each method."""

    res_list = list(results)
    labels = [r.method for r in res_list]
    mae = [r.mae for r in res_list]
    rmse = [r.rmse for r in res_list]

    x = range(len(res_list))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width / 2 for i in x], mae, width, label="MAE")
    ax.bar([i + width / 2 for i in x], rmse, width, label="RMSE")
    ax.set_ylabel("Error (deg C)")
    ax.set_title("Prediction error by method")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_pred_vs_actual(
    results: Iterable[PredictionResult],
    *,
    save_path: Optional[str] = None,
):
    """Scatter plot of predicted vs actual for each method."""
    res_list = list(results)

    fig, ax = plt.subplots()
    for r in res_list:
        ax.scatter(r.y_true, r.y_pred, label=r.method, alpha=0.7)

    lims = [
        min(min(r.y_true) for r in res_list),
        max(max(r.y_true) for r in res_list),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Observed Tmin (deg C)")
    ax.set_ylabel("Predicted Tmin (deg C)")
    ax.legend()
    ax.set_title("Predicted vs Observed Tmin")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


__all__ = ["plot_error_bars", "plot_pred_vs_actual"]
