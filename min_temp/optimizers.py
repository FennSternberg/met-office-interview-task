from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .base import MinTempPredictor
from .metrics import PredictionResult, compute_metrics
from .predictors import CraddockAndPritchardModel, KTable


@dataclass(frozen=True)
class FittedModel:
    model: MinTempPredictor
    metrics: PredictionResult
    params: Dict[str, float]


@dataclass(frozen=True)
class FitWorkspace:
    df: pd.DataFrame
    target_col: str
    design_cols: list[Tuple[str, np.ndarray]]
    residual: np.ndarray
    fixed: Dict[str, float]


class OptimizerFactory(ABC):
    """
    Generic least-squares fitter for models that are linear in their parameters.
    The solver supports "partial fitting" by holding a subset of parameters fixed.
    """

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "observed_min_temp_c",
        fixed: Mapping[str, float] | None = None,
    ) -> FittedModel:
        """
        Fit the model to the data in `df`, returning a fitted model and metrics.
        """
        workspace = self.prepare_fit(df, target_col=target_col, fixed=fixed)
        solved = self.solve(workspace)
        model = self._build_model(solved)
        metrics = self.evaluate(model, df, target_col)
        return FittedModel(model=model, metrics=metrics, params=solved)

    def prepare_fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "observed_min_temp_c",
        fixed: Mapping[str, float] | None = None,
    ) -> FitWorkspace:
        
        """Prepare the fitting workspace, applying any fixed parameter values."""

        parameter_columns = self._parameter_columns(df)
        y = df[target_col].astype(float).to_numpy()
        residual, design_cols, solved_fixed = self._build_residual_and_design(
            y,
            parameter_columns,
            fixed_values=fixed,
        )

        return FitWorkspace(
            df=df,
            target_col=target_col,
            design_cols=design_cols,
            residual=residual,
            fixed=solved_fixed,
        )

    def _build_residual_and_design(
        self,
        y: np.ndarray,
        parameter_columns: list[Tuple[str, np.ndarray]],
        *,
        fixed_values: Mapping[str, float],
    ) -> Tuple[np.ndarray, list[Tuple[str, np.ndarray]], Dict[str, float]]:
        """Apply fixed values and return residual + columns to be fit."""
        initial_values = self._initial_params()
        residual = y.copy()
        solved_fixed = dict(fixed_values)
        design_cols: list[Tuple[str, np.ndarray]] = []

        for name, col in parameter_columns:
            if name in solved_fixed:
                residual -= solved_fixed[name] * col
                continue

            # If a parameter has no support in the data (all zeros), you can't fit it
            if not np.any(col != 0):
                solved_fixed[name] = float(initial_values[name])
                raise ValueError(f"Parameter '{name}' has no support in the data; cannot fit")

            design_cols.append((name, col))

        if not design_cols:
            raise ValueError("All parameters are fixed, nothing to fit")

        return residual, design_cols, solved_fixed

    def solve(self, workspace: FitWorkspace) -> Dict[str, float]:
        """ Solve for the optimal parameters given the fit workspace."""
        X = np.column_stack([col for _, col in workspace.design_cols])
        coeffs, *_ = np.linalg.lstsq(X, workspace.residual, rcond=None)

        params = dict(workspace.fixed)
        for (name, _), coef in zip(workspace.design_cols, coeffs):
            params[name] = float(coef)
        return params

    def evaluate(self, model: MinTempPredictor, df: pd.DataFrame, target_col: str) -> PredictionResult:
        """Evaluate the fitted model on the given DataFrame."""
        preds = df.apply(model.predict, axis=1)
        metrics = compute_metrics(df[target_col], preds)
        return PredictionResult(
            method=getattr(model, "name", "model"),
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            errors=metrics.errors,
            mae=metrics.mae,
            rmse=metrics.rmse,
            bias=metrics.bias,
        )

    @abstractmethod
    def _parameter_columns(self, df: pd.DataFrame) -> list[Tuple[str, np.ndarray]]:
        """
        Return least-squares design columns as (param_name, column) pairs.

        - param_name is the coefficient name/key (and the key used in ``fixed``).
        - column is the 1D data vector that coefficient multiplies (aligned to ``df`` rows).
        """
        raise NotImplementedError

    @abstractmethod
    def _initial_params(self) -> Dict[str, float]:
        """Return default values for all parameters used by the model."""
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, params: Mapping[str, float]) -> MinTempPredictor:
        """Build a predictor instance from a solved parameter mapping."""
        raise NotImplementedError


class CraddockAndPritchardOptimizer(CraddockAndPritchardModel, OptimizerFactory):
    """
    Optimizer that derives coefficients and K-table values for the Craddock & Pritchard method.
    """

    def prepare_fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "observed_min_temp_c",
        fixed: Mapping[str, float] | None = None,
    ) -> FitWorkspace:
        fixed_dict = dict(fixed) if fixed is not None else {}
        normalized_fixed = self._normalize_fixed(fixed_dict)
        return super().prepare_fit(df, target_col=target_col, fixed=normalized_fixed)
    
    
    def _normalize_fixed(self, fixed: Dict[str, float]) -> Dict[str, float]:
        # To avoid perfect collinearity between intercept and K indicators, anchor one K cell if neither
        # c nor any K value is fixed.
        if not fixed:
            fixed = {}
        
        if "c" in fixed or any(k.startswith("k:") for k in fixed):
            return fixed

        wind_edges = self.k_table.wind_edges
        cloud_edges = self.k_table.cloud_edges

        for w_edge in wind_edges:
            for c_edge in cloud_edges:
                raw = self.k_table.values.loc[w_edge, c_edge]
                if not pd.isna(raw):
                    anchored = dict(fixed)
                    anchored[f"k:{w_edge}:{c_edge}"] = float(raw)
                    return anchored


    def _parameter_columns(self, df: pd.DataFrame) -> list[Tuple[str, np.ndarray]]:
        """
        Return design columns for fitting a, b, c and K-cell offsets.

        Subtlety:
        For each wind/cloud bin we add a one-hot indicator column (1 if the row
        falls in that bin, else 0) and fit a separate ``k:{wind_edge}:{cloud_edge}``
        coefficient parameter.
        """
        wind_bins, cloud_bins = self._compute_bins(df)

        cols: list[Tuple[str, np.ndarray]] = [
            ("a", df["midday_temp_c"].astype(float).to_numpy()),
            ("b", df["midday_dew_point_c"].astype(float).to_numpy()),
            ("c", np.ones(len(df), dtype=float)),
        ]

        for w_edge in self.k_table.wind_edges:
            for c_edge in self.k_table.cloud_edges:
                name = f"k:{w_edge}:{c_edge}"
                mask = (wind_bins == w_edge) & (cloud_bins == c_edge)
                cols.append((name, mask.astype(float).to_numpy()))

        return cols

    def _initial_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {"a": float(self.a), "b": float(self.b), "c": float(self.c)}
        for w_edge in self.k_table.wind_edges:
            for c_edge in self.k_table.cloud_edges:
                raw = self.k_table.values.loc[w_edge, c_edge]
                params[f"k:{w_edge}:{c_edge}"] = float(raw) if not pd.isna(raw) else float("nan")
        return params

    def _build_model(self, params: Mapping[str, float]) -> CraddockAndPritchardModel:
        a = float(params.get("a", self.a))
        b = float(params.get("b", self.b))
        c = float(params.get("c", self.c))

        new_table = self.k_table.values.copy()
        for w_edge in self.k_table.wind_edges:
            for c_edge in self.k_table.cloud_edges:
                key = f"k:{w_edge}:{c_edge}"
                if key in params:
                    new_table.loc[w_edge, c_edge] = float(params[key])
        return CraddockAndPritchardModel(a=a, b=b, c=c, k_table=KTable(new_table))

    def _compute_bins(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        wind_bins = df["wind_kn"].astype(float).apply(lambda v: self.k_table.bin_edge(v, axis=0))
        cloud_bins = df["cloud_oktas"].astype(float).apply(
            lambda v: self.k_table.bin_edge(v, axis=1)
        )
        return wind_bins, cloud_bins


__all__ = [
    "CraddockAndPritchardOptimizer",
    "FittedModel",
    "FitWorkspace",
    "OptimizerFactory",
]
