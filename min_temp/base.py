from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Tuple

import pandas as pd


@dataclass(frozen=True)
class FeatureKey:
    """Declares a required input feature and its expected unit."""

    name: str
    unit: str


class MinTempPredictor(ABC):
    """Abstract base for Tmin predictors that operate on a single row/Series."""

    name: str
    feature_keys: Tuple[FeatureKey, ...]

    @abstractmethod
    def predict(self, row: Mapping[str, float] | pd.Series) -> float:
        """Compute Tmin for a single row (mapping or Series)."""


__all__ = ["FeatureKey", "MinTempPredictor"]
