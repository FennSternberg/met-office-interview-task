from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple
import bisect
import pandas as pd

from .base import FeatureKey, MinTempPredictor


def _validate_strictly_increasing(edges: Tuple[float, ...], name: str) -> None:
    if len(edges) == 0:
        raise ValueError(f"{name} edges must not be empty")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(f"{name} edges must be strictly increasing: {edges}")


@dataclass(frozen=True)
class KTable:
    """
    Unambiguous binned K-lookup table.

    Convention:
    - edges are upper bounds (right-closed bins)
    - bin selection = first edge >= value
    """

    values: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        """Validate edges and sort index/columns."""
        wind_edges = tuple(float(x) for x in self.values.index)
        cloud_edges = tuple(float(x) for x in self.values.columns)

        _validate_strictly_increasing(wind_edges, "wind")
        _validate_strictly_increasing(cloud_edges, "cloud")

        if self.values.index.has_duplicates:
            raise ValueError("Wind edges (index) contain duplicates")
        if self.values.columns.has_duplicates:
            raise ValueError("Cloud edges (columns) contain duplicates")


    @property
    def wind_edges(self) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.values.index)

    @property
    def cloud_edges(self) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.values.columns)

    def bin_edge(self, x: float, axis:int) -> float:
        # right-closed: first edge >= x
        if axis not in (0,1):
            raise ValueError(f"axis must be 0 (wind) or 1 (cloud), got {axis}")
        edges = self.wind_edges if axis ==0 else self.cloud_edges
        i = bisect.bisect_left(edges, x)
        if i == len(edges):
            raise ValueError(f"value {x} exceeds max edge {edges[-1]}")
        return edges[i]

    def lookup(self, wind_kn: float, cloud_oktas: float) -> float:
        w_edge = self.bin_edge(wind_kn, axis=0)
        c_edge = self.bin_edge(cloud_oktas, axis=1)

        k = self.values.loc[w_edge, c_edge]
        if pd.isna(k):
            raise ValueError(f"K is undefined for wind<= {w_edge} and cloud<= {c_edge}")
        return float(k)


@dataclass(frozen=True)
class CraddockAndPritchardModel(MinTempPredictor):
    name: str = "craddock_and_pritchard"
    feature_keys: Tuple[FeatureKey, ...] = (
        FeatureKey("midday_temp_c", "deg C"),
        FeatureKey("midday_dew_point_c", "deg C"),
        FeatureKey("wind_kn", "kn"),
        FeatureKey("cloud_oktas", "oktas"),
    )
    a: float = 0.316
    b: float = 0.548
    c: float = -1.24
    k_table: KTable =  KTable(
            pd.DataFrame(
                data={
                    # Cloud Cover edges (Oktas) : K values
                    2.0: [-2.2, -1.1, -0.6, 1.1],
                    4.0: [-1.7, 0.0, 0.0, 1.7],
                    6.0: [-0.6, 0.6, 0.6, 2.8],
                    8.0: [0.0, 1.1, 1.1, float("nan")],
                },
                index=[12.5, 25.5, 38.5, 51],  # Geostrophic wind speed edges (kn)
            )
        )

    def __post_init__(self) -> None:
        # Avoid sharing mutable DataFrames across instances.
        object.__setattr__(self, "k_table", KTable(self.k_table.values.copy(deep=True)))

    def predict(self, features: Mapping[str, float]) -> float:
        t12 = features["midday_temp_c"]
        td12 = features["midday_dew_point_c"]
        wind_kn = features["wind_kn"]
        cloud_oktas = features["cloud_oktas"]
        return self.a * t12 + self.b * td12 + self.c + self.k_table.lookup(wind_kn, cloud_oktas)


__all__ = ["CraddockAndPritchardModel", "KTable"]
