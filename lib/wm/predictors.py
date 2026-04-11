"""
lib/wm/predictors.py — pluggable causal predictors for 1-D signals.

All predictors are causal: p_i depends only on samples at indices < i,
so they work correctly inside the sequential embed loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np
from numpy.typing import NDArray

from lib.wm.embedder import InvalidConfig


def _lagrange_coeffs(order: int) -> list[int]:
    """Lagrange extrapolation coefficients for `order` past equally-spaced samples.

    c_j = (-1)^(order-1-j) * C(order, j),  j = 0 … order-1.
    Produces exact integer predictions for integer-valued signals.
    """
    return [(-1) ** (order - 1 - j) * comb(order, j) for j in range(order)]


class _Predictor(ABC):
    @property
    @abstractmethod
    def order(self) -> int: ...

    @abstractmethod
    def predict_one(self, carr: NDArray, i: int) -> int:
        """Scalar prediction at index i (used in the causal embed loop)."""
        ...

    @abstractmethod
    def predict_all(self, c: NDArray) -> NDArray:
        """Vectorised prediction over the full signal (used in extract)."""
        ...


class LeftNeighbourPredictor(_Predictor):
    order = 1

    def predict_one(self, carr: NDArray, i: int) -> int:
        return int(carr[i - 1])

    def predict_all(self, c: NDArray) -> NDArray:
        p = np.zeros(len(c), dtype=np.int64)
        p[1:] = c[:-1]
        return p


class LagrangePredictor(_Predictor):
    def __init__(self, order: int) -> None:
        if order < 2:
            raise InvalidConfig("LagrangePredictor requires order >= 2")
        self._order  = order
        self._coeffs = _lagrange_coeffs(order)

    @property
    def order(self) -> int:
        return self._order

    def predict_one(self, carr: NDArray, i: int) -> int:
        return sum(c * int(carr[i - self._order + j])
                   for j, c in enumerate(self._coeffs))

    def predict_all(self, c: NDArray) -> NDArray:
        n, k = len(c), self._order
        p = np.zeros(n, dtype=np.int64)
        for j, coef in enumerate(self._coeffs):
            p[k:] += coef * c[j : n - k + j]
        return p
