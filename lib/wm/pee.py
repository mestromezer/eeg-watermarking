"""
lib/wm/pee.py — PEE (Prediction Error Expansion) embedder.

  Embed:   e = x_i − p_i;  x'_i = p_i + (e << block_len) + wm_val
  Extract: e' = x'_i − p_i;  wm_val = e' & mask;  x_i = p_i + (e' >> block_len)

Predictor is pluggable via _Predictor interface.
Embed requires a Python loop (p_i depends on the already-modified carrier).
Extract is vectorised (carrier is read-only during extraction).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from math import comb
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.wm.embedder import (
    WatermarkEmbedder, CantEmbed, CantExtract, InvalidConfig,
    _wm_preprocess, _wm_postprocess,
)


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


# https://link.springer.com/article/10.1007/s00500-018-3537-7
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


def _pack_bits(bits: NDArray, block_len: int) -> NDArray:
    pad    = (-len(bits)) % block_len
    b      = np.pad(bits.astype(np.int64), (0, pad)).reshape(-1, block_len)
    shifts = np.arange(block_len, dtype=np.int64)
    return (b * (1 << shifts)).sum(axis=1)


def _unpack_bits(values: NDArray, block_len: int) -> NDArray:
    shifts = np.arange(block_len, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()


class _PEEEngine:

    def __init__(
        self,
        *,
        block_len: int,
        redundancy: int,
        shuffle: bool,
        allow_partial: bool,
        wm_len: Optional[int],
        key: Optional[str],
        predictor: _Predictor,
    ) -> None:
        if block_len != 1:
            raise InvalidConfig("PEE поддерживает только block_len=1")

        self.block_len     = block_len
        self.redundancy    = redundancy
        self.shuffle       = shuffle
        self.allow_partial = allow_partial
        self.wm_len        = wm_len
        self.key           = key
        self.predictor     = predictor

        self.container:  Optional[NDArray] = None
        self.carrier:    Optional[NDArray] = None
        self.watermark:  Optional[NDArray] = None
        self.restored:   Optional[NDArray] = None
        self.bps:        Optional[float]   = None
        self.carr_range: tuple[int, int]   = (0, 0)

    def _rng(self) -> np.random.Generator:
        key = self.key
        if isinstance(key, str):
            key = list(key.encode())
        return np.random.default_rng(key)

    def embed(self, signal: NDArray, watermark: NDArray, carr_range: tuple[int, int]) -> NDArray:
        self.container  = np.array(signal)
        self.watermark  = np.array(watermark)
        self.carr_range = carr_range

        wm_flat   = _wm_preprocess(self.watermark, self.redundancy, self.shuffle, self._rng)
        wm_values = _pack_bits(wm_flat, self.block_len)

        n      = len(self.container)
        cont   = self.container.astype(np.int64)
        carr   = cont.copy()
        lo, hi = carr_range
        n_cap  = len(wm_values)
        first  = self.predictor.order

        wm_done = 0
        for i in range(first, n):
            if wm_done >= n_cap:
                break
            pred    = self.predictor.predict_one(carr, i)
            e       = int(cont[i]) - pred
            new_val = pred + (e << self.block_len) + int(wm_values[wm_done])
            if not (lo <= new_val <= hi):
                if self.allow_partial:
                    break
                raise CantEmbed(f"Range overflow at i={i}")
            carr[i]  = new_val
            wm_done += 1

        if wm_done < n_cap and not self.allow_partial:
            raise CantEmbed(f"Недостаточно сэмплов: встроено {wm_done}/{n_cap}")

        real_bits = min(wm_done * self.block_len, len(wm_flat))
        if self.wm_len is None:
            self.wm_len = real_bits // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps       = self.wm_len / n
        self.carrier   = carr.astype(signal.dtype)
        return self.carrier

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()

        n     = len(self.carrier)
        c     = self.carrier.astype(np.int64)
        mask  = np.int64((1 << self.block_len) - 1)
        first = self.predictor.order
        p     = self.predictor.predict_all(c)

        if self.wm_len is None:
            n_values = n - first
        else:
            bits_needed = self.wm_len * self.redundancy
            n_values    = int(np.ceil(bits_needed / self.block_len))

        if n_values > n - first:
            if not self.allow_partial:
                raise CantExtract(f"Недостаточно сэмплов для wm_len={self.wm_len}")
            n_values = n - first

        idx     = np.arange(first, first + n_values)
        e_prime = c[idx] - p[idx]

        wm_values = (e_prime & mask).astype(np.uint8)
        e_orig    = e_prime >> self.block_len

        self.restored[idx] = (p[idx] + e_orig).astype(signal.dtype)

        if self.wm_len is None:
            self.wm_len = (n_values * self.block_len) // self.redundancy

        bits_needed = self.wm_len * self.redundancy
        raw_bits    = _unpack_bits(wm_values.astype(np.int64), self.block_len)[:bits_needed]
        return _wm_postprocess(raw_bits, self.wm_len, self.redundancy, self.shuffle, self._rng)


class PEEEmbedder(WatermarkEmbedder):
    """PEE (Prediction Error Expansion) reversible embedder.

    Args:
        block_len:     Bits per sample; only 1 is supported.
        redundancy:    Repetition factor for the watermark.
        shuffle:       Shuffle watermark bits before embedding.
        allow_partial: Embed as many bits as possible on range overflow.
        key:           PRNG key for shuffle.
        predictor:     Pluggable predictor (LeftNeighbourPredictor or LagrangePredictor).
        log_level:     Logging level.
        metric_sink:   Metrics recorder.
    """

    codename: str = "pee"

    def __init__(
        self,
        *,
        block_len: int = 1,
        redundancy: int = 1,
        shuffle: bool = False,
        allow_partial: bool = False,
        key: Optional[str] = None,
        predictor: _Predictor = None,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        if block_len != 1:
            raise InvalidConfig("PEE поддерживает только block_len=1")
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._block_len     = block_len
        self._redundancy    = redundancy
        self._shuffle       = shuffle
        self._allow_partial = allow_partial
        self._key           = key
        self._predictor     = predictor if predictor is not None else LeftNeighbourPredictor()

    def algo_params(self) -> dict[str, object]:
        return {
            "block_len":       self._block_len,
            "redundancy":      self._redundancy,
            "shuffle":         self._shuffle,
            "predictor_order": self._predictor.order,
        }

    def _make_engine(self, wm_len: Optional[int] = None) -> _PEEEngine:
        return _PEEEngine(
            block_len=self._block_len,
            redundancy=self._redundancy,
            shuffle=self._shuffle,
            allow_partial=self._allow_partial,
            wm_len=wm_len,
            key=self._key,
            predictor=self._predictor,
        )
