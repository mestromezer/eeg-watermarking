"""
lib/wm/pee.py — PEE (Prediction Error Expansion) embedder.

Математика (по образцу вложенного pee.py):

  Предсказание: p_i = carrier[i-1]  (левый сосед; i=0 не обрабатывается)

  Embed (block_len бит на сэмпл):
    e    = x_i − p_i
    x'_i = p_i + (e << block_len) + wm_val
    где wm_val ∈ [0, 2^block_len − 1]

  Extract (обратимое):
    e'     = x'_i − p_i
    wm_val = e' & mask           mask = (1 << block_len) − 1
    e      = e' >> block_len     (арифметический сдвиг, восстанавливает e)
    x_i    = p_i + e

  Ёмкость = (n − 1) * block_len бит (все сэмплы кроме i=0 расширяются).
  BPS = block_len * n_embedded / n_total.
  Overflow → CantEmbed (нет histogram shifting).

  ВАЖНО: embed — Python-цикл (каузальная зависимость: carr[i] влияет на p_{i+1}).
          extract — векторизован (carrier при извлечении не изменяется).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView
from lib.wm.embedder import (
    WatermarkEmbedder, CantEmbed, CantExtract, InvalidConfig,
    _wm_preprocess, _wm_postprocess,
)


def _pack_bits(bits: NDArray, block_len: int) -> NDArray:
    """Плоский массив бит → массив block_len-битных целых (LSB первым).

    Дополняется нулями до кратного block_len.
    """
    pad = (-len(bits)) % block_len
    b   = np.pad(bits.astype(np.int64), (0, pad)).reshape(-1, block_len)
    shifts = np.arange(block_len, dtype=np.int64)
    return (b * (1 << shifts)).sum(axis=1)


def _unpack_bits(values: NDArray, block_len: int) -> NDArray:
    """Массив block_len-битных целых → плоский массив бит (LSB первым)."""
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
    ) -> None:
        # С левым соседом как предиктором block_len>1 вызывает геометрическое
        # расхождение carrier: каждая модификация меняет pred следующего сэмпла,
        # ошибка компаундируется с множителем 2^block_len.
        if block_len != 1:
            raise InvalidConfig("PEE с левым-соседом предиктором поддерживает только block_len=1")

        self.block_len    = block_len
        self.redundancy   = redundancy
        self.shuffle      = shuffle
        self.allow_partial = allow_partial
        self.wm_len       = wm_len
        self.key          = key

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

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(
        self,
        signal: NDArray,
        watermark: NDArray,
        carr_range: tuple[int, int],
    ) -> NDArray:
        self.container  = np.array(signal)
        self.watermark  = np.array(watermark)
        self.carr_range = carr_range

        wm_flat   = _wm_preprocess(self.watermark, self.redundancy, self.shuffle, self._rng)
        wm_values = _pack_bits(wm_flat, self.block_len)   # shape: (n_values,)

        n      = len(self.container)
        cont   = self.container.astype(np.int64)
        carr   = cont.copy()
        lo, hi = carr_range
        mask   = (1 << self.block_len) - 1

        wm_done = 0
        n_cap   = len(wm_values)

        # Python-цикл обязателен: pred[i] = carr[i-1] (уже модифицированный)
        for i in range(1, n):
            if wm_done >= n_cap:
                break
            pred    = int(carr[i - 1])
            e       = int(cont[i]) - pred
            new_val = pred + (e << self.block_len) + int(wm_values[wm_done])
            if not (lo <= new_val <= hi):
                if self.allow_partial:
                    break
                raise CantEmbed(
                    f"Range overflow at i={i}: увеличьте dig_range или уменьшите block_len"
                )
            carr[i]  = new_val
            wm_done += 1

        if wm_done < n_cap and not self.allow_partial:
            raise CantEmbed(
                f"Недостаточно сэмплов: встроено {wm_done}/{n_cap} значений"
            )

        # Реальные биты ЦВЗ без padding: min(wm_done * block_len, len(wm_flat))
        real_bits = min(wm_done * self.block_len, len(wm_flat))
        if self.wm_len is None:
            self.wm_len = real_bits // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps       = self.wm_len / n
        self.carrier   = carr.astype(signal.dtype)

        return self.carrier

    # ------------------------------------------------------------------
    # Extract (векторизован: carrier при извлечении не меняется)
    # ------------------------------------------------------------------

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()

        n    = len(self.carrier)
        c    = self.carrier.astype(np.int64)
        mask = np.int64((1 << self.block_len) - 1)

        # Предсказание: p[i] = carrier[i-1]; carrier не меняется → всё векторизуется
        p    = np.empty(n, dtype=np.int64)
        p[0] = 0
        p[1:] = c[:-1]

        # Сколько значений нужно извлечь
        if self.wm_len is None:
            n_values = n - 1
        else:
            bits_needed = self.wm_len * self.redundancy
            n_values    = int(np.ceil(bits_needed / self.block_len))

        if n_values > n - 1:
            if not self.allow_partial:
                raise CantExtract(f"Недостаточно сэмплов для wm_len={self.wm_len}")
            n_values = n - 1

        # Срез: только обработанные сэмплы [1 .. n_values]
        idx     = np.arange(1, 1 + n_values)
        e_prime = c[idx] - p[idx]

        wm_values = (e_prime & mask).astype(np.uint8)
        e_orig    = e_prime >> self.block_len

        self.restored[idx] = (p[idx] + e_orig).astype(signal.dtype)

        if self.wm_len is None:
            self.wm_len = (n_values * self.block_len) // self.redundancy

        bits_needed = self.wm_len * self.redundancy
        raw_bits    = _unpack_bits(wm_values.astype(np.int64), self.block_len)[:bits_needed]

        return _wm_postprocess(
            raw_bits, self.wm_len, self.redundancy, self.shuffle, self._rng
        )


# ---------------------------------------------------------------------------


class PEEEmbedder(WatermarkEmbedder):
    """PEE (Prediction Error Expansion) алгоритм встраивания / извлечения ЦВЗ.

    Обратимый алгоритм без histogram shifting.
    Предиктор: левый сосед (p_i = carrier[i-1]).

    Args:
        block_len:     Бит на сэмпл (1–8). Напрямую задаёт BPS.
                       block_len=1 → 1 бит/сэмпл (классическое DE).
        redundancy:    Кратность дублирования ЦВЗ.
        shuffle:       Перемешать биты ЦВЗ по ключу.
        allow_partial: Встраивать частично при переполнении диапазона.
        key:           Ключ ГПСЧ для shuffle.
        log_level:     Уровень логирования.
        metric_sink:   Объект для записи метрик.
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
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        if block_len != 1:
            raise InvalidConfig("PEE с левым-соседом предиктором поддерживает только block_len=1")
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._block_len     = block_len
        self._redundancy    = redundancy
        self._shuffle       = shuffle
        self._allow_partial = allow_partial
        self._key           = key

    def algo_params(self) -> dict[str, object]:
        return {
            "block_len":  self._block_len,
            "redundancy": self._redundancy,
            "shuffle":    self._shuffle,
        }

    def _make_engine(self, wm_len: Optional[int] = None) -> _PEEEngine:
        return _PEEEngine(
            block_len=self._block_len,
            redundancy=self._redundancy,
            shuffle=self._shuffle,
            allow_partial=self._allow_partial,
            wm_len=wm_len,
            key=self._key,
        )
