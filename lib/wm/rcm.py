"""
lib/wm/rcm.py — RCM (Reversible Contrast Mapping) embedder.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView
from lib.wm.embedder import WatermarkEmbedder


# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------

class InvalidConfig(Exception):
    pass

class CantEmbed(Exception):
    pass

class CantExtract(Exception):
    pass


# ---------------------------------------------------------------------------
# Воспроизводимый RNG
# ---------------------------------------------------------------------------

class _RNG(np.random.Generator):
    def __init__(self, key: Optional[str | bytes | int] = None):
        if isinstance(key, str):
            key = key.encode()
        if isinstance(key, bytes):
            key = list(key)
        super().__init__(np.random.PCG64(key))


# ---------------------------------------------------------------------------
# Внутренний движок
# ---------------------------------------------------------------------------

class _RCMEngine:
    """Низкоуровневый движок: chunk-loop + математика RCM.

    Не является публичным API.
    """

    packed_block_type = np.uint8

    def __init__(
        self,
        *,
        rcm_shift: int,
        rcm_rand_shift: bool,
        rcm_skip: bool,
        block_len: int,
        redundancy: int,
        shuffle: bool,
        contiguous: bool,
        allow_partial: bool,
        wm_len: Optional[int],
        key: Optional[str],
    ) -> None:
        if block_len > 8:
            raise InvalidConfig("block_len > 8 not supported")

        self.rcm_shift      = rcm_shift
        self.rcm_rand_shift = rcm_rand_shift
        self.rcm_skip       = rcm_skip
        self.block_len      = block_len
        self.redundancy     = redundancy
        self.shuffle        = shuffle
        self.contiguous     = contiguous
        self.allow_partial  = allow_partial
        self.wm_len         = wm_len
        self.key            = key

        self.rcm_n   = 2 ** (self.block_len - 1)
        self.rcm_mod = 2 * self.rcm_n + 1
        self.rcm_k1  = (self.rcm_n + 1) / self.rcm_mod
        self.rcm_k2  = self.rcm_n / self.rcm_mod

        self.container:  Optional[NDArray] = None
        self.carrier:    Optional[NDArray] = None
        self.watermark:  Optional[NDArray] = None
        self.restored:   Optional[NDArray] = None
        self.bps:        Optional[float]   = None
        self.carr_range: tuple[int, int]   = (0, 0)

    def _rng(self) -> _RNG:
        return _RNG(self.key)

    # ------------------------------------------------------------------ embed

    def embed(
        self,
        signal: NDArray,
        watermark: NDArray,
        carr_range: tuple[int, int],
    ) -> NDArray:
        self.container  = np.array(signal)
        self.watermark  = np.array(watermark)
        self.carr_range = carr_range
        self.carrier    = self.container.copy()

        coords = self._get_coords(self.container)
        wm     = self._preprocess_wm(self.watermark)

        wm_need     = len(wm)
        wm_done     = 0
        coords_done = 0

        while wm_need > 0:
            wm_chunk     = wm[wm_done:]
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk)]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantEmbed("Insufficient container length")

            done         = self._embed_chunk(wm_chunk, coords_chunk)
            done        *= self.block_len
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps       = self.wm_len / len(self.container) * self.block_len

        return self.carrier

    # ---------------------------------------------------------------- extract

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()

        coords = self._get_coords(self.carrier)

        if self.wm_len is None:
            raw_wm = self._alloc_wm(len(coords) * self.block_len)
        else:
            raw_wm = self._alloc_wm(self.wm_len * self.redundancy)

        wm_need     = len(raw_wm)
        wm_done     = 0
        coords_done = 0

        while wm_need > 0:
            wm_chunk     = raw_wm[wm_done:]
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk)]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantExtract("Could not find watermark with given length")

            done         = self._extract_chunk(wm_chunk, coords_chunk)
            done        *= self.block_len
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy

        return self._postprocess_wm(raw_wm)

    # ---------------------------------------------------- RCM: coords

    def _get_coords(self, carr: NDArray) -> NDArray:
        c1 = np.arange(0, len(carr) - self.rcm_shift, self.rcm_shift + 1)
        if not self.contiguous:
            self._rng().shuffle(c1)

        if self.rcm_rand_shift:
            c2 = c1 + self._rng().integers(1, 1 + self.rcm_shift, len(c1))
        else:
            c2 = c1 + self.rcm_shift

        return np.column_stack((c1, c2))

    # ------------------------------------------------ RCM: embed chunk

    def _embed_chunk(self, wm: NDArray, coords: NDArray) -> int:
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        x1 = self.container[c1].astype(np.int64)
        x2 = self.container[c2].astype(np.int64)
        n  = self.rcm_n
        y1 = (n + 1) * x1 - n * x2
        y2 = (n + 1) * x2 - n * x1

        min2 = self.carr_range[0]
        max2 = self.carr_range[1]
        min1 = min2 + n
        max1 = max2 - n
        embeddable = (min1 <= y1) & (y1 <= max1) & (min2 <= y2) & (y2 <= max2)
        y1e = y1[embeddable]
        y2e = y2[embeddable]
        x1n = x1[~embeddable]
        x2n = x2[~embeddable]

        w = wm[: y1e.size].astype(np.int16) + 1
        w[w > n] -= self.rcm_mod
        self.carrier[c1[embeddable]] = y1e + w
        self.carrier[c2[embeddable]] = y2e

        if x1n.size > 0:
            if not self.rcm_skip:
                raise CantEmbed("range overflow and rcm_skip is off")

            r       = (x1n - x2n) % self.rcm_mod
            v1      = x1n - r
            v1_fits = (min2 <= v1) & (v1 <= max2)
            v2      = v1 + self.rcm_mod
            v2_fits = (min2 <= v2) & (v2 <= max2)
            if not (v1_fits | v2_fits).all():
                raise CantEmbed("range overflow when trying to skip")
            self.carrier[c1[~embeddable]] = np.where(v1_fits, v1, v2)

        return w.size

    # ---------------------------------------------- RCM: extract chunk

    def _extract_chunk(self, wm: NDArray, coords: NDArray) -> int:
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        y1 = self.carrier[c1].astype(np.int64)
        y2 = self.carrier[c2].astype(np.int64)
        w  = (y1 - y2) % self.rcm_mod

        if self.rcm_skip:
            filled = w != 0
        else:
            filled = np.ones_like(w, dtype=bool)

        w  = w[filled]
        y1 = y1[filled]
        y2 = y2[filled]

        wm[: w.size] = (w - 1).astype(self.packed_block_type)

        w[w > self.rcm_n] -= self.rcm_mod
        y1 -= w
        x1 = np.round(self.rcm_k1 * y1 + self.rcm_k2 * y2)
        x2 = np.round(self.rcm_k2 * y1 + self.rcm_k1 * y2)

        self.restored[c1[filled]] = x1
        self.restored[c2[filled]] = x2
        return w.size

    # ---------------------------------------- wm pre/post processing

    def _alloc_wm(self, size: int) -> NDArray:
        packed_size = int(np.ceil(size / self.block_len))
        return np.empty(packed_size, dtype=self.packed_block_type)

    def _preprocess_wm(self, wm: NDArray) -> NDArray:
        if self.redundancy > 1:
            wm = np.repeat(wm, self.redundancy)
        if self.shuffle:
            self._rng().shuffle(wm)
        return _bits_to_packed(wm, bit_depth=self.block_len)

    def _postprocess_wm(self, wm: NDArray) -> NDArray:
        wm = _packed_to_bits(wm, bit_depth=self.block_len)

        wm_len = self.wm_len * self.redundancy
        wm     = wm[:wm_len]

        if self.shuffle:
            perm      = self._rng().permutation(wm_len)
            wm1       = np.empty_like(wm)
            wm1[perm] = wm
            wm        = wm1

        if self.redundancy > 1:
            wm = wm.reshape(-1, self.redundancy)
            c  = np.count_nonzero(wm, axis=1)
            wm = np.where(c + c >= self.redundancy, 1, 0)

        return wm


# ---------------------------------------------------------------------------
# Bit packing helpers
# ---------------------------------------------------------------------------

def _bits_to_packed(bits: NDArray, *, bit_depth: int) -> NDArray:
    dtype = np.dtype(np.uint8)
    bps   = dtype.itemsize * 8
    if bit_depth != bps:
        pad_width = bit_depth - (len(bits) % bit_depth)
        if pad_width != bit_depth:
            bits = np.pad(bits, (0, pad_width))
        bits = bits.reshape(-1, bit_depth)
        pad  = np.zeros((len(bits), bps - bit_depth), dtype=np.uint8)
        bits = np.hstack((bits, pad)).ravel()
    return np.packbits(bits, bitorder=sys.byteorder).view(dtype)


def _packed_to_bits(data: NDArray, *, bit_depth: int) -> NDArray:
    bps  = data.dtype.itemsize * 8
    bits = np.unpackbits(data.view(np.uint8), bitorder=sys.byteorder)
    if bit_depth != bps:
        bits = bits.reshape(-1, bps)[:, :bit_depth].flatten()
    return bits


# ---------------------------------------------------------------------------
# Публичный класс
# ---------------------------------------------------------------------------

class RCMEmbedder(WatermarkEmbedder):
    """RCM-алгоритм встраивания / извлечения ЦВЗ.

    Args:
        rcm_shift:      Фиксированный сдвиг между сэмплами пары.
        rcm_rand_shift: Случайный сдвиг ∈ [1, rcm_shift].
        rcm_skip:       Пропускать невстраиваемые пары.
        block_len:      Бит на блок (1–8).
        redundancy:     Кратность дублирования ЦВЗ.
        shuffle:        Перемешивать биты ЦВЗ.
        contiguous:     Последовательные координаты.
        allow_partial:  Допускать частичное встраивание.
        key:            Ключ (сид) для ГПСЧ.
        log_level:      Уровень логирования.
        metric_sink:    Куда писать метрики.
    """

    codename: str = "rcm"

    def __init__(
        self,
        *,
        rcm_shift: int = 1,
        rcm_rand_shift: bool = False,
        rcm_skip: bool = True,
        block_len: int = 1,
        redundancy: int = 1,
        shuffle: bool = False,
        contiguous: bool = True,
        allow_partial: bool = False,
        key: Optional[str] = None,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._rcm_shift      = rcm_shift
        self._rcm_rand_shift = rcm_rand_shift
        self._rcm_skip       = rcm_skip
        self._block_len      = block_len
        self._redundancy     = redundancy
        self._shuffle        = shuffle
        self._contiguous     = contiguous
        self._allow_partial  = allow_partial
        self._key            = key

    def algo_params(self) -> dict:
        return {
            "rcm_shift":      self._rcm_shift,
            "rcm_rand_shift": self._rcm_rand_shift,
            "rcm_skip":       self._rcm_skip,
            "block_len":      self._block_len,
            "redundancy":     self._redundancy,
            "shuffle":        self._shuffle,
            "contiguous":     self._contiguous,
        }

    def _make_engine(self, wm_len: Optional[int] = None) -> _RCMEngine:
        return _RCMEngine(
            rcm_shift=self._rcm_shift,
            rcm_rand_shift=self._rcm_rand_shift,
            rcm_skip=self._rcm_skip,
            block_len=self._block_len,
            redundancy=self._redundancy,
            shuffle=self._shuffle,
            contiguous=self._contiguous,
            allow_partial=self._allow_partial,
            wm_len=wm_len,
            key=self._key,
        )

    def _embed_channel(
        self,
        channel: ChannelView,
        watermark: NDArray,
    ) -> tuple[NDArray, NDArray, float, None]:
        engine  = self._make_engine()
        carrier = engine.embed(channel.signal, watermark, channel.dig_range)
        return carrier, engine.watermark, float(engine.bps), None

    def _extract_channel(
        self,
        channel: ChannelView,
        wm_len: int,
    ) -> tuple[NDArray, NDArray]:
        engine    = self._make_engine(wm_len=wm_len)
        extracted = engine.extract(channel.signal)
        return extracted, engine.restored