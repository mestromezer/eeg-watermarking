"""
lib/wm/rcm.py — RCM (Reversible Contrast Mapping) embedder.

Математика: пары сэмплов (x1, x2) → (y1, y2) через линейное преобразование,
бит встраивается как смещение y1. Полностью обратимо.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView
from lib.wm.embedder import WatermarkEmbedder


# Исключения

class InvalidConfig(Exception):
    pass

class CantEmbed(Exception):
    pass

class CantExtract(Exception):
    pass


# Воспроизводимый RNG

class _RNG(np.random.Generator):
    def __init__(self, key: Optional[str | bytes | int | list] = None) -> None:
        if isinstance(key, str):
            key = key.encode()
        if isinstance(key, bytes):
            key = list(key)
        super().__init__(np.random.PCG64(key))


# Внутренний движок

class _RCMEngine:
    """Низкоуровневый движок RCM.

    packed_block_type = np.uint8: ЦВЗ хранится как packed uint8,
    где каждый элемент несёт block_len бит.
    Цикл за один проход передаёт весь хвост wm[start:] в embed_chunk —
    как в оригинальном make_wm_chunk при packed_block_type is not None.
    """

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

        self.rcm_n   = 2 ** (block_len - 1)
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

    # embed

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
        # packed: каждый uint8 несёт block_len бит ЦВЗ
        wm_packed = self._preprocess_wm(self.watermark)

        wm_need      = len(wm_packed)
        wm_done      = 0
        coords_done  = 0

        while wm_need > 0:
            # make_wm_chunk при packed_block_type: wm[start:] — весь хвост
            wm_chunk     = wm_packed[wm_done:]
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk)]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantEmbed("Insufficient container length")

            done         = self._embed_chunk(wm_chunk, coords_chunk)
            # done = число packed uint8 (не умножаем на block_len — packed path)
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        # bps: wm_len / n_samples * block_len (packed path)
        self.bps = self.wm_len / len(self.container) * self.block_len

        return self.carrier

    # extract

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()

        coords = self._get_coords(self.carrier)

        # alloc: ceil(wm_len * redundancy / block_len) packed uint8
        if self.wm_len is None:
            alloc_bits = len(coords) * self.block_len
        else:
            alloc_bits = self.wm_len * self.redundancy
        raw_wm = np.empty(
            int(np.ceil(alloc_bits / self.block_len)), dtype=np.uint8
        )

        wm_need      = len(raw_wm)
        wm_done      = 0
        coords_done  = 0

        while wm_need > 0:
            wm_chunk     = raw_wm[wm_done:]
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk)]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantExtract("Could not find watermark with given length")

            done         = self._extract_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy

        return self._postprocess_wm(raw_wm)

    # coords

    def _get_coords(self, carr: NDArray) -> NDArray:
        c1 = np.arange(0, len(carr) - self.rcm_shift, self.rcm_shift + 1)
        if not self.contiguous:
            self._rng().shuffle(c1)

        if self.rcm_rand_shift:
            c2 = c1 + self._rng().integers(1, 1 + self.rcm_shift, len(c1))
        else:
            c2 = c1 + self.rcm_shift

        return np.column_stack((c1, c2))

    # embed chunk

    def _embed_chunk(self, wm: NDArray, coords: NDArray) -> int:
        """
        wm: packed uint8, каждый элемент = block_len бит.
        coords: пары (c1, c2), shape (N, 2).
        Возвращает число обработанных packed-элементов.
        """
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        x1 = self.container[c1].astype(np.int64)
        x2 = self.container[c2].astype(np.int64)
        n  = self.rcm_n

        y1 = (n + 1) * x1 - n * x2
        y2 = (n + 1) * x2 - n * x1

        lo, hi = self.carr_range
        min1   = lo + n
        max1   = hi - n
        embeddable = (min1 <= y1) & (y1 <= max1) & (lo <= y2) & (y2 <= hi)

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
            r        = (x1n - x2n) % self.rcm_mod
            v1       = x1n - r
            v1_fits  = (lo <= v1) & (v1 <= hi)
            v2       = v1 + self.rcm_mod
            v2_fits  = (lo <= v2) & (v2 <= hi)
            if not (v1_fits | v2_fits).all():
                raise CantEmbed("range overflow when trying to skip")
            self.carrier[c1[~embeddable]] = np.where(v1_fits, v1, v2)

        return w.size

    # extract chunk

    def _extract_chunk(self, wm: NDArray, coords: NDArray) -> int:
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        y1 = self.carrier[c1].astype(np.int64)
        y2 = self.carrier[c2].astype(np.int64)
        w  = (y1 - y2) % self.rcm_mod

        filled = (w != 0) if self.rcm_skip else np.ones_like(w, dtype=bool)

        w  = w[filled]
        y1 = y1[filled]
        y2 = y2[filled]

        wm[: w.size] = (w - 1).astype(np.uint8)

        w[w > self.rcm_n] -= self.rcm_mod
        y1 -= w
        x1 = np.round(self.rcm_k1 * y1 + self.rcm_k2 * y2)
        x2 = np.round(self.rcm_k2 * y1 + self.rcm_k1 * y2)

        self.restored[c1[filled]] = x1
        self.restored[c2[filled]] = x2
        return w.size

    # wm pre/post processing

    def _preprocess_wm(self, wm: NDArray) -> NDArray:
        """bits → packed uint8 (block_len бит на элемент), с redundancy и shuffle."""
        if self.redundancy > 1:
            wm = np.repeat(wm, self.redundancy)
        if self.shuffle:
            self._rng().shuffle(wm)
        return _bits_to_packed(wm, bit_depth=self.block_len)

    def _postprocess_wm(self, wm: NDArray) -> NDArray:
        """packed uint8 → bits, с de-shuffle и majority vote."""
        bits    = _packed_to_bits(wm, bit_depth=self.block_len)
        wm_len  = self.wm_len * self.redundancy
        bits    = bits[:wm_len]

        if self.shuffle:
            perm       = self._rng().permutation(wm_len)
            bits1      = np.empty_like(bits)
            bits1[perm] = bits
            bits        = bits1

        if self.redundancy > 1:
            bits = bits.reshape(-1, self.redundancy)
            c    = np.count_nonzero(bits, axis=1)
            bits = np.where(c + c >= self.redundancy, 1, 0).astype(np.uint8)

        return bits


# Bit packing helpers

def _bits_to_packed(bits: NDArray, *, bit_depth: int) -> NDArray:
    bps = 8  # uint8
    if bit_depth != bps:
        pad = bit_depth - (len(bits) % bit_depth)
        if pad != bit_depth:
            bits = np.pad(bits, (0, pad))
        bits = bits.reshape(-1, bit_depth)
        pad2 = np.zeros((len(bits), bps - bit_depth), dtype=np.uint8)
        bits = np.hstack((bits, pad2)).ravel()
    return np.packbits(bits, bitorder=sys.byteorder).view(np.uint8)


def _packed_to_bits(data: NDArray, *, bit_depth: int) -> NDArray:
    bps  = 8
    bits = np.unpackbits(data.view(np.uint8), bitorder=sys.byteorder)
    if bit_depth != bps:
        bits = bits.reshape(-1, bps)[:, :bit_depth].flatten()
    return bits


class RCMEmbedder(WatermarkEmbedder):
    """RCM (Reversible Contrast Mapping) алгоритм встраивания / извлечения ЦВЗ.

    Args:
        rcm_shift:      Фиксированный сдвиг между сэмплами пары.
        rcm_rand_shift: Случайный сдвиг ∈ [1, rcm_shift].
        rcm_skip:       Пропускать невстраиваемые пары.
        block_len:      Бит на блок (1–8).
        redundancy:     Кратность дублирования ЦВЗ.
        shuffle:        Перемешивать биты ЦВЗ.
        contiguous:     Последовательные координаты (иначе случайные).
        allow_partial:  Допускать частичное встраивание.
        key:            Ключ для ГПСЧ.
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

    def algo_params(self) -> dict[str, object]:
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