"""
lib/wm/itb.py — ITB (Integer Transform-Based) embedder.

Математика: сигнал масштабируется x*2 − floor(mean(x)),
биты пишутся в x[1:], x[0] — parity для обратного восстановления.

  Embed:
    x     = signal * 2 − floor(mean(signal))
    x[1:] += wm

  Extract:
    pb     = carrier[0] & 1
    wm     = (carrier[1:] − pb) & 1
    r[1:] −= wm
    r      = floor((n·r + Σr) / (2n))

Ограничения:
  - block_len фиксирован = 1
  - Масштабирование удваивает диапазон — возможен CantEmbed при малом dig_range
"""

from __future__ import annotations

import logging
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
# Внутренний движок
# ---------------------------------------------------------------------------

class _ITBEngine:
    """Низкоуровневый движок ITB.

    packed_block_type = np.uint8, block_len = 1:
    каждый uint8 несёт ровно 1 бит ЦВЗ.

    Цикл аналогичен оригиналу: make_wm_chunk при packed_block_type
    возвращает wm[start:] — весь хвост. make_coords_chunk запрашивает
    need+1 (лишний сэмпл для parity). Оба вызова — за один проход.
    """

    BLOCK_LEN = 1

    def __init__(
        self,
        *,
        redundancy: int,
        shuffle: bool,
        allow_partial: bool,
        wm_len: Optional[int],
        key: Optional[str],
    ) -> None:
        self.redundancy    = redundancy
        self.shuffle       = shuffle
        self.allow_partial = allow_partial
        self.wm_len        = wm_len
        self.key           = key

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

        coords    = self._get_coords(self.container)
        wm_packed = self._preprocess_wm(self.watermark)

        wm_need     = len(wm_packed)
        wm_done     = 0
        coords_done = 0

        while wm_need > 0:
            # make_wm_chunk (packed): весь хвост wm[start:]
            wm_chunk = wm_packed[wm_done:]
            # make_coords_chunk: need+1 (parity-сэмпл)
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk) + 1]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantEmbed("Insufficient container length")

            done         = self._embed_chunk(wm_chunk, coords_chunk)
            # done = packed uint8 count (не умножаем — packed path)
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps = self.wm_len / len(self.container) * self.BLOCK_LEN

        return self.carrier

    # ---------------------------------------------------------------- extract

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()

        coords = self._get_coords(self.carrier)

        if self.wm_len is None:
            alloc = max(0, len(coords) - 1)
        else:
            alloc = self.wm_len * self.redundancy
        raw_wm = np.empty(alloc, dtype=np.uint8)

        wm_need     = len(raw_wm)
        wm_done     = 0
        coords_done = 0

        while wm_need > 0:
            wm_chunk     = raw_wm[wm_done:]
            coords_chunk = coords[coords_done : coords_done + len(wm_chunk) + 1]

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

    # ------------------------------------------------------------ coords

    def _get_coords(self, signal: NDArray) -> NDArray:
        """ITB работает с непрерывными индексами."""
        return np.arange(len(signal))

    # ------------------------------------------------ embed chunk

    def _embed_chunk(self, wm: NDArray, coords: NDArray) -> int:
        """
        wm:     packed uint8 (1 бит/элемент), весь хвост от wm_done.
        coords: len = len(wm) + 1 (coords[0] — parity-сэмпл).
        Возвращает число встроенных packed-элементов.
        """
        chunk_len = min(len(wm), len(coords) - 1)
        x = self.container[coords].astype(np.int64)

        # x * 2 − floor(mean(x))
        x = x * 2 - int(np.floor(np.mean(x)))
        x[1 : chunk_len + 1] += wm[:chunk_len].astype(np.int64)

        lo, hi = self.carr_range
        if int(x.min()) < lo or int(x.max()) > hi:
            raise CantEmbed(
                f"Range overflow after ITB scaling: "
                f"[{int(x.min())}, {int(x.max())}] not in [{lo}, {hi}]"
            )

        self.carrier[coords] = x
        return chunk_len

    # ---------------------------------------- extract chunk

    def _extract_chunk(self, wm: NDArray, coords: NDArray) -> int:
        """
        Извлекает chunk_len бит и восстанавливает исходный сигнал.
        Возвращает число извлечённых packed-элементов.
        """
        chunk_len = min(len(wm), len(coords) - 1)

        s  = self.carrier[coords].astype(np.int64)
        pb = int(s[0]) & 1
        wm[:chunk_len] = (s[1 : chunk_len + 1] - pb) & 1

        r = self.restored[coords].astype(np.int64)
        n = len(r)
        r[1 : chunk_len + 1] -= wm[:chunk_len].astype(np.int64)
        r = np.floor((n * r + int(r.sum())) / (2 * n)).astype(np.int64)

        self.restored[coords] = r
        return chunk_len

    # ---------------------------------------- wm pre/post processing

    def _preprocess_wm(self, wm: NDArray) -> NDArray:
        """bits → packed uint8 (block_len=1 → 1:1), с redundancy и shuffle.

        В оригинале: bits_to_ndarray(wm, dtype=uint8, bit_depth=1)
        При dtype=uint8, bit_depth=1 это эквивалентно wm.astype(uint8) —
        каждый бит хранится в отдельном uint8.
        """
        if self.redundancy > 1:
            wm = np.repeat(wm, self.redundancy)
        if self.shuffle:
            self._rng().shuffle(wm)
        return wm.astype(np.uint8)

    def _postprocess_wm(self, wm: NDArray) -> NDArray:
        """packed uint8 (1 бит/элемент) → bits, с de-shuffle и majority vote.

        В оригинале: to_bits(wm, bit_depth=1) при dtype=uint8 → wm & 1.
        """
        bits   = (wm & 1).astype(np.uint8)
        wm_len = self.wm_len * self.redundancy
        bits   = bits[:wm_len]

        if self.shuffle:
            perm        = self._rng().permutation(wm_len)
            bits1       = np.empty_like(bits)
            bits1[perm] = bits
            bits        = bits1

        if self.redundancy > 1:
            bits = bits.reshape(-1, self.redundancy)
            c    = np.count_nonzero(bits, axis=1)
            bits = np.where(c + c >= self.redundancy, 1, 0).astype(np.uint8)

        return bits


# ---------------------------------------------------------------------------
# Публичный класс
# ---------------------------------------------------------------------------

class ITBEmbedder(WatermarkEmbedder):
    """ITB (Integer Transform-Based) алгоритм встраивания / извлечения ЦВЗ.

    Args:
        redundancy:    Кратность дублирования ЦВЗ.
        shuffle:       Перемешивать биты ЦВЗ.
        allow_partial: Допускать частичное встраивание.
        key:           Ключ для ГПСЧ (shuffle / redundancy).
        log_level:     Уровень логирования.
        metric_sink:   Куда писать метрики.

    Note:
        block_len фиксирован = 1 и не настраивается.
        Масштабирование удваивает диапазон сигнала — при малом dig_range
        возможен CantEmbed.
    """

    codename: str = "itb"

    def __init__(
        self,
        *,
        redundancy: int = 1,
        shuffle: bool = False,
        allow_partial: bool = False,
        key: Optional[str] = None,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._redundancy    = redundancy
        self._shuffle       = shuffle
        self._allow_partial = allow_partial
        self._key           = key

    def algo_params(self) -> dict[str, object]:
        return {
            "block_len":  1,
            "redundancy": self._redundancy,
            "shuffle":    self._shuffle,
        }

    def _make_engine(self, wm_len: Optional[int] = None) -> _ITBEngine:
        return _ITBEngine(
            redundancy=self._redundancy,
            shuffle=self._shuffle,
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