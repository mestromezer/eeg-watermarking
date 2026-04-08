"""
lib/wm/lsb.py — LSB (Least Significant Bit) embedder.

Математика:
  Embed:   carrier[i] = (signal[i] & ~mask) | (wm_group << lowest_bit)
  Extract: wm_group   = (carrier[i] >> lowest_bit) & lsb_mask

  где mask     = lsb_mask << lowest_bit
      lsb_mask = (1 << lsb_n) - 1
      wm_group — lsb_n последовательных бит ЦВЗ

Метод НЕ обратим: restored = carrier (оригинал восстановить нельзя).

BPS  = lsb_n / redundancy  (теоретический максимум = lsb_n при redundancy=1)
PSNR ≈ 20·log10(max_val / 2^lsb_n) — не зависит от содержимого ЦВЗ.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView
from lib.wm.embedder import (
    WatermarkEmbedder, InvalidConfig, CantEmbed, CantExtract,
    _wm_preprocess, _wm_postprocess,
)


class _LSBEngine:
    def __init__(
        self,
        *,
        lsb_n: int,
        lowest_bit: int,
        redundancy: int,
        shuffle: bool,
        contiguous: bool,
        allow_partial: bool,
        wm_len: Optional[int],
        key: Optional[str],
    ) -> None:
        if not (1 <= lsb_n <= 8):
            raise InvalidConfig("lsb_n должен быть в [1, 8]")
        if lowest_bit < 0:
            raise InvalidConfig("lowest_bit должен быть >= 0")

        self.lsb_n        = lsb_n
        self.lowest_bit   = lowest_bit
        self.redundancy   = redundancy
        self.shuffle      = shuffle
        self.contiguous   = contiguous
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

    def embed(self, signal: NDArray, watermark: NDArray, carr_range: tuple[int, int]) -> NDArray:
        self.container  = np.array(signal)
        self.watermark  = np.array(watermark)
        self.carr_range = carr_range
        self.carrier    = self.container.copy()

        # Проверяем, что модифицируемые биты лежат в пределах разрядности сигнала.
        # Для float-сигналов проверка не нужна — пользователь сам контролирует lowest_bit.
        if signal.dtype.kind in ("i", "u"):
            safe_bits = np.iinfo(signal.dtype).bits - (1 if signal.dtype.kind == "i" else 0)
            if self.lowest_bit + self.lsb_n > safe_bits:
                raise CantEmbed(
                    f"lowest_bit({self.lowest_bit}) + lsb_n({self.lsb_n}) = "
                    f"{self.lowest_bit + self.lsb_n} превышает разрядность сигнала {safe_bits}"
                )

        coords  = self._get_coords(self.container)
        wm_flat = self._preprocess_wm(self.watermark)   # кратно lsb_n (с padding)

        wm_need = len(wm_flat)
        wm_done = coords_done = 0

        while wm_need > 0:
            wm_chunk     = wm_flat[wm_done:]
            n_coords     = len(wm_chunk) // self.lsb_n   # точное деление — padding гарантирует
            coords_chunk = coords[coords_done : coords_done + n_coords]

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                raise CantEmbed("Insufficient container length")

            done         = self._embed_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done     += done
            wm_need     -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps = self.wm_len / len(self.container) * self.lsb_n

        return self.carrier

    def extract(self, signal: NDArray) -> NDArray:
        self.carrier  = np.array(signal)
        self.restored = self.carrier.copy()   # LSB не обратим

        coords = self._get_coords(self.carrier)

        if self.wm_len is None:
            alloc = len(coords) * self.lsb_n
        else:
            alloc = self.wm_len * self.redundancy
            alloc += (-alloc) % self.lsb_n    # выравниваем до кратного lsb_n

        raw_wm = np.empty(alloc, dtype=np.uint8)

        wm_need = len(raw_wm)
        wm_done = coords_done = 0

        while wm_need > 0:
            wm_chunk     = raw_wm[wm_done:]
            n_coords     = len(wm_chunk) // self.lsb_n
            coords_chunk = coords[coords_done : coords_done + n_coords]

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

    def _get_coords(self, signal: NDArray) -> NDArray:
        coords = np.arange(len(signal))
        if not self.contiguous:
            self._rng().shuffle(coords)
        return coords

    def _embed_chunk(self, wm_bits: NDArray, coords: NDArray) -> int:
        """Записывает lsb_n бит ЦВЗ в каждый сэмпл из coords.

        wm_bits — плоский массив: биты для coords[0], затем coords[1], …
        Возвращает число потреблённых бит.
        """
        n    = len(coords)
        s    = self.container[coords].astype(np.int64)
        wm2d = wm_bits[: n * self.lsb_n].reshape(n, self.lsb_n).astype(np.int64)

        for i in range(self.lsb_n):
            bit_pos = self.lowest_bit + i
            mask    = np.int64(1) << bit_pos
            s       = (s & ~mask) | (wm2d[:, i] << bit_pos)

        self.carrier[coords] = s
        return n * self.lsb_n

    def _extract_chunk(self, wm_bits: NDArray, coords: NDArray) -> int:
        n = len(coords)
        s = self.carrier[coords].astype(np.int64)

        bits2d = np.empty((n, self.lsb_n), dtype=np.uint8)
        for i in range(self.lsb_n):
            bits2d[:, i] = ((s >> (self.lowest_bit + i)) & 1).astype(np.uint8)

        wm_bits[: n * self.lsb_n] = bits2d.ravel()
        return n * self.lsb_n

    def _preprocess_wm(self, wm: NDArray) -> NDArray:
        result = _wm_preprocess(wm, self.redundancy, self.shuffle, self._rng).astype(np.uint8)
        # Дополняем нулями до кратного lsb_n: каждый coord должен получить ровно lsb_n бит.
        pad = (-len(result)) % self.lsb_n
        if pad:
            result = np.pad(result, (0, pad))
        return result

    def _postprocess_wm(self, raw_wm: NDArray) -> NDArray:
        bits = (raw_wm & 1).astype(np.uint8)
        return _wm_postprocess(bits, self.wm_len, self.redundancy, self.shuffle, self._rng)


class LSBEmbedder(WatermarkEmbedder):
    """LSB (Least Significant Bit) алгоритм встраивания / извлечения ЦВЗ.

    Args:
        lsb_n:         Количество бит на сэмпл (1–8). Напрямую задаёт BPS и влияет на PSNR.
        lowest_bit:    Начальная позиция бита (0 = истинный LSB).
        redundancy:    Кратность дублирования ЦВЗ (мажоритарное голосование при извлечении).
        shuffle:       Перемешивать биты ЦВЗ (требует key для воспроизводимости).
        contiguous:    True — последовательные сэмплы; False — случайный порядок (требует key).
        allow_partial: Встраивать частично при нехватке контейнера.
        key:           Ключ ГПСЧ для shuffle / contiguous=False.
        log_level:     Уровень логирования.
        metric_sink:   Объект для записи метрик.
    """

    codename: str = "lsb"

    def __init__(
        self,
        *,
        lsb_n: int = 1,
        lowest_bit: int = 0,
        redundancy: int = 1,
        shuffle: bool = False,
        contiguous: bool = True,
        allow_partial: bool = False,
        key: Optional[str] = None,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._lsb_n         = lsb_n
        self._lowest_bit    = lowest_bit
        self._redundancy    = redundancy
        self._shuffle       = shuffle
        self._contiguous    = contiguous
        self._allow_partial = allow_partial
        self._key           = key

    def algo_params(self) -> dict[str, object]:
        return {
            "lsb_n":      self._lsb_n,
            "lowest_bit": self._lowest_bit,
            "redundancy": self._redundancy,
            "shuffle":    self._shuffle,
            "contiguous": self._contiguous,
        }

    def _make_engine(self, wm_len: Optional[int] = None) -> _LSBEngine:
        return _LSBEngine(
            lsb_n=self._lsb_n,
            lowest_bit=self._lowest_bit,
            redundancy=self._redundancy,
            shuffle=self._shuffle,
            contiguous=self._contiguous,
            allow_partial=self._allow_partial,
            wm_len=wm_len,
            key=self._key,
        )
