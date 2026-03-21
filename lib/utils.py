"""
utils.py — общие вспомогательные утилиты.

Содержит:
  - WatermarkGenerator  — генерация случайного ЦВЗ заданной длины
  - signal_psnr         — вычисление PSNR между двумя сигналами
  - signal_ber          — вычисление BER между двумя битовыми массивами
  - signal_range        — динамический диапазон сигнала
  - to_bits / from_bits — конвертация bytes ↔ bit-array
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Watermark generation
# ---------------------------------------------------------------------------

class WatermarkGenerator:
    """Генерирует случайные ЦВЗ воспроизводимым образом.

    Args:
        seed: Сид для генератора. ``None`` — случайный каждый раз.
    """

    def __init__(self, seed: Optional[int | str | bytes] = None) -> None:
        self._seed = _normalise_seed(seed)

    def bits(self, length: int) -> NDArray[np.uint8]:
        """Вернуть ``length`` случайных бит (0/1) в виде uint8-массива."""
        rng = np.random.default_rng(self._seed)
        return rng.integers(0, 2, size=length, dtype=np.uint8)

    def bytes(self, length: int) -> bytes:
        """Вернуть ``length`` случайных байт."""
        rng = np.random.default_rng(self._seed)
        return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()

    @staticmethod
    def from_file(path) -> NDArray[np.uint8]:
        """Загрузить ЦВЗ из файла и конвертировать в биты."""
        from pathlib import Path
        data = Path(path).read_bytes()
        return to_bits(data)


# ---------------------------------------------------------------------------
# Bit conversion helpers
# ---------------------------------------------------------------------------

def to_bits(data: bytes | NDArray | str | int, *, bit_depth: Optional[int] = None) -> NDArray[np.uint8]:
    """Конвертировать bytes / str / int / ndarray → массив бит (0/1, uint8)."""
    bps = 8
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, int):
        bl = max(bit_depth or 0, data.bit_length())
        bps = int(np.ceil(bl / 8)) * 8
        data = data.to_bytes(bps // 8, sys.byteorder, signed=True)
    elif isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            bps = data.dtype.itemsize * 8
            data = data.tobytes()

    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    bits = np.unpackbits(data, bitorder=sys.byteorder)

    if bit_depth is not None and bit_depth != bps:
        assert bit_depth < bps
        bits = bits.reshape(-1, bps)[:, :bit_depth].flatten()
    return bits


def from_bits(bits: NDArray[np.uint8]) -> bytes:
    """Конвертировать массив бит обратно в bytes."""
    padded = np.pad(bits, (0, (8 - len(bits) % 8) % 8))
    return np.packbits(padded, bitorder=sys.byteorder).tobytes()


# ---------------------------------------------------------------------------
# Signal metrics
# ---------------------------------------------------------------------------

def signal_psnr(original: NDArray, modified: NDArray, *, rng: Optional[float] = None) -> float:
    """Peak Signal-to-Noise Ratio между двумя сигналами.

    Args:
        original: Исходный сигнал.
        modified: Модифицированный сигнал.
        rng:      Явный диапазон сигнала. Если None — вычисляется по ``original``.

    Returns:
        PSNR в dB. ``inf`` если сигналы идентичны.
    """
    mse = float(np.square(original.astype(np.int64) - modified.astype(np.int64)).mean())
    if mse == 0.0:
        return float("inf")
    if rng is None:
        rng = signal_range(original)
    return float(10.0 * np.log10(rng ** 2 / mse))


def signal_ber(original: NDArray[np.uint8], extracted: NDArray[np.uint8]) -> float:
    """Bit Error Rate между оригинальным и извлечённым ЦВЗ.

    Args:
        original:  Оригинальный битовый массив.
        extracted: Извлечённый битовый массив.

    Returns:
        BER ∈ [0, 1].
    """
    n = min(len(original), len(extracted))
    return float(np.count_nonzero(original[:n] != extracted[:n]) / n)


def signal_range(signal: NDArray) -> float:
    """Динамический диапазон сигнала (max − min + 1 для целых типов)."""
    rng = float(signal.max() - signal.min())
    if np.issubdtype(signal.dtype, np.integer):
        rng += 1.0
    return rng


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_seed(seed: Optional[int | str | bytes]) -> Optional[int | list]:
    if seed is None:
        return None
    if isinstance(seed, str):
        seed = seed.encode()
    if isinstance(seed, bytes):
        return list(seed)
    return seed
