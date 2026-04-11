"""
lib/wm/hs.py — HS (Histogram Shifting) reversible embedder.

Algorithm (prediction-error domain, Ni et al. 2006):
  1. Compute prediction errors e_i = x_i − p_i  (p_i from causal predictor).
  2. h = argmax of error histogram (peak bin, typically 0 for smooth EEG).
  3. z = nearest bin to h with count == 0  (zero bin).
  4. d = sign(z − h).

  Embed one bit b ∈ {0,1} per embeddable sample:
    e == h                         →  e' = h + d*b
    h < e < z  (d=+1)
    z < e < h  (d=-1)              →  e' = e + d   (shift, no bit stored)
    otherwise                      →  e' = e        (outside region, unchanged)

  x'_i = p_i + e'_i

  Extract (carrier read-only — prediction uses watermarked signal):
    e' == h                        →  bit=0, e=h
    e' == h+d                      →  bit=1, e=h
    h+d < e' ≤ z  (d=+1)
    z ≤ e' < h+d  (d=-1)          →  e = e'−d   (undo shift)
    otherwise                      →  e = e'      (unchanged)

Max distortion per sample: |x'_i − x_i| ≤ 1 → PSNR ≥ 96 dB for int16.

References:
  Ni Z. et al., "Reversible Data Hiding," IEEE Trans. Circuits Syst. Video Technol.,
  vol. 16, no. 3, pp. 354–362, Mar. 2006. DOI:10.1109/TCSVT.2006.867964
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.wm.embedder import (
    WatermarkEmbedder, CantEmbed, CantExtract, InvalidConfig,
    EmbedResult, ExtractResult,
    _wm_preprocess, _wm_postprocess,
)
from lib.wm.predictors import _Predictor, LeftNeighbourPredictor
from lib.records.base import ChannelView


def _find_peak_and_zero(errors: NDArray) -> tuple[int, int]:
    """Return (h, z): peak bin and nearest zero bin of the error histogram.

    Searches outward from h on both sides; always succeeds because
    bins beyond [min(e), max(e)] are implicitly zero.
    """
    e_min = int(errors.min())
    e_max = int(errors.max())
    counts = np.bincount(errors - e_min, minlength=e_max - e_min + 1)
    h = int(np.argmax(counts)) + e_min

    for delta in range(1, abs(e_min - h) + abs(e_max - h) + 2):
        for sign in (1, -1):
            z = h + sign * delta
            if z < e_min or z > e_max or counts[z - e_min] == 0:
                return h, z

    raise CantEmbed("No zero bin found in prediction-error histogram")


class _HSEngine:

    def __init__(
        self,
        *,
        redundancy: int,
        shuffle: bool,
        allow_partial: bool,
        wm_len: Optional[int],
        key: Optional[str],
        predictor: _Predictor,
        h: Optional[int],
        z: Optional[int],
    ) -> None:
        self.redundancy    = redundancy
        self.shuffle       = shuffle
        self.allow_partial = allow_partial
        self.wm_len        = wm_len
        self.key           = key
        self.predictor     = predictor
        self.h             = h
        self.z             = z

        self.carrier:  Optional[NDArray] = None
        self.watermark: Optional[NDArray] = None
        self.restored: Optional[NDArray] = None
        self.bps:      Optional[float]   = None

    def _rng(self) -> np.random.Generator:
        key = self.key
        if isinstance(key, str):
            key = list(key.encode())
        return np.random.default_rng(key)

    def embed(
        self,
        signal: NDArray,
        watermark: NDArray,
        carr_range: tuple[int, int],
    ) -> NDArray:
        n      = len(signal)
        cont   = signal.astype(np.int64)
        carr   = cont.copy()
        lo, hi = carr_range
        first  = self.predictor.order

        # Errors computed from original signal — consistent with histogram
        p_orig = self.predictor.predict_all(cont)
        errors = (cont[first:] - p_orig[first:]).astype(np.int64)

        if self.h is None or self.z is None:
            self.h, self.z = _find_peak_and_zero(errors)

        h, z = self.h, self.z
        d    = 1 if z > h else -1

        wm_flat = _wm_preprocess(np.array(watermark), self.redundancy, self.shuffle, self._rng)
        wm_iter = iter(wm_flat.tolist())
        wm_done = 0
        n_cap   = len(wm_flat)

        # Predict from original signal so errors match the histogram exactly.
        # Extract mirrors this by predicting from the causally restored signal.
        for i in range(first, n):
            if wm_done >= n_cap:
                break
            pred = self.predictor.predict_one(cont, i)
            e    = int(cont[i]) - pred

            if e == h:
                bit     = next(wm_iter)
                new_val = pred + h + d * bit
                wm_done += 1
            elif (d == 1 and h < e < z) or (d == -1 and z < e < h):
                new_val = pred + e + d
            else:
                new_val = pred + e

            if not (lo <= new_val <= hi):
                if self.allow_partial:
                    break
                raise CantEmbed(f"Range overflow at i={i}")
            carr[i] = new_val

        if wm_done < n_cap and not self.allow_partial:
            raise CantEmbed(f"Not enough peak-bin samples: embedded {wm_done}/{n_cap}")

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy

        self.watermark = watermark[: self.wm_len]
        self.bps       = self.wm_len / n
        self.carrier   = carr.astype(signal.dtype)
        return self.carrier

    def extract(self, signal: NDArray) -> NDArray:
        """Extract watermark and restore original signal.

        Processes samples in order, predicting from already-restored values.
        This mirrors the embed step (which predicts from the original signal):
        once position i-1 is restored, predict_one(restored, i) == predict_one(orig, i).
        """
        n      = len(signal)
        c      = signal.astype(np.int64)
        first  = self.predictor.order
        h, z   = self.h, self.z
        d      = 1 if z > h else -1

        restored = c.copy()
        bits: list[int] = []

        for i in range(first, n):
            # restored[0..i-1] are already the original values
            pred  = self.predictor.predict_one(restored, i)
            ep    = int(c[i]) - pred

            if ep == h:
                bits.append(0)
                e_orig = h
            elif ep == h + d:
                bits.append(1)
                e_orig = h
            elif (d == 1 and h + d < ep <= z) or (d == -1 and z <= ep < h + d):
                e_orig = ep - d
            else:
                e_orig = ep

            restored[i] = pred + e_orig

            if self.wm_len is not None and len(bits) >= self.wm_len * self.redundancy:
                break

        if self.wm_len is None:
            self.wm_len = len(bits) // self.redundancy

        self.restored = restored.astype(signal.dtype)

        bits_needed = self.wm_len * self.redundancy
        raw_bits    = np.array(bits[:bits_needed], dtype=np.uint8)
        return _wm_postprocess(raw_bits, self.wm_len, self.redundancy, self.shuffle, self._rng)


class HSEmbedder(WatermarkEmbedder):
    """Histogram-Shifting reversible embedder in the prediction-error domain.

    Max distortion per sample is 1 (|x'_i − x_i| ≤ 1), PSNR ≥ 96 dB for int16.

    Args:
        redundancy:    Repetition factor for the watermark.
        shuffle:       Shuffle watermark bits before embedding.
        allow_partial: Embed as many bits as possible if capacity is exceeded.
        key:           PRNG key for shuffle.
        predictor:     Pluggable predictor instance.
        log_level:     Logging level.
        metric_sink:   Metrics recorder.
    """

    codename: str = "hs"

    def __init__(
        self,
        *,
        redundancy: int = 1,
        shuffle: bool = False,
        allow_partial: bool = False,
        key: Optional[str] = None,
        predictor: _Predictor = None,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        super().__init__(log_level=log_level, metric_sink=metric_sink)
        self._redundancy    = redundancy
        self._shuffle       = shuffle
        self._allow_partial = allow_partial
        self._key           = key
        self._predictor     = predictor if predictor is not None else LeftNeighbourPredictor()
        self._h: Optional[int] = None
        self._z: Optional[int] = None

    def algo_params(self) -> dict[str, object]:
        return {
            "block_len":       1,
            "redundancy":      self._redundancy,
            "shuffle":         self._shuffle,
            "predictor_order": self._predictor.order,
        }

    def _make_engine(
        self,
        wm_len: Optional[int] = None,
        h: Optional[int] = None,
        z: Optional[int] = None,
    ) -> _HSEngine:
        return _HSEngine(
            redundancy=self._redundancy,
            shuffle=self._shuffle,
            allow_partial=self._allow_partial,
            wm_len=wm_len,
            key=self._key,
            predictor=self._predictor,
            h=h,
            z=z,
        )

    def _embed_channel(
        self,
        channel: ChannelView,
        watermark: NDArray,
    ) -> tuple[NDArray, NDArray, float, None]:
        engine  = self._make_engine()
        carrier = engine.embed(channel.signal, watermark, channel.dig_range)
        self._h = engine.h
        self._z = engine.z
        return carrier, engine.watermark, float(engine.bps), None

    def _extract_channel(
        self,
        channel: ChannelView,
        wm_len: int,
    ) -> tuple[NDArray, NDArray]:
        if self._h is None or self._z is None:
            raise CantExtract("embed() must be called before extract() to determine h and z")
        engine    = self._make_engine(wm_len=wm_len, h=self._h, z=self._z)
        extracted = engine.extract(channel.signal)
        return extracted, engine.restored
