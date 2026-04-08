"""
records/edf.py — реализация BaseRecord для формата EDF / EDF+.

Зависимости: pyedflib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import BaseRecord, ChannelView


class EDFRecord(BaseRecord):
    """Читает и пишет EDF / EDF+ файлы в цифровом (integer) режиме."""

    def _load(self, path: Path) -> None:
        import pyedflib
        import pyedflib.highlevel

        signals, sig_headers, header = pyedflib.highlevel.read_edf(
            str(path), digital=True
        )
        self._signals: list[NDArray] = list(signals)
        self._sig_headers = sig_headers
        self._header = header

        with pyedflib.EdfReader(str(path)) as r:
            self.duration = float(r.file_duration)
            full_patient: str = r.patient.decode()

        if full_patient:
            self.file_type = "EDF"
            self._header["patient_additional"] = full_patient.rstrip()
        else:
            self.file_type = "EDF+"

        self.signal_count = len(sig_headers)
        self.signal_labels = [h["label"] for h in sig_headers]

        self.log.debug(
            "EDF загружен: %d каналов, длительность %.1f с, метки: %s",
            self.signal_count, self.duration, self.signal_labels,
        )

    def _save(self, path: Path) -> None:
        from pyedflib.highlevel import write_edf

        write_edf(
            str(path),
            self._signals,
            self._sig_headers,
            self._header,
            digital=True,
        )

    def channel_view(self, index: int) -> ChannelView:
        h = self._sig_headers[index]
        return ChannelView(
            index=index,
            label=h["label"],
            signal=self._signals[index].copy(),
            dig_min=int(h["digital_min"]),
            dig_max=int(h["digital_max"]),
            sample_freq=float(h["sample_frequency"]),
            unit=h["dimension"],
        )

    def update_channel(self, index: int, signal: NDArray) -> None:
        self.log.debug("Обновление канала %d (%s)", index, self.signal_labels[index])
        self._signals[index] = signal