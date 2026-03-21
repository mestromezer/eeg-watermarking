"""
metrics/sink.py — запись метрик в CSV-файл.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .models import ChannelMetrics

_log = logging.getLogger("wm.metrics")


class CSVMetricSink:
    """Дописывает одну строку в CSV на каждый :class:`ChannelMetrics`.

    Заголовок создаётся автоматически при первой записи.
    Если файл уже существует и непустой — заголовок не дублируется.

    Args:
        path: Путь к файлу. Родительские директории создаются автоматически.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(self, metrics: ChannelMetrics) -> None:
        """Записать одну строку метрик."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = metrics.as_flat_dict()
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        try:
            with self._path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            _log.exception("CSVMetricSink: не удалось записать в %s", self._path)
