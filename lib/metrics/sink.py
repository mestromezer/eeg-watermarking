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
        """Записать одну строку метрик.

        Если строка содержит новые param-колонки, которых нет в заголовке,
        файл перезаписывается с расширенным заголовком (существующие строки
        получают пустое значение для новых колонок).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row    = metrics.as_flat_dict()
        is_new = not self._path.exists() or self._path.stat().st_size == 0

        try:
            if is_new:
                self._write_new(list(row.keys()), [row])
            else:
                with self._path.open("r", newline="", encoding="utf-8") as fh:
                    reader        = csv.DictReader(fh)
                    existing_cols = list(reader.fieldnames or [])
                    existing_rows = list(reader)

                new_cols   = [k for k in row.keys() if k not in existing_cols]
                fieldnames = existing_cols + new_cols

                if new_cols:
                    # Появились новые колонки — перезаписываем файл целиком.
                    self._write_new(fieldnames, existing_rows + [row])
                else:
                    with self._path.open("a", newline="", encoding="utf-8") as fh:
                        csv.DictWriter(
                            fh, fieldnames=fieldnames, restval="", extrasaction="ignore"
                        ).writerow(row)

        except OSError:
            _log.exception("CSVMetricSink: не удалось записать в %s", self._path)

    def _write_new(self, fieldnames: list[str], rows: list[dict]) -> None:
        with self._path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
