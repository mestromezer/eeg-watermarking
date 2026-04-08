"""
records/base.py — абстракция источника данных (одного файла).

Публичный API:
  - ChannelView  — срез одного канала: сигнал + метаданные
  - BaseRecord   — ABC для конкретных форматов (сейчас только EDF)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger("wm.records")

@dataclass
class ChannelView:
    """Срез одного канала файла.

    Передаётся алгоритму встраивания вместо голого массива — чтобы
    алгоритм знал допустимый диапазон, метку и частоту дискретизации.

    Attributes:
        index:       0-based индекс канала в файле.
        label:       Метка из заголовка (например «Fp1»).
        signal:      Цифровой сигнал (как правило int16 для EDF).
        dig_min:     Минимальное допустимое цифровое значение.
        dig_max:     Максимальное допустимое цифровое значение.
        sample_freq: Частота дискретизации (Гц).
        unit:        Единица измерения физического сигнала (мкВ и т.д.).
    """

    index: int
    label: str
    signal: NDArray
    dig_min: int
    dig_max: int
    sample_freq: float
    unit: str = ""

    @property
    def dig_range(self) -> tuple[int, int]:
        """Допустимый диапазон значений сигнала ``(dig_min, dig_max)``."""
        return (self.dig_min, self.dig_max)

    @property
    def sample_count(self) -> int:
        return len(self.signal)


class BaseRecord(ABC):
    """Абстракция одного файла с временными рядами.

    После вызова :meth:`load` доступны:

    - ``path``          — :class:`~pathlib.Path` к файлу
    - ``file_type``     — строка-идентификатор формата (``"EDF"``, …)
    - ``signal_count``  — число каналов
    - ``duration``      — длительность в секундах
    - ``signal_labels`` — список меток каналов
    - ``log``           — :class:`logging.LoggerAdapter`, привязанный к файлу;
                          используется внутри record и передаётся в embedder

    Субклассы реализуют:

    - :meth:`_load` — чтение файла
    - :meth:`_save` — запись на диск
    - :meth:`channel_view` — вернуть :class:`ChannelView` для канала ``i``
    - :meth:`update_channel` — записать модифицированный сигнал в канал ``i``
    """

    def __init__(self) -> None:
        self.path: Optional[Path] = None
        self.file_type: str = ""
        self.signal_count: int = 0
        self.duration: float = 0.0
        self.signal_labels: list[str] = []
        # До load() — безликий адаптер; после load() пересоздаётся с именем файла
        self.log: logging.LoggerAdapter = logging.LoggerAdapter(_log, {"file": "<unknown>"})

    # ------------------------------------------------------------------ I/O

    def load(self, path: str | Path) -> None:
        """Загрузить файл с диска."""
        self.path = Path(path)
        # Адаптер с именем файла — используется везде в record и embedder
        self.log = logging.LoggerAdapter(_log, {"file": self.path.name})
        self.log.info("Загрузка")
        self._load(self.path)
        self.log.info(
            "Загружено: %s  каналов=%d  длительность=%.1f с",
            self.file_type, self.signal_count, self.duration,
        )

    def save(self, path: str | Path) -> None:
        """Сохранить (возможно изменённые) сигналы на диск."""
        dest = Path(path)
        self.log.info("Сохранение в %s", dest)
        self._save(dest)
        self.log.info("Сохранено: %s", dest)

    # info

    def all_channels(self) -> list[int]:
        """0-based индексы всех каналов."""
        return list(range(self.signal_count))

    def print_info(self, channels: Optional[list[int]] = None) -> None:
        """Вывести краткую информацию о файле и каналах в stdout."""
        print(
            f"{self.file_type}  |  каналов: {self.signal_count}"
            f"  |  длительность: {self.duration:.1f} с"
        )
        for i in (channels or self.all_channels()):
            cv = self.channel_view(i)
            eff_min = int(cv.signal.min())
            eff_max = int(cv.signal.max())
            print(
                f"  [{i:>2}] {cv.label:<8}  {cv.sample_freq:.0f} Гц"
                f"  {cv.sample_count} сэмпл"
                f"  dig [{cv.dig_min}, {cv.dig_max}]"
                f"  eff [{eff_min}, {eff_max}]"
                f"  {cv.unit}"
            )

    # ------------------------------------------------------------ abstract

    @abstractmethod
    def _load(self, path: Path) -> None:
        """Прочитать файл и заполнить все публичные атрибуты."""
        ...

    @abstractmethod
    def _save(self, path: Path) -> None:
        """Записать файл на диск."""
        ...

    @abstractmethod
    def channel_view(self, index: int) -> ChannelView:
        """Вернуть :class:`ChannelView` для канала ``index``."""
        ...

    @abstractmethod
    def update_channel(self, index: int, signal: NDArray) -> None:
        """Заменить сигнал канала ``index`` на ``signal``."""
        ...
