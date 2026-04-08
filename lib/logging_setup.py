"""
lib/logging_setup.py — централизованная настройка логирования для дерева wm.*.

Вызывай один раз в точке входа (main.py, run_research.py, …):

    from lib.logging_setup import setup_logging
    setup_logging(logging.DEBUG)
"""

from __future__ import annotations

import logging


class _FileDefault(logging.Filter):
    """Подставляет ``file="-"`` если логгер не обёрнут в LoggerAdapter с extra["file"].

    Нужно для модульных логгеров внутри lib (например wm.records до вызова load()),
    чтобы форматтер с ``%(file)s`` не падал с KeyError.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "file"):
            record.file = "-"  # type: ignore[attr-defined]
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Настроить обработчик и форматтер для всего дерева ``wm.*``.

    Идемпотентна — повторный вызов не дублирует хендлеры.

    Args:
        level: Минимальный уровень логирования (``logging.DEBUG``, ``logging.INFO``, …).
    """
    logger = logging.getLogger("wm")
    logger.setLevel(level)

    if not any(isinstance(f, _FileDefault) for f in logger.filters):
        logger.addFilter(_FileDefault())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  [%(file)s]  %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)