"""
records — абстракция источника биомедицинских данных.

Быстрый старт::

    from records import load_record

    rec = load_record("data/S001R01.edf")
    cv  = rec.channel_view(0)          # ChannelView первого канала
    rec.update_channel(0, new_signal)
    rec.save("data/S001R01_wm.edf")
"""

from pathlib import Path
from .base import BaseRecord, ChannelView
from .edf import EDFRecord

__all__ = ["BaseRecord", "ChannelView", "EDFRecord", "load_record"]

_EXT_MAP: dict[str, type[BaseRecord]] = {
    ".edf": EDFRecord,
}


def load_record(path: str | Path) -> EDFRecord:
    """Загрузить EDF-файл.

    Args:
        path: Путь к файлу.

    Returns:
        Загруженный :class:`EDFRecord`.

    Raises:
        ValueError: Если расширение файла не ``.edf``.
    """
    p = Path(path)
    cls = _EXT_MAP.get(p.suffix.lower())
    if cls is None:
        raise ValueError(
            f"Неизвестный формат файла: {p.suffix!r}. "
            f"Поддерживаются: {list(_EXT_MAP)}"
        )
    rec = cls()
    rec.load(p)
    return rec