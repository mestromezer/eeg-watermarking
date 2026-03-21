"""
metrics/models.py — датакласс метрик для одного канала одного запуска.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChannelMetrics:
    """Метрики одного прохода embed / extract для конкретного канала.

    Поля соответствуют финальному списку требований:

    Attributes:
        filename:       Имя файла-контейнера (только basename).
        channel_index:  0-based индекс канала.
        channel_label:  Метка канала из заголовка EDF (например «Fp1»).
        algo:           Кодовое имя алгоритма (``"rcm"``, ``"lsb"``, …).
        operation:      ``"embed"`` или ``"extract"``.
        algo_params:    Параметры алгоритма — разворачиваются в CSV
                        как отдельные колонки с префиксом ``param_``.
        wm_len:         Длина ЦВЗ в битах.
        bps:            Бит на сэмпл (bits per sample).
        elapsed_sec:    Время выполнения операции в секундах.
        embed_psnr:     PSNR между оригинальным и модифицированным
                        сигналом после встраивания (dB). ``None`` при extract.
        restore_psnr:   PSNR между оригинальным и восстановленным
                        сигналом после извлечения (dB). ``None`` при embed.
        ber:            Bit Error Rate при извлечении.
                        ``None`` если оригинальный ЦВЗ неизвестен.
        comp_saving:    Степень сжатия: ``1 − compressed_len / original_len``.
                        ``None`` если алгоритм не использует сжатие.
        error:          Текст исключения, если операция завершилась с ошибкой.
    """

    # --- идентификаторы (всегда заполнены) ---
    filename: str
    channel_index: int
    channel_label: str
    algo: str
    operation: str

    # --- параметры алгоритма ---
    algo_params: dict = field(default_factory=dict)

    # --- основные метрики ---
    wm_len: int = 0
    bps: float = 0.0
    elapsed_sec: float = 0.0

    # --- качество сигнала ---
    embed_psnr: Optional[float] = None
    restore_psnr: Optional[float] = None

    # --- точность ЦВЗ ---
    ber: Optional[float] = None

    # --- сжатие ---
    comp_saving: Optional[float] = None

    # --- служебное ---
    error: str = ""

    # ----------------------------------------------------------------- export

    def as_flat_dict(self) -> dict:
        """Плоский словарь для записи в CSV.

        ``algo_params`` разворачивается в отдельные ключи ``param_<name>``.
        Порядок колонок фиксирован — удобно для pandas.
        """
        row: dict = {
            "filename":      self.filename,
            "channel_index": self.channel_index,
            "channel_label": self.channel_label,
            "algo":          self.algo,
            "operation":     self.operation,
            "wm_len":        self.wm_len,
            "bps":           self.bps,
            "elapsed_sec":   self.elapsed_sec,
            "embed_psnr":    self.embed_psnr,
            "restore_psnr":  self.restore_psnr,
            "ber":           self.ber,
            "comp_saving":   self.comp_saving,
            "error":         self.error,
        }
        for k, v in self.algo_params.items():
            row[f"param_{k}"] = v
        return row
