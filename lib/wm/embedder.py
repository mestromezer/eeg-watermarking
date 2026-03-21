"""
wm/embedder.py — абстракция алгоритма встраивания / извлечения ЦВЗ.

Публичный API:
  - EmbedResult         — результат встраивания (один канал)
  - ExtractResult       — результат извлечения (один канал)
  - WatermarkEmbedder   — ABC; конкретные алгоритмы наследуются от него
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView
from lib.metrics.models import ChannelMetrics
from lib.utils import signal_psnr, signal_ber


# ---------------------------------------------------------------------------
# Результаты операций
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    """Результат встраивания ЦВЗ в один канал.

    Attributes:
        channel:     Исходный :class:`ChannelView` (до встраивания).
        carrier:     Модифицированный сигнал с ЦВЗ.
        watermark:   Фактически встроенный ЦВЗ (может быть короче при
                     ``allow_partial=True``).
        bps:         Бит на сэмпл.
        elapsed_sec: Время выполнения.
        comp_saving: Степень сжатия ``1 − compressed/original``
                     (``None`` если алгоритм не сжимает).
    """

    channel: ChannelView
    carrier: NDArray
    watermark: NDArray
    bps: float
    elapsed_sec: float
    comp_saving: Optional[float] = None

    @property
    def embed_psnr(self) -> float:
        """PSNR между оригинальным и модифицированным сигналом."""
        return signal_psnr(self.channel.signal, self.carrier)


@dataclass
class ExtractResult:
    """Результат извлечения ЦВЗ из одного канала.

    Attributes:
        channel:     :class:`ChannelView` канала-носителя.
        extracted:   Извлечённый ЦВЗ.
        restored:    Контейнер после обратного преобразования.
        elapsed_sec: Время выполнения.
        orig_wm:     Оригинальный ЦВЗ для BER (``None`` — не считается).
        orig_signal: Оригинальный сигнал для restore PSNR (``None`` — не считается).
    """

    channel: ChannelView
    extracted: NDArray
    restored: NDArray
    elapsed_sec: float
    orig_wm: Optional[NDArray] = None
    orig_signal: Optional[NDArray] = None

    @property
    def ber(self) -> Optional[float]:
        if self.orig_wm is None:
            return None
        return signal_ber(self.orig_wm, self.extracted)

    @property
    def restore_psnr(self) -> Optional[float]:
        if self.orig_signal is None:
            return None
        return signal_psnr(self.orig_signal, self.restored)


# ---------------------------------------------------------------------------
# ABC алгоритма
# ---------------------------------------------------------------------------

class WatermarkEmbedder(ABC):
    """Абстрактный базовый класс алгоритма встраивания / извлечения ЦВЗ.

    Конкретный алгоритм реализует :meth:`_embed_channel` и
    :meth:`_extract_channel`. Логирование, тайминг и сборка метрик
    делаются здесь, один раз.

    Args:
        log_level:   Уровень логирования Python (``logging.DEBUG``, ``logging.INFO``, …).
        metric_sink: Объект с методом ``record(ChannelMetrics)``.
                     ``None`` — метрики не пишутся.

    Example::

        from wm import RCMEmbedder
        from metrics import CSVMetricSink
        import logging

        embedder = RCMEmbedder(
            block_len=4,
            log_level=logging.INFO,
            metric_sink=CSVMetricSink("out/metrics.csv"),
        )
        result = embedder.embed(channel_view, watermark, filename="S001R01.edf")
    """

    codename: str = "unknown"

    def __init__(
        self,
        *,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        self._metric_sink = metric_sink
        self._log = self._build_logger(log_level)

    # ------------------------------------------------------------------ embed

    def embed(
        self,
        channel: ChannelView,
        watermark: NDArray,
        *,
        filename: str = "",
    ) -> EmbedResult:
        """Встроить ЦВЗ в один канал.

        Args:
            channel:   :class:`ChannelView` — канал-контейнер.
            watermark: Битовый массив ЦВЗ (uint8, значения 0/1).
            filename:  Имя файла для метрик (опционально).

        Returns:
            :class:`EmbedResult`.
        """
        self._log.info(
            "embed  файл=%s  канал=[%d] %s  wm_len=%d  сэмплов=%d",
            filename or "—", channel.index, channel.label,
            len(watermark), channel.sample_count,
        )
        t0 = time.perf_counter()
        try:
            carrier, actual_wm, bps, comp_saving = self._embed_channel(channel, watermark)
            elapsed = time.perf_counter() - t0

            result = EmbedResult(
                channel=channel,
                carrier=carrier,
                watermark=actual_wm,
                bps=bps,
                elapsed_sec=elapsed,
                comp_saving=comp_saving,
            )

            self._log.info(
                "embed  готово  bps=%.6f  psnr=%.2f dB  elapsed=%.3f с",
                bps, result.embed_psnr, elapsed,
            )
            self._record_embed(result, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._log.error(
                "embed  ошибка  файл=%s  канал=[%d] %s: %s",
                filename or "—", channel.index, channel.label, exc,
            )
            self._record_error("embed", channel, elapsed, str(exc), filename)
            raise

    # --------------------------------------------------------------- extract

    def extract(
        self,
        channel: ChannelView,
        wm_len: int,
        *,
        filename: str = "",
        orig_wm: Optional[NDArray] = None,
        orig_signal: Optional[NDArray] = None,
    ) -> ExtractResult:
        """Извлечь ЦВЗ из одного канала.

        Args:
            channel:     :class:`ChannelView` канала-носителя.
            wm_len:      Ожидаемая длина ЦВЗ в битах.
            filename:    Имя файла для метрик (опционально).
            orig_wm:     Оригинальный ЦВЗ для BER (опционально).
            orig_signal: Оригинальный сигнал для restore PSNR (опционально).

        Returns:
            :class:`ExtractResult`.
        """
        self._log.info(
            "extract  файл=%s  канал=[%d] %s  wm_len=%d",
            filename or "—", channel.index, channel.label, wm_len,
        )
        t0 = time.perf_counter()
        try:
            extracted, restored = self._extract_channel(channel, wm_len)
            elapsed = time.perf_counter() - t0

            result = ExtractResult(
                channel=channel,
                extracted=extracted,
                restored=restored,
                elapsed_sec=elapsed,
                orig_wm=orig_wm,
                orig_signal=orig_signal,
            )

            self._log.info(
                "extract  готово  elapsed=%.3f с%s%s",
                elapsed,
                f"  ber={result.ber:.4f}"           if result.ber         is not None else "",
                f"  restore_psnr={result.restore_psnr:.2f} dB" if result.restore_psnr is not None else "",
            )
            self._record_extract(result, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._log.error(
                "extract  ошибка  файл=%s  канал=[%d] %s: %s",
                filename or "—", channel.index, channel.label, exc,
            )
            self._record_error("extract", channel, elapsed, str(exc), filename)
            raise

    # ------------------------------------------------------------ abstract

    @abstractmethod
    def _embed_channel(
        self,
        channel: ChannelView,
        watermark: NDArray,
    ) -> tuple[NDArray, NDArray, float, Optional[float]]:
        """Реализация встраивания.

        Returns:
            ``(carrier, actual_wm, bps, comp_saving)``
            Если алгоритм не сжимает — ``comp_saving = None``.
        """
        ...

    @abstractmethod
    def _extract_channel(
        self,
        channel: ChannelView,
        wm_len: int,
    ) -> tuple[NDArray, NDArray]:
        """Реализация извлечения.

        Returns:
            ``(extracted_wm, restored_signal)``
        """
        ...

    @abstractmethod
    def algo_params(self) -> dict:
        """Параметры алгоритма для записи в метрики."""
        ...

    # ---------------------------------------------------------- metrics write

    def _record_embed(self, result: EmbedResult, filename: str) -> None:
        if self._metric_sink is None:
            return
        self._metric_sink.record(ChannelMetrics(
            filename=filename,
            channel_index=result.channel.index,
            channel_label=result.channel.label,
            algo=self.codename,
            operation="embed",
            algo_params=self.algo_params(),
            wm_len=len(result.watermark),
            bps=result.bps,
            elapsed_sec=result.elapsed_sec,
            embed_psnr=result.embed_psnr,
            comp_saving=result.comp_saving,
        ))

    def _record_extract(self, result: ExtractResult, filename: str) -> None:
        if self._metric_sink is None:
            return
        self._metric_sink.record(ChannelMetrics(
            filename=filename,
            channel_index=result.channel.index,
            channel_label=result.channel.label,
            algo=self.codename,
            operation="extract",
            algo_params=self.algo_params(),
            wm_len=len(result.extracted),
            elapsed_sec=result.elapsed_sec,
            ber=result.ber,
            restore_psnr=result.restore_psnr,
        ))

    def _record_error(
        self,
        operation: str,
        channel: ChannelView,
        elapsed: float,
        error: str,
        filename: str,
    ) -> None:
        if self._metric_sink is None:
            return
        self._metric_sink.record(ChannelMetrics(
            filename=filename,
            channel_index=channel.index,
            channel_label=channel.label,
            algo=self.codename,
            operation=operation,
            algo_params=self.algo_params(),
            elapsed_sec=elapsed,
            error=error,
        ))

    # ------------------------------------------------------------ logger

    def _build_logger(self, level: int) -> logging.Logger:
        name = f"wm.{self.codename}"
        logger = logging.getLogger(name)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(h)
        logger.setLevel(level)
        return logger
