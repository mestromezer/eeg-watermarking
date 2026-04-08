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
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from lib.records.base import ChannelView, BaseRecord
from lib.metrics.models import ChannelMetrics
from lib.utils import signal_psnr, signal_ber


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


# ABC алгоритма

# Тип аргумента source: record или просто имя файла
_Source = Union[BaseRecord, str]


class WatermarkEmbedder(ABC):
    """Абстрактный базовый класс алгоритма встраивания / извлечения ЦВЗ.

    Конкретный алгоритм реализует :meth:`_embed_channel` и
    :meth:`_extract_channel`. Логирование, тайминг и сборка метрик
    делаются здесь, один раз.

    Args:
        log_level:   Уровень логирования Python.
        metric_sink: Объект с методом ``record(ChannelMetrics)``.
                     ``None`` — метрики не пишутся.

    Example::

        result = embedder.embed(channel_view, watermark, source=rec)
        # или для совместимости:
        result = embedder.embed(channel_view, watermark, source="S001R01.edf")
    """

    codename: str = "unknown"

    def __init__(
        self,
        *,
        log_level: int = logging.WARNING,
        metric_sink=None,
    ) -> None:
        self._metric_sink = metric_sink
        self._log_level   = log_level
        self._base_log    = self._build_base_logger(log_level)

    # helpers

    def _resolve_source(self, source: _Source) -> tuple[str, logging.LoggerAdapter]:
        """Вернуть (filename, logger) из record или строки."""
        if isinstance(source, BaseRecord):
            filename = source.path.name if source.path else ""
            return filename, source.log
        # строка — обратная совместимость
        filename = source
        adapter  = logging.LoggerAdapter(
            self._base_log, {"file": filename or "<unknown>"}
        )
        return filename, adapter

    # embed

    def embed(
        self,
        channel: ChannelView,
        watermark: NDArray,
        *,
        source: _Source = "",
    ) -> EmbedResult:
        """Встроить ЦВЗ в один канал.

        Args:
            channel:   :class:`ChannelView` — канал-контейнер.
            watermark: Битовый массив ЦВЗ (uint8, значения 0/1).
            source:    :class:`~lib.records.base.BaseRecord` **или** имя файла
                       ``str`` (для обратной совместимости).
        """
        filename, log = self._resolve_source(source)

        log.info(
            "[%s] embed  канал=[%d] %s  wm_len=%d  сэмплов=%d",
            self.codename, channel.index, channel.label,
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

            log.info(
                "[%s] embed  готово  канал=[%d] %s  bps=%.6f  psnr=%.2f dB  elapsed=%.3f с",
                self.codename, channel.index, channel.label,
                bps, result.embed_psnr, elapsed,
            )
            self._record_embed(result, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error(
                "[%s] embed  ошибка  канал=[%d] %s: %s",
                self.codename, channel.index, channel.label, exc,
            )
            self._record_error("embed", channel, elapsed, str(exc), filename)
            raise

    # extract

    def extract(
        self,
        channel: ChannelView,
        wm_len: int,
        *,
        source: _Source = "",
        orig_wm: Optional[NDArray] = None,
        orig_signal: Optional[NDArray] = None,
    ) -> ExtractResult:
        """Извлечь ЦВЗ из одного канала.

        Args:
            channel:     :class:`ChannelView` канала-носителя.
            wm_len:      Ожидаемая длина ЦВЗ в битах.
            source:      :class:`~lib.records.base.BaseRecord` **или** имя файла.
            orig_wm:     Оригинальный ЦВЗ для BER (опционально).
            orig_signal: Оригинальный сигнал для restore PSNR (опционально).
        """
        filename, log = self._resolve_source(source)

        log.info(
            "[%s] extract  канал=[%d] %s  wm_len=%d",
            self.codename, channel.index, channel.label, wm_len,
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

            log.info(
                "[%s] extract  готово  канал=[%d] %s  elapsed=%.3f с%s%s",
                self.codename, channel.index, channel.label, elapsed,
                f"  ber={result.ber:.4f}"                          if result.ber         is not None else "",
                f"  restore_psnr={result.restore_psnr:.2f} dB"    if result.restore_psnr is not None else "",
            )
            self._record_extract(result, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error(
                "[%s] extract  ошибка  канал=[%d] %s: %s",
                self.codename, channel.index, channel.label, exc,
            )
            self._record_error("extract", channel, elapsed, str(exc), filename)
            raise

    # abstract

    @abstractmethod
    def _embed_channel(
        self,
        channel: ChannelView,
        watermark: NDArray,
    ) -> tuple[NDArray, NDArray, float, Optional[float]]: ...

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

    # metrics write

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

    # logger

    def _build_base_logger(self, level: int) -> logging.Logger:
        """Базовый логгер без привязки к файлу.

        Хендлер и форматтер настраиваются на уровне ``wm`` через
        :func:`lib.logging_setup.setup_logging` — здесь только уровень.
        """
        logger = logging.getLogger(f"wm.{self.codename}")
        logger.setLevel(level)
        return logger