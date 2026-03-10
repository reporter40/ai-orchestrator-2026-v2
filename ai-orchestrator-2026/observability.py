"""
observability.py — Телеметрия и трейсинг LLM-вызовов
=====================================================
OpenTelemetry интеграция + декоратор @trace_llm_call для
автоматического трейсинга всех LLM-вызовов.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Попытка инициализации OpenTelemetry (graceful fallback если недоступен)
_tracer = None
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    logger.info("OpenTelemetry не установлен — трейсинг в режиме логирования.")


class TelemetrySystem:
    """
    Система наблюдаемости с поддержкой OpenTelemetry.
    Если OTEL недоступен — логирует метрики через стандартный logging.
    """

    def __init__(self) -> None:
        global _tracer
        self.endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT", "")

        if _OTEL_AVAILABLE:
            try:
                provider = TracerProvider()
                # По умолчанию — вывод в консоль (для отладки)
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)
                _tracer = trace.get_tracer("ai-orchestrator-2026")
                logger.info("✅ TelemetrySystem: OpenTelemetry инициализирован")
            except Exception as e:
                logger.warning(
                    "⚠️  TelemetrySystem: не удалось инициализировать OTEL: %s", e
                )
                _tracer = None
        else:
            logger.info(
                "📊 TelemetrySystem: работаем без OpenTelemetry (только logging)"
            )

    @staticmethod
    def get_tracer():
        """Получить текущий tracer (или None)."""
        return _tracer


def trace_llm_call(func: Callable) -> Callable:
    """
    Декоратор для трейсинга LLM-вызовов.
    Записывает: имя модели, task_type, latency, статус.
    Работает и без OpenTelemetry — просто логирует.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Извлекаем параметры вызова из kwargs
        task_type = kwargs.get("task_type", "unknown")
        temperature = kwargs.get("temperature", 0.0)

        span_name = f"llm_call.{task_type}"
        start_time = time.time()

        if _tracer is not None:
            # С OpenTelemetry — создаём span
            with _tracer.start_as_current_span(span_name) as span:
                span.set_attribute("llm.task_type", task_type)
                span.set_attribute("llm.temperature", temperature)
                try:
                    result = await func(*args, **kwargs)
                    latency_ms = int((time.time() - start_time) * 1000)
                    span.set_attribute("llm.latency_ms", latency_ms)
                    span.set_attribute("llm.model", result.get("model", "unknown"))
                    span.set_attribute("llm.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("llm.status", "error")
                    span.set_attribute("llm.error", str(e))
                    raise
        else:
            # Без OpenTelemetry — просто логируем
            try:
                result = await func(*args, **kwargs)
                latency_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    "📊 TRACE: %s | model=%s | latency=%dms",
                    span_name,
                    result.get("model", "unknown"),
                    latency_ms,
                )
                return result
            except Exception as e:
                logger.error("📊 TRACE ERROR: %s | error=%s", span_name, str(e))
                raise

    return wrapper
