"""
state_types.py — Контракты данных (Pydantic модели)
====================================================
Определяет структуры состояния для всего пайплайна оркестрации.
Эти модели используются всеми модулями системы.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Решение о маршрутизации запроса."""
    destination: str        # creative | structured | profiling | external
    tasks: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class SwarmResponse(BaseModel):
    """Ответ от внешнего или внутреннего агента в формате Swarm."""
    ok: bool = True
    destination: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    score: float = 0.0
    data: Optional[Any] = None

    @property
    def generated_text(self) -> str:
        """Извлечь основной текст промпта из результата."""
        return self.result.get("text") or self.result.get("content", "")


class PromptDraft(BaseModel):
    """Черновик промпта, сгенерированный одним из MoE-агентов."""

    agent_id: str = ""
    agent_role: str = ""
    generated_text: str = ""
    model_used: str = ""
    task_type: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Результат оценки промпта судьёй (LLM-based evaluation)."""

    task_fulfillment: float = 0.0       # Насколько промпт выполняет задачу (0..1)
    rag_accuracy: float = 0.0           # Фактическая точность (0..1)
    hallucination_penalty: float = 0.0  # Штраф за галлюцинации (0..1, выше = хуже)
    latency_ms: int = 0
    approved: bool = False
    judge_model: str = ""
    feedback: str = ""


class RedTeamResult(BaseModel):
    """Результат red-team тестирования промпта."""

    vulnerabilities_found: int = 0
    attack_results: list[dict[str, Any]] = Field(default_factory=list)
    overall_robustness: float = 1.0  # 0..1 (1 = полностью устойчив)
    recommendation: str = ""


class ContextState(BaseModel):
    """
    Глобальное состояние оркестратора — передаётся между узлами графа.
    Это основной контракт для state-machine.
    """

    # Идентификация сессии
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Входные данные
    request: str = ""                        # Исходный запрос пользователя
    request_type: str = ""                   # Тип запроса (routing decision)
    rag_context: str = ""                    # Контекст из RAG/KB

    # Цепочка промптов (результаты MoE-агентов)
    prompt_chain: list[dict[str, Any]] = Field(default_factory=list)

    # Оценка
    evaluation: Optional[EvaluationResult] = None
    current_score: float = 0.0

    # Оптимизация
    optimized_prompt: str = ""

    # Red-team
    red_team_result: Optional[RedTeamResult] = None

    # Управление итерациями
    iteration: int = 0
    max_iterations: int = 3
    approved: bool = False

    # Финальный результат
    final_prompt: str = ""

    # Метаданные для логирования
    logs: list[str] = Field(default_factory=list)
    
    # Гибкое хранилище для новых полей
    memory_kv: dict[str, Any] = Field(default_factory=dict)
