"""
prompt_serializer.py — Сериализация промптов для обмена между агентами
=====================================================================
Преобразует ContextState и PromptDraft в JSON-формат для передачи внешним
агентам и обратно.
"""

from __future__ import annotations

import json
from typing import Any

from state_types import ContextState, PromptDraft, SwarmResponse


class PromptSerializer:
    """
    Класс для трансформации внутренних структур данных в транспортный JSON и обратно.
    """

    @staticmethod
    def to_transport_json(state: ContextState) -> str:
        """
        Сериализовать состояние в JSON для передачи внешнему агенту.
        Включает только необходимые для генерации поля.
        """
        payload = {
            "session_id": state.session_id,
            "request": state.request,
            "rag_context": state.rag_context,
            "iteration": state.iteration,
            "prompt_chain": [
                {
                    "agent": d.get("agent_role", "Unknown"),
                    "text": d.get("generated_text", ""),
                    "metadata": d.get("metadata", {})
                }
                for d in state.prompt_chain[-3:]  # Передаём последние 3 шага для контекста
            ],
            # Добавляем memory_kv если там есть что-то важное
            "context": state.logs[-5:] if state.logs else []
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def from_swarm_response(raw_response: dict[str, Any]) -> SwarmResponse:
        """
        Преобразовать сырой JSON-ответ от внешнего агента в SwarmResponse.
        """
        return SwarmResponse(
            ok=raw_response.get("ok", True),
            destination=raw_response.get("destination", "external"),
            result=raw_response.get("result", raw_response),
            summary=raw_response.get("summary", "External agent execution completed"),
            score=float(raw_response.get("score", 0.0))
        )
