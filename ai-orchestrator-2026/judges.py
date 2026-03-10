"""
judges.py — LLM-based оценка промптов (Evaluation Judges)
===========================================================
Реальная LLM-оценка черновиков промптов вместо захардкоженных метрик.
Использует дешёвую модель (task_type='cheap') с temperature=0.0.
После одобрения — сохраняет промпт в ChromaDB через rag_manager.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from llm_router import LiteLLMClient
from state_types import ContextState, EvaluationResult

logger = logging.getLogger(__name__)

# Промпт для судьи (компактный, < 300 токенов)
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator with temperature 0.0. "
    "Evaluate prompts precisely and objectively. "
    "Return ONLY valid JSON, no explanations."
)

_JUDGE_EVAL_TEMPLATE = """Evaluate the following prompt draft on a scale 0.0 to 1.0 for these criteria:

ORIGINAL REQUEST: {request}

PROMPT DRAFT:
---
{draft}
---

Criteria:
1) task_fulfillment: how well the draft fulfills the original request (0.0-1.0)
2) rag_accuracy: factual precision and consistency (0.0-1.0)
3) hallucination_penalty: likelihood of hallucination (0.0-1.0, higher = worse)

Return ONLY valid JSON:
{{"task_fulfillment": float, "rag_accuracy": float, "hallucination_penalty": float, "latency_ms": int, "feedback": "brief feedback"}}"""

# Fallback значения при ошибке парсинга
_FALLBACK_SCORES: dict[str, Any] = {
    "task_fulfillment": 0.5,
    "rag_accuracy": 0.5,
    "hallucination_penalty": 0.5,
    "latency_ms": 999,
    "feedback": "Не удалось распарсить ответ судьи.",
}


class EvaluationJudge:
    """
    LLM-based судья для оценки качества промптов.
    Использует дешёвую модель с temperature=0.0 для консистентности.
    После approve — сохраняет промпт в ChromaDB через rag_manager.
    """

    def __init__(
        self,
        llm_router: Optional[LiteLLMClient] = None,
        rag_manager: Any = None,
    ) -> None:
        self.router = llm_router or LiteLLMClient()
        self.rag_manager = rag_manager
        self.approval_threshold = float(
            os.getenv("APPROVAL_THRESHOLD", "0.85")
        )

    async def evaluate(self, state: ContextState) -> EvaluationResult:
        """
        Оценить текущий черновик промпта.

        Args:
            state: Текущее состояние с prompt_chain

        Returns:
            EvaluationResult с оценками и решением approved/rejected
        """
        # Извлекаем последний черновик
        draft_text = ""
        if state.prompt_chain:
            draft_text = state.prompt_chain[-1].get("generated_text", "")

        if not draft_text:
            logger.warning("⚠️  Нет черновика для оценки — возвращаем fallback")
            return EvaluationResult(**_FALLBACK_SCORES, approved=False)

        # Формируем промпт для судьи
        eval_prompt = _JUDGE_EVAL_TEMPLATE.format(
            request=state.request[:500],
            draft=draft_text[:1500],
        )

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt},
        ]

        # Вызов LLM (cheap, temperature=0.0)
        start_time = time.time()
        response = await self.router.generate(
            messages=messages,
            temperature=0.0,
            task_type="cheap",
        )
        eval_latency = int((time.time() - start_time) * 1000)

        # Парсинг JSON-ответа
        scores = self._parse_scores(response["content"])
        scores["latency_ms"] = scores.get("latency_ms", eval_latency)

        # Расчёт финального score
        final_score = self._calculate_final_score(scores)

        # Решение approved/rejected
        approved = final_score >= self.approval_threshold

        result = EvaluationResult(
            task_fulfillment=scores.get("task_fulfillment", 0.5),
            rag_accuracy=scores.get("rag_accuracy", 0.5),
            hallucination_penalty=scores.get("hallucination_penalty", 0.5),
            latency_ms=scores.get("latency_ms", eval_latency),
            approved=approved,
            judge_model=response.get("model", "unknown"),
            feedback=scores.get("feedback", ""),
        )

        logger.info(
            "⚖️  Оценка: fulfillment=%.2f, rag=%.2f, hallucination=%.2f, "
            "final_score=%.2f, approved=%s (threshold=%.2f)",
            result.task_fulfillment,
            result.rag_accuracy,
            result.hallucination_penalty,
            final_score,
            result.approved,
            self.approval_threshold,
        )

        # === Сохранение одобрённого промпта в ChromaDB для DSPy trainset ===
        if approved and self.rag_manager is not None:
            try:
                await self.rag_manager.save_approved_prompt(
                    state=state,
                    draft_text=draft_text,
                    score=final_score,
                )
                logger.info(
                    "📥 Промпт сохранён в ChromaDB для trainset (score=%.2f)",
                    final_score,
                )
            except Exception as e:
                logger.warning("⚠️  Не удалось сохранить промпт в ChromaDB: %s", e)

        return result

    def _parse_scores(self, content: str) -> dict[str, Any]:
        """Парсинг JSON-ответа судьи с fallback при ошибках."""
        try:
            # Пытаемся найти JSON в ответе
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                data = json.loads(json_str)
                # Валидация: все значения должны быть числами 0..1
                for key in ("task_fulfillment", "rag_accuracy", "hallucination_penalty"):
                    if key in data:
                        data[key] = max(0.0, min(1.0, float(data[key])))
                return data
            else:
                logger.warning("⚠️  Ответ судьи не содержит JSON: %s", content[:200])
                return _FALLBACK_SCORES.copy()
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("⚠️  Ошибка парсинга ответа судьи: %s", e)
            return _FALLBACK_SCORES.copy()

    @staticmethod
    def _calculate_final_score(scores: dict[str, Any]) -> float:
        """
        Расчёт итогового score: среднее fulfillment и accuracy
        минус штраф за галлюцинации.
        """
        fulfillment = scores.get("task_fulfillment", 0.5)
        accuracy = scores.get("rag_accuracy", 0.5)
        hallucination = scores.get("hallucination_penalty", 0.5)

        # Формула: (fulfillment + accuracy) / 2 - hallucination * 0.3
        final = (fulfillment + accuracy) / 2 - hallucination * 0.3
        return max(0.0, min(1.0, final))
