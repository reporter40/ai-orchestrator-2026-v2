"""
orchestrator.py — State-machine граф оркестрации
==================================================
Определяет граф: router → generator → evaluator → optimizer → red_team → end
Каждый узел обрабатывает ContextState и передаёт дальше.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from state_types import ContextState

logger = logging.getLogger(__name__)


class OrchestratorGraph:
    """
    State-machine оркестратор.
    Граф: router → generator → evaluator → optimizer → (red_team) → end
    Итеративный цикл: evaluator может вернуть к optimizer если score < threshold.
    """

    def __init__(
        self,
        router_fn,
        generator_fn,
        evaluator_fn,
        optimizer_fn,
        red_team_fn=None,
        save_fn=None,
    ) -> None:
        """
        Args:
            router_fn:    async (state) -> state — классификация запроса
            generator_fn: async (state) -> state — генерация промптов (MoE)
            evaluator_fn: async (state) -> state — оценка (judges)
            optimizer_fn: async (state) -> state — оптимизация (DSPy)
            red_team_fn:  async (state) -> state — red-team (adversary)
            save_fn:      async (state) -> None  — сохранение в DB
        """
        self.router_fn = router_fn
        self.generator_fn = generator_fn
        self.evaluator_fn = evaluator_fn
        self.optimizer_fn = optimizer_fn
        self.red_team_fn = red_team_fn
        self.save_fn = save_fn

        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))

    async def run(self, state: ContextState) -> ContextState:
        """
        Запустить полный цикл оркестрации.

        Args:
            state: Начальное состояние с request

        Returns:
            Финальное состояние с результатами
        """
        state.max_iterations = self.max_iterations

        logger.info("=" * 60)
        logger.info("🚀 Оркестратор запущен: '%s'", state.request[:100])
        logger.info("=" * 60)

        # === Узел 1: Router — классификация запроса ===
        logger.info("\n📡 [1/5] ROUTER — Классификация запроса...")
        state = await self.router_fn(state)
        state.logs.append(f"[Router] Тип запроса: {state.request_type}")

        # === Итеративный цикл ===
        for iteration in range(self.max_iterations):
            state.iteration = iteration
            logger.info(
                "\n🔄 ИТЕРАЦИЯ %d/%d",
                iteration + 1,
                self.max_iterations,
            )

            # === Узел 2: Generator — генерация промптов (MoE) ===
            logger.info("🎨 [2/5] GENERATOR — Генерация промптов...")
            state = await self.generator_fn(state)
            state.logs.append(
                f"[Generator] Итерация {iteration + 1}: "
                f"{len(state.prompt_chain)} черновиков"
            )

            # === Узел 3: Evaluator — оценка ===
            logger.info("⚖️  [3/5] EVALUATOR — Оценка промпта...")
            state = await self.evaluator_fn(state)

            if state.evaluation:
                score = (
                    state.evaluation.task_fulfillment
                    + state.evaluation.rag_accuracy
                ) / 2 - state.evaluation.hallucination_penalty * 0.3
                state.current_score = max(0.0, min(1.0, score))
                state.approved = state.evaluation.approved

            state.logs.append(
                f"[Evaluator] Score: {state.current_score:.2f}, "
                f"Approved: {state.approved}"
            )

            # Сохранение версии после каждой итерации
            if self.save_fn:
                await self.save_fn(state)

            # Если промпт одобрен — переходим к red-team
            if state.approved:
                logger.info(
                    "✅ Промпт одобрен (score=%.2f) на итерации %d",
                    state.current_score,
                    iteration + 1,
                )
                # Do not break here, proceed to Red Team testing.

            # Если не последняя итерация и промпт не одобрен, или Red Team отклонил — оптимизируем
            if not state.approved and iteration < self.max_iterations - 1:
                logger.info("🔧 [4/5] OPTIMIZER — Оптимизация промпта...")
                state = await self.optimizer_fn(state)
                state.logs.append(
                    f"[Optimizer] Оптимизация завершена "
                    f"(итерация {iteration + 1})"
                )

            # === Узел 5: Red Team (только если одобрено или это последняя итерация и мы "одобряем" лучший) ===
            # Red Team запускается, если промпт одобрен Evaluator'ом, или если это последняя итерация
            # и мы принудительно "одобряем" лучший результат для финальной проверки.
            if (state.approved or iteration == self.max_iterations - 1) and self.red_team_fn:
                logger.info("🔴 [5/5] RED TEAM — Стресс-тестирование...")
                state = await self.red_team_fn(state)
                
                # Если Red Team нашел критические уязвимости и у нас есть итерации — возвращаемся к оптимизации
                # Проверка: если в state.memory_kv["routing_target"] == "optimizer", значит надо фиксить
                if state.memory_kv.get("routing_target") == "optimizer" and iteration < self.max_iterations - 1:
                    logger.warning("🔴 Red Team отклонил промпт. Возврат к оптимизации...")
                    state.approved = False # Сбрасываем одобрение, чтобы цикл продолжился с оптимизацией
                    continue # Переходим к следующей итерации для оптимизации
                
                if state.red_team_result:
                    state.logs.append(
                        f"[RedTeam] Уязвимостей: "
                        f"{state.red_team_result.vulnerabilities_found}, "
                        f"Robustness: {state.red_team_result.overall_robustness:.2f}"
                    )
                
                # Если промпт прошел Red Team (или не было критических уязвимостей, требующих оптимизации) — выходим из цикла
                if state.approved: # Если state.approved все еще True после Red Team, значит он прошел.
                    break
        else:
            # Исчерпали все итерации — берём лучший результат
            logger.warning(
                "⚠️  Достигнут лимит итераций (%d). "
                "Используем лучший доступный результат (score=%.2f).",
                self.max_iterations,
                state.current_score,
            )
            state.approved = True  # Принудительно одобряем лучший результат

        # === Финализация ===
        if state.prompt_chain:
            state.final_prompt = state.prompt_chain[-1].get("generated_text", "")

        # Финальное сохранение
        if self.save_fn:
            await self.save_fn(state)

        logger.info("\n" + "=" * 60)
        logger.info("🏁 Оркестрация завершена")
        logger.info("   Итераций: %d", state.iteration + 1)
        logger.info("   Score: %.2f", state.current_score)
        logger.info("   Approved: %s", state.approved)
        logger.info("=" * 60)

        return state
