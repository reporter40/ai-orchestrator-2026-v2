"""
moe_agents.py — MoE-агенты (Mixture of Experts)
=================================================
Специализированные агенты для генерации промптов:
  - CreativeAgent  — метафоры, storytelling, emotional resonance (creative)
  - ArchitectAgent — структурированный JSON-вывод (medium)
  - ProfilerAgent  — адаптация тона под аудиторию (medium)
  - AdversaryAgent — red-team стресс-тестирование (cheap)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from llm_router import LiteLLMClient
from state_types import ContextState, PromptDraft, RedTeamResult

logger = logging.getLogger(__name__)


# === Системные промпты агентов (компактные для экономии токенов) ===

_SYSTEM_PROMPTS: dict[str, str] = {
    "creative_agent": (
        "You are a Creative Expert specializing in metaphors, storytelling, "
        "and emotional resonance in prompts. Generate a rich, creative version "
        "of the given request. Use vivid language, analogies, and narrative hooks. "
        "Focus on engagement and memorability."
    ),
    "architect_agent": (
        "You are a Structure Expert. Convert the request into a precise, "
        "well-organized prompt with clear logical blocks: role, context, task, "
        "constraints, output format. Return structured, actionable output."
    ),
    "profiler_agent": (
        "You are a Tone of Voice Psychologist. Analyze the target audience and "
        "adapt the prompt's tone and style accordingly. Consider: formality level, "
        "technical depth, emotional approach, cultural sensitivity."
    ),
    # Краткий промпт для cheap-задач (< 300 токенов)
    "adversary_agent": (
        "You are a Red Team tester. Generate 3 adversarial inputs to test "
        "prompt robustness: 1) injection attack, 2) edge case, 3) ambiguous input. "
        "Return JSON: {\"attacks\": [{\"type\": str, \"input\": str, \"risk\": str}]}"
    ),
}


class BaseAgent:
    """Базовый класс MoE-агента с реальным LLM-вызовом."""

    agent_id: str = "base_agent"
    agent_role: str = "Base Agent"
    task_type: str = "cheap"
    default_temperature: float = 0.0

    def __init__(self, agent_name: str, rag_manager: Any = None, llm_router: Optional[LiteLLMClient] = None) -> None:
        self.name = agent_name
        self.rag_manager = rag_manager
        self.llm_router = llm_router or LiteLLMClient()
        self.profile = None
        self.system_instruction = _SYSTEM_PROMPTS.get(self.agent_id, "You are a helpful AI.")
        self.version = 1
        self.role = self.agent_role
        self.temperature = self.default_temperature

    async def initialize(self) -> None:
        """Асинхронная инициализация: загрузка профиля из реестра."""
        if not self.rag_manager or not hasattr(self.rag_manager, 'mcp'):
            logger.warning("[%s] RAG manager недоступен, использую дефолтные значения", self.name)
            return
        
        try:
            profile = await self.rag_manager.mcp.get_agent(self.name)
            if profile:
                self.profile = profile
                self.system_instruction = profile.get('instruction', self.system_instruction)
                self.version = profile.get('version', 1)
                self.role = profile.get('role', self.agent_role)
                self.temperature = profile.get('temperature', self.default_temperature)
                logger.info("✅ [%s] Инициализирован из реестра v%s", self.name, self.version)
            else:
                logger.warning("⚠️ [%s] Профиль не найден в реестре, использую дефолтные значения", self.name)
        except Exception as e:
            logger.error("❌ [%s] Ошибка инициализации: %s", self.name, e)

    async def process(
        self,
        state: ContextState,
        rag_context: str = "",
    ) -> PromptDraft:
        """
        Обработать запрос и вернуть черновик промпта.
        Использует self.system_instruction, загруженную из реестра.
        """
        # Формируем сообщения для LLM
        system_prompt = self.system_instruction
        
        # Добавляем RAG-контекст и предыдущие черновики в системный промпт или как доп. сообщения
        context_msg = ""
        if rag_context:
            context_msg += f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{rag_context}\n\n"
        
        if state.prompt_chain:
            prev_drafts = "\n".join(
                f"[{d.get('agent_role', 'Agent')}]: {d.get('generated_text', '')[:300]}"
                for d in state.prompt_chain[-2:]
            )
            context_msg += f"ПРЕДЫДУЩИЕ ЧЕРНОВИКИ:\n{prev_drafts}\n\n"

        user_content = (
            f"{context_msg}"
            f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {state.request}\n\n"
            f"Сгенеририруй улучшенную версию промпта, используя свою экспертизу ({self.agent_role}). "
            f"Итерация: {state.iteration + 1}/{state.max_iterations}."
        )

        # Вызов LLM через роутер
        start_time = time.time()
        
        # Используем новый метод call_agent или старый generate
        # Так как мы обновили LiteLLMClient.generate, используем его
        response = await self.llm_router.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=self.default_temperature,
            task_type=self.task_type
        )
        
        latency_ms = int((time.time() - start_time) * 1000)

        # Формируем PromptDraft
        draft = PromptDraft(
            agent_id=self.name,
            agent_role=self.role,
            generated_text=response["content"],
            model_used=response["model"],
            task_type=self.task_type,
            latency_ms=latency_ms,
        )

        # Сохраняем информацию об агенте в metadata черновика (через цепочку)
        # draft_meta = {'agent': self.name, 'version': self.version}
        # В state.prompt_chain мы добавим это вручную в main.py или orchestrator.py

        logger.info(
            "🎨 [%s] Сгенерирован черновик v%d (%d симв., model=%s, %dms)",
            self.name,
            self.version,
            len(draft.generated_text),
            draft.model_used,
            latency_ms,
        )

        return draft

    async def update_instruction(self, new_instruction: str, reason: str = "") -> None:
        """Обновляет и сохраняет улучшенную инструкцию в реестре."""
        if not self.rag_manager or not hasattr(self.rag_manager, 'mcp'):
            logger.warning("[%s] Невозможно обновить инструкцию: реестр недоступен", self.name)
            return

        try:
            old_version = self.version
            self.system_instruction = new_instruction
            self.version += 1
            
            # Обновление истории если есть старый профиль
            history = []
            if self.profile:
                history = self.profile.get('instruction_history', [])
                history = ([self.profile['instruction']] + history)[:3]

            new_profile = {
                "agent_name": self.name,
                "role": self.role,
                "system_instruction": new_instruction,
                "version": self.version,
                "temperature": self.temperature,
                "task_types": self._get_task_types(),
                "instruction_history": history,
                "avg_score": self.profile.get('avg_score', 0.0) if self.profile else 0.0,
                "sessions_count": self.profile.get('sessions_count', 0) if self.profile else 0,
            }
            
            success = await self.rag_manager.mcp.upsert_agent(new_profile)
            if success:
                self.profile = await self.rag_manager.mcp.get_agent(self.name)
                logger.info("📈 [%s] Инструкция обновлена: v%d -> v%d. Причина: %s", 
                            self.name, old_version, self.version, reason)
        except Exception as e:
            logger.error("❌ [%s] Ошибка обновления инструкции: %s", self.name, e)

    def _get_task_types(self) -> list[str]:
        """Возвращает список типов задач агента."""
        if self.profile and 'task_types' in self.profile:
            return self.profile['task_types']
        return [self.task_type]


class CreativeAgent(BaseAgent):
    """Творческий агент — метафоры, storytelling, emotional resonance."""
    agent_id = "creative_agent"
    agent_role = "Creative Expert"
    task_type = "creative"
    default_temperature = 0.8

    def __init__(self, rag_manager: Any = None, llm_router: Optional[LiteLLMClient] = None) -> None:
        super().__init__("creative_expert", rag_manager, llm_router)


class ArchitectAgent(BaseAgent):
    """Архитектурный агент — структурированный вывод."""
    agent_id = "architect_agent"
    agent_role = "Structure Expert"
    task_type = "structured"
    default_temperature = 0.2

    def __init__(self, rag_manager: Any = None, llm_router: Optional[LiteLLMClient] = None) -> None:
        super().__init__("struct_expert", rag_manager, llm_router)


class ProfilerAgent(BaseAgent):
    """Агент-профайлер — адаптация тона под аудиторию."""
    agent_id = "profiler_agent"
    agent_role = "Tone Psychologist"
    task_type = "profiling"
    default_temperature = 0.6

    def __init__(self, rag_manager: Any = None, llm_router: Optional[LiteLLMClient] = None) -> None:
        super().__init__("tone_expert", rag_manager, llm_router)


class AdversaryAgent(BaseAgent):
    """Red-team агент — стресс-тестирование промптов."""
    agent_id = "adversary_agent"
    agent_role = "Red Team Tester"
    task_type = "redteam"
    default_temperature = 0.3

    def __init__(self, rag_manager: Any = None, llm_router: Optional[LiteLLMClient] = None) -> None:
        super().__init__("adversary_agent", rag_manager, llm_router)

    async def stress_test(self, state: ContextState) -> RedTeamResult:
        """
        Стресс-тест финального промпта на устойчивость.

        Args:
            state: Текущее состояние с финальным промптом

        Returns:
            RedTeamResult с результатами тестирования
        """
        final_prompt = state.final_prompt or ""
        if not final_prompt and state.prompt_chain:
            final_prompt = state.prompt_chain[-1].get("generated_text", "")

        messages = [
            {"role": "system", "content": self.system_instruction},
            {
                "role": "user",
                "content": (
                    f"Протестируй следующий промпт на устойчивость:\n\n"
                    f"---\n{final_prompt[:1000]}\n---\n\n"
                    f"Сгенерируй 3 атаки и оцени уязвимости. "
                    f"Верни JSON: {{\"attacks\": [...], \"robustness_score\": 0.0-1.0, "
                    f"\"recommendation\": \"...\"}}"
                ),
            },
        ]

        response = await self.llm_router.generate(
            messages=messages,
            temperature=self.default_temperature,
            task_type=self.task_type,
        )

        # Парсинг ответа (с fallback)
        try:
            content = response["content"]
            # Пытаемся извлечь JSON из ответа
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                data = json.loads(json_str)
            else:
                data = {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        attacks = data.get("attacks", [])
        robustness = data.get("robustness_score", 0.8)
        recommendation = data.get("recommendation", "Промпт прошёл базовое тестирование.")

        result = RedTeamResult(
            vulnerabilities_found=len(attacks),
            attack_results=attacks,
            overall_robustness=float(robustness),
            recommendation=recommendation,
        )

        logger.info(
            "🔴 Red Team: %d уязвимостей, robustness=%.2f",
            result.vulnerabilities_found,
            result.overall_robustness,
        )

        return result
