"""
llm_router.py — Умный LLM-роутер с выбором модели по типу задачи
=================================================================
Реализует реальные вызовы через litellm.acompletion() с поддержкой
smart routing: cheap/medium/creative модели выбираются автоматически.
Если API-ключ не установлен — работает в mock-режиме.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import litellm

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    Клиент для LLM-вызовов с интеллектуальным выбором модели.

    Маппинг task_type → модель:
      - 'cheap'    → MODEL_CHEAP    (routing, evaluation, red-team)
      - 'medium'   → MODEL_MEDIUM   (structured output, DSPy optimization)
      - 'creative' → MODEL_CREATIVE (генерация промптов, творчество)
    """

    def __init__(self) -> None:
        # Очищаем proxy переменные, которые могут мешать LLM вызовам
        for proxy_var in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)

        self.api_key: str = os.getenv("LLM_ROUTER_API_KEY", "")
        self.base_url: str = os.getenv("LLM_ROUTER_BASE_URL", "")
        self.is_mock: bool = not bool(self.api_key)

        # Маппинг типов задач на модели из переменных окружения
        self.model_map = {
            "routing": os.getenv("MODEL_ROUTER", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "creative": os.getenv("MODEL_CREATIVE", os.getenv("MODEL_FALLBACK", "openai/gpt-4o")),
            "structured": os.getenv("MODEL_MEDIUM", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "profiling": os.getenv("MODEL_CHEAP", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "evaluation": os.getenv("MODEL_CHEAP", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "optimization": os.getenv("MODEL_OPTIMIZER", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "redteam": os.getenv("MODEL_REDTEAM", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
            "cheap": os.getenv("MODEL_CHEAP", os.getenv("MODEL_FALLBACK", "openai/gpt-4o-mini")),
        }

        if self.is_mock:
            logger.warning(
                "⚠️  LLM_ROUTER_API_KEY не установлен — работа в mock-режиме. "
                "Для реальных вызовов создайте .env файл (см. .env.example)."
            )
        else:
            logger.info(
                "✅ LLM Router инициализирован. Модели: cheap=%s, creative=%s",
                self.model_map["cheap"],
                self.model_map["creative"],
            )
            # Отключаем логирование litellm по умолчанию (слишком verbose)
            litellm.suppress_debug_info = True

    def _select_model(self, task_type: str) -> str:
        """Выбрать модель по типу задачи."""
        model = self.model_map.get(task_type, self.model_map['cheap'])
        return model

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        task_type: str = "cheap",
        response_format: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = 60.0,
    ) -> dict[str, Any]:
        """
        Генерация ответа LLM с выбором модели по task_type.

        Args:
            messages: Список сообщений [{"role": ..., "content": ...}]
            temperature: Температура генерации
            task_type: Тип задачи ('cheap', 'medium', 'creative')
            response_format: Опциональный формат ответа (JSON mode и т.д.)
            timeout: Тайм-аут запроса в секундах

        Returns:
            dict с полями: content, model, task_type, latency_ms
        """
        selected_model = self._select_model(task_type)

        # === Mock-режим (без API-ключа) ===
        if self.is_mock:
            return self._mock_response(messages, selected_model, task_type)

        # === Реальный LLM-вызов через litellm ===
        start_time = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": temperature,
                "api_key": self.api_key,
                "timeout": timeout,
            }
            # base_url передаём только если задан (иначе litellm сам определит)
            if self.base_url:
                kwargs["base_url"] = self.base_url
            # response_format — для JSON mode (если поддерживается)
            if response_format:
                kwargs["response_format"] = response_format

            response = await litellm.acompletion(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.choices[0].message.content or ""
            
            # Считаем примерный расход токенов
            prompt_tokens = len(str(messages).split()) * 1.3
            completion_tokens = len(content.split()) * 1.3
            total_tokens = getattr(response, "usage", {}).get("total_tokens", int(prompt_tokens + completion_tokens))

            logger.info(
                "🤖 [Router] task_type=%s → model=%s | latency=%dms | tokens=%s",
                task_type,
                selected_model,
                latency_ms,
                total_tokens,
            )

            return {
                "content": content,
                "model": selected_model,
                "task_type": task_type,
                "latency_ms": latency_ms,
                "usage": {"total_tokens": total_tokens}
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "❌ Ошибка LLM вызова: model=%s, task_type=%s, error=%s",
                selected_model,
                task_type,
                str(e),
            )
            # Fallback на mock при ошибке, чтобы не ронять весь пайплайн
            logger.warning("⚠️  Используем mock-ответ как fallback")
            return self._mock_response(messages, selected_model, task_type)

    def _mock_response(
        self,
        messages: list[dict[str, str]],
        model: str,
        task_type: str,
    ) -> dict[str, Any]:
        """
        Mock-ответ для тестирования без API-ключа.
        Генерирует правдоподобный ответ на основе последнего сообщения.
        """
        # Извлекаем последнее сообщение пользователя для контекста
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")[:200]
                break

        # Разные mock-ответы в зависимости от task_type
        mock_content_map = {
            "creative": (
                f"[MOCK-CREATIVE] Сгенерированный творческий промпт для запроса: "
                f"'{last_user_msg[:100]}...'\n\n"
                f"Этот промпт использует метафоры, storytelling и эмоциональный резонанс "
                f"для максимального вовлечения аудитории. Структура включает: "
                f"1) Яркий хук, 2) Развитие через аналогии, 3) Призыв к действию."
            ),
            "medium": (
                f'{{"role": "assistant", "task": "{last_user_msg[:80]}", '
                f'"structure": {{"sections": ["intro", "body", "conclusion"]}}, '
                f'"output_format": "structured_json", "confidence": 0.85}}'
            ),
            "cheap": (
                f'{{"task_fulfillment": 0.82, "rag_accuracy": 0.78, '
                f'"hallucination_penalty": 0.15, "latency_ms": 150}}'
            ),
        }

        content = mock_content_map.get(task_type, mock_content_map["cheap"])

        logger.info(
            "🧪 MOCK вызов: model=%s, task_type=%s (API ключ не установлен)",
            model,
            task_type,
        )

        return {
            "content": content,
            "model": f"{model} (mock)",
            "task_type": task_type,
            "latency_ms": 0,
            "usage": {"total_tokens": 0}
        }

    async def call_agent(
        self, 
        agent_name: str, 
        system_prompt: str = "", 
        user_message: str = "", 
        prompt: str = "", 
        task_type: str = "cheap",
        temperature: float = 0.0,
        response_model: Optional[type[BaseModel]] = None
    ) -> Any:
        """
        Метод для вызова агента (враппер над generate).
        Поддерживает передачу task_type для маршрутизации и валидацию через Pydantic.
        """
        from state_types import SwarmResponse
        import json
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if response_model and not system_prompt:
             # Добавляем инструкцию о формате, если есть модель
             messages.append({
                 "role": "system", 
                 "content": f"You are {agent_name}. Return valid JSON matching the schema."
             })

        main_content = prompt or user_message
        if main_content:
            messages.append({"role": "user", "content": main_content})
            
        response = await self.generate(
            messages=messages,
            temperature=temperature,
            task_type=task_type,
            response_format={"type": "json_object"} if response_model else None
        )
        
        content = response["content"]
        data = None
        
        if response_model:
            try:
                # Пытаемся распарсить JSON
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}") + 1]
                    raw_data = json.loads(json_str)
                    data = response_model.model_validate(raw_data)
                else:
                    logger.warning("🤖 [Router] JSON not found in response for %s", agent_name)
            except Exception as e:
                logger.error("🤖 [Router] Error parsing response for %s: %s", agent_name, e)

        # Возвращаем Swarm-подобный объект
        return SwarmResponse(
            ok=True,
            destination=agent_name,
            result={"text": content, "model": response["model"]},
            summary=f"Agent {agent_name} responded using {response['model']}",
            score=0.0,
            data=data  # Добавляем поле для хранения распарсенных данных
        )
