"""
wrapper_translator.py — Санитайзер и транслятор ответов внешних агентов
========================================================================
Использует LLM для очистки и структуруирования ответов от сторонних API,
которые могут возвращать невалидный JSON или лишний текст.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from llm_router import LiteLLMClient
from state_types import SwarmResponse

logger = logging.getLogger(__name__)


class WrapperTranslator:
    """
    Класс для очистки (sanitizing) ответов от внешних агентов.
    """

    def __init__(self, llm_router: Optional[LiteLLMClient] = None) -> None:
        self.llm_router = llm_router or LiteLLMClient()

    async def clean_external_response(
        self, 
        raw_text: str, 
        agent_url: str = "unknown"
    ) -> SwarmResponse:
        """
        Превращает "мусорный" вывод стороннего API в валидный SwarmResponse через LLM.
        """
        logger.info("🧹 [Translator] Очистка ответа от %s", agent_url)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a response sanitizer. Extract the meaningful core from the "
                    "external agent's response. Remove any system logs, boilerplate, or meta-talk. "
                    "Return a JSON object with fields: 'text' (the clean prompt), "
                    "'score' (perceived quality 0.0-1.0), and 'summary' (short description)."
                )
            },
            {"role": "user", "content": f"RAW RESPONSE TO CLEAN:\n{raw_text[:2000]}"}
        ]

        try:
            # Используем 'cheap' модель для быстрой очистки
            response = await self.llm_router.generate(
                messages=messages,
                temperature=0.0,
                task_type="cheap"
            )
            
            content = response["content"]
            
            # Извлекаем JSON
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                data = json.loads(json_str)
            else:
                data = {"text": content, "score": 0.5, "summary": "Extracted as raw text"}

            return SwarmResponse(
                ok=True,
                destination=agent_url,
                result={"text": data.get("text", content), "model": "external-translated"},
                summary=data.get("summary", "Cleaned by WrapperTranslator"),
                score=float(data.get("score", 0.0))
            )

        except Exception as e:
            logger.error("❌ [Translator] Ошибка очистки ответа: %s", e)
            return SwarmResponse(
                ok=True,  # Всё равно помечаем как OK, но с текстом ошибки в результате (или исходным текстом)
                result={"text": raw_text, "model": "raw-fallback"},
                summary=f"Failed to clean: {str(e)}",
                score=0.1
            )
