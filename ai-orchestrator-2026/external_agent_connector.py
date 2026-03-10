"""
external_agent_connector.py — HTTP-коннектор для внешних агентов
================================================================
Позволяет вызывать сторонние сервисы (например, GenSpark) по HTTP-протоколу.
Поддерживает сериализацию промптов и нормализацию ответов.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from prompt_serializer import PromptSerializer
from state_types import ContextState, SwarmResponse

logger = logging.getLogger(__name__)


class ExternalAgentConnector:
    """
    Коннектор для взаимодействия с внешними агентами через REST API.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.serializer = PromptSerializer()

    async def call_external(
        self, 
        url: str, 
        state: ContextState, 
        headers: Optional[dict[str, str]] = None
    ) -> SwarmResponse:
        """
        Выполнить POST-запрос к внешнему агенту.
        """
        start_time = time.time()
        
        # Подготовка полезной нагрузки
        payload_json = self.serializer.to_transport_json(state)
        payload = {
            "prompt_data": payload_json,
            "session_id": state.session_id,
            "metadata": {
                "iteration": state.iteration,
                "timestamp": time.time()
            }
        }

        logger.info("🌐 [Connector] Вызов внешнего агента: %s", url)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers or {"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                raw_data = response.json()
                
                # Десериализация ответа
                swarm_resp = self.serializer.from_swarm_response(raw_data)
                
                latency = int((time.time() - start_time) * 1000)
                logger.info(
                    "🌐 [Connector] Ответ получен (status=%d, latency=%dms)",
                    response.status_code,
                    latency
                )
                
                return swarm_resp

        except Exception as e:
            logger.error("❌ [Connector] Ошибка вызова внешнего агента (%s): %s", url, e)
            return SwarmResponse(
                ok=False,
                summary=f"External agent call failed: {str(e)}",
                result={"error": str(e)}
            )
