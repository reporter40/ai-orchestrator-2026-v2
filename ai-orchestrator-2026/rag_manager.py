"""
rag_manager.py — Менеджер контекста RAG (Retrieval-Augmented Generation)
=========================================================================
Оборачивает MCPService для предоставления контекста агентам,
сохранения одобрённых промптов и подготовки trainset для DSPy.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from mcp_servers import MCPService
from state_types import ContextState

logger = logging.getLogger(__name__)

# Маппинг агентов на предпочтительные узлы KB
_AGENT_KB_MAP: dict[str, list[str]] = {
    "creative_agent": ["creative_writing", "music_theory", "psychology"],
    "architect_agent": ["prompt_engineering", "technical_docs"],
    "profiler_agent": ["psychology", "prompt_engineering"],
    "adversary_agent": ["prompt_engineering"],
}


class RAGManager:
    """
    Менеджер контекста: RAG-поиск, сохранение одобрённых промптов,
    подготовка trainset для DSPy из ChromaDB.
    """

    def __init__(self, mcp: Optional[MCPService] = None) -> None:
        self.mcp = mcp or MCPService()
        self._prompts_collection_name = os.getenv(
            "CHROMA_COLLECTION_PROMPTS", "prompt_versions"
        )

    async def get_context(self, agent_id: str, query: str = "") -> str:
        """
        Получить контекст из KB для конкретного агента.

        Args:
            agent_id: Идентификатор агента (creative_agent, architect_agent и т.д.)
            query: Текст запроса пользователя

        Returns:
            Строка с релевантным контекстом из KB
        """
        preferred_nodes = _AGENT_KB_MAP.get(agent_id, [])
        context_parts: list[str] = []

        for node_id in preferred_nodes:
            result = await self.mcp.query(node_id=node_id, query_text=query)
            if result and result != "НЕТ_ДАННЫХ":
                context_parts.append(result)

        # Если ничего не нашли — пробуем по query напрямую
        if not context_parts and query:
            fallback = await self.mcp.query(node_id="", query_text=query)
            if fallback and fallback != "НЕТ_ДАННЫХ":
                context_parts.append(fallback)

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        logger.info(
            "📖 RAG контекст для %s: %d символов (query: '%s')",
            agent_id,
            len(context),
            query[:50],
        )

        return context

    async def save_approved_prompt(
        self,
        state: ContextState,
        draft_text: str,
        score: float,
    ) -> None:
        """
        Сохранить одобрённый промпт в ChromaDB для будущего DSPy trainset.

        Args:
            state: Текущее состояние оркестратора
            draft_text: Текст одобрённого промпта
            score: Финальный score промпта
        """
        doc_id = str(uuid.uuid4())

        metadata: dict[str, Any] = {
            "session_id": state.session_id,
            "score": score,
            "iteration": state.iteration,
            "original_request": state.request[:500],
            "approved": "true",
            "created_at": datetime.utcnow().isoformat(),
        }

        await self.mcp.add_document(
            collection_name=self._prompts_collection_name,
            doc_id=doc_id,
            text=draft_text,
            metadata=metadata,
        )

        logger.info(
            "📥 [RAG] Промпт сохранён в ChromaDB для trainset (id=%s, score=%.2f)",
            doc_id[:8],
            score,
        )

    async def get_trainset_for_dspy(
        self, query: str, n: int = 5
    ) -> list[dict]:
        """
        Получить одобрённые промпты из ChromaDB для DSPy trainset.

        Args:
            query: Текст запроса для семантического поиска
            n: Количество результатов

        Returns:
            Список [{text: str, metadata: dict}, ...]
        """
        trainset_size = int(os.getenv("DSPY_OPTIMIZER_TRAINSET_SIZE", str(n)))
        results = await self.mcp.search_similar_prompts(
            query_text=query,
            n_results=trainset_size,
            min_score=0.85,
        )

        logger.info(
            "📊 [RAG] Trainset для DSPy: %d одобрённых промптов (query: '%s')",
            len(results),
            query[:50],
        )

        return results
