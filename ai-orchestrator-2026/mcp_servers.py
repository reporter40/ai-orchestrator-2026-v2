"""
mcp_servers.py — Сервис базы знаний (MCP — Model Context Protocol)
====================================================================
Два режима работы:
  1. ChromaDB — персистентное векторное хранилище (семантический поиск)
     - Коллекция knowledge_base: RAG-контекст для агентов
     - Коллекция prompt_versions: одобрённые промпты для DSPy trainset
  2. FileBasedKB — чтение из knowledge_base.json (fallback)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Автоопределение ChromaDB при импорте модуля (один раз)
try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# SentenceTransformerEmbeddingFunction — опциональная (нужен torch >= 2.4)
_ST_EMBEDDING_AVAILABLE = False
try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    # Проверяем что torch реально работает для ST
    import torch  # noqa: F401
    _ST_EMBEDDING_AVAILABLE = True
except (ImportError, Exception):
    pass

# Путь к файлу KB по умолчанию
_KB_FILE = Path(__file__).parent / "knowledge_base.json"

# Дефолтный контент KB (если файл не найден)
_DEFAULT_KB: dict[str, str] = {
    "music_theory": (
        "Fibonacci sequence in music: intervals 1,1,2,3,5,8,13,21 relate to "
        "harmonic ratios. Golden ratio phi=1.618 appears in chord progressions."
    ),
    "psychology": (
        "Psychological profiles of electronic music listeners: high openness "
        "to experience, prefer complexity and novelty."
    ),
    "prompt_engineering": (
        "Best practices: clear role definition, explicit output format, "
        "few-shot examples, chain-of-thought reasoning, temperature calibration."
    ),
}


class MCPService:
    """
    Сервис базы знаний с поддержкой ChromaDB и файлового режимов.

    ChromaDB mode:
      - knowledge_base коллекция: RAG-контекст для агентов
      - prompt_versions коллекция: одобрённые промпты для DSPy trainset
    FileBasedKB mode (fallback):
      - Чтение из knowledge_base.json
    """

    def __init__(self, kb_path: Optional[str] = None) -> None:
        self.kb_path = Path(kb_path) if kb_path else _KB_FILE
        self._kb_data: dict[str, str] = {}
        self._memory_prompts: dict[str, dict] = {}  # fallback storage for prompts

        # ChromaDB компоненты
        self._client: Any = None
        self._kb_collection: Any = None
        self._prompts_collection: Any = None
        self._agents_collection: Any = None  # NEW: Agent Registry
        self._embedding_fn: Any = None
        self.chroma_mode: bool = False

        # Загружаем JSON KB (нужен и для ChromaDB seed, и для fallback)
        self._load_kb_file()

        # Инициализация ChromaDB если доступен
        if CHROMA_AVAILABLE:
            self._init_chromadb()
        else:
            logger.info("📁 [MCP] FileBasedKB mode — ChromaDB не установлен")

    def _load_kb_file(self) -> None:
        """Загрузить базу знаний из JSON-файла."""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    self._kb_data = json.load(f)
                logger.info(
                    "📚 KB загружена из %s (%d записей)",
                    self.kb_path,
                    len(self._kb_data),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("⚠️  Не удалось загрузить KB: %s", e)
                self._kb_data = _DEFAULT_KB.copy()
        else:
            logger.info("📝 KB файл не найден — используем дефолтные данные")
            self._kb_data = _DEFAULT_KB.copy()
            try:
                with open(self.kb_path, "w", encoding="utf-8") as f:
                    json.dump(self._kb_data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.warning("⚠️  Не удалось сохранить KB: %s", e)

    def _init_chromadb(self) -> None:
        """Инициализировать ChromaDB PersistentClient и две коллекции."""
        try:
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            embedding_model = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            kb_collection_name = os.getenv("CHROMA_COLLECTION_KNOWLEDGE", "knowledge_base")
            prompts_collection_name = os.getenv("CHROMA_COLLECTION_PROMPTS", "prompt_versions")
            agents_collection_name = os.getenv("CHROMA_COLLECTION_AGENTS", "agents_registry")

            # Embedding function — пробуем ST, fallback на default
            if _ST_EMBEDDING_AVAILABLE:
                logger.info(
                    "⏳ Инициализация ChromaDB (embedding: %s via SentenceTransformer)... "
                    "При первом запуске модель (~90MB) будет скачана.",
                    embedding_model,
                )
                try:
                    # self._embedding_fn = SentenceTransformerEmbeddingFunction(
                    #     model_name=embedding_model
                    # )
                    self._embedding_fn = None
                except Exception as st_err:
                    logger.warning(
                        "⚠️  SentenceTransformer init error: %s — используем default embeddings",
                        st_err,
                    )
                    self._embedding_fn = None
            else:
                logger.info(
                    "⏳ Инициализация ChromaDB (default embeddings — "
                    "SentenceTransformer недоступен, torch >= 2.4 required)..."
                )
                self._embedding_fn = None

            # PersistentClient — создаём ОДИН раз
            self._client = chromadb.PersistentClient(path=persist_dir)

            # Две коллекции
            coll_kwargs_kb = {"name": kb_collection_name}
            coll_kwargs_pr = {"name": prompts_collection_name}
            coll_kwargs_ag = {"name": agents_collection_name}
            if self._embedding_fn is not None:
                coll_kwargs_kb["embedding_function"] = self._embedding_fn
                coll_kwargs_pr["embedding_function"] = self._embedding_fn
                coll_kwargs_ag["embedding_function"] = self._embedding_fn

            self._kb_collection = self._client.get_or_create_collection(**coll_kwargs_kb)
            self._prompts_collection = self._client.get_or_create_collection(**coll_kwargs_pr)
            self._agents_collection = self._client.get_or_create_collection(**coll_kwargs_ag)

            self.chroma_mode = True
            logger.info(
                "✅ [MCP] ChromaDB mode — коллекции: %s (%d docs), %s (%d docs), %s (%d docs)",
                kb_collection_name,
                self._kb_collection.count(),
                prompts_collection_name,
                self._prompts_collection.count(),
                agents_collection_name,
                self._agents_collection.count(),
            )

        except Exception as e:
            logger.warning(
                "⚠️  ChromaDB init error: %s — откат на FileBasedKB mode", e
            )
            self.chroma_mode = False
            self._client = None
            self._kb_collection = None
            self._prompts_collection = None
            self._agents_collection = None

    async def seed_knowledge_base(self) -> None:
        """
        При первом запуске заполнить ChromaDB из knowledge_base.json
        если коллекция knowledge_base пустая.
        """
        if not self.chroma_mode or self._kb_collection is None:
            return

        if self._kb_collection.count() > 0:
            logger.info("📚 KB уже заполнена (%d docs) — seed пропущен", self._kb_collection.count())
            return

        if not self._kb_data:
            return

        ids = list(self._kb_data.keys())
        documents = list(self._kb_data.values())
        metadatas = [
            {
                "node_id": node_id,
                "topic": node_id,
                "source": "knowledge_base.json",
                "added_at": datetime.utcnow().isoformat(),
            }
            for node_id in ids
        ]

        self._kb_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("🌱 seed_knowledge_base: загружено %d документов в ChromaDB", len(ids))

        # NEW: Сеем реестр агентов
        await self.seed_agents_registry()

    async def get_agent(self, agent_name: str) -> Optional[dict]:
        """Загружает профиль агента из ChromaDB по имени."""
        if not self.chroma_mode or self._agents_collection is None:
            return None
        try:
            res = self._agents_collection.get(ids=[agent_name], include=['documents', 'metadatas'])
            if res and res.get('documents') and res['documents']:
                profile = res['metadatas'][0].copy()
                profile['instruction'] = res['documents'][0]
                # Десериализация JSON полей
                for field in ['task_types', 'instruction_history']:
                    if field in profile and isinstance(profile[field], str):
                        try:
                            profile[field] = json.loads(profile[field])
                        except:
                            pass
                return profile
        except Exception as e:
            logger.warning("⚠️  get_agent error [%s]: %s", agent_name, e)
        return None

    async def upsert_agent(self, profile: dict) -> bool:
        """Создаёт или обновляет запись агента в реестре."""
        if not self.chroma_mode or self._agents_collection is None:
            return False
        try:
            name = profile.get('agent_name') or profile.get('name')
            instruction = profile.get('system_instruction') or profile.get('instruction')
            
            if not name or not instruction:
                logger.warning("⚠️  upsert_agent: name or instruction missing")
                return False
            
            # Приводим к стандарту для хранения
            profile['agent_name'] = name
            profile['system_instruction'] = instruction
            
            # Подготовка метаданных (только плоские типы для Chroma)
            meta = profile.copy()
            meta.pop('system_instruction', None)
            meta['last_updated'] = datetime.utcnow().isoformat()
            if 'created_at' not in meta:
                meta['created_at'] = meta['last_updated']
            
            # Сериализация списков
            for field in ['task_types', 'instruction_history']:
                if field in meta and not isinstance(meta[field], str):
                    meta[field] = json.dumps(meta[field])

            self._agents_collection.upsert(
                ids=[name],
                documents=[instruction],
                metadatas=[meta]
            )
            logger.info("💾 [MCP] Agent upserted: %s v%s", name, meta.get('version', '?'))
            return True
        except Exception as e:
            logger.warning("⚠️  upsert_agent error: %s", e)
            return False

    async def list_agents(self) -> list[dict]:
        """Возвращает список всех агентов из реестра."""
        if not self.chroma_mode or self._agents_collection is None:
            return []
        try:
            res = self._agents_collection.get(include=['metadatas'])
            if res and res.get('metadatas'):
                agents = []
                for m in res['metadatas']:
                    agent = m.copy()
                    if 'task_types' in agent and isinstance(agent['task_types'], str):
                        try:
                            agent['task_types'] = json.loads(agent['task_types'])
                        except:
                            pass
                    agents.append(agent)
                return agents
        except Exception as e:
            logger.warning("⚠️  list_agents error: %s", e)
        return []

    async def update_agent_score(self, agent_name: str, new_score: float) -> None:
        """Обновляет средний балл и счетчик сессий агента."""
        profile = await self.get_agent(agent_name)
        if not profile:
            return
        
        try:
            old_avg = float(profile.get('avg_score', 0.0))
            old_count = int(profile.get('sessions_count', 0))
            
            new_count = old_count + 1
            new_avg = (old_avg * old_count + new_score) / new_count
            
            profile['avg_score'] = new_avg
            profile['sessions_count'] = new_count
            profile['system_instruction'] = profile.pop('instruction')
            
            await self.upsert_agent(profile)
        except Exception as e:
            logger.warning("⚠️  update_agent_score error [%s]: %s", agent_name, e)

    async def search_agent_by_task(self, task_description: str, n_results: int = 3) -> list[dict]:
        """Семантический поиск агента по описанию задачи."""
        if not self.chroma_mode or self._agents_collection is None:
            return []
        try:
            res = self._agents_collection.query(
                query_texts=[task_description],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            if not res or not res.get('documents') or not res['documents'][0]:
                return []
            
            agents = []
            for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
                agent = meta.copy()
                agent['instruction'] = doc
                agent['search_distance'] = dist
                if 'task_types' in agent and isinstance(agent['task_types'], str):
                    agent['task_types'] = json.loads(agent['task_types'])
                agents.append(agent)
            return agents
        except Exception as e:
            logger.warning("⚠️  search_agent_by_task error: %s", e)
            return []

    async def seed_agents_registry(self) -> None:
        """Заполнение реестра дефолтными агентами."""
        if not self.chroma_mode or self._agents_collection is None:
            return
        
        if self._agents_collection.count() > 0:
            return

        default_agents = [
            {
                "agent_name": "creative_expert",
                "role": "Генератор креативных метафор и сторителлинга",
                "system_instruction": "Ты — CreativeExpert, специализированный агент для создания выразительных, образных и эмоционально насыщенных промтов. Твоя задача: использовать метафоры, аналогии, нарративные структуры и психологические триггеры. Запрос пользователя: {request}. RAG контекст: {rag_context}. Создай структурированный JSON промт с блоками: hook (захват внимания), metaphor (ключевая метафора), narrative (нарратив), emotional_trigger (эмоциональный якорь).",
                "version": 1,
                "temperature": 0.8,
                "task_types": ["creative", "storytelling", "metaphor", "emotional"],
                "avg_score": 0.0,
                "sessions_count": 0,
                "instruction_history": []
            },
            {
                "agent_name": "struct_expert",
                "role": "Архитектор JSON-структур и логических блоков",
                "system_instruction": "Ты — StructExpert, специализированный агент для создания логически выверенных, структурированных промтов. Твоя задача: разбить задачу на чёткие логические блоки, создать JSON-архитектуру, обеспечить последовательность и полноту. Запрос: {request}. RAG контекст: {rag_context}. Создай структурированный JSON промт с блоками: objective (цель), constraints (ограничения), steps (шаги), output_format (формат вывода), validation (критерии проверки).",
                "version": 1,
                "temperature": 0.2,
                "task_types": ["structured", "json", "logical", "architecture", "technical"],
                "avg_score": 0.0,
                "sessions_count": 0,
                "instruction_history": []
            },
            {
                "agent_name": "tone_expert",
                "role": "Психолог и эксперт Tone of Voice",
                "system_instruction": "Ты — ToneExpert, специализированный агент для профилирования аудитории и адаптации тона. Твоя задача: определить целевую аудиторию, подобрать оптимальный тон голоса, учесть психографические особенности. Запрос: {request}. RAG контекст: {rag_context}. Создай структурированный JSON промт с блоками: audience_profile (профиль аудитории), tone_parameters (параметры тона), language_register (языковой регистр), psychological_triggers (психологические триггеры).",
                "version": 1,
                "temperature": 0.6,
                "task_types": ["profiling", "tone", "audience", "psychology", "brand_voice"],
                "avg_score": 0.0,
                "sessions_count": 0,
                "instruction_history": []
            }
        ]

        for agent in default_agents:
            await self.upsert_agent(agent)
        
        logger.info("🌱 [MCP] Registry seeded: %d agents", len(default_agents))

    async def query(
        self, node_id: str, query_text: str = "", n_results: int = 3
    ) -> str:
        """
        Поиск релевантной информации в KB.

        Args:
            node_id: Идентификатор узла KB (используется как fallback / фильтр)
            query_text: Текст запроса для семантического поиска
            n_results: Количество результатов

        Returns:
            Строка с найденным контекстом, или 'НЕТ_ДАННЫХ'
        """
        # === ChromaDB mode ===
        if self.chroma_mode and self._kb_collection is not None:
            search_text = query_text or node_id
            if not search_text:
                return "НЕТ_ДАННЫХ"

            try:
                count = self._kb_collection.count()
                actual_n = min(n_results, count) if count > 0 else 1

                kwargs: dict[str, Any] = {
                    "query_texts": [search_text],
                    "n_results": actual_n,
                }
                # Фильтр по node_id если задан
                if node_id:
                    kwargs["where"] = {"node_id": node_id}

                try:
                    results = self._kb_collection.query(**kwargs)
                except Exception:
                    # Если where-фильтр не найден — ищем без фильтра
                    results = self._kb_collection.query(
                        query_texts=[search_text],
                        n_results=actual_n,
                    )

                if results and results.get("documents") and results["documents"][0]:
                    return "\n\n".join(results["documents"][0])

            except Exception as e:
                logger.warning("⚠️  ChromaDB query error: %s", e)

            return "НЕТ_ДАННЫХ"

        # === FileBasedKB mode ===
        # Точное совпадение по node_id
        if node_id in self._kb_data:
            return self._kb_data[node_id]

        # Поиск по вхождению query_text
        if query_text:
            query_lower = query_text.lower()
            matches: list[str] = []
            for key, value in self._kb_data.items():
                if query_lower in key.lower() or query_lower in value.lower():
                    matches.append(value)
            if matches:
                return "\n\n".join(matches[:n_results])

        # Fallback: все записи
        all_entries = list(self._kb_data.values())
        if all_entries:
            return "\n\n".join(all_entries[:n_results])

        return "НЕТ_ДАННЫХ"

    async def add_document(
        self, collection_name: str, doc_id: str, text: str, metadata: dict
    ) -> None:
        """
        Добавить документ в указанную коллекцию.

        Args:
            collection_name: Имя коллекции (prompt_versions / knowledge_base)
            doc_id: Уникальный ID документа
            text: Текст документа
            metadata: Метаданные (score, session_id, etc.)
        """
        if self.chroma_mode and self._client is not None:
            try:
                coll_kwargs = {"name": collection_name}
                if self._embedding_fn is not None:
                    coll_kwargs["embedding_function"] = self._embedding_fn
                collection = self._client.get_or_create_collection(**coll_kwargs)
                # Приводим все значения metadata к строкам/числам (chromadb requirement)
                clean_meta = {}
                for k, v in metadata.items():
                    if isinstance(v, bool):
                        clean_meta[k] = str(v).lower()
                    elif isinstance(v, (int, float, str)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)

                collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[clean_meta],
                )
                logger.info("💾 [MCP] Документ %s добавлен в %s", doc_id[:8], collection_name)
            except Exception as e:
                logger.warning("⚠️  add_document error: %s", e)
        else:
            # FileBasedKB fallback — сохраняем в memory
            self._memory_prompts[doc_id] = {"text": text, "metadata": metadata}
            logger.info(
                "💾 [MCP] Документ %s сохранён в memory (FileBasedKB mode)", doc_id[:8]
            )

    async def search_similar_prompts(
        self, query_text: str, n_results: int = 5, min_score: float = 0.0
    ) -> list[dict]:
        """
        Семантический поиск одобрённых промптов для DSPy trainset.

        Args:
            query_text: Текст запроса
            n_results: Количество результатов
            min_score: Минимальный score промпта

        Returns:
            Список [{text: str, metadata: dict}, ...]
        """
        if not self.chroma_mode or self._prompts_collection is None:
            # FileBasedKB fallback
            results = []
            for doc_id, data in self._memory_prompts.items():
                meta = data.get("metadata", {})
                score = float(meta.get("score", 0))
                if score >= min_score and str(meta.get("approved", "")).lower() == "true":
                    results.append({"text": data["text"], "metadata": meta})
            return results[:n_results]

        try:
            count = self._prompts_collection.count()
            if count == 0:
                return []

            actual_n = min(n_results, count)

            results = self._prompts_collection.query(
                query_texts=[query_text],
                n_results=actual_n,
                where={"approved": "true"},
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            items = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                doc_score = float(meta.get("score", 0))
                if doc_score >= min_score:
                    items.append({"text": doc, "metadata": meta})

            return items

        except Exception as e:
            logger.warning("⚠️  search_similar_prompts error: %s", e)
            return []

    def list_nodes(self) -> list[str]:
        """Список всех доступных узлов (ключей) в KB."""
        return list(self._kb_data.keys())
