"""
db_versioning.py — Версионирование промптов (aiosqlite)
========================================================
Персистентное хранение версий промптов в SQLite.
Поддержка: save_state, rollback, get_history.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import aiosqlite

from state_types import ContextState

logger = logging.getLogger(__name__)

# SQL-схема таблицы версий
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    iteration INTEGER,
    final_score REAL,
    approved INTEGER,
    created_at TEXT,
    data_json TEXT
)
"""


class PromptVersioningManager:
    """
    Менеджер версионирования промптов с персистентным хранением в SQLite.
    Каждая итерация оркестратора сохраняется как отдельная версия.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.getenv("DB_PATH", "orchestrator.db")
        self._initialized = False

    async def init_db(self) -> None:
        """Создать таблицу если не существует."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(_CREATE_TABLE_SQL)
                await db.commit()
            self._initialized = True
            logger.info("✅ DB инициализирована: %s", self.db_path)
        except Exception as e:
            logger.error("❌ Ошибка инициализации DB: %s", e)
            raise

    async def _ensure_init(self) -> None:
        """Убедиться, что DB инициализирована."""
        if not self._initialized:
            await self.init_db()

    async def save_state(self, state: ContextState) -> str:
        """
        Сохранить текущее состояние как новую версию.

        Args:
            state: Текущее состояние оркестратора

        Returns:
            version_id созданной записи
        """
        await self._ensure_init()

        version_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Сериализация состояния в JSON
        data_json = state.model_dump_json()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO prompt_versions
                    (version_id, session_id, iteration, final_score, approved, created_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        state.session_id,
                        state.iteration,
                        state.current_score,
                        1 if state.approved else 0,
                        now,
                        data_json,
                    ),
                )
                await db.commit()

            logger.info(
                "💾 Версия сохранена: %s (сессия=%s, итерация=%d, score=%.2f)",
                version_id[:8],
                state.session_id[:8],
                state.iteration,
                state.current_score,
            )
            return version_id

        except Exception as e:
            logger.error("❌ Ошибка сохранения версии: %s", e)
            raise

    async def rollback(self, version_id: str) -> Optional[ContextState]:
        """
        Откатить состояние к указанной версии.

        Args:
            version_id: ID версии для отката

        Returns:
            ContextState восстановленного состояния или None
        """
        await self._ensure_init()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT data_json FROM prompt_versions WHERE version_id = ?",
                    (version_id,),
                )
                row = await cursor.fetchone()

            if row is None:
                logger.warning("⚠️  Версия %s не найдена", version_id)
                return None

            state = ContextState.model_validate_json(row[0])
            logger.info("⏪ Откат к версии: %s", version_id[:8])
            return state

        except Exception as e:
            logger.error("❌ Ошибка отката: %s", e)
            return None

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Получить историю всех версий для сессии.

        Args:
            session_id: ID сессии

        Returns:
            Список словарей с метаданными версий
        """
        await self._ensure_init()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT version_id, session_id, iteration, final_score,
                           approved, created_at
                    FROM prompt_versions
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                )
                rows = await cursor.fetchall()

            history = [
                {
                    "version_id": row["version_id"],
                    "session_id": row["session_id"],
                    "iteration": row["iteration"],
                    "final_score": row["final_score"],
                    "approved": bool(row["approved"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

            logger.info(
                "📜 История сессии %s: %d версий", session_id[:8], len(history)
            )
            return history

        except Exception as e:
            logger.error("❌ Ошибка чтения истории: %s", e)
            return []
