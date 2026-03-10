# AI Orchestrator 2026 — Инструкция по запуску

## Быстрый старт

### 1. Установка зависимостей

```bash
cd /Users/Orlova/ai-orchestrator-2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Настройка (опционально — для реальных LLM-вызовов)

```bash
cp .env.example .env
# Отредактируйте .env: укажите LLM_ROUTER_API_KEY и LLM_ROUTER_BASE_URL
```

### 3. Запуск

```bash
# Mock-режим (без API ключа)
python main.py

# С реальными LLM (нужен .env с ключами)
python main.py

# С кастомным запросом
python main.py --request "Создай промпт для анализа данных"
```

## Архитектура

```
Request → Router → Generator(MoE) → Evaluator → Optimizer → RedTeam → Result
            ↑                            │
            └────── итерация ←───────────┘
```

**Модели по задачам**:
| Task Type | Env Variable | Default | Применение |
|-----------|-------------|---------|-----------|
| creative | MODEL_CREATIVE | gpt-4o | Генерация промптов |
| medium | MODEL_MEDIUM | gpt-4o-mini | Структура, оптимизация |
| cheap | MODEL_CHEAP | gpt-4o-mini | Routing, оценка, red-team |

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `main.py` | Точка входа, инициализация |
| `orchestrator.py` | State-machine граф |
| `llm_router.py` | Smart LLM routing |
| `moe_agents.py` | MoE-агенты (Creative, Architect, Profiler, Adversary) |
| `judges.py` | LLM-based оценка |
| `dspy_optimizer.py` | DSPy/LLM оптимизация |
| `mcp_servers.py` | База знаний (File/ChromaDB) |
| `rag_manager.py` | RAG контекст |
| `db_versioning.py` | SQLite версионирование |
| `state_types.py` | Pydantic контракты |
| `observability.py` | OpenTelemetry трейсинг |
