"""
main.py — Точка входа AI Orchestrator 2026
=============================================
Инициализация всех компонентов и запуск оркестрации.
Поддержка mock-режима (без API ключа) и real-режима.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Загрузка переменных окружения из .env
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Импорт модулей системы
from cli_interface import run_cli
from db_versioning import PromptVersioningManager
from dspy_optimizer import PromptOptimizer
from judges import EvaluationJudge
from llm_router import LiteLLMClient
from mcp_servers import MCPService
from moe_agents import (
    AdversaryAgent,
    ArchitectAgent,
    CreativeAgent,
    ProfilerAgent,
)
from observability import TelemetrySystem
from orchestrator import OrchestratorGraph
from rag_manager import RAGManager
from state_types import ContextState, RoutingDecision, SwarmResponse
from external_agent_connector import ExternalAgentConnector
from wrapper_translator import WrapperTranslator
from prompt_serializer import PromptSerializer


# ============================================================================
# УЗЛЫ ГРАФА (функции для OrchestratorGraph)
# ============================================================================


def create_router_node(llm_router: LiteLLMClient):
    """Создать узел маршрутизации запросов с использованием RoutingDecision."""

    async def router_node(state: ContextState) -> ContextState:
        """Реальный LLM анализ запроса для выбора стратегии."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Analyze the user request and determine the best specialized agent. "
                    "Return a JSON object: {\"destination\": \"creative\"|\"structured\"|\"profiling\"|\"external\", "
                    "\"tasks\": [\"exact task 1\", ...], \"summary\": \"why this decision\"}. "
                    "Use 'external' only if the user provides a URL or explicitly asks for an outside search/service."
                ),
            },
            {"role": "user", "content": state.request[:500]},
        ]

        response = await llm_router.generate(
            messages=messages,
            temperature=0.0,
            task_type="routing",
            response_format={"type": "json_object"}
        )

        try:
            content = response["content"]
            data = json.loads(content)
            decision = RoutingDecision(**data)
        except Exception as e:
            logger.warning("⚠️ Ошибка парсинга RoutingDecision: %s. Fallback to creative.", e)
            decision = RoutingDecision(destination="creative", summary="Fallback due to error")

        # Сохранение в memory_kv и request_type
        state.memory_kv["routing"] = decision.model_dump()
        state.memory_kv["routing_target"] = decision.destination
        state.request_type = decision.destination

        logger.info("📡 [Router] Назначен таргет: %s (%s)", decision.destination, decision.summary)
        return state

    return router_node


def create_generator_node(
    creative_agent: CreativeAgent,
    architect_agent: ArchitectAgent,
    profiler_agent: ProfilerAgent,
    rag_manager: RAGManager,
    connector: ExternalAgentConnector,
    translator: WrapperTranslator
):
    """Создать узел генерации промптов (MoE + External)."""

    async def generator_node(state: ContextState) -> ContextState:
        """Перенаправление в нужный агентный пул на основе routing_target."""
        target = state.memory_kv.get("routing_target", "creative")
        
        # Получаем RAG-контекст
        rag_context = await rag_manager.get_context(
            agent_id=f"{target}_agent", query=state.request
        )
        state.rag_context = rag_context

        # === 1. Внешний агент ===
        if target == "external":
            # Извлекаем URL если он вшит или в метаданных
            ext_url = state.memory_kv.get("external_agent_url")
            if not ext_url:
                import re
                urls = re.findall(r'https?://[^\s]+', state.request)
                ext_url = urls[0] if urls else "http://localhost:8080/generate"
            
            logger.info("🌐 [Generator] Вызов ВНЕШНЕГО агента: %s", ext_url)
            raw_swarm = await connector.call_external(ext_url, state)
            
            # Если ответ "мусорный", чистим транслятором
            if raw_swarm.ok:
                clean_resp = await translator.clean_external_response(
                    str(raw_swarm.result), agent_url=ext_url
                )
                state.prompt_chain.append({
                    "agent_id": "external_agent",
                    "agent_role": "External Specialist",
                    "generated_text": clean_resp.result.get("text", ""),
                    "model_used": "external",
                    "task_type": "external",
                    "latency_ms": 0,
                    "metadata": {"url": ext_url, "score": clean_resp.score}
                })
            return state

        # === 2. Внутренние MoE агенты ===
        agents = []
        if target == "creative":
            agents = [creative_agent]
        elif target == "structured":
            agents = [architect_agent]
        elif target == "profiling":
            agents = [profiler_agent]
        else:
            # Fallback на полный пул
            agents = [creative_agent, architect_agent, profiler_agent]

        logger.info("🎨 [Generator] Запуск %d MoE агентов (target=%s)...", len(agents), target)
        
        async def run_agent(agent_obj):
            try:
                return await agent_obj.process(state=state, rag_context=rag_context)
            except Exception as e:
                logger.error("❌ Ошибка агента %s: %s", agent_obj.agent_id, e)
                return None

        tasks = [run_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks)

        for draft in results:
            if draft:
                state.prompt_chain.append(draft.model_dump())

        return state

    return generator_node


def create_evaluator_node(judge: EvaluationJudge):
    """Создать узел оценки промптов."""

    async def evaluator_node(state: ContextState) -> ContextState:
        """Оценка последнего черновика."""
        result = await judge.evaluate(state)
        state.evaluation = result
        return state

    return evaluator_node


def create_optimizer_node(optimizer: PromptOptimizer, agents_dict: dict):
    """Создать узел оптимизации промптов."""

    async def optimizer_node(state: ContextState) -> ContextState:
        """Оптимизация промпта на основе оценки."""
        # V2: Теперь оптимизатор может править системные инструкции агентов
        state = await optimizer.optimize_prompt(state, agents=agents_dict)
        
        # После оптимизации возвращаемся на роутер для перепроверки новым агентом/инструкцией
        state.routing_target = "router"
        return state

    return optimizer_node


def create_red_team_node(adversary: AdversaryAgent):
    """Создать узел red-team тестирования."""

    async def red_team_node(state: ContextState) -> ContextState:
        """Стресс-тест финального промпта."""
        result = await adversary.stress_test(state)
        state.red_team_result = result
        
        # Анализ надежности: если robustness < 0.7 — отправляем на оптимизацию
        if result.overall_robustness < 0.7:
            logger.warning("🔴 [RedTeam] Robustness low (%.2f). Routing to optimizer.", result.overall_robustness)
            state.memory_kv["routing_target"] = "optimizer"
            state.approved = False
        else:
            logger.info("🟢 [RedTeam] Robustness OK (%.2f).", result.overall_robustness)
            state.memory_kv["routing_target"] = "end"
            # state.approved остается True
            
        return state

    return red_team_node


def create_save_fn(db: PromptVersioningManager):
    """Создать функцию сохранения состояния."""

    async def save_fn(state: ContextState) -> None:
        try:
            await db.save_state(state)
        except Exception as e:
            logger.error("❌ Ошибка сохранения в DB: %s", e)

    return save_fn


# ============================================================================
# MAIN
# ============================================================================


async def main(request: str | None = None, interactive: bool = False) -> ContextState:
    """
    Главная функция: инициализация и запуск оркестратора.

    Args:
        request: Запрос пользователя (если None — используется default)
        interactive: Запуск в интерактивном режиме
    """
    external_url = None
    
    if interactive:
        request, external_url = await run_cli()
    
    # Запрос по умолчанию
    if not request:
        request = (
            "Создай промпт для генерации музыкальной композиции в стиле "
            "электронного ambient с элементами классической гармонии. "
            "Промпт должен быть детальным и вдохновляющим."
        )

    print("\n" + "=" * 70)
    print("🤖 AI ORCHESTRATOR 2026")
    print("   Мультиагентная система оркестрации LLM-промптов")
    print("=" * 70)
    print(f"\n📋 ЗАПРОС: {request}\n")

    # === Инициализация компонентов ===

    # 1. Телеметрия
    telemetry = TelemetrySystem()

    # 2. LLM Router (smart routing)
    llm_router = LiteLLMClient()
    mode = "MOCK" if llm_router.is_mock else "REAL"
    print(f"🔧 Режим: {mode}")

    # 3. Агенты MoE (с передачей rag_manager и llm_router)
    mcp = MCPService()
    await mcp.seed_knowledge_base()
    rag = RAGManager(mcp)

    creative_agent = CreativeAgent(rag_manager=rag, llm_router=llm_router)
    architect_agent = ArchitectAgent(rag_manager=rag, llm_router=llm_router)
    profiler_agent = ProfilerAgent(rag_manager=rag, llm_router=llm_router)
    adversary_agent = AdversaryAgent(rag_manager=rag, llm_router=llm_router)

    # V2: Асинхронная инициализация профилей из реестра
    await creative_agent.initialize()
    await architect_agent.initialize()
    await profiler_agent.initialize()
    await adversary_agent.initialize()

    agents_dict = {
        "creative_expert": creative_agent,
        "struct_expert": architect_agent,
        "tone_expert": profiler_agent,
        "adversary_agent": adversary_agent
    }

    # 5. Judge (оценка) — с rag_manager для сохранения approved промптов
    judge = EvaluationJudge(llm_router=llm_router, rag_manager=rag)

    # 6. Optimizer (DSPy + LLM fallback) — с rag_manager для trainset из ChromaDB
    optimizer = PromptOptimizer(llm_router=llm_router, rag_manager=rag)

    # 8. DB Versioning (персистентность)
    db = PromptVersioningManager()
    await db.init_db()

    # 8. External Integration
    connector = ExternalAgentConnector()
    translator = WrapperTranslator(llm_router=llm_router)

    # === Создание графа оркестрации ===
    graph = OrchestratorGraph(
        router_fn=create_router_node(llm_router),
        generator_fn=create_generator_node(
            creative_agent, architect_agent, profiler_agent, rag, connector, translator
        ),
        evaluator_fn=create_evaluator_node(judge),
        optimizer_fn=create_optimizer_node(optimizer, agents_dict),
        red_team_fn=create_red_team_node(adversary_agent),
        save_fn=create_save_fn(db),
    )

    # === Запуск ===
    initial_state = ContextState(request=request)
    if external_url:
        initial_state.memory_kv["routing_target"] = "external"
        initial_state.memory_kv["external_agent_url"] = external_url
        
    final_state = await graph.run(initial_state)

    # === Вывод результата ===
    print("\n" + "=" * 70)
    print("✅ РЕЗУЛЬТАТ ОРКЕСТРАЦИИ")
    print("=" * 70)

    print(f"\n📊 Статистика:")
    print(f"   Итераций: {final_state.iteration + 1}")
    print(f"   Score: {final_state.current_score:.2f}")
    print(f"   Approved: {final_state.approved}")
    print(f"   Черновиков в цепочке: {len(final_state.prompt_chain)}")

    if final_state.red_team_result:
        rt = final_state.red_team_result
        print(f"\n🔴 Red Team:")
        print(f"   Уязвимостей: {rt.vulnerabilities_found}")
        print(f"   Robustness: {rt.overall_robustness:.2f}")

    print(f"\n📝 ФИНАЛЬНЫЙ ПРОМПТ:")
    print("-" * 70)
    final_text = final_state.final_prompt or "(пусто)"
    print(final_text)
    print("-" * 70)

    # Логи
    if final_state.logs:
        print(f"\n📜 ЛОГ ОРКЕСТРАЦИИ:")
        for log_entry in final_state.logs:
            print(f"   • {log_entry}")

    # История версий
    history = await db.get_history(final_state.session_id)
    if history:
        print(f"\n💾 ИСТОРИЯ ВЕРСИЙ ({len(history)} записей):")
        for v in history:
            print(
                f"   v{v['iteration']+1}: score={v['final_score']:.2f}, "
                f"approved={v['approved']}, id={v['version_id'][:8]}"
            )

    print("\n" + "=" * 70)
    print("🏁 Готово!")
    print("=" * 70 + "\n")

    return final_state


# ============================================================================
# ENTRY POINT
# ============================================================================

async def cli_add_agent(args):
    """Команда добавления агента в реестр."""
    mcp = MCPService()
    await mcp.seed_knowledge_base()
    
    profile = {
        "agent_name": args.name,
        "role": args.role,
        "system_instruction": args.instruction,
        "version": 1,
        "temperature": args.temperature,
        "task_types": args.task_types.split(",") if args.task_types else ["general"],
        "avg_score": 0.0,
        "sessions_count": 0,
        "instruction_history": []
    }
    
    success = await mcp.upsert_agent(profile)
    if success:
        print(f"✅ Агент '{args.name}' v1 успешно добавлен в реестр.")
    else:
        print(f"❌ Ошибка при добавлении агента '{args.name}'.")

async def cli_list_agents():
    """Команда вывода списка агентов."""
    mcp = MCPService()
    await mcp.seed_knowledge_base()
    agents = await mcp.list_agents()
    
    if not agents:
        print("📭 Реестр агентов пуст.")
        return
    
    print("\n📋 РЕЕСТР АГЕНТОВ:")
    print("-" * 100)
    print(f"{'Имя':<20} | {'Роль':<35} | {'Ver':<3} | {'Score':<5} | {'Sess':<4}")
    print("-" * 100)
    for a in agents:
        name = a.get('agent_name', '?')
        role = a.get('role', '?')[:35]
        ver = a.get('version', 1)
        score = a.get('avg_score', 0.0)
        sess = a.get('sessions_count', 0)
        print(f"{name:<20} | {role:<35} | v{ver:<2} | {score:0.2f}  | {sess:<4}")
    print("-" * 100 + "\n")

async def cli_show_agent(name: str):
    """Команда вывода детальной информации об агенте."""
    mcp = MCPService()
    await mcp.seed_knowledge_base()
    profile = await mcp.get_agent(name)
    
    if not profile:
        print(f"❌ Агент '{name}' не найден.")
        return
    
    print(f"\n👤 ПРОФИЛЬ АГЕНТА: {name}")
    print("=" * 70)
    print(f"Роль:      {profile.get('role', '?')}")
    print(f"Версия:    v{profile.get('version', 1)}")
    print(f"Score:     {profile.get('avg_score', 0.0):.2f}")
    print(f"Сессий:    {profile.get('sessions_count', 0)}")
    print(f"Задачи:    {', '.join(profile.get('task_types', []))}")
    print(f"Темп-ра:   {profile.get('temperature', 0.5)}")
    print("\n📜 СИСТЕМНАЯ ИНСТРУКЦИЯ:")
    print("-" * 70)
    print(profile.get('instruction', ''))
    print("-" * 70 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Orchestrator 2026 — Мультиагентная оркестрация промптов"
    )
    # Группа запуска
    parser.add_argument("--request", type=str, default=None, help="Запрос для генерации промпта")
    parser.add_argument("--interactive", action="store_true", help="Запустить в интерактивном режиме")
    
    # Группа управления реестром
    parser.add_argument("--list-agents", action="store_true", help="Показать список всех агентов")
    parser.add_argument("--show-agent", type=str, metavar="NAME", help="Показать профиль конкретного агента")
    parser.add_argument("--add-agent", action="store_true", help="Добавить нового агента (требует доп. аргументы)")
    
    # Аргументы для добавления агента
    parser.add_argument("--name", type=str, help="Имя нового агента")
    parser.add_argument("--role", type=str, help="Роль агента")
    parser.add_argument("--instruction", type=str, help="Системная инструкция")
    parser.add_argument("--task-types", type=str, help="Типы задач через запятую")
    parser.add_argument("--temperature", type=float, default=0.5, help="Температура (0.0-1.0)")

    args = parser.parse_args()

    if args.list_agents:
        asyncio.run(cli_list_agents())
    elif args.show_agent:
        asyncio.run(cli_show_agent(args.show_agent))
    elif args.add_agent:
        if not all([args.name, args.role, args.instruction]):
            print("❌ Для добавления агента нужны: --name, --role, --instruction")
        else:
            asyncio.run(cli_add_agent(args))
    else:
        asyncio.run(main(request=args.request, interactive=args.interactive))
