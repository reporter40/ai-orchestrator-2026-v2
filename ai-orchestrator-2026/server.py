"""
server.py — FastAPI Web Server for AI Orchestrator 2026
========================================================
REST API + SSE streaming for real-time orchestration progress.
Serves the static frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Загрузка переменных окружения
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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Импорт системных модулей
from db_versioning import PromptVersioningManager
from dspy_optimizer import PromptOptimizer
from external_agent_connector import ExternalAgentConnector
from judges import EvaluationJudge
from llm_router import LiteLLMClient
from mcp_servers import MCPService
from moe_agents import AdversaryAgent, ArchitectAgent, CreativeAgent, ProfilerAgent
from observability import TelemetrySystem
from orchestrator import OrchestratorGraph
from rag_manager import RAGManager
from state_types import ContextState, RoutingDecision
from wrapper_translator import WrapperTranslator

# ============================================================================
# Глобальные компоненты (инициализируются при старте)
# ============================================================================

app = FastAPI(title="AI Orchestrator 2026", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Компоненты — лениво инициализируются
_components: dict[str, Any] = {}
_initialized = False


async def _ensure_init():
    """Инициализировать компоненты при первом запросе."""
    global _initialized
    if _initialized:
        return

    telemetry = TelemetrySystem()
    llm_router = LiteLLMClient()
    mcp = MCPService()
    await mcp.seed_knowledge_base()
    await mcp.seed_agents_registry()
    rag = RAGManager(mcp)
    judge = EvaluationJudge(llm_router=llm_router, rag_manager=rag)
    optimizer = PromptOptimizer(llm_router=llm_router, rag_manager=rag)
    
    # V2: Инициализация агентов с rag_manager
    creative = CreativeAgent(rag_manager=rag, llm_router=llm_router)
    architect = ArchitectAgent(rag_manager=rag, llm_router=llm_router)
    profiler = ProfilerAgent(rag_manager=rag, llm_router=llm_router)
    adversary = AdversaryAgent(rag_manager=rag, llm_router=llm_router)
    
    await creative.initialize()
    await architect.initialize()
    await profiler.initialize()
    await adversary.initialize()

    db = PromptVersioningManager()
    await db.init_db()
    
    connector = ExternalAgentConnector()
    translator = WrapperTranslator(llm_router=llm_router)

    _components.update({
        "llm_router": llm_router,
        "mcp": mcp,
        "rag": rag,
        "judge": judge,
        "optimizer": optimizer,
        "creative": creative,
        "architect": architect,
        "profiler": profiler,
        "adversary": adversary,
        "db": db,
        "telemetry": telemetry,
        "connector": connector,
        "translator": translator,
        "agents_dict": {
            "creative_expert": creative,
            "struct_expert": architect,
            "tone_expert": profiler,
            "adversary_agent": adversary
        }
    })
    _initialized = True
    logger.info("✅ All components initialized (V2 Adaptive Engine active)")


# ============================================================================
# Маршруты
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI Orchestrator 2026</h1><p>Static files not found</p>")


@app.post("/api/orchestrate")
async def orchestrate(request: Request):
    """
    Run orchestration with SSE streaming for real-time progress.
    Request body: {"request": "...", "max_iterations": 3, "threshold": 0.85}
    """
    await _ensure_init()
    body = await request.json()
    user_request = body.get("request", "").strip()
    max_iterations = int(body.get("max_iterations", 3))
    threshold = float(body.get("threshold", 0.85))

    if not user_request:
        return JSONResponse({"error": "Empty request"}, status_code=400)

    async def event_stream():
        """SSE event generator."""
        llm = _components["llm_router"]
        rag = _components["rag"]
        judge = _components["judge"]
        optimizer = _components["optimizer"]
        creative = _components["creative"]
        architect = _components["architect"]
        profiler = _components["profiler"]
        adversary = _components["adversary"]
        db = _components["db"]

        state = ContextState(request=user_request)
        state.max_iterations = max_iterations

        # Override threshold
        judge.approval_threshold = threshold

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # --- ROUTER ---
            yield sse("progress", {"step": "router", "status": "running", "message": "📡 Анализ запроса и маршрутизация..."})

            routing_result = await llm.call_agent(
                prompt=state.request,
                agent_name="node_router",
                response_model=RoutingDecision,
                task_type="routing"
            )
            
            decision = routing_result.data
            if decision:
                state.request_type = decision.destination
                state.memory_kv["routing_target"] = decision.destination
                state.memory_kv["reasoning"] = decision.reasoning
                state.memory_kv["confidence"] = decision.confidence
            else:
                # Fallback
                state.request_type = "creative"
                state.memory_kv["routing_target"] = "creative"

            # Переопределение на внешний агент, если передан URL
            external_url = body.get("external_agent_url")
            if external_url:
                state.memory_kv["routing_target"] = "external"
                state.memory_kv["external_agent_url"] = external_url

            yield sse("router", {
                "request_type": state.request_type,
                "routing_target": state.memory_kv["routing_target"],
                "reasoning": state.memory_kv.get("reasoning", ""),
                "confidence": state.memory_kv.get("confidence", 1.0)
            })

            # --- ITERATIVE LOOP ---
            for iteration in range(max_iterations):
                state.iteration = iteration

                # --- GENERATOR ---
                yield sse("progress", {
                    "step": "generator",
                    "status": "running",
                    "message": f"Итерация {iteration + 1}: запуск MoE-агентов...",
                    "iteration": iteration + 1,
                })

                rag_context = await rag.get_context(agent_id="creative_agent", query=state.request)
                state.rag_context = rag_context

                target = state.memory_kv.get("routing_target", "creative")
                
                if target == "external":
                    url = state.memory_kv.get("external_agent_url")
                    yield sse("progress", {
                        "step": "generator",
                        "status": "running",
                        "message": f"🔗 Вызов внешнего агента: {url}",
                        "iteration": iteration + 1,
                    })
                    
                    swarm_resp = await _components["connector"].call_external(url, state)
                    
                    # Snapshotting: сохраняем внешний агент в локальный реестр для стабильности
                    await _components["mcp"].upsert_agent({
                        "agent_name": f"external_{url.split('/')[-1]}",
                        "role": "External Agent",
                        "instruction": "External agent at " + url,
                        "external_url": url,
                        "is_external": True
                    })
                    drafts_info = [{
                        "agent": "External Agent",
                        "length": len(swarm_resp.generated_text),
                        "model": "external",
                        "latency_ms": swarm_resp.latency_ms,
                    }]
                else:
                    # MoE Pool
                    if target == "creative":
                        agents = [creative, profiler]
                    elif target == "structured":
                        agents = [architect, profiler]
                    elif target == "profiling":
                        agents = [profiler]
                    else:
                        agents = [creative, architect, profiler]

                    tasks = [a.process(state=state, rag_context=rag_context) for a in agents]
                    drafts_info = []
                    for future in asyncio.as_completed(tasks):
                        try:
                            draft = await future
                            if draft:
                                state.prompt_chain.append(draft.model_dump())
                                drafts_info.append({
                                    "agent": draft.agent_role,
                                    "length": len(draft.generated_text),
                                    "model": draft.model_used,
                                    "latency_ms": draft.latency_ms,
                                })
                                yield sse("progress", {
                                    "step": "generator",
                                    "status": "running",
                                    "message": f"✅ {draft.agent_role} готов!",
                                    "iteration": iteration + 1,
                                })
                        except Exception as e:
                            logger.error("❌ MoE Agent error: %s", e)
                            yield sse("progress", {
                                "step": "generator",
                                "status": "error",
                                "message": f"⚠️ Ошибка одного из агентов: {str(e)[:50]}",
                            })

                yield sse("generator", {
                    "iteration": iteration + 1,
                    "drafts": drafts_info,
                    "target": target,
                    "rag_context_length": len(rag_context),
                })

                # --- EVALUATOR ---
                yield sse("progress", {
                    "step": "evaluator",
                    "status": "running",
                    "message": f"Итерация {iteration + 1}: оценка промпта...",
                })

                result = await judge.evaluate(state)
                state.evaluation = result
                if result:
                    score = (result.task_fulfillment + result.rag_accuracy) / 2 - result.hallucination_penalty * 0.3
                    state.current_score = max(0.0, min(1.0, score))
                    state.approved = result.approved

                yield sse("evaluator", {
                    "iteration": iteration + 1,
                    "task_fulfillment": result.task_fulfillment if result else 0,
                    "rag_accuracy": result.rag_accuracy if result else 0,
                    "hallucination_penalty": result.hallucination_penalty if result else 0,
                    "score": state.current_score,
                    "approved": state.approved,
                    "feedback": result.feedback if result else "",
                })

                # Save version
                await db.save_state(state)

                if state.approved:
                    # --- RED TEAM ---
                    yield sse("progress", {"step": "red_team", "status": "running", "message": f"Итерация {iteration + 1}: стресс-тестирование..."})

                    rt_result = await adversary.stress_test(state)
                    state.red_team_result = rt_result

                    yield sse("red_team", {
                        "vulnerabilities": rt_result.vulnerabilities_found if rt_result else 0,
                        "robustness": rt_result.overall_robustness if rt_result else 1.0,
                        "recommendation": rt_result.recommendation if rt_result else "",
                        "attacks": rt_result.attack_results if rt_result else [],
                        "routing_target": state.memory_kv.get("routing_target", "end")
                    })

                    # Если Red Team одобряет — выходим из цикла
                    if state.memory_kv.get("routing_target") != "optimizer":
                        break
                    
                    yield sse("progress", {
                        "step": "optimizer",
                        "status": "running",
                        "message": "🔴 Red Team: возврат к оптимизации...",
                    })

                # --- OPTIMIZER ---
                if iteration < max_iterations - 1:
                    yield sse("progress", {
                        "step": "optimizer",
                        "status": "running",
                        "message": f"Итерация {iteration + 1}: оптимизация и самосовершенствование...",
                    })
                    
                    # V2: Вызов комплексного оптимизатора
                    state = await optimizer.optimize_prompt(state, agents=_components["agents_dict"])
                    optimized = state.optimized_prompt
                    
                    yield sse("optimizer", {
                        "iteration": iteration + 1, 
                        "optimized_length": len(optimized),
                        "agent_updated": state.current_score < 0.7
                    })
            else:
                state.approved = True


            # Финализация
            if state.prompt_chain:
                state.final_prompt = state.prompt_chain[-1].get("generated_text", "")
            await db.save_state(state)

            # --- COMPLETE ---
            yield sse("complete", {
                "session_id": state.session_id,
                "final_prompt": state.final_prompt,
                "score": state.current_score,
                "approved": state.approved,
                "iterations": state.iteration + 1,
                "total_drafts": len(state.prompt_chain),
                "request_type": state.request_type,
                "logs": state.logs,
            })

        except Exception as e:
            logger.error("❌ Orchestration error: %s", e, exc_info=True)
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def list_sessions():
    """Get all unique sessions."""
    await _ensure_init()
    db = _components["db"]
    try:
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """SELECT session_id, MAX(iteration) as max_iter,
                          MAX(final_score) as best_score,
                          MAX(approved) as approved,
                          MIN(created_at) as started_at,
                          MAX(created_at) as completed_at,
                          COUNT(*) as versions
                   FROM prompt_versions
                   GROUP BY session_id
                   ORDER BY MAX(created_at) DESC
                   LIMIT 50"""
            )
            rows = await cursor.fetchall()

        sessions = []
        for row in rows:
            # Get the request from the latest version
            async with aiosqlite.connect(db.db_path) as conn:
                cur2 = await conn.execute(
                    "SELECT data_json FROM prompt_versions WHERE session_id = ? ORDER BY created_at ASC LIMIT 1",
                    (row["session_id"],),
                )
                data_row = await cur2.fetchone()
                request_text = ""
                if data_row:
                    try:
                        state_data = json.loads(data_row[0])
                        request_text = state_data.get("request", "")[:100]
                    except Exception:
                        pass

            sessions.append({
                "session_id": row["session_id"],
                "iterations": row["max_iter"] + 1 if row["max_iter"] is not None else 0,
                "best_score": row["best_score"] or 0,
                "approved": bool(row["approved"]),
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "versions": row["versions"],
                "request": request_text,
            })

        return JSONResponse(sessions)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history/{session_id}")
async def get_session_history(session_id: str):
    """Get version history for a session."""
    await _ensure_init()
    db = _components["db"]
    history = await db.get_history(session_id)
    return JSONResponse(history)


@app.post("/api/rollback/{version_id}")
async def rollback(version_id: str):
    """Rollback to a specific version."""
    await _ensure_init()
    db = _components["db"]
    state = await db.rollback(version_id)
    if state:
        return JSONResponse({
            "success": True,
            "state": state.model_dump(),
        })
    return JSONResponse({"success": False, "error": "Version not found"}, status_code=404)


@app.get("/api/kb")
async def list_kb_nodes():
    """List knowledge base nodes."""
    await _ensure_init()
    mcp = _components["mcp"]
    nodes = mcp.list_nodes()
    kb_data = {}
    for node in nodes:
        result = await mcp.query(node_id=node)
        kb_data[node] = result[:200] if result != "НЕТ_ДАННЫХ" else ""
    return JSONResponse({"nodes": nodes, "data": kb_data})


@app.post("/api/kb/query")
async def query_kb(request: Request):
    """Query knowledge base."""
    await _ensure_init()
    body = await request.json()
    query = body.get("query", "")
    mcp = _components["mcp"]
    result = await mcp.query(node_id="", query_text=query, n_results=5)
    return JSONResponse({"query": query, "result": result})


@app.get("/api/chromadb/stats")
async def chromadb_stats():
    """Get ChromaDB collection statistics."""
    await _ensure_init()
    mcp = _components["mcp"]
    stats = {
        "mode": "chromadb" if mcp.chroma_mode else "file",
        "knowledge_base": {"count": 0},
        "prompt_versions": {"count": 0},
    }
    if mcp.chroma_mode:
        if mcp._kb_collection:
            stats["knowledge_base"]["count"] = mcp._kb_collection.count()
        if mcp._prompts_collection:
            stats["prompt_versions"]["count"] = mcp._prompts_collection.count()
        if hasattr(mcp, '_agents_collection') and mcp._agents_collection:
            stats["agents_registry"] = {"count": mcp._agents_collection.count()}
    return JSONResponse(stats)


@app.get("/api/agents")
async def list_agents():
    """List all agents in the registry."""
    await _ensure_init()
    mcp = _components["mcp"]
    agents = await mcp.list_agents()
    return JSONResponse(agents)


@app.post("/api/agents/add")
async def add_agent(request: Request):
    """Add or update an agent in the registry."""
    await _ensure_init()
    body = await request.json()
    mcp = _components["mcp"]
    success = await mcp.upsert_agent(body)
    return JSONResponse({"success": success})


@app.get("/api/external/catalog")
async def get_external_catalog():
    """Get the catalog of external agent URLs."""
    catalog_path = Path(__file__).parent / "external_agents.json"
    if not catalog_path.exists():
        return JSONResponse({})
    try:
        return JSONResponse(json.loads(catalog_path.read_text(encoding="utf-8")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent system logs."""
    log_file = Path("server.log")
    if not log_file.exists():
        return JSONResponse({"logs": []})
    
    try:
        lines = log_file.read_text().splitlines()
        return JSONResponse({"logs": lines[-limit:]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/settings")
async def get_settings():
    """Get current orchestrator settings."""
    return JSONResponse({
        "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
        "approval_threshold": float(os.getenv("APPROVAL_THRESHOLD", "0.85")),
        "model_creative": os.getenv("MODEL_CREATIVE", "gpt-4o"),
        "model_medium": os.getenv("MODEL_MEDIUM", "gpt-4o-mini"),
        "model_cheap": os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
        "dspy_model": os.getenv("DSPY_MODEL", "gpt-4o-mini"),
        "chroma_mode": _components.get("mcp", None) and _components["mcp"].chroma_mode,
        "chroma_embedding": os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "llm_mode": "REAL" if _initialized and not _components.get("llm_router", None) or (
            _initialized and not _components["llm_router"].is_mock
        ) else "MOCK",
    })


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
