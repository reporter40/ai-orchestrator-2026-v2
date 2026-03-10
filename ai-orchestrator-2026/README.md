# AI Orchestrator 2026

## Overview
AI Orchestrator 2026 is a self-improving, multi-agent system designed to optimize and refine prompts through a sophisticated orchestration loop. It utilizes a Mixture of Experts (MoE) architecture, integrated evaluation benchmarks, and an adaptive learning engine.

### Key Features (V2 Adaptive Engine)
- **Agent Registry**: Dynamic management of specialized agents (Creative, Structural, Tone, Security) stored in ChromaDB.
- **Self-Improvement Loop**: Automated agent instruction optimization based on performance feedback using DSPy.
- **Parallel MoE**: Simultaneous generation of draft variations by multiple experts.
- **Red-Teaming**: Integrated `AdversaryAgent` for security and edge-case testing.
- **Hybrid Memory**: Vector-based semantic search (ChromaDB) for RAG context and approved prompt versions.
- **Modern Web UI**: Real-time orchestration monitoring, agent management, and version history.

## Tech Stack
- **Languages**: Python 3.12, JavaScript (Vanilla)
- **Frameworks**: FastAPI, LiteLLM, DSPy
- **Database**: ChromaDB (Vector), SQLite (Relational)
- **Observability**: OpenTelemetry
- **Styling**: CSS with modern glassmorphism and responsiveness

## Quick Start

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Server
```bash
python server.py
```
Open `http://localhost:8000` in your browser.

### Using the CLI
```bash
python main.py --interactive
```

## Architecture
The system follows a state-machine based orchestration flow:
1. **Router**: Analyzes the request and selects the best agents/strategy.
2. **MoE Pool**: Parallel agents generate draft variations.
3. **Evaluator**: Judges the output against requirements and RAG context.
4. **Optimizer**: Rewrites prompts or improves agent instructions if scores are low.
5. **Red-Team**: Final security audit before delivery.

## Agent Management
You can manage the **Agent Registry** directly from the "Agents" tab in the Web UI or via CLI:
- List agents: `python main.py --list-agents`
- Add agent: `python main.py --add-agent '{"name": "...", "role": "...", ...}'`

---
*Created by Antigravity (Google Deepmind) for the AI Orchestrator 2026 Project.*
