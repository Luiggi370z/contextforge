# ContextForge

ContextForge is an agentic RAG portfolio project: FastAPI + LangGraph orchestration, hybrid retrieval (Qdrant dense + BM25 + RRF), and a React chat UI with debug visibility.

- Plan: [PLAN.md](PLAN.md)
- Ollama integration (planned): [OLLAMA_INTEGRATION.md](OLLAMA_INTEGRATION.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Coding standards: [CODING_STANDARDS.md](CODING_STANDARDS.md)

## Why this project

This repo demonstrates production-minded RAG behaviors:
- route selection (`direct`, `single_hop_rag`, `multi_hop`)
- retrieval grading + abstain path
- grounded answers with citations
- measurable quality with RAGAS/heuristic eval runs
- full-stack developer workflow (API + UI + tests + CI)

## Quick start

Prereqs: [Docker](https://docs.docker.com/), [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), [pnpm](https://pnpm.io/).

```bash
cp .env.example .env
just up
just migrate
cd backend && uv sync --extra dev
just api-dev                 # terminal 1: http://localhost:8000/docs
just web-install && just web-dev  # terminal 2: http://localhost:5173
just seed                    # terminal 3, optional: load sample docs
```

Default Qdrant hybrid retrieval dependencies (MiniLM dense embeddings + FastEmbed BM25)
are installed by the base backend package. Use `uv sync --extra ml` only for BGE-M3
local-first mode.

## Run with Ollama

Provider selection is environment-only for now (no per-request or UI toggle).

1. Start Ollama and pull a model:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```
2. In `.env`, set:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```
3. Start the API + web app as usual:
   ```bash
   just api-dev
   just web-dev
   ```

## Demo (2 minutes)

Use the helper script:

```bash
just demo
```

Or run manually:
1. Open `http://localhost:5173` and keep Debug on.
2. Ask `hi` (direct route).
3. Ask `How many PTO days do full-time employees accrue per year?` (single-hop RAG).
4. Ask `Compare PTO policy steps with remote work approval steps` (multi-hop).
5. Ask `What is the lunar habitat budget for 2099?` (abstain path).

## API contract (important)

- Backend Python fields are `snake_case`.
- JSON contract is `camelCase` via `BaseRequest`/`BaseResponse` aliases.
- Frontend TypeScript models are `camelCase`.

Example request body:

```json
{
  "message": "How many PTO days?",
  "threadId": "00000000-0000-0000-0000-000000000002"
}
```

## Evaluation

```bash
just eval-dry         # validate golden set only
just eval-heuristic   # no OpenAI, lexical metrics
just eval             # full RAGAS (requires OPENAI_API_KEY + running API)
```

Reports are written under `reports/` (gitignored).

## Useful commands

```bash
just check         # backend + web lint/tests
just lint          # backend: ruff + pyright
just test          # backend: pytest
just web-lint      # web: biome
just web-test      # web: vitest
```

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI, Pydantic v2, structlog, SSE |
| Agent | LangGraph `StateGraph` |
| Retrieval | Qdrant dense + BM25 + RRF + rerank |
| DB | PostgreSQL 16 (host port `5434`) |
| UI | Vite, React, Tailwind v4, Biome, Vitest, pnpm |
| Eval | RAGAS + heuristic metrics (`eval/`) |
| Tasks | just |

## Project layout

```text
contextforge/
├── backend/         # FastAPI + graph/retrieval; ships app/eval (harness, golden set, corpus)
├── web/             # React chat UI
├── eval/            # dev-only RAGAS/heuristic reporters (hit live API, write reports/)
├── scripts/         # seed + demo helpers
├── PLAN.md
└── justfile
```
