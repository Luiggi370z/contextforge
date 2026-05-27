# ContextForge

Agentic RAG platform for portfolio demos: **LangGraph-style pipeline**, **FastAPI**, **hybrid retrieval** (dense + BM25 + RRF), **PostgreSQL**, **Qdrant**, and a **Vite/React** chat UI with route debug visibility.

**Implementation plan:** [docs/PLAN.md](docs/PLAN.md) · **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

## Quick start

Prerequisites: [Docker](https://docs.docker.com/), [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), [pnpm](https://pnpm.io/).

```bash
cp .env.example .env
just up
just migrate
cd backend && uv sync --extra dev
just api-dev          # terminal 1 — http://localhost:8000/docs
just web-install && just web-dev   # terminal 2 — http://localhost:5173
just seed             # ingest sample_corpus/ (API must be running)
```

## Demo script (~2 min)

1. Open http://localhost:5173 — toggle **Debug** on.
2. Upload or run `just seed` for sample HR policies.
3. Ask **"hi"** → route `direct`.
4. Ask **"How many PTO days per year?"** → `single_hop_rag` + citations.
5. Ask **"Compare PTO policy steps and remote work steps"** → `multi_hop`.
6. Ask about something not in corpus → abstain path.

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI, SSE, Pydantic v2, structlog |
| Agent | Route → retrieve → grade → generate → validate |
| Retrieval | Qdrant + BM25 + RRF + rerank |
| DB | PostgreSQL 16 (Docker, host port **5433**) |
| UI | Vite, React, Tailwind v4, Biome, Vitest, **pnpm** |
| QA | pytest, pyright, ruff · Biome · Vitest |
| Tasks | **just** |

## Commands

```bash
just check      # lint + test (backend + web)
just test       # pytest
just lint       # ruff + pyright
just web-test   # vitest
just seed       # load sample_corpus via API
```

## Evaluation (RAGAS)

```bash
# After seed + API running, with OPENAI_API_KEY set:
cd backend && uv run python ../eval/run_ragas.py --dry-run
cd backend && uv run python ../eval/run_ragas.py
```

Reports are written under `reports/`.

## Project layout

```
contextforge/
├── backend/          # FastAPI + agent pipeline
├── web/              # React chat UI
├── sample_corpus/    # Demo markdown docs
├── eval/             # golden.jsonl + RAGAS runner
├── scripts/          # seed_corpus.py
├── docs/PLAN.md
└── justfile
```

## Configuration

| Variable | Default | Notes |
|----------|---------|--------|
| `DATABASE_URL` | `localhost:5433` | Matches docker-compose host port |
| `EMBEDDING_BACKEND` | `sentence-transformers` | Set `hash` for fast tests |
| `LLM_PROVIDER` | `heuristic` | Set `openai` + `OPENAI_API_KEY` for Instructor routing |
| `GRADE_MIN_SCORE` | `0.25` | Below this → abstain |

## Status

MVP implemented: ingestion, hybrid retrieval, agent pipeline, chat UI, CI, RAGAS runner stub. See [docs/PLAN.md](docs/PLAN.md) for stretch goals (HITL, Redis cache, A/B retrieval).
