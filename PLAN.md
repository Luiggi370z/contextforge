# ContextForge — Implementation Plan

> Canonical plan for this repo. Parent folder: `Personal/ai-projects/contextforge/`.
> Updated: 2026-05-26

## Conventions

- **Backend (Python):** **pytest** (tests), **pyright** (type checking), **ruff** (lint + format), **structlog** (logging), **Pydantic v2** (all schemas/settings). Structured LLM outputs via **Pydantic AI** and/or **Instructor** (shared v2 models). Managed with **uv**.
- **Frontend (UI):** **pnpm** only (never npm), **Biome** (lint + format), **Vitest** (unit/component tests), **Tailwind CSS v4** (styling via `@tailwindcss/vite`).
- **Task runner:** **[just](https://github.com/casey/just)** (`justfile` at repo root) — not Makefile. Requires `just` installed locally.
- **Database:** local PostgreSQL via Docker Compose (host port `5434`; `5432` is often used by other projects).
- **Docs:** this file lives at repo root for README-linked documentation. Backlog items remain local under ignored `docs/` unless promoted.
- **Git:** root [`.gitignore`](.gitignore) — Python, Node, env secrets, IDE, caches, local data (see [`.gitignore`](#gitignore)).

---

# ContextForge: Agentic RAG Portfolio Project

## Recommendation (strongest for hiring in 2026)

Ship a **domain-agnostic Agentic Document Intelligence platform** — FastAPI backend plus a **Vite + React chat UI** (not Streamlit). Recruiters and hiring managers in 2026 repeatedly surface the same gap: candidates who followed a Chroma + LangChain tutorial vs. candidates who can explain **retrieval quality, orchestration, evaluation, full-stack integration, and production concerns**.

Your existing work on [Amaru/nexus](Amaru/nexus) (FastAPI, agents, structured backends) is a differentiator. This project should **reuse that muscle** (async API, tests, Docker) while adding the GenAI-specific layers employers ask for: **hybrid search, reranking, agentic routing, grounded answers with citations, and measurable quality (RAGAS)**.

**Inspiration (study, do not fork blindly):**

- [cbratkovics/rag-pipeline](https://github.com/cbratkovics/rag-pipeline) — hybrid RRF, RAGAS, Prometheus, load tests
- [saimdinky/fastapi-rag](https://github.com/saimdinky/fastapi-rag) — FastAPI + Qdrant hybrid + rerank + swappable providers
- [Akansha051991/AI_Research_Agent-LangGraph-RAG](https://github.com/Akansha051991/AI_Research_Agent-LangGraph-RAG) — LangGraph + tools + checkpoints (good patterns; avoid “kitchen sink” tools unless you implement them well)
- Incident RCA write-up on DEV — LangGraph multi-node workflow narrative ([DEV article](https://dev.to/zeroshotanu/how-i-built-an-ai-powered-incident-rca-platform-with-langgraph-and-rag-423j))

---

## Why this beats a naive RAG tutorial


| What tutorials show               | What ContextForge demonstrates                                                |
| --------------------------------- | ----------------------------------------------------------------------------- |
| Fixed retrieve → generate chain   | **LangGraph** state machine with conditional edges                            |
| Vector-only search                | **Hybrid** dense + BM25, **RRF** fusion, **cross-encoder rerank**             |
| Answers even when context is weak | **CRAG/Self-RAG-style** grading: refuse or reformulate when retrieval is poor |
| No metrics                        | **RAGAS** faithfulness / context precision / answer relevancy on a golden set |
| Notebook or Streamlit only        | **FastAPI** + **Vite/React** chat UI + SSE + **Docker Compose** + CI          |
| Black box                         | **Citations** (doc id, page/chunk, score) + optional **Langfuse** traces      |


Industry direction (2025–2026): production RAG is **hybrid retrieval + reranking + agentic planning**, not bigger context windows alone ([Neo4j advanced RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/), [Atlan 12 techniques](https://atlan.com/know/advanced-rag-techniques/)).

---

## Product scope (agentic tier — your choice)

**One-liner for README/resume:**
*“Agentic RAG API that routes queries, retrieves with hybrid search + reranking, grounds answers with citations, and scores itself with RAGAS.”*

### Core user stories (MVP that still feels “agentic”)

1. **Ingest** PDF/Markdown/TXT via API; chunk with structure-aware splitting (headings, page metadata).
2. **Ask** a question; receive **streamed** answer with **inline citations**.
3. **Router** classifies: `direct` (no retrieval), `single_hop_rag`, `multi_hop` (decompose → sub-queries → merge).
4. **Grader node** checks retrieval quality; if below threshold → “I don’t have enough information” (no hallucination).
5. **Eval CLI** runs RAGAS on a committed golden JSONL; results written to `reports/` and compared across git tags.

### Stretch (post-MVP, still resume-worthy)

- **HITL**: LangGraph `interrupt_before` on ingest approval or low-confidence answers.
- **Semantic cache** (Redis) for repeated queries.
- **A/B retrieval profiles** (e.g. `k=8` vs `k=20`, with/without rerank) compared via RAGAS.
- **React debug panel** in chat UI: show `route`, retrieval scores, and graph branch taken (helps manually test direct / RAG / multi-hop / abstain paths).

### Demo corpus (domain-agnostic)

Bundle a small **public sample corpus** so reviewers can run without your private files:

- 10–20 short Markdown “policy” docs you author, **or**
- A tiny subset of public docs (e.g. a few pages from a well-known open handbook)

Avoid depending on paid APIs for the default path; support **OpenAI-compatible** + **local** (Ollama) via env config.

---

## Architecture

```mermaid
flowchart TB
  subgraph client [Clients]
    Web[Vite React chat UI]
    CLI[CLI ingest and eval]
  end

  subgraph api [FastAPI service]
    IngestAPI[POST /v1/documents]
    QueryAPI[POST /v1/query stream]
    ThreadsAPI[GET/POST /v1/threads]
  end

  subgraph graph [LangGraph orchestrator]
    Router[Route intent]
    Decompose[Query decompose optional]
    Retrieve[Hybrid retrieve]
    Rerank[Cross-encoder rerank]
    Grade[Context grader]
    Generate[Grounded generate]
    Validate[Answer validator]
  end

  subgraph data [Data layer]
    Qdrant[(Qdrant hybrid collection)]
    PG[(PostgreSQL)]
  end

  subgraph obs [Observability]
    RAGAS[RAGAS eval pipeline]
    Langfuse[Langfuse traces optional]
  end

  Web -->|SSE + REST| api
  CLI --> api
  QueryAPI --> graph
  IngestAPI --> Qdrant
  IngestAPI --> PG
  Router --> Decompose --> Retrieve --> Rerank --> Grade --> Generate --> Validate
  Retrieve --> Qdrant
  graph --> PG
  RAGAS --> QueryAPI
```



**Graph nodes (minimum viable agentic loop):**

1. `route` — LLM or classifier: direct / rag / multi_hop
2. `retrieve` — parallel dense (Qdrant) + sparse (BM25 via `rank_bm25` or Qdrant sparse vectors) → **RRF**
3. `rerank` — cross-encoder top-N (e.g. `BAAI/bge-reranker-base` or smaller for laptop)
4. `grade_context` — binary/ternary relevance; branch to regenerate query (once) or abstain
5. `generate` — prompt with numbered sources; require citation markers
6. `validate_answer` — check claims map to sources; optional second-pass fix or abstain

State: `TypedDict` with `messages`, `documents`, `retrieval_scores`, `route`, `citations`.

---

## Modern stack

| Layer         | Choice                                                                     | Resume keyword                 |
| ------------- | -------------------------------------------------------------------------- | ------------------------------ |
| Language      | Python 3.11+ (backend), TypeScript (frontend)                              | full-stack                     |
| API           | **FastAPI**, **Pydantic v2** schemas, SSE streaming                        | async, OpenAPI                 |
| LLM structured output | **Pydantic AI** + **Instructor** + shared Pydantic v2 result models    | typed LLM outputs              |
| Settings      | **pydantic-settings** `BaseSettings`                                       | env-based config               |
| Frontend      | **Vite + React** chat app (`web/`), **pnpm only** (no npm)                 | SSE client, product UX         |
| JS tooling    | **pnpm** — lockfile `pnpm-lock.yaml`, `packageManager` in `package.json`   | consistent installs            |
| Orchestration | **LangGraph** + LangChain integrations                                     | agentic workflows, checkpoints |
| Relational DB | **PostgreSQL 16** — local only via **Docker Compose** (demo-friendly, no cloud DB) | SQLAlchemy 2, Alembic |
| Vector DB     | **Qdrant** (hybrid dense + sparse in one collection)                       | production vector search       |
| Embeddings    | `sentence-transformers` / `fastembed` (local default)                      | cost-aware design              |
| Sparse        | BM25 or Qdrant sparse vectors                                              | hybrid retrieval               |
| Fusion        | **RRF**                                                                    | industry standard              |
| Rerank        | Cross-encoder (`sentence-transformers` CrossEncoder)                       | precision@k                    |
| LLM           | Pluggable: Ollama (default) + OpenAI-compatible env                        | provider abstraction           |
| Persistence   | Postgres: documents, chunks metadata, **threads/messages**, LangGraph checkpoints via `langgraph-checkpoint-postgres` | session memory |
| Eval          | **RAGAS** + small golden dataset in repo                                   | LLM-as-judge metrics           |
| Logging       | **structlog** (sole app logger; JSON in prod, console in dev)               | structured logs, context       |
| Observability | **Langfuse** (optional); `/metrics` stub or Prometheus                     | traces and metrics             |
| Packaging     | **uv**, **Docker Compose**, GitHub Actions                                 | engineering hygiene            |
| Backend QA    | **pytest**, **pyright**, **ruff** (`check`, `format`)                      | tested, typed Python           |
| Frontend QA   | **Vitest**, **Biome** (`check`, `format`)                                  | tested, consistent UI code     |
| CSS           | **Tailwind CSS v4** (`@tailwindcss/vite`)                                  | utility-first styling          |

### Python tooling (backend)

| Tool | Role | Config / commands |
|------|------|-------------------|
| **pytest** | Unit + API tests (`tests/`) | `[tool.pytest.ini_options]` in `backend/pyproject.toml` · `uv run pytest` |
| **pyright** | Static types for `app/` | `backend/pyrightconfig.json` or `[tool.pyright]` in `pyproject.toml` · `uv run pyright` |
| **ruff** | Lint + format | `[tool.ruff]` / `[tool.ruff.lint]` in `pyproject.toml` · `uv run ruff check .` · `uv run ruff format .` |

- Dev deps: `pytest`, `pytest-asyncio`, `pyright`, `ruff` under `[project.optional-dependencies] dev`.
- **just** recipes: `just test`, `just lint`, `just format` (see [Justfile recipes](#justfile-recipes)).

### Logging (backend — structlog)

**structlog** is the only logging API used in application code (`app/`). Do not use `print`, ad-hoc `logging.getLogger`, or unstructured log strings.

| Piece | Location / behavior |
|-------|---------------------|
| Setup | `app/core/logging.py` — `configure_logging(level)` called from FastAPI `lifespan` |
| Logger | `import structlog` → `log = structlog.get_logger(__name__)` per module |
| Dev | `structlog.dev.ConsoleRenderer()` when `APP_ENV=development` |
| Prod | `structlog.processors.JSONRenderer()` when `APP_ENV` is not development (machine-parseable logs) |
| Context | `log.bind(request_id=..., thread_id=..., route=...)` in middleware and LangGraph nodes |
| Stdlib bridge | `logging.basicConfig` only to satisfy third-party libs; app code uses structlog only |

Example:

```python
import structlog

log = structlog.get_logger(__name__)

async def ingest_document(...):
    log.info("ingest_started", filename=filename, document_id=str(doc_id))
    ...
    log.warning("ingest_retry", attempt=attempt, error=str(exc))
```

Dependency: `structlog>=24.0` in `backend/pyproject.toml` (already listed).

### Schemas and structured LLM outputs (Pydantic v2 + Pydantic AI + Instructor)

**Pydantic v2** is mandatory for all request/response models, graph state DTOs exposed to the API, and settings. Use v2 idioms only: `model_config = ConfigDict(...)`, `Field(...)`, `model_dump()` / `model_validate()`, `@field_validator` — not v1 `class Config` or `.dict()`.

| Layer | Tool | Location |
|-------|------|----------|
| HTTP API | Pydantic v2 `BaseModel` | `app/schemas/` — `QueryRequest`, `QueryResponse`, `Citation`, etc. |
| Config | `pydantic-settings` | `app/core/config.py` |
| LLM output models | Pydantic v2 `BaseModel` (shared) | `app/llm/models.py` — `RouteDecision`, `RetrievalGrade`, `AnswerValidation` |
| LangGraph state | `TypedDict` for graph; validated outputs stored as `model_dump()` in metadata |
| SQLAlchemy | ORM in `app/db/models.py` — separate from API schemas |

**One model, two clients.** Define each structured output **once** as a Pydantic v2 model, then call it through **Pydantic AI** or **Instructor** — never duplicate schema definitions.

| Node / feature | Shared model (example) |
|----------------|------------------------|
| `route` | `RouteDecision(route: Literal["direct", "single_hop_rag", "multi_hop"], ...)` |
| `grade_context` | `RetrievalGrade(relevant: bool, score: float, should_abstain: bool)` |
| `validate_answer` | `AnswerValidation(grounded: bool, issues: list[str])` |
| Query decompose (stretch) | `SubQueries(queries: list[str])` |

#### When to use Pydantic AI vs Instructor

| Use **Pydantic AI** | Use **Instructor** |
|---------------------|-------------------|
| Multi-turn agent step with built-in retries / message history | Single-shot completion → structured object |
| `Agent(..., output_type=Model).run(user_prompt)` | `client.chat.completions.create(..., response_model=Model)` |
| Default for graph nodes that may grow (tools, deps later) | Fast path for router / grader when provider is OpenAI-compatible or Ollama via patched client |

**Rules:**
- Pick **one** library per graph node (not both in the same node).
- Centralize provider wiring in `app/llm/` (`factory.py`, `agents.py`, `instructor_client.py`).
- **Not required for:** health, CRUD, RRF/rerank, Qdrant/Postgres — plain Python only.
- Never hand-parse JSON from LLM text; always `response_model` / `output_type`.

**Dependencies** (Phase 2): `pydantic-ai`, `instructor` in `backend/pyproject.toml`. Run `uv lock` and resolve any provider pin conflicts (e.g. OpenAI SDK version).

**Pydantic AI example:**

```python
from pydantic_ai import Agent
from app.llm.models import RouteDecision

router_agent = Agent("openai:gpt-4o-mini", output_type=RouteDecision)
result = await router_agent.run(user_message)
decision: RouteDecision = result.output
```

**Instructor example:**

```python
import instructor
from openai import AsyncOpenAI
from app.llm.models import RouteDecision

client = instructor.from_openai(AsyncOpenAI())
decision: RouteDecision = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_message}],
    response_model=RouteDecision,
)
```

**LangGraph:** nodes write the validated Pydantic model into graph state; FastAPI maps state → `QueryResponse`.

### Frontend tooling (web)

| Tool | Role | Config / commands |
|------|------|-------------------|
| **pnpm** | Package manager | `web/package.json` + `pnpm-lock.yaml` · never npm |
| **Biome** | Lint + format (replaces ESLint + Prettier) | `web/biome.json` · `pnpm exec biome check .` · `pnpm exec biome format --write .` |
| **Vitest** | Unit / component tests | `web/vitest.config.ts` + `@testing-library/react` · `pnpm test` |
| **Tailwind v4** | Styling | `@tailwindcss/vite` in `vite.config.ts` · `web/src/index.css` with `@import "tailwindcss"` |

- Scaffold: `pnpm create vite web --template react-ts`, then add Tailwind v4, Biome, Vitest per official docs.
- **just** recipes: `just web-dev`, `just web-test`, `just web-lint`.

### PostgreSQL strategy (local demo)

- **Single environment:** `docker-compose.yml` runs **Postgres 16** with a named volume; no managed cloud DB required for reviewers or interviews.
- **Connection:** `DATABASE_URL=postgresql+asyncpg://contextforge:contextforge@localhost:5434/contextforge` on host (Compose maps `5434:5432`; in-network services use `postgres:5432`).
- **Bootstrap:** Compose `depends_on` + healthcheck; optional `scripts/wait-for-postgres.sh` or retry in Alembic migrate on first `compose up`.
- **Single source of truth in Postgres:** `documents`, `ingestion_jobs`, `chunks` (metadata + Qdrant point ids), `threads`, `messages`, optional `eval_runs`.
- **Vectors stay in Qdrant** — Postgres stores relational metadata and foreign keys to Qdrant point IDs; pgvector is out of scope for v1.
- **LangGraph:** Postgres checkpointer (`AsyncPostgresSaver`) so conversation state survives container restarts.
- **Later (optional, not v1):** swap `DATABASE_URL` to any hosted Postgres if you deploy publicly — same schema, no code change.

### React chat UI (path testing)

Purpose: manual end-to-end testing of **all graph paths**, not just a pretty demo.

- **Chat view:** message list, input, streamed assistant tokens (EventSource / `fetch` + ReadableStream).
- **Citations:** expandable source cards (doc title, chunk snippet, score).
- **Upload:** drag-and-drop PDF/Markdown → `POST /v1/documents`; show ingestion status from Postgres job row.
- **Debug drawer** (toggle): display API metadata per turn — `route` (`direct` | `single_hop_rag` | `multi_hop`), `abstained`, `retrieval_scores`, nodes visited — so you can deliberately prompt each branch during development.
- **Thread picker:** list threads from `GET /v1/threads`; resume via `thread_id` header or body.
- **Dev proxy:** Vite `server.proxy` → FastAPI `:8000` to avoid CORS pain locally.

### Frontend package manager (pnpm only)

- Use **pnpm** for all Node workflows — never `npm install`, `npm run`, or `npx`.
- Scaffold: `pnpm create vite web --template react-ts` (from repo root) or `cd web && pnpm init` + manual Vite setup.
- Install: `pnpm install` · Dev: `pnpm dev` · Build: `pnpm build` · CI: `pnpm install --frozen-lockfile`.
- Commit `pnpm-lock.yaml`; add `"packageManager": "pnpm@9.x"` to `web/package.json` when the app is created.

**Deliberately skip for v1 (add only if time):** Kubernetes, Ray Serve, Pinecone paid tier, GraphRAG/Neo4j — high complexity, weaker ROI for first portfolio ship.

### Justfile recipes

Root [`justfile`](justfile) replaces Makefile. Example recipes to implement:

```just
# Infra
up:       docker compose up -d postgres qdrant
down:     docker compose down
stack:    docker compose up --build
migrate:  cd backend && uv run alembic upgrade head

# Backend
test:     cd backend && uv run pytest -q
lint:     cd backend && uv run ruff check . && uv run pyright
format:   cd backend && uv run ruff format .
api-dev:  cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
web-install: cd web && pnpm install
web-dev:     cd web && pnpm dev
web-test:    cd web && pnpm test
web-lint:    cd web && pnpm exec biome check .
web-format:  cd web && pnpm exec biome format --write .

# All checks (CI locally)
check: lint web-lint test web-test
```

README quick start uses `just up`, `just migrate`, `just api-dev`, `just web-dev`.

### `.gitignore`

Root `.gitignore` covers the monorepo (backend + web + local infra). **Never commit** `.env`, API keys, or generated model weights.

```gitignore
# --- Environment & secrets ---
.env
.env.*
!.env.example

# --- Python / backend ---
__pycache__/
*.py[cod]
*$py.class
.Python
.venv/
venv/
env/
*.egg-info/
.eggs/
dist/
build/
*.egg
.pytest_cache/
.mypy_cache/
.pyrightcache/
.ruff_cache/
htmlcov/
.coverage
.coverage.*
*.cover
.hypothesis/
uv.lock.bak

# --- Node / web ---
node_modules/
dist/
dist-ssr/
*.local
.pnpm-store/
.turbo/
coverage/

# --- Vite / Vitest ---
*.tsbuildinfo

# --- IDE & OS ---
.idea/
.vscode/
*.swp
*.swo
.DS_Store
Thumbs.db

# --- Docker / local data ---
postgres_data/
qdrant_data/
.qdrant_storage/

# --- ML / embeddings cache (large) ---
.cache/
models/
*.pt
*.bin
*.onnx

# --- Project outputs ---
reports/
*.log
logs/

# --- Misc ---
*.bak
.cursor/
```

On implementation: keep this file at repo root; extend only if a tool adds new artifact dirs.

---

## Repo layout (new folder under Personal)

Project path: [`Personal/ai-projects/contextforge/`](Personal/ai-projects/contextforge/)

```
contextforge/
├── .gitignore             # Python, Node, env, IDE, caches (see below)
├── README.md
├── ARCHITECTURE.md
├── PLAN.md                # implementation plan (source of truth)
├── justfile               # task runner (just, not make)
├── docker-compose.yml     # api, qdrant, postgres
├── .env.example           # DATABASE_URL, QDRANT_URL, LLM keys
├── backend/
│   ├── pyproject.toml     # pytest, ruff, pyright config
│   ├── pyrightconfig.json
│   ├── alembic/           # migrations
│   ├── app/
│   │   ├── main.py
│   │   ├── core/logging.py  # structlog configure_logging
│   │   ├── api/v1/        # documents, query (SSE), threads, health
│   │   ├── db/            # SQLAlchemy models, session
│   │   ├── graph/         # LangGraph + Postgres checkpointer
│   │   ├── retrieval/
│   │   ├── ingestion/
│   │   ├── llm/             # models.py, pydantic-ai agents, instructor clients
│   │   └── schemas/         # Pydantic v2 API models
│   └── tests/
├── web/                   # Vite + React + TypeScript (pnpm)
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── biome.json
│   ├── vitest.config.ts
│   ├── vite.config.ts     # @tailwindcss/vite + proxy → :8000
│   └── src/
│       ├── index.css      # @import "tailwindcss"
│       ├── components/Chat.tsx
│       ├── components/Citations.tsx
│       ├── components/DebugPanel.tsx
│       └── hooks/useSSE.ts
├── eval/
├── sample_corpus/
└── scripts/
```

---

## Phased delivery (10+ weeks part-time)

### Phase 1 — Retrieval foundation (weeks 1–3)

- Ingestion pipeline: load → chunk (recursive + metadata) → embed → upsert Qdrant
- Hybrid retrieve + RRF + rerank behind a **pure Python function** (unit tested, no LLM yet)
- Golden set: 30–50 Q/A pairs with expected source doc ids

### Phase 2 — LangGraph + Postgres (weeks 4–6)

- Alembic migrations: documents, chunks, threads, messages, ingestion_jobs
- LangGraph graph + **Postgres checkpointer** (`AsyncPostgresSaver`)
- Structured LLM nodes (`route`, `grade_context`, `validate_answer`) via **Pydantic AI** and/or **Instructor** + shared v2 models in `app/llm/models.py`
- FastAPI `POST /v1/query` with SSE; response includes `route` + citation metadata for UI debug panel
- `GET/POST /v1/threads` backed by Postgres

### Phase 3 — React UI + production polish (weeks 7–8)

- Scaffold **Vite + React** (`web/`) with **pnpm**, **Tailwind v4**, **Biome**, **Vitest**
- Chat UI: upload, citations, debug drawer; SSE streaming; thread persistence via API
- Docker Compose: api + qdrant + **postgres** — one command for full local demo stack
- Config via `pydantic-settings`; CORS for Vite dev; abstain path + rate limiting (`slowapi`)
- **structlog** used in API routes, ingestion, retrieval, and LangGraph nodes (no raw `logging` in `app/`)
- **pytest** + **pyright** + **ruff** green on backend; **Vitest** + **Biome** green on `web/`

### Phase 4 — Evaluation & resume package (weeks 9–10)

- GitHub Actions: `uv run pytest`, `uv run ruff check`, `uv run pyright`; `pnpm test`, `pnpm exec biome check`
- RAGAS pipeline in CI (nightly or on PR with smaller subset)
- README: architecture, **metric table** (before/after rerank), **design decisions**
- 2–3 min demo: React UI ingest → chat questions that hit each route → debug panel shows path
- Optional: Langfuse dashboard screenshot for README

### Phase 5 — Stretch

- Multi-hop decomposition node
- HITL interrupt on low grade
- A/B config for retrieval params

---

## Resume bullets (after ship)

- Built an **agentic RAG platform**: **LangGraph** orchestration, **FastAPI** SSE API, **Vite/React** chat UI with route-level debug visibility.
- Implemented **hybrid search** (dense + BM25), **RRF**, and **cross-encoder reranking** on **Qdrant**, with document/thread state in **PostgreSQL**.
- Integrated **RAGAS** evaluation (faithfulness, answer relevancy) in CI; documented abstain behavior when retrieval confidence is low.
- Containerized with **Docker Compose**; Alembic migrations; provider-agnostic LLM layer (local Ollama + cloud-compatible APIs).

Replace **X%** with real numbers from your `eval/reports/` — measured beats buzzwords.

---

## Key challenges to call out in interviews (built into the project)

1. **Chunking quality** — tables, headers, code blocks; parent-child chunk strategy optional.
2. **Retrieval misses exact IDs** — why hybrid beats dense-only.
3. **Latency vs quality** — rerank only top-50 from fusion, not full corpus.
4. **Hallucination** — grader + validator + explicit abstain.
5. **Eval leakage** — hold out questions; version embeddings with collection name.
6. **Cost** — local embeddings default; cache repeated queries.

---

## What to avoid

- Clone-and-tweak a single tutorial repo with no eval or architecture doc.
- Streamlit-only or API with no client (hard to demo integrated product).
- Pinecone + 6 external APIs as required dependencies (reviewers can’t run it).
- Claiming “production” without tests, Docker, or abstain behavior.

---

## Success criteria (definition of done)

- `docker compose up` → React UI ingest sample corpus → streamed answer with citations and debug path visible
- LangGraph diagram in README matches implemented nodes
- Postgres migrations apply cleanly via `just migrate` (or Alembic in api entrypoint on `just stack`)
- RAGAS report committed or generated in CI with baseline scores
- 15+ meaningful **pytest** tests (retrieval + API + graph branches); **pyright** + **ruff** pass in CI
- **Vitest** covers chat hooks/components; **Biome** passes on `web/`
- ARCHITECTURE.md explains 3 tradeoffs (Qdrant vs Postgres roles, hybrid retrieval, abstain vs always-answer)

---

## Optional later bridge to Nexus (not required for v1)

If you want a second narrative thread: add a **“telemetry runbook”** sample pack and a router intent `ops_docs` — ties to your SRE domain without making the whole project DevOps-only. Keep the core domain-agnostic.
