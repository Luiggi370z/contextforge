# Coding standards — ContextForge

Project conventions for backend (Python) and frontend (TypeScript). Apply on every change; refactor existing code when you touch a file.

## Workflow

- Work **one plan TODO at a time**, then **pause** for review and your own commit.
- Do not batch unrelated TODOs in a single commit unless you choose to.

## Frontend state (React)

When one feature owns **many related fields** that change together (e.g. chat: threads, messages, loading, stream stage), prefer:

| Approach | When to use |
|----------|-------------|
| **`useReducer` + pure reducer** | Single screen/hook; state transitions are event-driven (send, load thread, stream token). **Default for ContextForge feature hooks.** |
| **One `useState` per independent concern** | Isolated toggles (e.g. `showDebug`) that do not participate in the same async flow. |
| **Jotai / Zustand / Context** | State shared across **many** routes/components, or needed outside React. **Avoid** for a single `Chat` tree (YAGNI). |

Example: `web/src/hooks/chatReducer.ts` + `useChat.ts` (reducer for chat domain, separate `useState` only for debug panel visibility).

Do **not** add a global store until a second consumer appears (e.g. header + chat both need live thread id).

## Imports

- **Module-level imports only** — put imports at the top of the file.
- **No inline imports inside functions** unless there is a **proven circular import** (document why in a one-line comment).
- Do **not** use inline imports to lazy-load optional packages (e.g. `from pydantic_ai import Agent` inside a method). Prefer:
  - top-level `try/except ImportError` with a clear runtime error (see `app/retrieval/embeddings.py`), or
  - a **provider submodule** with third-party imports at the top of that file only (see `app/llm/instructor_route.py`, `app/llm/pydantic_ai_route.py`, `app/llm/ollama_provider.py`).

```python
# Avoid — third-party import inside a function
async def decide_route(message: str) -> RouteDecision:
    from pydantic_ai import Agent
    ...

# OK — optional provider isolated in its own module
# app/llm/pydantic_ai_route.py
from pydantic_ai import Agent

# app/llm/structured.py (imports local package only, not third-party provider SDKs)
async def _decide_route_pydantic_ai(message: str) -> RouteDecision:
    from app.llm.pydantic_ai_route import decide_route
    return await decide_route(message)
```

## Naming

### API JSON casing (backend ↔ frontend)

| Layer | Field naming | Example |
|-------|----------------|---------|
| **Python (API models)** | `snake_case` | `thread_id`, `content_type`, `nodes_visited` |
| **JSON over HTTP** | `camelCase` | `threadId`, `contentType`, `nodesVisited` |
| **TypeScript types** | `camelCase` | same as JSON |

- All **HTTP request/response** Pydantic models inherit from [`BaseRequest`](backend/app/schemas/base.py) / [`BaseResponse`](backend/app/schemas/base.py) (via `WithAliasModel` + `alias_generator=to_camel`).
- **`populate_by_name=True`** — accepts camelCase from the React client and snake_case in pytest/internal callers.
- **Internal-only models** (`app/llm/models.py`, graph state, etc.) stay plain `BaseModel` — not exposed on the wire.
- Manual JSON (`SSE`, exception handlers): use `model_dump(mode="json", by_alias=True)`.

```python
# app/schemas/base.py
class BaseResponse(WithAliasModel):
    """Serializes snake_case Python fields as camelCase JSON keys."""
```

```typescript
// web/src/types.ts — camelCase only
export interface QueryResponse {
  threadId?: string;
  metadata: QueryMetadata;
}
```

- **No single-character variable names** (except accepted idioms: `_` for unused, `i`/`j` only in trivial numeric loops if unavoidable — prefer `index`, `chunk`, `document`).
- Use descriptive names: `session` not `db` in broad scope; `chunk` not `c`; `query_text` not `q`.

## Docstrings

- **Public** functions, classes, and modules: short docstring (one–three sentences).
- **Include an example** when behavior is non-obvious or has a specific input/output shape.

```python
def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked lists with Reciprocal Rank Fusion.

    Example:
        >>> reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        [("b", ...), ("a", ...), ("c", ...)]
    """
```

- Private helpers (`_foo`): docstring optional; add if logic is non-trivial.

## Types over plain dicts

Prefer **Pydantic v2 models** or **`@dataclass`** over `dict[str, Any]` for:

- API request/response bodies (`app/schemas/`)
- LLM structured outputs (`app/llm/models.py`)
- Retrieval hits, graph node outputs, config payloads

**Exceptions (OK as dict/TypedDict):**

- LangGraph `GraphState` (`TypedDict`) where the framework expects it
- JSON metadata columns (`metadata_` on ORM rows)
- Third-party APIs that only accept dicts (convert at the boundary)

Pattern: parse early → model → pass models inward; `model_dump()` only at DB/JSON edges.

```python
# Good
class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float

# Avoid in application logic
documents: list[dict]  # use list[RetrievedChunk] or list[ChunkPayload]
```

## Design principles

| Principle | Practice here |
|-----------|----------------|
| **DRY** | Shared retrieval/LLM helpers in one module; one Pydantic model per concept |
| **YAGNI** | No abstractions until a second use case (e.g. don’t add plugin systems for one provider) |
| **SOLID / separation** | Layered: `api/` → `services/` → `repositories/`; domain modules (`ingestion/`, `retrieval/`, `graph/`, `llm/`) stay HTTP-free |
| **File size** | **&lt; 300 lines per file**; split when approaching limit (e.g. graph nodes vs builder) |

## Constants (no magic strings or numbers)

- Put literals in [`backend/app/core/constants.py`](backend/app/core/constants.py): route names, limits, statuses, client error messages, RRF `k`, etc.
- Import constants in services, routes, and graph code — do not inline `100`, `"/v1"`, or `"processing"` in business logic.

## API layout (`api/v1/<entity>/`)

Each HTTP resource is a **folder** under `backend/app/api/v1/`:

```
api/v1/documents/
  router.py        # FastAPI routes only
  service.py       # business logic
  repository.py    # SQLAlchemy (when persisted)
  schemas.py       # Pydantic request/response
  models.py        # dataclasses / internal DTOs (optional)
  dependencies.py  # Depends() factories
```

Entities today: `documents`, `threads`, `query`, `health`, `metrics`. Global error schema stays in `app/schemas/errors.py`.

Wire routers from [`backend/app/router.py`](backend/app/router.py) via each package’s `router` export.

## API layering (repository → service → route)

| Layer | Responsibility | Raises |
|-------|----------------|--------|
| **`api/v1/<entity>/repository.py`** | SQLAlchemy / persistence only | DB/driver errors only (no HTTP) |
| **`api/v1/<entity>/service.py`** | Business rules, orchestration | **`DomainError`** subclasses (e.g. `DocumentNotFoundError`, `IngestionError`) |
| **`api/v1/<entity>/router.py`** | HTTP mapping, Pydantic responses | **`AppException`** only (map domain errors with `domain_error_to_app_exception`) |

- Global handlers live in [`backend/app/core/exception_handlers.py`](backend/app/core/exception_handlers.py); register from `create_app()`.
- **`AppException`** carries `status_code`, `detail`, and `correlation_id` (from request context when present).
- Unhandled exceptions → `500` + generic `ErrorResponse` (no internal leak).

## App wiring

- **[`backend/app/router.py`](backend/app/router.py)** — `build_api_router()` mounts all v1 routers (single entry for HTTP routes).
- **[`backend/app/startup.py`](backend/app/startup.py)** — `application_lifespan`: logging, LangGraph checkpointer, structlog checkpoints (`application_starting`, `langgraph_checkpointer_attached`, `application_shutdown`).
- **[`backend/app/main.py`](backend/app/main.py)** — `create_app()` only: middleware, exception handlers, include router.

When adding a new resource, copy the **documents** folder layout under `api/v1/<entity>/`.

## Database session (request scope)

**Today (keep this):** Routes acquire the session with `Depends(get_db)` and pass `session` into each service/repository call. Repositories never open their own sessions.

```python
@router.get("")
async def list_documents(
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    result = await document_service.list_documents(session)
```

**Future (optional refactor):** Inject the session into the service constructor via `Depends` in `dependencies.py`, so routes stay thinner and services hold `self._session` for the request. Repositories still receive `session` per method (or a shared unit-of-work) — do not hide sessions inside repos.

```python
# Planned pattern — not implemented yet (TODO: session-in-service-di)
def get_document_service(
    session: AsyncSession = Depends(get_db),
) -> DocumentService:
    return DocumentService(session)  # not @lru_cache — one service instance per request
```

Non-HTTP entry points (LangGraph, scripts) create a session at their own boundary and pass it in, same as a route would.

## Stack reminders

- **Backend:** pytest, pyright, ruff, structlog, Pydantic v2, Pydantic AI / Instructor for LLM outputs
- **Frontend:** pnpm, Biome, Vitest, Tailwind v4
- **Tasks:** `just` (not Make)

## Refactor backlog (existing code)

When resuming TODOs, align touched files with this doc:

- Move graph `run_query` body into `api/v1/query/service.py` (runner stays graph-only)
- **Postgres retrieval profile** — implement items in `app/retrieval/factory.py` (grep `TODO(retrieval-backend)`)
- **UI retrieval backend toggle** — `TODO(retrieval-backend-ui)` in `factory.py`; settings API + React control
- **Session-in-service DI** — `Depends(get_db)` → `DocumentService(session)` factories; remove `session` arg from service methods (grep `session-in-service-di`)
- **Inline imports** — audit any remaining function-level imports; allow only documented circular-import exceptions
- Replace `list[dict]` document/citation payloads with Pydantic models in graph + runner
- Rename short locals (`cfg`, `c`, `d`, `s`, `k` in non-trivial code) on edit
- Add docstrings + examples to public retrieval/graph/LLM functions
- Split [`web/src/components/Chat.tsx`](web/src/components/Chat.tsx) if it grows past 300 lines
