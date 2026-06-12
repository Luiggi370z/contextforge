# ContextForge — task runner (https://github.com/casey/just)

# Infra
up:
    docker compose up -d postgres qdrant redis

down:
    docker compose down

stack:
    docker compose up --build

migrate:
    cd backend && uv run alembic upgrade head

seed:
    cd backend && uv run python ../scripts/seed_corpus.py

# Dump documents + chunks (Postgres) joined with embeddings (Qdrant) -> scripts/db_dump.json
dump *args:
    cd backend && uv run python ../scripts/dump_db.py {{args}}

# Wipe the corpus from Postgres + Qdrant. Pass --include-threads / --yes via `just wipe --yes`
wipe *args:
    cd backend && uv run python ../scripts/wipe_db.py {{args}}

# Full reset: truncate EVERY Postgres table (keeps schema) + drop EVERY Qdrant collection
nuke *args:
    cd backend && uv run python ../scripts/wipe_db.py --all {{args}}

demo:
    bash scripts/demo.sh

eval-dry:
    cd backend && uv run python ../eval/run_ragas.py --dry-run

eval:
    cd backend && uv run python ../eval/run_ragas.py

eval-heuristic:
    cd backend && uv run python ../eval/run_ragas.py --heuristic-only

# Backend
test:
    cd backend && uv run pytest -q -n 8

# End-to-end golden-set eval (heuristic provider, in-memory corpus, deterministic)
test-eval:
    cd backend && uv run pytest -q -m eval

lint:
    cd backend && uv run ruff check . && uv run pyright

format:
    cd backend && uv run ruff format .

api-dev:
    cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Async ingestion worker (needs redis + postgres up)
worker:
    cd backend && uv run arq app.workers.settings.WorkerSettings

# Frontend
web-install:
    cd web && pnpm install

web-dev:
    cd web && pnpm dev

web-test:
    cd web && pnpm test

web-lint:
    cd web && pnpm lint

web-format:
    cd web && pnpm format

# All checks (CI locally)
check: lint web-lint test web-test
