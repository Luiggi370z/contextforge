# Running ContextForge fully offline

Same code, different env — ContextForge runs on hosted APIs or 100% locally on a Mac.

## Fully local (no external API calls)

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma2
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_BACKEND=bge-m3      # dense (1024) + sparse from ONE local model (FlagEmbedding BGE-M3)
EMBEDDING_DIM=1024
RERANK_BACKEND=cross_encoder  # local cross-encoder
```

Steps:
1. `ollama pull gemma2`
2. `uv sync --extra ml`            # FlagEmbedding (BGE-M3) + sentence-transformers + fastembed
3. `just up` (postgres + qdrant + redis)
4. `just migrate && just seed`     # seeds the 1024-dim collection (contextforge_1024)
5. `just worker` and `just api-dev`

Switching `EMBEDDING_DIM` creates a new dimension-keyed Qdrant collection
(`contextforge_<dim>`), so re-seed after changing it. BGE-M3 downloads ~2.3GB on
first use.

## Hosted (OpenAI)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_DIM=384
RERANK_BACKEND=cross_encoder
```

## Deterministic CI (what `pytest -m eval` uses)

`EMBEDDING_BACKEND=sentence-transformers`, `LLM_PROVIDER=heuristic`, 384-dim — pinned
by the test fixture regardless of surrounding env. Collection `contextforge_384`.
