#!/usr/bin/env python3
"""Dump documents + chunks (Postgres) joined with their embeddings (Qdrant).

Reads connection info from the same Settings the API uses (``backend/.env`` →
env vars → defaults), so it just works against the running docker-compose
stack without any manual wiring.

Outputs:

- A human-readable summary on stdout (counts + a sample chunk per document).
- A full JSON dump to the path passed via ``--out`` (default
  ``scripts/db_dump.json``) shaped as:

  {
    "summary": {...},
    "documents": [
      {
        "id": "...",
        "filename": "...",
        "status": "...",
        "created_at": "...",
        "chunks": [
          {
            "id": "...",
            "chunk_index": 0,
            "content": "...",
            "metadata": {...},
            "qdrant_point_id": "...",
            "embedding": {
              "dim": 384,
              "preview": [...first 8 dims...],
              "vector": [...]   // only when --full-vectors is passed
            }
          },
          ...
        ]
      },
      ...
    ]
  }

Usage:

  uv run python ../scripts/dump_db.py                 # preview vectors only
  uv run python ../scripts/dump_db.py --full-vectors  # include all 384 dims
  uv run python ../scripts/dump_db.py --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Make ``app.*`` imports work whether the script is invoked from repo root or
# from the backend dir (the justfile runs the latter).
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.orm import selectinload  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.models import Chunk, Document  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


async def _load_postgres() -> list[Document]:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(Document)
                .options(selectinload(Document.chunks))
                .order_by(Document.created_at, Document.id)
            )
            documents = list(result.scalars().all())
            for doc in documents:
                doc.chunks.sort(key=lambda c: c.chunk_index)
            return documents
    finally:
        await engine.dispose()


async def _load_qdrant() -> dict[str, dict[str, Any]]:
    """Return a ``{chunk_id: {vector, payload}}`` map for every point in the collection."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        # Match QdrantStore's naming: base name suffixed with the embedding dim.
        collection = f"{settings.qdrant_collection}_{settings.embedding_dim}"
        if not await client.collection_exists(collection):
            return {}
        out: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                key = str(point.id)
                payload = dict(point.payload or {})
                if isinstance(payload.get("chunk_id"), str):
                    key = payload["chunk_id"]
                out[key] = {
                    "point_id": str(point.id),
                    "vector": list(point.vector) if point.vector is not None else [],
                    "payload": payload,
                }
            if offset is None:
                break
        return out
    finally:
        await client.close()


def _build_dump(
    documents: list[Document],
    embeddings: dict[str, dict[str, Any]],
    *,
    include_full_vectors: bool,
    preview_dims: int,
) -> dict[str, Any]:
    out_documents: list[dict[str, Any]] = []
    chunks_with_embedding = 0
    chunks_without_embedding = 0

    for doc in documents:
        doc_chunks: list[dict[str, Any]] = []
        for chunk in doc.chunks:
            entry = embeddings.get(str(chunk.id))
            embedding_block: dict[str, Any] | None = None
            if entry and entry.get("vector"):
                chunks_with_embedding += 1
                vector = entry["vector"]
                embedding_block = {
                    "dim": len(vector),
                    "preview": [round(float(v), 6) for v in vector[:preview_dims]],
                }
                if include_full_vectors:
                    embedding_block["vector"] = [float(v) for v in vector]
                embedding_block["qdrant_point_id"] = entry.get("point_id")
                embedding_block["qdrant_payload"] = entry.get("payload")
            else:
                chunks_without_embedding += 1

            doc_chunks.append(
                {
                    "id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": dict(chunk.metadata_ or {}),
                    "qdrant_point_id": chunk.qdrant_point_id,
                    "embedding": embedding_block,
                }
            )
        out_documents.append(
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "content_type": doc.content_type,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "chunk_count": len(doc_chunks),
                "chunks": doc_chunks,
            }
        )

    settings = get_settings()
    orphan_qdrant_ids = [
        cid
        for cid in embeddings
        if not any(str(chunk.id) == cid for doc in documents for chunk in doc.chunks)
    ]

    summary = {
        "database_url": settings.database_url.split("@")[-1],  # drop user:pass
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": f"{settings.qdrant_collection}_{settings.embedding_dim}",
        "embedding_model": settings.embedding_model,
        "embedding_backend": settings.embedding_backend,
        "document_count": len(out_documents),
        "chunk_count": sum(doc["chunk_count"] for doc in out_documents),
        "chunks_with_embedding": chunks_with_embedding,
        "chunks_without_embedding": chunks_without_embedding,
        "qdrant_points_total": len(embeddings),
        "qdrant_points_without_chunk": len(orphan_qdrant_ids),
    }
    return {"summary": summary, "documents": out_documents}


def _print_summary(dump: dict[str, Any]) -> None:
    s = dump["summary"]
    print("=== ContextForge DB dump ===")
    print(f"  postgres        : {s['database_url']}")
    print(f"  qdrant          : {s['qdrant_url']}  (collection={s['qdrant_collection']})")
    print(f"  embedding model : {s['embedding_model']} ({s['embedding_backend']})")
    print(
        f"  documents={s['document_count']}  chunks={s['chunk_count']}  "
        f"qdrant_points={s['qdrant_points_total']}"
    )
    print(
        f"  chunks_with_embedding={s['chunks_with_embedding']}  "
        f"chunks_without_embedding={s['chunks_without_embedding']}  "
        f"orphan_qdrant_points={s['qdrant_points_without_chunk']}"
    )
    print()
    for doc in dump["documents"]:
        print(f"- {doc['filename']}  (id={doc['id']}, status={doc['status']})")
        print(f"    chunks: {doc['chunk_count']}")
        if doc["chunks"]:
            sample = doc["chunks"][0]
            content_preview = sample["content"].replace("\n", " ")
            if len(content_preview) > 140:
                content_preview = content_preview[:137] + "..."
            embedding = sample.get("embedding")
            preview = embedding["preview"] if embedding else "—"
            dim = embedding["dim"] if embedding else "—"
            print(f"    sample chunk[0]: {content_preview}")
            print(f"    sample embedding (dim={dim}): {preview}")
        print()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "db_dump.json",
        help="Where to write the full JSON dump (default: scripts/db_dump.json).",
    )
    parser.add_argument(
        "--full-vectors",
        action="store_true",
        help="Include the full embedding vectors (default: only a preview slice).",
    )
    parser.add_argument(
        "--preview-dims",
        type=int,
        default=8,
        help="How many dims of each vector to include in the 'preview' field.",
    )
    args = parser.parse_args()

    documents, embeddings = await asyncio.gather(_load_postgres(), _load_qdrant())
    dump = _build_dump(
        documents,
        embeddings,
        include_full_vectors=args.full_vectors,
        preview_dims=args.preview_dims,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dump, indent=2, default=_json_default), encoding="utf-8")

    _print_summary(dump)
    print(f"Full dump written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
