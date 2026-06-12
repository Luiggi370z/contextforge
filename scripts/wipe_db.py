#!/usr/bin/env python3
"""Wipe the document corpus from Postgres and Qdrant.

Truncates ``documents`` and ``chunks`` (CASCADE) in Postgres and drops the
configured Qdrant collection so the next ingest starts from an empty store.

By default ``threads`` / ``messages`` are preserved — pass ``--include-threads``
to also drop conversation history.

Pass ``--all`` for a full reset: truncates EVERY Postgres table (except the
Alembic version table, so the schema/migrations survive) and drops EVERY Qdrant
collection (covering every dimension-keyed ``contextforge_<dim>`` collection),
not just the configured one.

Usage:

  uv run python ../scripts/wipe_db.py                  # corpus only (asks first)
  uv run python ../scripts/wipe_db.py --yes            # corpus only, skip prompt
  uv run python ../scripts/wipe_db.py --include-threads --yes
  uv run python ../scripts/wipe_db.py --all --yes      # nuke everything in both DBs
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.core.config import get_settings  # noqa: E402


async def _wipe_postgres(*, include_threads: bool) -> tuple[int, int, int, int]:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, future=True)
    try:
        async with engine.begin() as conn:
            chunk_count = (await conn.execute(text("SELECT COUNT(*) FROM chunks"))).scalar_one()
            doc_count = (await conn.execute(text("SELECT COUNT(*) FROM documents"))).scalar_one()
            thread_count = 0
            message_count = 0
            if include_threads:
                message_count = (
                    await conn.execute(text("SELECT COUNT(*) FROM messages"))
                ).scalar_one()
                thread_count = (
                    await conn.execute(text("SELECT COUNT(*) FROM threads"))
                ).scalar_one()

            tables = ["chunks", "documents"]
            if include_threads:
                tables += ["messages", "threads"]
            await conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
            return doc_count, chunk_count, thread_count, message_count
    finally:
        await engine.dispose()


def _qdrant_collection_name(settings) -> str:
    """Match QdrantStore's naming: base name suffixed with the embedding dim."""
    return f"{settings.qdrant_collection}_{settings.embedding_dim}"


async def _wipe_qdrant() -> tuple[bool, int]:
    settings = get_settings()
    collection = _qdrant_collection_name(settings)
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        if not await client.collection_exists(collection):
            return False, 0
        info = await client.get_collection(collection)
        point_count = int(info.points_count or 0)
        await client.delete_collection(collection)
        return True, point_count
    finally:
        await client.close()


# Keep migration state so the schema survives a full reset (re-seed, not re-migrate).
_PRESERVE_TABLES = frozenset({"alembic_version"})


async def _wipe_all_postgres() -> list[str]:
    """Truncate every table except the Alembic version table; return wiped names."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, future=True)
    try:
        async with engine.begin() as conn:
            rows = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = current_schema() ORDER BY tablename"
                )
            )
            tables = [name for (name,) in rows if name not in _PRESERVE_TABLES]
            if tables:
                quoted = ", ".join(f'"{name}"' for name in tables)
                await conn.execute(text(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"))
            return tables
    finally:
        await engine.dispose()


async def _wipe_all_qdrant() -> list[str]:
    """Drop every Qdrant collection (all dimension-keyed ones); return dropped names."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        for name in names:
            await client.delete_collection(name)
        return names
    finally:
        await client.close()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-threads",
        action="store_true",
        help="Also wipe ``threads`` and ``messages`` (conversation history).",
    )
    parser.add_argument(
        "--all",
        dest="wipe_all",
        action="store_true",
        help="Full reset: truncate EVERY Postgres table (keeps Alembic version) "
        "and drop EVERY Qdrant collection.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (use in scripts / CI).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.wipe_all:
        target = "EVERYTHING (all Postgres tables + all Qdrant collections)"
    else:
        target = "documents + chunks"
        if args.include_threads:
            target += " + threads + messages"
    print(f"About to wipe {target}.")
    print(f"  postgres : {settings.database_url.split('@')[-1]}")
    print(f"  qdrant   : {settings.qdrant_url}")

    if not args.yes:
        confirmation = input("Type 'wipe' to confirm: ").strip().lower()
        if confirmation != "wipe":
            print("Aborted.")
            return 1

    if args.wipe_all:
        tables = await _wipe_all_postgres()
        collections = await _wipe_all_qdrant()
        print()
        print(f"Postgres: truncated {len(tables)} table(s): {', '.join(tables) or '(none)'}")
        dropped = ", ".join(collections) or "(none)"
        print(f"Qdrant: dropped {len(collections)} collection(s): {dropped}")
        return 0

    docs, chunks, threads, messages = await _wipe_postgres(include_threads=args.include_threads)
    existed, points = await _wipe_qdrant()

    print()
    print("Postgres:")
    print(f"  documents removed: {docs}")
    print(f"  chunks removed   : {chunks}")
    if args.include_threads:
        print(f"  threads removed  : {threads}")
        print(f"  messages removed : {messages}")
    print("Qdrant:")
    if existed:
        print(f"  collection '{_qdrant_collection_name(settings)}' dropped ({points} points)")
    else:
        print(f"  collection '{_qdrant_collection_name(settings)}' did not exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
