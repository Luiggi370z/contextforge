#!/usr/bin/env python3
"""Wipe the document corpus from Postgres and Qdrant.

Truncates ``documents`` and ``chunks`` (CASCADE) in Postgres and drops the
configured Qdrant collection so the next ingest starts from an empty store.

By default ``threads`` / ``messages`` are preserved — pass ``--include-threads``
to also drop conversation history.

Usage:

  uv run python ../scripts/wipe_db.py                  # corpus only (asks first)
  uv run python ../scripts/wipe_db.py --yes            # corpus only, skip prompt
  uv run python ../scripts/wipe_db.py --include-threads --yes
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


async def _wipe_qdrant() -> tuple[bool, int]:
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        if not await client.collection_exists(settings.qdrant_collection):
            return False, 0
        info = await client.get_collection(settings.qdrant_collection)
        point_count = int(info.points_count or 0)
        await client.delete_collection(settings.qdrant_collection)
        return True, point_count
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
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (use in scripts / CI).",
    )
    args = parser.parse_args()

    settings = get_settings()
    target = "documents + chunks"
    if args.include_threads:
        target += " + threads + messages"
    print(f"About to wipe {target} from Postgres and drop Qdrant collection.")
    print(f"  postgres : {settings.database_url.split('@')[-1]}")
    print(f"  qdrant   : {settings.qdrant_url}  (collection={settings.qdrant_collection})")

    if not args.yes:
        confirmation = input("Type 'wipe' to confirm: ").strip().lower()
        if confirmation != "wipe":
            print("Aborted.")
            return 1

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
        print(f"  collection '{settings.qdrant_collection}' dropped ({points} points)")
    else:
        print(f"  collection '{settings.qdrant_collection}' did not exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
