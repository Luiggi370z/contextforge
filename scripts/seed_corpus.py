#!/usr/bin/env python3
"""Seed sample corpus into ContextForge via the ingest API."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parent.parent / "backend" / "app" / "eval" / "corpus"
API = "http://localhost:8000/v1/documents"


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"Missing corpus dir: {CORPUS_DIR}", file=sys.stderr)
        return 1
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print(f"No .md files in {CORPUS_DIR}", file=sys.stderr)
        return 1
    with httpx.Client(timeout=60.0) as client:
        for path in files:
            content = path.read_text(encoding="utf-8")
            resp = client.post(
                API,
                json={
                    "filename": path.name,
                    "content": content,
                    "content_type": "text/markdown",
                },
            )
            resp.raise_for_status()
            print(f"ingested {path.name} -> {resp.json()['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
