#!/usr/bin/env python3
"""Run RAGAS evaluation against golden.jsonl (requires API + seeded corpus)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

GOLDEN = Path(__file__).parent / "golden.jsonl"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
API_QUERY = "http://localhost:8000/v1/query"


def load_golden(limit: int | None) -> list[dict]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_golden(args.limit)
    if args.dry_run:
        print(f"Would evaluate {len(rows)} rows")
        return 0

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("Install dev deps: cd backend && uv sync --extra dev", file=sys.stderr)
        return 1

    questions, answers, contexts, grounds = [], [], [], []
    with httpx.Client(timeout=120.0) as client:
        for row in rows:
            resp = client.post(API_QUERY, json={"message": row["question"]})
            resp.raise_for_status()
            data = resp.json()
            questions.append(row["question"])
            answers.append(data["answer"])
            contexts.append([c.get("snippet", "") for c in data.get("citations", [])] or [""])
            grounds.append(row["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": grounds,
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"ragas_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result.to_pandas().to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
