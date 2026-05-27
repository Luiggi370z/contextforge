#!/usr/bin/env python3
"""Run RAGAS + heuristic evaluation against eval/golden.jsonl.

Requires a running API with seeded corpus for live runs:
  just up && just migrate && just seed && just api-dev
  just eval

CI / local without OpenAI: use --heuristic-only (no RAGAS LLM judge).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import httpx

from golden_loader import GoldenRow, load_golden
from heuristic_metrics import score_row, summarize_heuristic
from report_writer import write_json_report, write_markdown_summary

DEFAULT_API = "http://localhost:8000"
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def check_api(base_url: str) -> None:
    health_url = f"{base_url.rstrip('/')}/v1/health"
    with httpx.Client(timeout=10.0) as client:
        response = client.get(health_url)
        response.raise_for_status()


def fetch_answers(base_url: str, rows: list[GoldenRow]) -> list[dict[str, Any]]:
    query_url = f"{base_url.rstrip('/')}/v1/query"
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=120.0) as client:
        for row in rows:
            response = client.post(query_url, json={"message": row.question})
            response.raise_for_status()
            payload = response.json()
            citations = payload.get("citations") or []
            metadata = payload.get("metadata") or {}
            results.append(
                {
                    "question": row.question,
                    "answer": payload.get("answer", ""),
                    "contexts": [citation.get("snippet", "") for citation in citations] or [""],
                    "ground_truth": row.ground_truth,
                    "reference_doc": row.reference_doc,
                    "abstained": bool(metadata.get("abstained", False)),
                    "route": metadata.get("route"),
                }
            )
    return results


def run_ragas_judge(records: list[dict[str, Any]]) -> dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in records],
            "answer": [row["answer"] for row in records],
            "contexts": [row["contexts"] for row in records],
            "ground_truth": [row["ground_truth"] for row in records],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    frame = result.to_pandas()
    means: dict[str, float] = {}
    for column in frame.columns:
        if column in ("question", "contexts", "answer", "ground_truth"):
            continue
        try:
            means[column] = float(frame[column].mean())
        except (TypeError, ValueError):
            continue
    return means


def run_heuristic(rows: list[GoldenRow], records: list[dict[str, Any]]) -> dict[str, float]:
    scores = [
        score_row(
            row,
            answer=record["answer"],
            contexts=record["contexts"],
            abstained=record["abstained"],
        )
        for row, record in zip(rows, records, strict=True)
    ]
    return summarize_heuristic(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ContextForge against golden.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Max golden rows to run")
    parser.add_argument("--api-url", default=DEFAULT_API, help="ContextForge API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Print row count only")
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip RAGAS (no OpenAI); write lexical heuristic metrics only",
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = parser.parse_args()

    rows = load_golden(limit=args.limit)
    if args.dry_run:
        print(f"Would evaluate {len(rows)} golden rows")
        return 0

    try:
        check_api(args.api_url)
    except Exception as error:
        print(f"API not reachable at {args.api_url}: {error}", file=sys.stderr)
        print("Start stack: just up && just migrate && just seed && just api-dev", file=sys.stderr)
        return 1

    records = fetch_answers(args.api_url, rows)
    heuristic_means = run_heuristic(rows, records)

    ragas_means: dict[str, float] | None = None
    if not args.heuristic_only:
        try:
            ragas_means = run_ragas_judge(records)
        except ImportError:
            print(
                "RAGAS not installed; re-run with --heuristic-only or: "
                "cd backend && uv sync --extra dev",
                file=sys.stderr,
            )
            return 1
        except Exception as error:
            print(f"RAGAS failed ({error}); writing heuristic report only", file=sys.stderr)

    report_payload: dict[str, Any] = {
        "row_count": len(rows),
        "heuristic": heuristic_means,
        "ragas": ragas_means,
        "rows": records,
    }
    json_path = write_json_report(report_payload, reports_dir=args.reports_dir)
    md_path = write_markdown_summary(
        row_count=len(rows),
        ragas_means=ragas_means,
        heuristic_means=heuristic_means,
        reports_dir=args.reports_dir,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if ragas_means:
        print("RAGAS means:", ragas_means)
    print("Heuristic means:", heuristic_means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
