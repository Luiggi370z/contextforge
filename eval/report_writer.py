"""Write eval reports to reports/ (JSON + markdown summary)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_json_report(payload: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"ragas_{timestamp_slug()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_markdown_summary(
    *,
    row_count: int,
    ragas_means: dict[str, float] | None,
    heuristic_means: dict[str, float] | None,
    ir_means: dict[str, float] | None = None,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"ragas_{timestamp_slug()}.md"
    lines = [
        "# ContextForge eval report",
        "",
        f"- **Rows evaluated:** {row_count}",
        f"- **Generated (UTC):** {datetime.now(UTC).isoformat()}",
        "",
    ]
    if ir_means:
        lines.append("## Retrieval IR metrics (mean) — offline gate only; live numbers are not meaningful")  # noqa: E501
        lines.append("")
        for key, value in sorted(ir_means.items()):
            lines.append(f"- **{key}:** {value:.4f}")
        lines.append("")
    if ragas_means:
        lines.append("## RAGAS metrics (mean)")
        lines.append("")
        for key, value in sorted(ragas_means.items()):
            lines.append(f"- **{key}:** {value:.4f}")
        lines.append("")
    if heuristic_means:
        lines.append("## Heuristic metrics (mean)")
        lines.append("")
        for key, value in sorted(heuristic_means.items()):
            lines.append(f"- **{key}:** {value:.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
