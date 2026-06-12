"""Load and validate the bundled golden.jsonl rows (ships in the app.eval package)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


@dataclass(frozen=True)
class GoldenRow:
    question: str
    ground_truth: str
    reference_doc: str
    expect_abstain: bool = False
    relevant_docs: tuple[str, ...] = ()

    @property
    def relevant_set(self) -> set[str]:
        """Docs that actually contain the answer (for IR scoring).

        Most questions have a single owning document (``reference_doc``). A few
        facts are genuinely stated in more than one corpus file (e.g. full-disk
        encryption, production-access rules); those list every owning file in
        ``relevant_docs`` so retrieving any of them counts as a hit instead of
        being unfairly scored as a miss under single-doc relevance.
        """
        return set(self.relevant_docs) if self.relevant_docs else {self.reference_doc}


def load_golden(path: Path = GOLDEN_PATH, limit: int | None = None) -> list[GoldenRow]:
    """Parse golden JSONL into validated rows.

    Example:
        >>> rows = load_golden(limit=1)
        >>> rows[0].question
        'How many PTO days...'
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    rows: list[GoldenRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        for field in ("question", "ground_truth", "reference_doc"):
            if field not in payload or not str(payload[field]).strip():
                raise ValueError(f"Line {line_number}: missing or empty '{field}'")
        relevant = payload.get("relevant_docs") or []
        rows.append(
            GoldenRow(
                question=str(payload["question"]).strip(),
                ground_truth=str(payload["ground_truth"]).strip(),
                reference_doc=str(payload["reference_doc"]).strip(),
                expect_abstain=bool(payload.get("expect_abstain", False)),
                relevant_docs=tuple(str(doc).strip() for doc in relevant if str(doc).strip()),
            )
        )

    return rows[:limit] if limit else rows
