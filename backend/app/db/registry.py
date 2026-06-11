"""Single import point that registers every ORM model on ``Base.metadata``.

Model classes live beside their domain (``app/api/v1/<entity>/models.py``); importing
this module pulls them all in so ``Base.metadata`` is complete for Alembic
autogenerate / migrations and for any ``create_all`` in tests. Import it for its
side effect only.
"""

from __future__ import annotations

from app.api.v1.documents.models import Chunk, Document, IngestionJob
from app.api.v1.eval.models import EvalResult, EvalRun
from app.api.v1.threads.models import Message, Thread

__all__ = [
    "Chunk",
    "Document",
    "EvalResult",
    "EvalRun",
    "IngestionJob",
    "Message",
    "Thread",
]
