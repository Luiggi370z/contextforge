"""Run the deterministic golden eval across retrieval configs and persist results.

Reuses the in-memory corpus + IR metrics so a run needs no Qdrant or live API.
Each config toggles which retrieval legs are active (dense-only vs hybrid vs
hybrid+rerank), producing the comparison table the dashboard renders.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.eval.repository import EvalRepository
from app.eval import build_reference_corpus
from app.eval.golden_loader import load_golden
from app.eval.harness import InMemoryCorpus, hybrid_retrieve_in_memory
from app.eval.ir_metrics import summarize_ir

# config name -> (use_sparse, use_rerank)
CONFIG_LEGS: dict[str, tuple[bool, bool]] = {
    "dense_only": (False, False),
    "hybrid": (True, False),
    "hybrid_rerank": (True, True),
}


class EvalService:
    def __init__(self, repository: EvalRepository | None = None) -> None:
        self._repository = repository or EvalRepository()

    async def run(self, session: AsyncSession, *, configs: list[str] | None = None):
        selected = configs or list(CONFIG_LEGS)
        corpus = self._build_corpus()
        rows = [row for row in load_golden() if not row.expect_abstain]

        created = []
        for config_name in selected:
            use_sparse, use_rerank = CONFIG_LEGS.get(config_name, (True, True))
            ir_rows = []
            persisted = []
            for row in rows:
                chunks = await hybrid_retrieve_in_memory(
                    corpus, row.question, use_sparse=use_sparse, use_rerank=use_rerank
                )
                retrieved_docs = [corpus.filename_for(chunk.chunk_id) for chunk in chunks]
                ir_rows.append({"retrieved": retrieved_docs, "relevant": row.relevant_set})
                persisted.append(
                    {
                        "question": row.question,
                        "reference_doc": row.reference_doc,
                        "retrieved_docs": retrieved_docs,
                        "abstained": False,
                    }
                )
            metrics = summarize_ir(ir_rows, ks=(1, 3, 5))
            run = await self._repository.create_run(
                session,
                label=config_name,
                config={"name": config_name, "use_sparse": use_sparse, "use_rerank": use_rerank},
                metrics=metrics,
                row_count=len(persisted),
            )
            for result in persisted:
                await self._repository.add_result(session, run_id=run.id, **result)
            created.append(run)
        await session.commit()
        return created

    def _build_corpus(self) -> InMemoryCorpus:
        """Ingest the bundled reference corpus (md + pdf + txt) into a fresh corpus."""
        return build_reference_corpus()

    async def list_runs(self, session: AsyncSession):
        items = await self._repository.list_runs(session)
        total = await self._repository.count_runs(session)
        return items, total
