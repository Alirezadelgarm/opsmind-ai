from pathlib import Path

from opsmind.models.retrieval import RetrievalHit, RetrievalQueryResult
from opsmind.rag.generators import ExtractiveGenerator
from opsmind.rag.service import RAGService


class FakeRetrieval:
    def search(self, query: str, *, top_k: int | None = None):
        return RetrievalQueryResult(
            query=query,
            embedding_model_name="fake-embedding",
            top_k=top_k or 1,
            hits=(
                RetrievalHit(
                    chunk_id="1",
                    source_path=Path("database-runbook.md"),
                    title="Database",
                    chunk_index=0,
                    text="Database connection pool saturation can cause request timeouts.",
                    score=0.91,
                ),
            ),
        )


def test_rag_service_returns_grounded_sources() -> None:
    service = RAGService(
        retrieval_service=FakeRetrieval(),
        generator=ExtractiveGenerator(max_chars=500),
        top_k=3,
        max_context_chars=2000,
    )
    result = service.ask("Why are database requests timing out?")
    assert "connection pool" in result.answer.lower()
    assert result.sources[0].source_path.name == "database-runbook.md"
    assert result.provider == "extractive"
