import json
from pathlib import Path

from opsmind.models.rag import RAGAnswer, RAGSource
from opsmind.rag.evaluation import evaluate_rag


class FakeRAG:
    def ask(self, query: str, *, top_k: int | None = None):
        return RAGAnswer(
            query=query,
            answer="Check the database connection pool and database health.",
            provider="fake",
            model_name="fake",
            retrieval_model_name="fake-embedding",
            latency_ms=1.0,
            sources=(
                RAGSource(
                    rank=1,
                    score=0.9,
                    source_path=Path("database-runbook.md"),
                    title="Database",
                    chunk_index=0,
                    text="context",
                ),
            ),
        )


def test_rag_evaluation_computes_source_and_keyword_metrics(tmp_path: Path) -> None:
    path = tmp_path / "rag.json"
    path.write_text(
        json.dumps([
            {
                "query": "database timeout",
                "expected_source": "database-runbook.md",
                "required_keywords": ["database", "connection", "pool"],
            }
        ]),
        encoding="utf-8",
    )
    result = evaluate_rag(FakeRAG(), evaluation_path=path, top_k=3)
    assert result.source_hit_rate == 1.0
    assert result.mean_keyword_recall == 1.0
