from pathlib import Path

from opsmind.models.retrieval import (
    RetrievalHit,
    RetrievalQueryResult,
)
from opsmind.retrieval.evaluation import evaluate_retrieval


class _FakeService:
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> RetrievalQueryResult:
        del top_k

        mapping = {
            "q1": ("other.md", "expected.md"),
            "q2": ("expected.md",),
        }

        hits = tuple(
            RetrievalHit(
                chunk_id=f"id-{index}",
                source_path=Path(source),
                title=source,
                chunk_index=0,
                text="text",
                score=1.0 / index,
            )
            for index, source
            in enumerate(
                mapping[query],
                start=1,
            )
        )

        return RetrievalQueryResult(
            query=query,
            embedding_model_name="fake",
            top_k=len(hits),
            hits=hits,
        )


def test_retrieval_evaluation_calculates_hit_and_mrr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        '[{"query":"q1","expected_source":"expected.md"},'
        '{"query":"q2","expected_source":"expected.md"}]',
        encoding="utf-8",
    )

    result = evaluate_retrieval(
        _FakeService(),  # type: ignore[arg-type]
        evaluation_path=path,
        top_k=2,
    )

    assert result.hit_at_1 == 0.5
    assert result.hit_at_k == 1.0
    assert result.mean_reciprocal_rank == 0.75
