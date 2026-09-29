from pathlib import Path

from opsmind.models.retrieval import RetrievalHit
from opsmind.rag.prompting import build_grounded_prompt


def test_prompt_contains_query_and_source_metadata() -> None:
    hit = RetrievalHit(
        chunk_id="x",
        source_path=Path("database-runbook.md"),
        title="Database",
        chunk_index=1,
        text="Check database connection pool saturation.",
        score=0.9,
    )
    prompt = build_grounded_prompt(
        "Why are requests timing out?",
        (hit,),
        max_context_chars=1000,
    )
    assert "Why are requests timing out?" in prompt
    assert "database-runbook.md" in prompt
    assert "connection pool" in prompt
    assert "Do not invent" in prompt
