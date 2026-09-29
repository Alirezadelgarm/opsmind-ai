"""Deterministic RAG evaluation using source-hit and keyword-recall metrics."""

from __future__ import annotations

import json
from pathlib import Path

from opsmind.exceptions import RAGEvaluationError
from opsmind.models.rag import RAGEvaluationCase, RAGEvaluationResult
from opsmind.rag.service import RAGService


def load_rag_evaluation_cases(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        raise RAGEvaluationError(f"RAG evaluation file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGEvaluationError(f"Could not read RAG evaluation data: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise RAGEvaluationError("RAG evaluation data must be a non-empty JSON list.")

    normalized: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RAGEvaluationError(f"RAG evaluation item {index} must be an object.")
        query = str(item.get("query", "")).strip()
        expected_source = str(item.get("expected_source", "")).strip()
        raw_keywords = item.get("required_keywords", [])
        if not isinstance(raw_keywords, list):
            raise RAGEvaluationError(
                f"RAG evaluation item {index} required_keywords must be a list."
            )
        keywords = tuple(str(value).strip().lower() for value in raw_keywords if str(value).strip())
        if not query or not expected_source or not keywords:
            raise RAGEvaluationError(
                f"RAG evaluation item {index} requires query, expected_source, and required_keywords."
            )
        normalized.append(
            {"query": query, "expected_source": expected_source, "required_keywords": keywords}
        )
    return tuple(normalized)


def evaluate_rag(
    service: RAGService,
    *,
    evaluation_path: Path,
    top_k: int,
) -> RAGEvaluationResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    items = load_rag_evaluation_cases(evaluation_path)
    cases: list[RAGEvaluationCase] = []

    for item in items:
        query = str(item["query"])
        expected_source = str(item["expected_source"])
        required_keywords = tuple(item["required_keywords"])
        result = service.ask(query, top_k=top_k)

        source_rank: int | None = None
        for source in result.sources:
            if source.source_path.name == expected_source:
                source_rank = source.rank
                break

        answer_lower = result.answer.lower()
        matched = sum(keyword in answer_lower for keyword in required_keywords)
        keyword_recall = matched / len(required_keywords)

        cases.append(
            RAGEvaluationCase(
                query=query,
                expected_source=expected_source,
                required_keywords=required_keywords,
                source_hit=source_rank is not None,
                source_rank=source_rank,
                keyword_recall=float(keyword_recall),
                answer=result.answer,
            )
        )

    total = len(cases)
    source_hit_rate = sum(case.source_hit for case in cases) / total
    mean_keyword_recall = sum(case.keyword_recall for case in cases) / total

    return RAGEvaluationResult(
        query_count=total,
        source_hit_rate=float(source_hit_rate),
        mean_keyword_recall=float(mean_keyword_recall),
        cases=tuple(cases),
    )
