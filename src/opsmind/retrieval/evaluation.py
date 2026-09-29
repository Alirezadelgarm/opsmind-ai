"""Retrieval evaluation utilities for Stage 4."""

from __future__ import annotations

import json
from pathlib import Path

from opsmind.exceptions import RetrievalEvaluationError
from opsmind.models.retrieval import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
)
from opsmind.retrieval.service import SemanticRetrievalService


def load_evaluation_queries(
    path: Path,
) -> tuple[tuple[str, str], ...]:
    """Load query/expected-source pairs from JSON."""

    if not path.exists():
        raise RetrievalEvaluationError(
            f"Retrieval evaluation file does not exist: {path}"
        )

    try:
        raw = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalEvaluationError(
            f"Could not read retrieval evaluation data: {path}"
        ) from exc

    if not isinstance(raw, list) or not raw:
        raise RetrievalEvaluationError(
            "Retrieval evaluation data must be a non-empty JSON list."
        )

    pairs: list[tuple[str, str]] = []

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RetrievalEvaluationError(
                f"Evaluation item {index} must be an object."
            )

        query = str(
            item.get("query", "")
        ).strip()
        expected_source = str(
            item.get("expected_source", "")
        ).strip()

        if not query or not expected_source:
            raise RetrievalEvaluationError(
                f"Evaluation item {index} requires query and expected_source."
            )

        pairs.append(
            (query, expected_source)
        )

    return tuple(pairs)


def evaluate_retrieval(
    service: SemanticRetrievalService,
    *,
    evaluation_path: Path,
    top_k: int,
) -> RetrievalEvaluationResult:
    """Compute Hit@1, Hit@K, and MRR over curated retrieval queries."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    pairs = load_evaluation_queries(
        evaluation_path
    )

    cases: list[RetrievalEvaluationCase] = []

    for query, expected_source in pairs:
        result = service.search(
            query,
            top_k=top_k,
        )

        retrieved_sources = tuple(
            hit.source_path.name
            for hit in result.hits
        )

        rank: int | None = None

        for index, source in enumerate(
            retrieved_sources,
            start=1,
        ):
            if source == expected_source:
                rank = index
                break

        reciprocal_rank = (
            0.0
            if rank is None
            else 1.0 / rank
        )

        cases.append(
            RetrievalEvaluationCase(
                query=query,
                expected_source=(
                    expected_source
                ),
                reciprocal_rank=(
                    reciprocal_rank
                ),
                rank=rank,
                retrieved_sources=(
                    retrieved_sources
                ),
            )
        )

    total = len(cases)
    hit_at_1 = sum(
        case.rank == 1
        for case in cases
    ) / total
    hit_at_k = sum(
        case.rank is not None
        and case.rank <= top_k
        for case in cases
    ) / total
    mrr = sum(
        case.reciprocal_rank
        for case in cases
    ) / total

    return RetrievalEvaluationResult(
        query_count=total,
        top_k=top_k,
        hit_at_1=float(hit_at_1),
        hit_at_k=float(hit_at_k),
        mean_reciprocal_rank=float(mrr),
        cases=tuple(cases),
    )
