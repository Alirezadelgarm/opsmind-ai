"""Domain models for retrieval-augmented generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RAGSource:
    """One retrieved source exposed with a grounded answer."""

    rank: int
    score: float
    source_path: Path
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """One grounded answer plus the sources used to produce it."""

    query: str
    answer: str
    provider: str
    model_name: str
    retrieval_model_name: str
    latency_ms: float
    sources: tuple[RAGSource, ...]


@dataclass(frozen=True, slots=True)
class RAGEvaluationCase:
    """Evaluation details for one curated RAG query."""

    query: str
    expected_source: str
    required_keywords: tuple[str, ...]
    source_hit: bool
    source_rank: int | None
    keyword_recall: float
    answer: str


@dataclass(frozen=True, slots=True)
class RAGEvaluationResult:
    """Aggregate deterministic RAG evaluation metrics."""

    query_count: int
    source_hit_rate: float
    mean_keyword_recall: float
    cases: tuple[RAGEvaluationCase, ...]
