"""Typed domain models for semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    source_path: Path
    title: str
    chunk_index: int
    text: str
    word_count: int


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    source_path: Path
    title: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalIndexResult:
    source_documents: int
    failed_documents: int
    chunk_count: int
    embedding_model_name: str
    vector_dimension: int
    collection_name: str
    vector_store_path: Path
    vector_store_url: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalQueryResult:
    query: str
    embedding_model_name: str
    top_k: int
    hits: tuple[RetrievalHit, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    query: str
    expected_source: str
    reciprocal_rank: float
    rank: int | None
    retrieved_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    query_count: int
    top_k: int
    hit_at_1: float
    hit_at_k: float
    mean_reciprocal_rank: float
    cases: tuple[RetrievalEvaluationCase, ...]
