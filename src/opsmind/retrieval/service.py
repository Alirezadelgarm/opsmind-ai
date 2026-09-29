"""Application service that builds and queries the semantic index."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from opsmind.models import ProcessedDocument
from opsmind.models.retrieval import RetrievalIndexResult, RetrievalQueryResult
from opsmind.retrieval.chunking import chunk_documents
from opsmind.retrieval.embeddings import SentenceTransformerEmbedder
from opsmind.retrieval.vector_store import QdrantVectorStore


class SemanticRetrievalService:
    def __init__(
        self,
        *,
        embedding_model_name: str,
        embedding_batch_size: int,
        vector_store_path: Path,
        vector_store_url: str | None,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        score_threshold: float | None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self._embedder = SentenceTransformerEmbedder(
            embedding_model_name,
            batch_size=embedding_batch_size,
        )
        self._store = QdrantVectorStore(
            path=vector_store_path,
            url=vector_store_url,
            collection_name=collection_name,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._score_threshold = score_threshold

    @property
    def embedding_model_name(self) -> str:
        return self._embedder.model_name

    def build_index(
        self,
        *,
        documents: Sequence[ProcessedDocument],
        failed_documents: int = 0,
    ) -> RetrievalIndexResult:
        chunks = chunk_documents(
            documents,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        vectors = self._embedder.encode_documents([chunk.text for chunk in chunks])
        self._store.rebuild(
            chunks=chunks,
            vectors=vectors,
            vector_dimension=self._embedder.dimension,
        )
        return RetrievalIndexResult(
            source_documents=len(documents),
            failed_documents=failed_documents,
            chunk_count=len(chunks),
            embedding_model_name=self._embedder.model_name,
            vector_dimension=self._embedder.dimension,
            collection_name=self._store.collection_name,
            vector_store_path=self._store.path,
            vector_store_url=self._store.url,
        )

    def search(self, query: str, *, top_k: int | None = None) -> RetrievalQueryResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty.")
        requested_top_k = self._top_k if top_k is None else top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        query_vector = self._embedder.encode_query(cleaned_query)
        hits = self._store.search(
            query_vector=query_vector,
            top_k=requested_top_k,
            score_threshold=self._score_threshold,
        )
        return RetrievalQueryResult(
            query=cleaned_query,
            embedding_model_name=self._embedder.model_name,
            top_k=requested_top_k,
            hits=hits,
        )

    def close(self) -> None:
        self._store.close()
