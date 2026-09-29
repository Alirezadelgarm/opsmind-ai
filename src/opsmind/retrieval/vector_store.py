"""Qdrant-backed vector store supporting local persistence and server mode."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opsmind.exceptions import VectorStoreError
from opsmind.models.retrieval import DocumentChunk, RetrievalHit


class QdrantVectorStore:
    def __init__(
        self,
        *,
        path: Path,
        collection_name: str,
        url: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is not installed. Run `python -m pip install -e \".[dev]\"`."
            ) from exc

        self._path = path
        self._url = url.strip() if url else None
        self._collection_name = collection_name

        try:
            if self._url:
                self._client = QdrantClient(url=self._url)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
        except Exception as exc:
            location = self._url or str(path)
            raise VectorStoreError(f"Could not open Qdrant vector store at {location}.") from exc

    @property
    def path(self) -> Path:
        return self._path

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def rebuild(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        vectors: np.ndarray,
        vector_dimension: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk count and vector count must match.")
        if vector_dimension <= 0:
            raise VectorStoreError("vector_dimension must be greater than zero.")

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            if self._client.collection_exists(self._collection_name):
                self._client.delete_collection(collection_name=self._collection_name)

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
            )

            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=np.asarray(vector).astype(float).tolist(),
                    payload={
                        "source_path": str(chunk.source_path),
                        "title": chunk.title,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "word_count": chunk.word_count,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            if points:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError("Could not rebuild the Qdrant collection.") from exc

    def search(
        self,
        *,
        query_vector: np.ndarray,
        top_k: int,
        score_threshold: float | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        vector = np.asarray(query_vector, dtype=np.float32).astype(float).tolist()
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Vector search failed. Build the index first with `opsmind index-retrieval`."
            ) from exc

        hits: list[RetrievalHit] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload if point.payload is not None else {}
            hits.append(
                RetrievalHit(
                    chunk_id=str(point.id),
                    source_path=Path(str(payload.get("source_path", ""))),
                    title=str(payload.get("title", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return tuple(hits)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
