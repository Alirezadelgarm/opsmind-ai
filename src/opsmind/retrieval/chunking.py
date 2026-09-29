"""Deterministic overlapping chunking for Stage 4 retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from opsmind.models import ProcessedDocument
from opsmind.models.retrieval import DocumentChunk


def _chunk_id(
    *,
    source_path: Path,
    chunk_index: int,
    text: str,
) -> str:
    value = f"{source_path.as_posix()}::{chunk_index}::{text}"
    return str(uuid5(NAMESPACE_URL, value))


def chunk_document(
    document: ProcessedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[DocumentChunk, ...]:
    """Split one processed document into overlapping word chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to zero.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = document.content.split()

    if not words:
        return ()

    step = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []

    for chunk_index, start in enumerate(range(0, len(words), step)):
        selected = words[start : start + chunk_size]

        if not selected:
            break

        text = " ".join(selected)

        chunks.append(
            DocumentChunk(
                chunk_id=_chunk_id(
                    source_path=document.metadata.source_path,
                    chunk_index=chunk_index,
                    text=text,
                ),
                source_path=document.metadata.source_path,
                title=document.title,
                chunk_index=chunk_index,
                text=text,
                word_count=len(selected),
            )
        )

        if start + chunk_size >= len(words):
            break

    return tuple(chunks)


def chunk_documents(
    documents: Sequence[ProcessedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[DocumentChunk, ...]:
    """Chunk multiple processed documents while preserving metadata."""

    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return tuple(chunks)
