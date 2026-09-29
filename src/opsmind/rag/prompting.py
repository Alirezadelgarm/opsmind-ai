"""Prompt construction for grounded OpsMind answers."""

from __future__ import annotations

from collections.abc import Sequence

from opsmind.models.retrieval import RetrievalHit


def build_grounded_prompt(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    max_context_chars: int,
) -> str:
    """Build a compact prompt that instructs the model to use only retrieved context."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be empty.")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero.")

    blocks: list[str] = []
    remaining = max_context_chars

    for rank, hit in enumerate(hits, start=1):
        block = (
            f"[Source {rank}: {hit.source_path.name} / chunk {hit.chunk_index}]\n"
            f"{hit.text.strip()}"
        )
        if len(block) > remaining:
            block = block[:remaining]
        if block:
            blocks.append(block)
            remaining -= len(block)
        if remaining <= 0:
            break

    context = "\n\n".join(blocks) if blocks else "No relevant context was retrieved."

    return (
        "You are OpsMind, an incident-response assistant. "
        "Answer using only the provided context. "
        "If the context is insufficient, say that the available knowledge base is insufficient. "
        "Be concise and operational. Do not invent commands, causes, or facts.\n\n"
        f"Question:\n{cleaned_query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
