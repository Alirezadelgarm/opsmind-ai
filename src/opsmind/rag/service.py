"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

from time import perf_counter

from opsmind.models.rag import RAGAnswer, RAGSource
from opsmind.rag.generators import TextGenerator
from opsmind.rag.prompting import build_grounded_prompt


class RAGService:
    """Retrieve relevant chunks, build a grounded prompt, and generate an answer."""

    def __init__(
        self,
        *,
        retrieval_service,
        generator: TextGenerator,
        top_k: int,
        max_context_chars: int,
        owns_retrieval: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")
        self._retrieval = retrieval_service
        self._generator = generator
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        self._owns_retrieval = owns_retrieval

    @property
    def provider(self) -> str:
        return self._generator.provider

    @property
    def model_name(self) -> str:
        return self._generator.model_name

    def ask(self, query: str, *, top_k: int | None = None) -> RAGAnswer:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty.")

        requested_top_k = self._top_k if top_k is None else top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        started = perf_counter()
        retrieval_result = self._retrieval.search(cleaned_query, top_k=requested_top_k)
        prompt = build_grounded_prompt(
            cleaned_query,
            retrieval_result.hits,
            max_context_chars=self._max_context_chars,
        )
        answer_text = self._generator.generate(prompt)
        latency_ms = (perf_counter() - started) * 1000.0

        sources = tuple(
            RAGSource(
                rank=rank,
                score=hit.score,
                source_path=hit.source_path,
                title=hit.title,
                chunk_index=hit.chunk_index,
                text=hit.text,
            )
            for rank, hit in enumerate(retrieval_result.hits, start=1)
        )

        return RAGAnswer(
            query=cleaned_query,
            answer=answer_text,
            provider=self._generator.provider,
            model_name=self._generator.model_name,
            retrieval_model_name=retrieval_result.embedding_model_name,
            latency_ms=float(latency_ms),
            sources=sources,
        )

    def close(self) -> None:
        if not self._owns_retrieval:
            return
        close = getattr(self._retrieval, "close", None)
        if callable(close):
            close()

