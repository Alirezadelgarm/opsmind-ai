"""Dependency composition for retrieval, RAG, and API services."""

from __future__ import annotations

from opsmind.config import Settings
from opsmind.rag.generators import (
    ExtractiveGenerator,
    HuggingFaceLocalGenerator,
    OpenAICompatibleGenerator,
)
from opsmind.rag.service import RAGService
from opsmind.retrieval.service import SemanticRetrievalService


def build_retrieval_service(settings: Settings) -> SemanticRetrievalService:
    return SemanticRetrievalService(
        embedding_model_name=settings.retrieval_embedding_model_name,
        embedding_batch_size=settings.retrieval_embedding_batch_size,
        vector_store_path=settings.retrieval_vector_store_path,
        vector_store_url=settings.retrieval_qdrant_url,
        collection_name=settings.retrieval_collection_name,
        chunk_size=settings.retrieval_chunk_size,
        chunk_overlap=settings.retrieval_chunk_overlap,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )


def build_generator(settings: Settings):
    provider = settings.llm_provider
    if provider == "extractive":
        return ExtractiveGenerator(max_chars=settings.rag_extractive_max_chars)
    if provider == "hf_local":
        return HuggingFaceLocalGenerator(
            settings.llm_model_name,
            max_new_tokens=settings.rag_max_new_tokens,
            max_input_tokens=settings.rag_max_input_tokens,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleGenerator(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model_name=settings.llm_model_name,
            max_new_tokens=settings.rag_max_new_tokens,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def build_rag_service(
    settings: Settings,
    *,
    retrieval_service: SemanticRetrievalService | None = None,
) -> RAGService:
    retrieval = retrieval_service or build_retrieval_service(settings)
    return RAGService(
        retrieval_service=retrieval,
        generator=build_generator(settings),
        top_k=settings.retrieval_top_k,
        max_context_chars=settings.rag_max_context_chars,
        owns_retrieval=retrieval_service is None,
    )
