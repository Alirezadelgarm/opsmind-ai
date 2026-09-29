"""FastAPI application exposing classification, retrieval, and RAG endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from opsmind import __version__
from opsmind.api.schemas import (
    AskRequest,
    AskResponse,
    ClassificationResponse,
    ClassifyRequest,
    HealthResponse,
    RetrievalHitResponse,
    RetrievalRequest,
    RetrievalResponse,
    SourceResponse,
)
from opsmind.config import Settings, load_settings
from opsmind.exceptions import OpsMindError
from opsmind.ml import IncidentClassifier
from opsmind.ml.neural import NeuralIncidentClassifier
from opsmind.models import DocumentMetadata, ProcessedDocument
from opsmind.services import build_rag_service, build_retrieval_service


logger = logging.getLogger(__name__)


def _api_document(title: str, text: str) -> ProcessedDocument:
    cleaned = " ".join(text.split())
    return ProcessedDocument(
        title=title.strip() or "API incident",
        content=cleaned,
        metadata=DocumentMetadata(
            source_path=Path("api-input.txt"),
            file_type="txt",
            size_bytes=len(text.encode("utf-8")),
        ),
        word_count=len(cleaned.split()),
        character_count=len(cleaned),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or load_settings()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        rag = getattr(app.state, "rag_service", None)
        rag_close = getattr(rag, "close", None)
        if callable(rag_close):
            rag_close()

        retrieval = getattr(app.state, "retrieval_service", None)
        retrieval_close = getattr(retrieval, "close", None)
        if callable(retrieval_close):
            retrieval_close()

    app = FastAPI(
        title="OpsMind AI",
        version=__version__,
        description=(
            "Incident classification, semantic retrieval, and grounded RAG for operational knowledge."
        ),
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.retrieval_service = None
    app.state.rag_service = None
    app.state.classifiers = {}

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        started = perf_counter()
        response = await call_next(request)
        latency_ms = (perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{latency_ms:.2f}"
        logger.info(
            "api_request method=%s path=%s status=%s latency_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response

    def retrieval_service():
        if app.state.retrieval_service is None:
            app.state.retrieval_service = build_retrieval_service(runtime_settings)
        return app.state.retrieval_service

    def rag_service():
        if app.state.rag_service is None:
            app.state.rag_service = build_rag_service(
                runtime_settings,
                retrieval_service=retrieval_service(),
            )
        return app.state.rag_service

    def classifier(model_type: str):
        if model_type in app.state.classifiers:
            return app.state.classifiers[model_type]
        if model_type == "classical":
            instance = IncidentClassifier.load(runtime_settings.model_path)
        elif model_type == "neural":
            instance = NeuralIncidentClassifier.load(runtime_settings.neural_model_path)
        elif model_type == "transformer":
            from opsmind.ml.transformer.classifier import TransformerIncidentClassifier
            instance = TransformerIncidentClassifier.load(
                runtime_settings.transformer_model_dir,
                max_length=runtime_settings.transformer_max_length,
            )
        else:
            raise ValueError(f"Unsupported classifier: {model_type}")
        app.state.classifiers[model_type] = instance
        return instance

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.post("/v1/retrieve", response_model=RetrievalResponse)
    def retrieve(payload: RetrievalRequest) -> RetrievalResponse:
        try:
            result = retrieval_service().search(payload.query, top_k=payload.top_k)
        except (OpsMindError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RetrievalResponse(
            query=result.query,
            embedding_model_name=result.embedding_model_name,
            top_k=result.top_k,
            hits=[
                RetrievalHitResponse(
                    rank=rank,
                    score=hit.score,
                    source_path=str(hit.source_path),
                    title=hit.title,
                    chunk_index=hit.chunk_index,
                    text=hit.text,
                )
                for rank, hit in enumerate(result.hits, start=1)
            ],
        )

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            result = rag_service().ask(payload.query, top_k=payload.top_k)
        except (OpsMindError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AskResponse(
            query=result.query,
            answer=result.answer,
            provider=result.provider,
            model_name=result.model_name,
            retrieval_model_name=result.retrieval_model_name,
            latency_ms=result.latency_ms,
            sources=[
                SourceResponse(
                    rank=source.rank,
                    score=source.score,
                    source_path=str(source.source_path),
                    title=source.title,
                    chunk_index=source.chunk_index,
                    text=source.text,
                )
                for source in result.sources
            ],
        )

    @app.post("/v1/classify", response_model=ClassificationResponse)
    def classify(payload: ClassifyRequest) -> ClassificationResponse:
        try:
            prediction = classifier(payload.model_type).predict_documents(
                (_api_document(payload.title, payload.text),)
            )[0]
        except (OpsMindError, ValueError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ClassificationResponse(
            model_type=payload.model_type,
            predicted_category=prediction.predicted_category,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        )


    return app


app = create_app()
