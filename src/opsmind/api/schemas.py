"""Pydantic request and response schemas for the OpsMind API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievalHitResponse(BaseModel):
    rank: int
    score: float
    source_path: str
    title: str
    chunk_index: int
    text: str


class RetrievalResponse(BaseModel):
    query: str
    embedding_model_name: str
    top_k: int
    hits: list[RetrievalHitResponse]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    rank: int
    score: float
    source_path: str
    title: str
    chunk_index: int
    text: str


class AskResponse(BaseModel):
    query: str
    answer: str
    provider: str
    model_name: str
    retrieval_model_name: str
    latency_ms: float
    sources: list[SourceResponse]


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    title: str = Field(default="API incident", max_length=300)
    model_type: Literal["classical", "neural", "transformer"] = "transformer"


class ClassificationResponse(BaseModel):
    model_type: str
    predicted_category: str
    confidence: float
    probabilities: dict[str, float]
