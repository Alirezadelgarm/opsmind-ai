"""Serialize OpsMind reports as JSON."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from opsmind.models import (
    IncidentPrediction,
    NeuralTrainingResult,
    PipelineReport,
    RAGAnswer,
    RAGEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalIndexResult,
    RetrievalQueryResult,
    TrainingResult,
    TransformerTrainingResult,
)


def _write_json(payload: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_report(report: PipelineReport, output_dir: Path) -> Path:
    payload = {
        "summary": {
            "total_files": report.total_files,
            "successful_files": report.successful_files,
            "failed_files": report.failed_files,
        },
        "processed": [
            {
                **asdict(document),
                "metadata": {
                    **asdict(document.metadata),
                    "source_path": str(document.metadata.source_path),
                },
            }
            for document in report.processed
        ],
        "failures": [
            {
                "source_path": str(failure.source_path),
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in report.failures
        ],
    }
    return _write_json(payload, output_dir / "stage1-report.json")


def save_training_report(result: TrainingResult, output_dir: Path) -> Path:
    payload = asdict(result)
    payload["model_path"] = str(result.model_path)
    return _write_json(payload, output_dir / "stage2-training-report.json")


def _prediction_payload(
    *,
    predictions: tuple[IncidentPrediction, ...],
    model_type: str,
) -> dict[str, object]:
    return {
        "summary": {
            "model_type": model_type,
            "classified_documents": len(predictions),
        },
        "predictions": [
            {
                "source_path": str(prediction.source_path),
                "title": prediction.title,
                "predicted_category": prediction.predicted_category,
                "confidence": round(prediction.confidence, 6),
                "probabilities": {
                    category: round(probability, 6)
                    for category, probability in prediction.probabilities.items()
                },
            }
            for prediction in predictions
        ],
    }


def save_prediction_report(
    predictions: tuple[IncidentPrediction, ...], output_dir: Path
) -> Path:
    return _write_json(
        _prediction_payload(predictions=predictions, model_type="classical"),
        output_dir / "stage2-predictions.json",
    )


def save_neural_training_report(
    result: NeuralTrainingResult, output_dir: Path
) -> Path:
    payload = asdict(result)
    payload["model_path"] = str(result.model_path)
    return _write_json(payload, output_dir / "stage3-neural-training-report.json")


def save_neural_prediction_report(
    predictions: tuple[IncidentPrediction, ...], output_dir: Path
) -> Path:
    return _write_json(
        _prediction_payload(predictions=predictions, model_type="neural"),
        output_dir / "stage3-neural-predictions.json",
    )


def save_transformer_training_report(
    result: TransformerTrainingResult, output_dir: Path
) -> Path:
    payload = asdict(result)
    payload["model_dir"] = str(result.model_dir)
    return _write_json(payload, output_dir / "stage3-transformer-training-report.json")


def save_transformer_prediction_report(
    predictions: tuple[IncidentPrediction, ...], output_dir: Path
) -> Path:
    return _write_json(
        _prediction_payload(predictions=predictions, model_type="transformer"),
        output_dir / "stage3-transformer-predictions.json",
    )


def save_retrieval_index_report(
    result: RetrievalIndexResult, output_dir: Path
) -> Path:
    payload = asdict(result)
    payload["vector_store_path"] = str(result.vector_store_path)
    return _write_json(payload, output_dir / "stage4-index-report.json")


def save_retrieval_query_report(
    result: RetrievalQueryResult, output_dir: Path
) -> Path:
    payload = {
        "query": result.query,
        "embedding_model_name": result.embedding_model_name,
        "top_k": result.top_k,
        "hits": [
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "source_path": str(hit.source_path),
                "title": hit.title,
                "chunk_index": hit.chunk_index,
                "score": round(hit.score, 6),
                "text": hit.text,
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
    }
    return _write_json(payload, output_dir / "stage4-search-report.json")


def save_retrieval_evaluation_report(
    result: RetrievalEvaluationResult, output_dir: Path
) -> Path:
    return _write_json(asdict(result), output_dir / "stage4-retrieval-evaluation.json")


def save_rag_answer_report(result: RAGAnswer, output_dir: Path) -> Path:
    payload = {
        "query": result.query,
        "answer": result.answer,
        "provider": result.provider,
        "model_name": result.model_name,
        "retrieval_model_name": result.retrieval_model_name,
        "latency_ms": round(result.latency_ms, 3),
        "sources": [
            {
                "rank": source.rank,
                "score": round(source.score, 6),
                "source_path": str(source.source_path),
                "title": source.title,
                "chunk_index": source.chunk_index,
                "text": source.text,
            }
            for source in result.sources
        ],
    }
    return _write_json(payload, output_dir / "stage5-rag-answer.json")


def save_rag_evaluation_report(
    result: RAGEvaluationResult, output_dir: Path
) -> Path:
    return _write_json(asdict(result), output_dir / "stage5-rag-evaluation.json")
