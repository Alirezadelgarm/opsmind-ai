"""Command-line entry point for the complete OpsMind application."""

from __future__ import annotations

import argparse
import asyncio
import logging

from opsmind.config import Settings, load_settings
from opsmind.exceptions import OpsMindError
from opsmind.loaders import LoaderRegistry, MarkdownLoader, TextLoader
from opsmind.logging_config import configure_logging
from opsmind.ml import IncidentClassifier, train_incident_classifier
from opsmind.ml.neural import NeuralIncidentClassifier, train_neural_incident_classifier
from opsmind.pipeline import DocumentPipeline
from opsmind.processing import DocumentProcessor
from opsmind.reporting import (
    save_neural_prediction_report,
    save_neural_training_report,
    save_prediction_report,
    save_rag_answer_report,
    save_rag_evaluation_report,
    save_report,
    save_retrieval_evaluation_report,
    save_retrieval_index_report,
    save_retrieval_query_report,
    save_training_report,
    save_transformer_prediction_report,
    save_transformer_training_report,
)
from opsmind.services import build_rag_service, build_retrieval_service


logger = logging.getLogger(__name__)

_COMMANDS = (
    "run",
    "train",
    "run-neural",
    "train-neural",
    "run-transformer",
    "train-transformer",
    "run-all",
    "index-retrieval",
    "search-retrieval",
    "eval-retrieval",
    "rag-ask",
    "eval-rag",
    "serve",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="opsmind",
        description=(
            "OpsMind: document processing, incident classification, semantic retrieval, "
            "grounded RAG, and a FastAPI serving layer."
        ),
    )
    parser.add_argument("command", nargs="?", choices=_COMMANDS, default="run")
    parser.add_argument("--query", help="Natural-language query for search or RAG.")
    parser.add_argument("--top-k", type=int, help="Override retrieval top-k.")
    args = parser.parse_args()

    if args.command in {"search-retrieval", "rag-ask"} and not (args.query or "").strip():
        parser.error(f"{args.command} requires --query.")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")
    return args


def _pipeline(settings: Settings) -> DocumentPipeline:
    return DocumentPipeline(
        registry=LoaderRegistry(loaders=[TextLoader(), MarkdownLoader()]),
        processor=DocumentProcessor(),
        max_concurrency=settings.max_concurrency,
    )


async def _process_documents(settings: Settings):
    report = await _pipeline(settings).process_directory(settings.input_dir)
    report_path = save_report(report, settings.output_dir)
    logger.info(
        "Stage 1 finished | total=%d success=%d failed=%d",
        report.total_files,
        report.successful_files,
        report.failed_files,
    )
    logger.info("Stage 1 report written to %s", report_path)
    return report


def _log_predictions(*, model_name: str, predictions) -> None:
    for prediction in predictions:
        logger.info(
            "%s classified %s | category=%s confidence=%.3f",
            model_name,
            prediction.source_path,
            prediction.predicted_category,
            prediction.confidence,
        )


async def run_pipeline(settings: Settings) -> int:
    report = await _process_documents(settings)
    predictions = IncidentClassifier.load(settings.model_path).predict_documents(report.processed)
    path = save_prediction_report(predictions, settings.output_dir)
    _log_predictions(model_name="Classical", predictions=predictions)
    logger.info("Stage 2 predictions written to %s", path)
    return 0 if report.failed_files == 0 else 1


async def run_neural_pipeline(settings: Settings) -> int:
    report = await _process_documents(settings)
    predictions = NeuralIncidentClassifier.load(
        settings.neural_model_path
    ).predict_documents(report.processed)
    path = save_neural_prediction_report(predictions, settings.output_dir)
    _log_predictions(model_name="Neural", predictions=predictions)
    logger.info("Stage 3 neural predictions written to %s", path)
    return 0 if report.failed_files == 0 else 1


async def run_transformer_pipeline(settings: Settings) -> int:
    from opsmind.ml.transformer.classifier import TransformerIncidentClassifier

    report = await _process_documents(settings)
    classifier = TransformerIncidentClassifier.load(
        settings.transformer_model_dir,
        max_length=settings.transformer_max_length,
    )
    predictions = classifier.predict_documents(report.processed)
    path = save_transformer_prediction_report(predictions, settings.output_dir)
    _log_predictions(model_name="Transformer", predictions=predictions)
    logger.info("Stage 3 Transformer predictions written to %s", path)
    return 0 if report.failed_files == 0 else 1


async def run_all_pipelines(settings: Settings) -> int:
    from opsmind.ml.transformer.classifier import TransformerIncidentClassifier

    report = await _process_documents(settings)
    classical = IncidentClassifier.load(settings.model_path)
    neural = NeuralIncidentClassifier.load(settings.neural_model_path)
    transformer = TransformerIncidentClassifier.load(
        settings.transformer_model_dir,
        max_length=settings.transformer_max_length,
    )
    classical_predictions = classical.predict_documents(report.processed)
    neural_predictions = neural.predict_documents(report.processed)
    transformer_predictions = transformer.predict_documents(report.processed)
    save_prediction_report(classical_predictions, settings.output_dir)
    save_neural_prediction_report(neural_predictions, settings.output_dir)
    save_transformer_prediction_report(transformer_predictions, settings.output_dir)
    _log_predictions(model_name="Classical", predictions=classical_predictions)
    _log_predictions(model_name="Neural", predictions=neural_predictions)
    _log_predictions(model_name="Transformer", predictions=transformer_predictions)
    logger.info("All classification reports were written to %s", settings.output_dir)
    return 0 if report.failed_files == 0 else 1


async def run_retrieval_indexing(settings: Settings) -> int:
    logger.info(
        "Starting retrieval indexing | source=%s model=%s",
        settings.retrieval_source_dir,
        settings.retrieval_embedding_model_name,
    )
    report = await _pipeline(settings).process_directory(settings.retrieval_source_dir)
    service = build_retrieval_service(settings)
    try:
        result = service.build_index(
            documents=report.processed,
            failed_documents=report.failed_files,
        )
    finally:
        service.close()
    path = save_retrieval_index_report(result, settings.output_dir)
    logger.info(
        "Retrieval index built | documents=%d chunks=%d dimension=%d collection=%s",
        result.source_documents,
        result.chunk_count,
        result.vector_dimension,
        result.collection_name,
    )
    logger.info("Retrieval index report written to %s", path)
    return 0 if report.failed_files == 0 else 1


def run_retrieval_search(
    settings: Settings,
    *,
    query: str,
    top_k: int | None,
) -> int:
    service = build_retrieval_service(settings)
    try:
        result = service.search(query, top_k=top_k)
    finally:
        service.close()
    path = save_retrieval_query_report(result, settings.output_dir)
    for rank, hit in enumerate(result.hits, start=1):
        logger.info(
            "Retrieval rank=%d score=%.3f source=%s chunk=%d",
            rank,
            hit.score,
            hit.source_path,
            hit.chunk_index,
        )
    logger.info("Retrieval search report written to %s", path)
    return 0


def run_retrieval_evaluation(settings: Settings, *, top_k: int | None) -> int:
    from opsmind.retrieval.evaluation import evaluate_retrieval

    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    service = build_retrieval_service(settings)
    try:
        result = evaluate_retrieval(
            service,
            evaluation_path=settings.retrieval_evaluation_path,
            top_k=requested_top_k,
        )
    finally:
        service.close()
    path = save_retrieval_evaluation_report(result, settings.output_dir)
    logger.info(
        "Retrieval evaluation | queries=%d hit@1=%.3f hit@%d=%.3f mrr=%.3f",
        result.query_count,
        result.hit_at_1,
        result.top_k,
        result.hit_at_k,
        result.mean_reciprocal_rank,
    )
    logger.info("Retrieval evaluation report written to %s", path)
    return 0


def run_rag_ask(
    settings: Settings,
    *,
    query: str,
    top_k: int | None,
) -> int:
    service = build_rag_service(settings)
    try:
        result = service.ask(query, top_k=top_k)
    finally:
        service.close()
    path = save_rag_answer_report(result, settings.output_dir)
    logger.info(
        "RAG answer | provider=%s model=%s sources=%d latency_ms=%.1f",
        result.provider,
        result.model_name,
        len(result.sources),
        result.latency_ms,
    )
    logger.info("Answer: %s", result.answer)
    for source in result.sources:
        logger.info(
            "Source rank=%d score=%.3f source=%s chunk=%d",
            source.rank,
            source.score,
            source.source_path,
            source.chunk_index,
        )
    logger.info("RAG answer report written to %s", path)
    return 0


def run_rag_evaluation(settings: Settings, *, top_k: int | None) -> int:
    from opsmind.rag.evaluation import evaluate_rag

    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    service = build_rag_service(settings)
    try:
        result = evaluate_rag(
            service,
            evaluation_path=settings.rag_evaluation_path,
            top_k=requested_top_k,
        )
    finally:
        service.close()
    path = save_rag_evaluation_report(result, settings.output_dir)
    logger.info(
        "RAG evaluation | queries=%d source_hit_rate=%.3f keyword_recall=%.3f",
        result.query_count,
        result.source_hit_rate,
        result.mean_keyword_recall,
    )
    logger.info("RAG evaluation report written to %s", path)
    return 0


def run_training(settings: Settings) -> int:
    result = train_incident_classifier(
        dataset_path=settings.training_data_path,
        model_path=settings.model_path,
        test_size=settings.ml_test_size,
        random_state=settings.ml_random_state,
    )
    path = save_training_report(result, settings.output_dir)
    logger.info(
        "Stage 2 training finished | rows=%d test_macro_f1=%.3f model=%s report=%s",
        result.dataset_rows,
        result.test_macro_f1,
        result.model_path,
        path,
    )
    return 0


def run_neural_training(settings: Settings) -> int:
    result = train_neural_incident_classifier(
        dataset_path=settings.training_data_path,
        model_path=settings.neural_model_path,
        validation_size=settings.neural_validation_size,
        random_state=settings.ml_random_state,
        batch_size=settings.neural_batch_size,
        epochs=settings.neural_epochs,
        learning_rate=settings.neural_learning_rate,
        weight_decay=settings.neural_weight_decay,
        embedding_dim=settings.neural_embedding_dim,
        hidden_dim=settings.neural_hidden_dim,
        max_length=settings.neural_max_length,
        min_frequency=settings.neural_min_frequency,
        dropout=settings.neural_dropout,
        patience=settings.neural_patience,
    )
    path = save_neural_training_report(result, settings.output_dir)
    logger.info(
        "Neural training finished | rows=%d best_epoch=%d val_macro_f1=%.3f report=%s",
        result.dataset_rows,
        result.best_epoch,
        result.best_validation_macro_f1,
        path,
    )
    return 0


def run_transformer_training(settings: Settings) -> int:
    from opsmind.ml.transformer.training import train_transformer_incident_classifier

    result = train_transformer_incident_classifier(
        dataset_path=settings.training_data_path,
        model_dir=settings.transformer_model_dir,
        pretrained_model_name=settings.transformer_model_name,
        validation_size=settings.transformer_validation_size,
        random_state=settings.ml_random_state,
        batch_size=settings.transformer_batch_size,
        epochs=settings.transformer_epochs,
        learning_rate=settings.transformer_learning_rate,
        weight_decay=settings.transformer_weight_decay,
        max_length=settings.transformer_max_length,
        patience=settings.transformer_patience,
        freeze_base=settings.transformer_freeze_base,
    )
    path = save_transformer_training_report(result, settings.output_dir)
    logger.info(
        "Transformer training finished | best_epoch=%d val_macro_f1=%.3f trainable=%d/%d report=%s",
        result.best_epoch,
        result.best_validation_macro_f1,
        result.trainable_parameter_count,
        result.total_parameter_count,
        path,
    )
    return 0


def run_api(settings: Settings) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required for `opsmind serve`.") from exc
    logger.info("Starting API server on http://%s:%d", settings.api_host, settings.api_port)
    uvicorn.run(
        "opsmind.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
    return 0


def main() -> None:
    args = _parse_args()
    try:
        settings = load_settings()
        configure_logging(settings.log_level)

        if args.command == "train":
            exit_code = run_training(settings)
        elif args.command == "train-neural":
            exit_code = run_neural_training(settings)
        elif args.command == "train-transformer":
            exit_code = run_transformer_training(settings)
        elif args.command == "index-retrieval":
            exit_code = asyncio.run(run_retrieval_indexing(settings))
        elif args.command == "search-retrieval":
            exit_code = run_retrieval_search(settings, query=args.query, top_k=args.top_k)
        elif args.command == "eval-retrieval":
            exit_code = run_retrieval_evaluation(settings, top_k=args.top_k)
        elif args.command == "rag-ask":
            exit_code = run_rag_ask(settings, query=args.query, top_k=args.top_k)
        elif args.command == "eval-rag":
            exit_code = run_rag_evaluation(settings, top_k=args.top_k)
        elif args.command == "serve":
            exit_code = run_api(settings)
        elif args.command == "run-neural":
            exit_code = asyncio.run(run_neural_pipeline(settings))
        elif args.command == "run-transformer":
            exit_code = asyncio.run(run_transformer_pipeline(settings))
        elif args.command == "run-all":
            exit_code = asyncio.run(run_all_pipelines(settings))
        else:
            exit_code = asyncio.run(run_pipeline(settings))
    except OpsMindError:
        logger.exception("OpsMind terminated because of a domain error.")
        raise SystemExit(1)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
