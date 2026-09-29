"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from opsmind.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the complete OpsMind application."""

    environment: str
    log_level: str
    input_dir: Path
    output_dir: Path
    max_concurrency: int

    training_data_path: Path
    model_path: Path
    ml_test_size: float
    ml_random_state: int

    neural_model_path: Path
    neural_validation_size: float
    neural_batch_size: int
    neural_epochs: int
    neural_learning_rate: float
    neural_weight_decay: float
    neural_embedding_dim: int
    neural_hidden_dim: int
    neural_max_length: int
    neural_min_frequency: int
    neural_dropout: float
    neural_patience: int

    transformer_model_name: str
    transformer_model_dir: Path
    transformer_validation_size: float
    transformer_batch_size: int
    transformer_epochs: int
    transformer_learning_rate: float
    transformer_weight_decay: float
    transformer_max_length: int
    transformer_patience: int
    transformer_freeze_base: bool

    retrieval_source_dir: Path
    retrieval_embedding_model_name: str
    retrieval_embedding_batch_size: int
    retrieval_vector_store_path: Path
    retrieval_qdrant_url: str | None
    retrieval_collection_name: str
    retrieval_chunk_size: int
    retrieval_chunk_overlap: int
    retrieval_top_k: int
    retrieval_score_threshold: float | None
    retrieval_evaluation_path: Path

    llm_provider: str
    llm_model_name: str
    rag_max_new_tokens: int
    rag_max_input_tokens: int
    rag_max_context_chars: int
    rag_extractive_max_chars: int
    rag_evaluation_path: Path
    openai_base_url: str
    openai_api_key: str
    openai_timeout_seconds: float

    api_host: str
    api_port: int


def _load_positive_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def _load_non_negative_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be greater than or equal to zero.")
    return value


def _load_non_negative_float(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if value < 0.0:
        raise ConfigurationError(f"{name} must be greater than or equal to zero.")
    return value


def _load_positive_float(name: str, default: str) -> float:
    value = _load_non_negative_float(name, default)
    if value <= 0.0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def _load_fraction(name: str, default: str) -> float:
    value = _load_non_negative_float(name, default)
    if not 0.0 < value < 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1.")
    return value


def _load_bool(name: str, default: str) -> bool:
    raw_value = os.getenv(name, default).strip().lower()
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    if raw_value in truthy:
        return True
    if raw_value in falsy:
        return False
    raise ConfigurationError(
        f"{name} must be one of: true/false, yes/no, on/off, 1/0."
    )


def _load_optional_fraction(name: str, default: str) -> float | None:
    raw_value = os.getenv(name, default).strip()
    if not raw_value:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be empty or a number.") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1.")
    return value


def _load_optional_string(name: str, default: str = "") -> str | None:
    value = os.getenv(name, default).strip()
    return value or None


def load_settings() -> Settings:
    """Load and validate settings from .env / process environment."""

    load_dotenv()

    log_level = os.getenv("OPSMIND_LOG_LEVEL", "INFO").upper()
    allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in allowed_levels:
        raise ConfigurationError(f"Unsupported OPSMIND_LOG_LEVEL: {log_level!r}.")

    neural_dropout = _load_non_negative_float("OPSMIND_NEURAL_DROPOUT", "0.10")
    if not 0.0 <= neural_dropout < 1.0:
        raise ConfigurationError("OPSMIND_NEURAL_DROPOUT must be in [0, 1).")

    retrieval_chunk_size = _load_positive_int("OPSMIND_RETRIEVAL_CHUNK_SIZE", "80")
    retrieval_chunk_overlap = _load_non_negative_int(
        "OPSMIND_RETRIEVAL_CHUNK_OVERLAP", "20"
    )
    if retrieval_chunk_overlap >= retrieval_chunk_size:
        raise ConfigurationError(
            "OPSMIND_RETRIEVAL_CHUNK_OVERLAP must be smaller than OPSMIND_RETRIEVAL_CHUNK_SIZE."
        )

    llm_provider = os.getenv("OPSMIND_LLM_PROVIDER", "hf_local").strip().lower()
    allowed_providers = {"extractive", "hf_local", "openai_compatible"}
    if llm_provider not in allowed_providers:
        raise ConfigurationError(
            f"OPSMIND_LLM_PROVIDER must be one of {sorted(allowed_providers)}."
        )

    return Settings(
        environment=os.getenv("OPSMIND_ENV", "development"),
        log_level=log_level,
        input_dir=Path(os.getenv("OPSMIND_INPUT_DIR", "data/input")),
        output_dir=Path(os.getenv("OPSMIND_OUTPUT_DIR", "data/output")),
        max_concurrency=_load_positive_int("OPSMIND_MAX_CONCURRENCY", "4"),

        training_data_path=Path(
            os.getenv("OPSMIND_TRAINING_DATA", "data/training/incidents.csv")
        ),
        model_path=Path(
            os.getenv("OPSMIND_MODEL_PATH", "data/models/incident-classifier.joblib")
        ),
        ml_test_size=_load_fraction("OPSMIND_ML_TEST_SIZE", "0.20"),
        ml_random_state=_load_positive_int("OPSMIND_ML_RANDOM_STATE", "42"),

        neural_model_path=Path(
            os.getenv(
                "OPSMIND_NEURAL_MODEL_PATH", "data/models/neural-incident-classifier.pt"
            )
        ),
        neural_validation_size=_load_fraction(
            "OPSMIND_NEURAL_VALIDATION_SIZE", "0.20"
        ),
        neural_batch_size=_load_positive_int("OPSMIND_NEURAL_BATCH_SIZE", "8"),
        neural_epochs=_load_positive_int("OPSMIND_NEURAL_EPOCHS", "60"),
        neural_learning_rate=_load_positive_float(
            "OPSMIND_NEURAL_LEARNING_RATE", "0.003"
        ),
        neural_weight_decay=_load_non_negative_float(
            "OPSMIND_NEURAL_WEIGHT_DECAY", "0.0001"
        ),
        neural_embedding_dim=_load_positive_int(
            "OPSMIND_NEURAL_EMBEDDING_DIM", "48"
        ),
        neural_hidden_dim=_load_positive_int("OPSMIND_NEURAL_HIDDEN_DIM", "48"),
        neural_max_length=_load_positive_int("OPSMIND_NEURAL_MAX_LENGTH", "48"),
        neural_min_frequency=_load_positive_int(
            "OPSMIND_NEURAL_MIN_FREQUENCY", "1"
        ),
        neural_dropout=neural_dropout,
        neural_patience=_load_positive_int("OPSMIND_NEURAL_PATIENCE", "10"),

        transformer_model_name=os.getenv(
            "OPSMIND_TRANSFORMER_MODEL_NAME", "distilbert-base-uncased"
        ),
        transformer_model_dir=Path(
            os.getenv(
                "OPSMIND_TRANSFORMER_MODEL_DIR",
                "data/models/transformer-incident-classifier",
            )
        ),
        transformer_validation_size=_load_fraction(
            "OPSMIND_TRANSFORMER_VALIDATION_SIZE", "0.20"
        ),
        transformer_batch_size=_load_positive_int(
            "OPSMIND_TRANSFORMER_BATCH_SIZE", "8"
        ),
        transformer_epochs=_load_positive_int("OPSMIND_TRANSFORMER_EPOCHS", "5"),
        transformer_learning_rate=_load_positive_float(
            "OPSMIND_TRANSFORMER_LEARNING_RATE", "0.0005"
        ),
        transformer_weight_decay=_load_non_negative_float(
            "OPSMIND_TRANSFORMER_WEIGHT_DECAY", "0.01"
        ),
        transformer_max_length=_load_positive_int(
            "OPSMIND_TRANSFORMER_MAX_LENGTH", "96"
        ),
        transformer_patience=_load_positive_int(
            "OPSMIND_TRANSFORMER_PATIENCE", "2"
        ),
        transformer_freeze_base=_load_bool(
            "OPSMIND_TRANSFORMER_FREEZE_BASE", "true"
        ),

        retrieval_source_dir=Path(
            os.getenv("OPSMIND_RETRIEVAL_SOURCE_DIR", "data/knowledge")
        ),
        retrieval_embedding_model_name=os.getenv(
            "OPSMIND_RETRIEVAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        retrieval_embedding_batch_size=_load_positive_int(
            "OPSMIND_RETRIEVAL_EMBEDDING_BATCH_SIZE", "32"
        ),
        retrieval_vector_store_path=Path(
            os.getenv(
                "OPSMIND_RETRIEVAL_VECTOR_STORE_PATH", "data/vector_store/qdrant"
            )
        ),
        retrieval_qdrant_url=_load_optional_string("OPSMIND_QDRANT_URL"),
        retrieval_collection_name=os.getenv(
            "OPSMIND_RETRIEVAL_COLLECTION_NAME", "opsmind_knowledge"
        ),
        retrieval_chunk_size=retrieval_chunk_size,
        retrieval_chunk_overlap=retrieval_chunk_overlap,
        retrieval_top_k=_load_positive_int("OPSMIND_RETRIEVAL_TOP_K", "3"),
        retrieval_score_threshold=_load_optional_fraction(
            "OPSMIND_RETRIEVAL_SCORE_THRESHOLD", ""
        ),
        retrieval_evaluation_path=Path(
            os.getenv(
                "OPSMIND_RETRIEVAL_EVALUATION_PATH",
                "data/evaluation/retrieval_queries.json",
            )
        ),

        llm_provider=llm_provider,
        llm_model_name=os.getenv("OPSMIND_LLM_MODEL", "google/flan-t5-small"),
        rag_max_new_tokens=_load_positive_int("OPSMIND_RAG_MAX_NEW_TOKENS", "160"),
        rag_max_input_tokens=_load_positive_int("OPSMIND_RAG_MAX_INPUT_TOKENS", "512"),
        rag_max_context_chars=_load_positive_int(
            "OPSMIND_RAG_MAX_CONTEXT_CHARS", "6000"
        ),
        rag_extractive_max_chars=_load_positive_int(
            "OPSMIND_RAG_EXTRACTIVE_MAX_CHARS", "900"
        ),
        rag_evaluation_path=Path(
            os.getenv("OPSMIND_RAG_EVALUATION_PATH", "data/evaluation/rag_queries.json")
        ),
        openai_base_url=os.getenv("OPSMIND_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=os.getenv("OPSMIND_OPENAI_API_KEY", ""),
        openai_timeout_seconds=_load_positive_float(
            "OPSMIND_OPENAI_TIMEOUT_SECONDS", "60"
        ),

        api_host=os.getenv("OPSMIND_API_HOST", "127.0.0.1"),
        api_port=_load_positive_int("OPSMIND_API_PORT", "8000"),
    )
