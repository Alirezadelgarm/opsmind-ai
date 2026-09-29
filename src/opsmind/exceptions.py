"""Domain-specific exceptions for OpsMind."""


class OpsMindError(Exception):
    """Base exception for expected OpsMind domain failures."""


class ConfigurationError(OpsMindError):
    """Raised when application configuration is invalid."""


class DocumentLoadError(OpsMindError):
    """Raised when a document cannot be loaded."""


class DocumentValidationError(OpsMindError):
    """Raised when a loaded document is not valid for processing."""


class UnsupportedFileTypeError(OpsMindError):
    """Raised when no loader supports a file extension."""


class TrainingDataError(OpsMindError):
    """Raised when the labeled ML dataset is missing or invalid."""


class ModelArtifactError(OpsMindError):
    """Raised when a trained model artifact cannot be loaded or saved."""


class EmbeddingModelError(OpsMindError):
    """Raised when a Stage 4 embedding model cannot be loaded or executed."""


class VectorStoreError(OpsMindError):
    """Raised when the Stage 4 vector store cannot be created or queried."""


class RetrievalEvaluationError(OpsMindError):
    """Raised when retrieval evaluation data is missing or invalid."""


class GenerationModelError(OpsMindError):
    """Raised when a RAG generation provider cannot be loaded or called."""


class RAGEvaluationError(OpsMindError):
    """Raised when RAG evaluation data is missing or invalid."""
