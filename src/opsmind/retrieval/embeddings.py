"""Sentence-Transformer embedding adapter for Stage 4 retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from opsmind.exceptions import EmbeddingModelError


class SentenceTransformerEmbedder:
    """Encode documents and queries into normalized dense vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. "
                "Run `python -m pip install -e \".[dev]\"`."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model: {model_name!r}."
            ) from exc

        dimension_getter = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        )

        if callable(dimension_getter):
            dimension = dimension_getter()
        else:
            # Compatibility with older sentence-transformers releases.
            dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise EmbeddingModelError(
                "The embedding model did not expose an embedding dimension."
            )

        self._model_name = model_name
        self._batch_size = batch_size
        self._dimension = int(dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self._encode(texts)

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        vectors = self._encode([query])
        return vectors[0]

    def _encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self._dimension),
                dtype=np.float32,
            )

        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingModelError(
                "Embedding generation failed."
            ) from exc

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        expected_shape = (
            len(texts),
            self._dimension,
        )

        if array.shape != expected_shape:
            raise EmbeddingModelError(
                "Unexpected embedding shape: "
                f"expected {expected_shape}, got {array.shape}."
            )

        return array
