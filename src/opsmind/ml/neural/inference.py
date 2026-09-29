"""Inference service for the Stage 3 PyTorch neural classifier."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from opsmind.exceptions import ModelArtifactError
from opsmind.ml.neural.model import NeuralTextClassifier
from opsmind.ml.neural.vocabulary import Vocabulary
from opsmind.models import (
    IncidentPrediction,
    ProcessedDocument,
)


def _select_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _document_text(
    document: ProcessedDocument,
) -> str:
    return (
        f"{document.title}\n"
        f"{document.content}"
    )


class NeuralIncidentClassifier:
    """Load and run the Stage 3 from-scratch neural model."""

    def __init__(
        self,
        *,
        model: NeuralTextClassifier,
        vocabulary: Vocabulary,
        classes: tuple[str, ...],
        max_length: int,
        device: torch.device,
    ) -> None:
        self._model = model
        self._vocabulary = vocabulary
        self._classes = classes
        self._max_length = max_length
        self._device = device

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "NeuralIncidentClassifier":
        if not path.exists():
            raise ModelArtifactError(
                f"Neural model artifact not found: {path}. "
                "Run `opsmind train-neural` first."
            )

        device = _select_device()

        try:
            artifact = torch.load(
                path,
                map_location=device,
                weights_only=True,
            )
        except Exception as exc:
            raise ModelArtifactError(
                f"Could not load neural model artifact: {path}"
            ) from exc

        vocabulary = Vocabulary.from_dict(
            artifact["vocabulary"]
        )

        classes = tuple(
            str(value)
            for value
            in artifact["classes"]
        )

        config = artifact["config"]

        model = NeuralTextClassifier(
            vocab_size=len(vocabulary),
            embedding_dim=int(
                config["embedding_dim"]
            ),
            hidden_dim=int(
                config["hidden_dim"]
            ),
            num_classes=len(classes),
            padding_idx=vocabulary.pad_id,
            dropout=float(
                config["dropout"]
            ),
        ).to(
            device
        )

        model.load_state_dict(
            artifact["state_dict"]
        )

        model.eval()

        return cls(
            model=model,
            vocabulary=vocabulary,
            classes=classes,
            max_length=int(
                config["max_length"]
            ),
            device=device,
        )

    def predict_documents(
        self,
        documents: Sequence[
            ProcessedDocument
        ],
    ) -> tuple[
        IncidentPrediction,
        ...
    ]:
        if not documents:
            return ()

        encoded = [
            self._vocabulary.encode(
                _document_text(document),
                max_length=self._max_length,
            )
            for document in documents
        ]

        input_ids = torch.tensor(
            [
                token_ids
                for token_ids, _
                in encoded
            ],
            dtype=torch.long,
            device=self._device,
        )

        attention_mask = torch.tensor(
            [
                mask
                for _, mask
                in encoded
            ],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.inference_mode():
            logits = self._model(
                input_ids,
                attention_mask,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        rows = probabilities.cpu()

        predictions: list[
            IncidentPrediction
        ] = []

        for document, row in zip(
            documents,
            rows,
            strict=True,
        ):
            predicted_index = int(
                row.argmax().item()
            )

            probability_map = {
                class_name: float(
                    row[index].item()
                )
                for index, class_name
                in enumerate(
                    self._classes
                )
            }

            predictions.append(
                IncidentPrediction(
                    source_path=(
                        document
                        .metadata
                        .source_path
                    ),
                    title=document.title,
                    predicted_category=(
                        self._classes[
                            predicted_index
                        ]
                    ),
                    confidence=float(
                        row[
                            predicted_index
                        ].item()
                    ),
                    probabilities=(
                        probability_map
                    ),
                )
            )

        return tuple(
            predictions
        )
