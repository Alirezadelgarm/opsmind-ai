"""Inference service for the Stage 3 pretrained Transformer classifier."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from opsmind.exceptions import ModelArtifactError
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


class TransformerIncidentClassifier:
    """Load and run a locally saved Hugging Face sequence classifier."""

    def __init__(
        self,
        *,
        tokenizer: object,
        model: torch.nn.Module,
        classes: tuple[str, ...],
        max_length: int,
        device: torch.device,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._classes = classes
        self._max_length = max_length
        self._device = device

    @classmethod
    def load(
        cls,
        model_dir: Path,
        max_length: int,
    ) -> "TransformerIncidentClassifier":
        if not model_dir.exists():
            raise ModelArtifactError(
                f"Transformer model directory not found: {model_dir}. "
                "Run `opsmind train-transformer` first."
            )

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ModelArtifactError(
                "The `transformers` package is required. "
                "Reinstall Stage 3 dependencies."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    model_dir,
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise ModelArtifactError(
                "Could not load the saved Transformer artifact "
                f"from: {model_dir}"
            ) from exc

        device = _select_device()
        model = model.to(
            device
        )
        model.eval()

        id_to_label = {
            int(index): str(label)
            for index, label
            in model.config.id2label.items()
        }

        classes = tuple(
            id_to_label[index]
            for index
            in sorted(
                id_to_label
            )
        )

        return cls(
            tokenizer=tokenizer,
            model=model,
            classes=classes,
            max_length=max_length,
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

        texts = [
            _document_text(document)
            for document
            in documents
        ]

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                self._device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            outputs = self._model(
                **encoded
            )
            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            ).cpu()

        predictions: list[
            IncidentPrediction
        ] = []

        for document, row in zip(
            documents,
            probabilities,
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
