"""Training for the Stage 3 pretrained Transformer classifier."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from opsmind.exceptions import (
    ModelArtifactError,
    TrainingDataError,
)
from opsmind.ml.dataset import load_training_dataset
from opsmind.models import TransformerTrainingResult


class _EncodedTextDataset(
    Dataset[dict[str, torch.Tensor]]
):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: list[int],
    ) -> None:
        self._encodings = encodings
        self._labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(
            self._labels
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        item = {
            key: value[index]
            for key, value
            in self._encodings.items()
        }

        item["labels"] = (
            self._labels[index]
        )

        return item


def _set_seed(
    random_state: int,
) -> None:
    random.seed(
        random_state
    )
    np.random.seed(
        random_state
    )
    torch.manual_seed(
        random_state
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_state
        )


def _select_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    float,
    float,
    float,
]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            outputs = model(
                **batch
            )

            if outputs.loss is None:
                raise RuntimeError(
                    "Transformer model did not return a loss."
                )

            labels = batch["labels"]
            batch_size = labels.size(
                0
            )

            total_loss += (
                outputs.loss.item()
                *
                batch_size
            )
            total_samples += batch_size

            predictions = (
                outputs.logits.argmax(
                    dim=1
                )
            )

            all_targets.extend(
                labels.cpu().tolist()
            )
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    return (
        total_loss / total_samples,
        float(
            accuracy_score(
                all_targets,
                all_predictions,
            )
        ),
        float(
            f1_score(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0,
            )
        ),
    )


def train_transformer_incident_classifier(
    *,
    dataset_path: Path,
    model_dir: Path,
    pretrained_model_name: str,
    validation_size: float,
    random_state: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    patience: int,
    freeze_base: bool,
) -> TransformerTrainingResult:
    """Train a small task head on a pretrained Transformer."""

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

    _set_seed(
        random_state
    )

    frame = load_training_dataset(
        dataset_path
    )

    texts = frame["text"].tolist()
    labels = frame["category"].tolist()

    classes = tuple(
        sorted(
            frame["category"].unique()
        )
    )

    label_to_id = {
        label: index
        for index, label
        in enumerate(classes)
    }

    id_to_label = {
        index: label
        for label, index
        in label_to_id.items()
    }

    numeric_labels = [
        label_to_id[label]
        for label in labels
    ]

    try:
        (
            train_texts,
            validation_texts,
            train_labels,
            validation_labels,
        ) = train_test_split(
            texts,
            numeric_labels,
            test_size=validation_size,
            random_state=random_state,
            stratify=numeric_labels,
        )
    except ValueError as exc:
        raise TrainingDataError(
            "Could not create a stratified Transformer "
            "train/validation split."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name
    )

    train_encodings = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    validation_encodings = tokenizer(
        validation_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = _EncodedTextDataset(
        train_encodings,
        train_labels,
    )

    validation_dataset = _EncodedTextDataset(
        validation_encodings,
        validation_labels,
    )

    generator = torch.Generator().manual_seed(
        random_state
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            pretrained_model_name,
            num_labels=len(classes),
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
    )

    if freeze_base:
        for parameter in (
            model.base_model.parameters()
        ):
            parameter.requires_grad = False

    device = _select_device()
    model = model.to(
        device
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_epoch = 0
    best_validation_loss = float(
        "inf"
    )
    best_validation_accuracy = 0.0
    best_validation_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[
        dict[str, float]
    ] = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        if freeze_base:
            model.base_model.eval()

        train_total_loss = 0.0
        train_total_samples = 0
        train_targets: list[int] = []
        train_predictions: list[int] = []

        for batch in train_loader:
            batch = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(
                **batch
            )

            if outputs.loss is None:
                raise RuntimeError(
                    "Transformer model did not return a loss."
                )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            labels_batch = batch[
                "labels"
            ]

            batch_size_value = (
                labels_batch.size(
                    0
                )
            )

            train_total_loss += (
                loss.item()
                *
                batch_size_value
            )
            train_total_samples += (
                batch_size_value
            )

            predictions = (
                outputs.logits.argmax(
                    dim=1
                )
            )

            train_targets.extend(
                labels_batch
                .detach()
                .cpu()
                .tolist()
            )

            train_predictions.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

        train_loss = (
            train_total_loss
            /
            train_total_samples
        )

        train_accuracy = float(
            accuracy_score(
                train_targets,
                train_predictions,
            )
        )

        train_macro_f1 = float(
            f1_score(
                train_targets,
                train_predictions,
                average="macro",
                zero_division=0,
            )
        )

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
        ) = _evaluate(
            model=model,
            loader=validation_loader,
            device=device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_macro_f1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
        )

        improved = (
            validation_macro_f1
            >
            best_validation_macro_f1
        ) or (
            validation_macro_f1
            ==
            best_validation_macro_f1
            and validation_loss
            <
            best_validation_loss
        )

        if improved:
            best_epoch = epoch
            best_validation_loss = (
                validation_loss
            )
            best_validation_accuracy = (
                validation_accuracy
            )
            best_validation_macro_f1 = (
                validation_macro_f1
            )
            epochs_without_improvement = 0

            try:
                model_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                model.save_pretrained(
                    model_dir
                )
                tokenizer.save_pretrained(
                    model_dir
                )
            except OSError as exc:
                raise ModelArtifactError(
                    "Could not save Transformer model "
                    f"artifact to: {model_dir}"
                ) from exc
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    return TransformerTrainingResult(
        dataset_rows=len(frame),
        train_rows=len(train_texts),
        validation_rows=len(
            validation_texts
        ),
        classes=classes,
        pretrained_model_name=(
            pretrained_model_name
        ),
        freeze_base=freeze_base,
        total_parameter_count=(
            total_parameter_count
        ),
        trainable_parameter_count=(
            trainable_parameter_count
        ),
        best_epoch=best_epoch,
        best_validation_loss=float(
            best_validation_loss
        ),
        best_validation_accuracy=float(
            best_validation_accuracy
        ),
        best_validation_macro_f1=float(
            best_validation_macro_f1
        ),
        model_dir=model_dir,
        history=tuple(
            history
        ),
    )
