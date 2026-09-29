"""Training pipeline for the Stage 3 PyTorch neural text classifier."""

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
from torch import nn
from torch.utils.data import DataLoader

from opsmind.exceptions import (
    ModelArtifactError,
    TrainingDataError,
)
from opsmind.ml.dataset import load_training_dataset
from opsmind.ml.neural.dataset import IncidentTextDataset
from opsmind.ml.neural.model import (
    NeuralTextClassifier,
    count_trainable_parameters,
)
from opsmind.ml.neural.vocabulary import Vocabulary
from opsmind.models import NeuralTrainingResult


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


def _run_validation(
    model: NeuralTextClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
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
        for (
            input_ids,
            attention_mask,
            targets,
        ) in loader:
            input_ids = input_ids.to(
                device
            )
            attention_mask = attention_mask.to(
                device
            )
            targets = targets.to(
                device
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = loss_fn(
                logits,
                targets,
            )

            batch_size = targets.size(
                0
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )
            total_samples += batch_size

            predictions = logits.argmax(
                dim=1
            )

            all_targets.extend(
                targets.cpu().tolist()
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


def _save_neural_artifact(
    *,
    model: NeuralTextClassifier,
    vocabulary: Vocabulary,
    classes: tuple[str, ...],
    model_path: Path,
    embedding_dim: int,
    hidden_dim: int,
    max_length: int,
    dropout: float,
) -> None:
    try:
        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocabulary": vocabulary.to_dict(),
                "classes": list(classes),
                "config": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "max_length": max_length,
                    "dropout": dropout,
                },
            },
            model_path,
        )
    except OSError as exc:
        raise ModelArtifactError(
            f"Could not save neural model artifact: {model_path}"
        ) from exc


def train_neural_incident_classifier(
    *,
    dataset_path: Path,
    model_path: Path,
    validation_size: float,
    random_state: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    embedding_dim: int,
    hidden_dim: int,
    max_length: int,
    min_frequency: int,
    dropout: float,
    patience: int,
) -> NeuralTrainingResult:
    """Train and persist the from-scratch neural incident classifier."""

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
            "Could not create a stratified neural "
            "train/validation split."
        ) from exc

    vocabulary = Vocabulary.build(
        train_texts,
        min_frequency=min_frequency,
    )

    train_dataset = IncidentTextDataset(
        train_texts,
        train_labels,
        vocabulary,
        max_length,
    )

    validation_dataset = IncidentTextDataset(
        validation_texts,
        validation_labels,
        vocabulary,
        max_length,
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

    device = _select_device()

    model = NeuralTextClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        padding_idx=vocabulary.pad_id,
        dropout=dropout,
    ).to(
        device
    )

    parameter_count = (
        count_trainable_parameters(
            model
        )
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        train_total_loss = 0.0
        train_total_samples = 0
        train_targets: list[int] = []
        train_predictions: list[int] = []

        for (
            input_ids,
            attention_mask,
            targets,
        ) in train_loader:
            input_ids = input_ids.to(
                device
            )
            attention_mask = attention_mask.to(
                device
            )
            targets = targets.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = loss_fn(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            batch_size_value = targets.size(
                0
            )

            train_total_loss += (
                loss.item()
                *
                batch_size_value
            )
            train_total_samples += (
                batch_size_value
            )

            predictions = logits.argmax(
                dim=1
            )

            train_targets.extend(
                targets.detach().cpu().tolist()
            )
            train_predictions.extend(
                predictions.detach().cpu().tolist()
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
        ) = _run_validation(
            model,
            validation_loader,
            loss_fn,
            device,
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

            _save_neural_artifact(
                model=model,
                vocabulary=vocabulary,
                classes=classes,
                model_path=model_path,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                max_length=max_length,
                dropout=dropout,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    # Refit the production artifact on all labeled data using the
    # epoch count selected on the validation split. Validation metrics
    # above remain the honest estimate from unseen validation samples.
    _set_seed(
        random_state
    )

    final_vocabulary = Vocabulary.build(
        texts,
        min_frequency=min_frequency,
    )

    final_dataset = IncidentTextDataset(
        texts,
        numeric_labels,
        final_vocabulary,
        max_length,
    )

    final_loader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=(
            torch.Generator().manual_seed(
                random_state
            )
        ),
    )

    final_model = NeuralTextClassifier(
        vocab_size=len(
            final_vocabulary
        ),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        padding_idx=(
            final_vocabulary.pad_id
        ),
        dropout=dropout,
    ).to(
        device
    )

    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(
        max(
            1,
            best_epoch,
        )
    ):
        final_model.train()

        for (
            input_ids,
            attention_mask,
            targets,
        ) in final_loader:
            input_ids = input_ids.to(
                device
            )
            attention_mask = attention_mask.to(
                device
            )
            targets = targets.to(
                device
            )

            final_optimizer.zero_grad()

            logits = final_model(
                input_ids,
                attention_mask,
            )

            loss = loss_fn(
                logits,
                targets,
            )

            loss.backward()
            final_optimizer.step()

    _save_neural_artifact(
        model=final_model,
        vocabulary=final_vocabulary,
        classes=classes,
        model_path=model_path,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        max_length=max_length,
        dropout=dropout,
    )

    final_parameter_count = (
        count_trainable_parameters(
            final_model
        )
    )

    return NeuralTrainingResult(
        dataset_rows=len(frame),
        train_rows=len(train_texts),
        validation_rows=len(
            validation_texts
        ),
        classes=classes,
        vocab_size=len(
            final_vocabulary
        ),
        parameter_count=(
            final_parameter_count
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
        model_path=model_path,
        history=tuple(
            history
        ),
    )
