"""Typed domain models for Stage 2 and Stage 3 classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IncidentPrediction:
    """Classification result for one processed incident document."""

    source_path: Path
    title: str
    predicted_category: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metrics and metadata produced by one Stage 2 training run."""

    dataset_rows: int
    train_rows: int
    test_rows: int
    classes: tuple[str, ...]
    cv_splits: int
    baseline_accuracy: float
    baseline_macro_f1: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    best_params: dict[str, object]
    test_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    classification_report: dict[str, object]
    model_path: Path


@dataclass(frozen=True, slots=True)
class NeuralTrainingResult:
    """Metrics and metadata from Stage 3 neural text training."""

    dataset_rows: int
    train_rows: int
    validation_rows: int
    classes: tuple[str, ...]
    vocab_size: int
    parameter_count: int
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    best_validation_macro_f1: float
    model_path: Path
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True, slots=True)
class TransformerTrainingResult:
    """Metrics and metadata from Stage 3 pretrained-transformer training."""

    dataset_rows: int
    train_rows: int
    validation_rows: int
    classes: tuple[str, ...]
    pretrained_model_name: str
    freeze_base: bool
    total_parameter_count: int
    trainable_parameter_count: int
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    best_validation_macro_f1: float
    model_dir: Path
    history: tuple[dict[str, float], ...]
