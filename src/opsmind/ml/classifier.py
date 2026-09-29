"""Build, save, load, and run the Stage 2 text classifier."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from opsmind.exceptions import ModelArtifactError
from opsmind.models import IncidentPrediction, ProcessedDocument


def build_text_classifier(random_state: int) -> Pipeline:
    """Create the unfitted TF-IDF + Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def save_classifier(pipeline: Pipeline, path: Path) -> None:
    """Persist the complete fitted ML pipeline as one artifact."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path)
    except OSError as exc:
        raise ModelArtifactError(
            f"Could not save model artifact: {path}"
        ) from exc


class IncidentClassifier:
    """Inference service around a fitted scikit-learn Pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def load(cls, path: Path) -> "IncidentClassifier":
        """Load a previously trained classifier artifact."""

        if not path.exists():
            raise ModelArtifactError(
                f"Model artifact not found: {path}. "
                "Run `opsmind train` first."
            )

        try:
            pipeline = joblib.load(path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Could not load model artifact: {path}"
            ) from exc

        if not isinstance(pipeline, Pipeline):
            raise ModelArtifactError(
                f"Unexpected model artifact type: {type(pipeline).__name__}"
            )

        return cls(pipeline)

    def predict_documents(
        self,
        documents: Sequence[ProcessedDocument],
    ) -> tuple[IncidentPrediction, ...]:
        """Classify processed Stage 1 documents."""

        if not documents:
            return ()

        texts = [
            _build_classifier_text(document)
            for document in documents
        ]

        predicted_categories = self._pipeline.predict(texts)
        probability_rows = self._pipeline.predict_proba(texts)

        model = self._pipeline.named_steps["model"]
        classes = tuple(str(value) for value in model.classes_)

        predictions: list[IncidentPrediction] = []

        for document, category, probabilities in zip(
            documents,
            predicted_categories,
            probability_rows,
            strict=True,
        ):
            probability_map = {
                class_name: float(probability)
                for class_name, probability in zip(
                    classes,
                    probabilities,
                    strict=True,
                )
            }

            predictions.append(
                IncidentPrediction(
                    source_path=document.metadata.source_path,
                    title=document.title,
                    predicted_category=str(category),
                    confidence=float(max(probabilities)),
                    probabilities=probability_map,
                )
            )

        return tuple(predictions)


def _build_classifier_text(document: ProcessedDocument) -> str:
    """Build the exact text representation used during inference."""

    return f"{document.title}\n{document.content}"
