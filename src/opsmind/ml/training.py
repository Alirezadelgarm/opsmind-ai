"""Train and evaluate the Stage 2 incident text classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from opsmind.exceptions import TrainingDataError
from opsmind.ml.classifier import (
    build_text_classifier,
    save_classifier,
)
from opsmind.ml.dataset import load_training_dataset
from opsmind.models import TrainingResult


def train_incident_classifier(
    dataset_path: Path,
    model_path: Path,
    test_size: float,
    random_state: int,
) -> TrainingResult:
    """Evaluate, select, refit, and persist the Stage 2 classifier."""

    frame = load_training_dataset(dataset_path)

    texts = frame["text"]
    labels = frame["category"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as exc:
        raise TrainingDataError(
            "Could not create a stratified train/test split. "
            "Add more labeled samples per class or adjust the test size."
        ) from exc

    minimum_train_class_size = int(
        y_train.value_counts().min()
    )
    cv_splits = min(5, minimum_train_class_size)

    if cv_splits < 2:
        raise TrainingDataError(
            "Not enough training samples per class for cross-validation."
        )

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_X_train = np.zeros(
        (len(y_train), 1),
        dtype=float,
    )
    baseline_X_test = np.zeros(
        (len(y_test), 1),
        dtype=float,
    )

    baseline.fit(
        baseline_X_train,
        y_train,
    )

    baseline_predictions = baseline.predict(
        baseline_X_test
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    pipeline = build_text_classifier(
        random_state=random_state
    )

    parameter_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],
        "model__C": [
            0.5,
            1.0,
            2.0,
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    test_predictions = search.predict(
        X_test
    )

    classes = tuple(
        str(value)
        for value in search.best_estimator_
        .named_steps["model"]
        .classes_
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=classes,
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    best_index = int(search.best_index_)
    cv_std = float(
        search.cv_results_["std_test_score"][best_index]
    )

    # Evaluation is now frozen. For the deployable artifact we refit the
    # selected configuration on every labeled sample available to us.
    final_pipeline = clone(
        search.best_estimator_
    )
    final_pipeline.fit(
        texts,
        labels,
    )

    save_classifier(
        final_pipeline,
        model_path,
    )

    return TrainingResult(
        dataset_rows=len(frame),
        train_rows=len(X_train),
        test_rows=len(X_test),
        classes=classes,
        cv_splits=cv_splits,
        baseline_accuracy=float(
            accuracy_score(
                y_test,
                baseline_predictions,
            )
        ),
        baseline_macro_f1=float(
            f1_score(
                y_test,
                baseline_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        cv_macro_f1_mean=float(
            search.best_score_
        ),
        cv_macro_f1_std=cv_std,
        best_params={
            key: value
            for key, value in search.best_params_.items()
        },
        test_accuracy=float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        test_macro_f1=float(
            f1_score(
                y_test,
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        test_weighted_f1=float(
            f1_score(
                y_test,
                test_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in matrix
        ),
        classification_report=report,
        model_path=model_path,
    )
