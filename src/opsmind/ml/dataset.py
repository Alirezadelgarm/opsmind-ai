"""Load and validate labeled incident data for Stage 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opsmind.exceptions import TrainingDataError


_REQUIRED_COLUMNS = ("text", "category")


def load_training_dataset(path: Path) -> pd.DataFrame:
    """Return a clean two-column labeled incident dataset."""

    if not path.exists():
        raise TrainingDataError(
            f"Training dataset not found: {path}"
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TrainingDataError(
            f"Could not read training dataset: {path}"
        ) from exc

    missing_columns = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        raise TrainingDataError(
            "Training dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

    frame = frame.loc[:, list(_REQUIRED_COLUMNS)].copy()
    frame = frame.dropna(subset=list(_REQUIRED_COLUMNS))

    frame["text"] = frame["text"].astype(str).str.strip()
    frame["category"] = (
        frame["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    frame = frame[
        (frame["text"] != "")
        & (frame["category"] != "")
    ].copy()

    conflicting_labels = (
        frame.groupby("text")["category"]
        .nunique()
        .gt(1)
    )

    if conflicting_labels.any():
        conflicting_text = conflicting_labels[
            conflicting_labels
        ].index[0]

        raise TrainingDataError(
            "The same training text has multiple labels: "
            f"{conflicting_text!r}"
        )

    frame = frame.drop_duplicates(
        subset=["text", "category"]
    ).reset_index(drop=True)

    if len(frame) < 10:
        raise TrainingDataError(
            "Training dataset must contain at least 10 valid rows."
        )

    class_counts = frame["category"].value_counts()

    if len(class_counts) < 2:
        raise TrainingDataError(
            "Training dataset must contain at least two classes."
        )

    if int(class_counts.min()) < 3:
        raise TrainingDataError(
            "Every class must contain at least three samples."
        )

    return frame
