"""PyTorch datasets for Stage 3 neural text classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from opsmind.ml.neural.vocabulary import Vocabulary


class IncidentTextDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Fixed-length tokenized incidents for neural classification."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocabulary: Vocabulary,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have the same length."
            )

        self._texts = tuple(
            str(text)
            for text in texts
        )
        self._labels = tuple(
            int(label)
            for label in labels
        )
        self._vocabulary = vocabulary
        self._max_length = max_length

    def __len__(self) -> int:
        return len(
            self._texts
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        token_ids, attention_mask = (
            self._vocabulary.encode(
                self._texts[index],
                max_length=self._max_length,
            )
        )

        return (
            torch.tensor(
                token_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                attention_mask,
                dtype=torch.float32,
            ),
            torch.tensor(
                self._labels[index],
                dtype=torch.long,
            ),
        )
