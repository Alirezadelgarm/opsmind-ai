"""Neural text classifier built from PyTorch primitives."""

from __future__ import annotations

import torch
from torch import nn


class NeuralTextClassifier(nn.Module):
    """Embedding + masked mean pooling + MLP incident classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw class logits."""

        embeddings = self.embedding(
            input_ids
        )

        mask = attention_mask.unsqueeze(
            dim=-1
        ).to(
            dtype=embeddings.dtype
        )

        masked_embeddings = (
            embeddings
            *
            mask
        )

        summed = masked_embeddings.sum(
            dim=1
        )

        valid_token_count = mask.sum(
            dim=1
        ).clamp(
            min=1.0
        )

        pooled = (
            summed
            /
            valid_token_count
        )

        return self.classifier(
            pooled
        )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
