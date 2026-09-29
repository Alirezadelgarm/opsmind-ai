"""Vocabulary and lightweight tokenization for the Stage 3 neural model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import re


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)*")


def simple_tokenize(text: str) -> list[str]:
    """Return a deterministic lower-cased token sequence."""

    return _TOKEN_PATTERN.findall(
        text.lower()
    )


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Token-to-id mapping used by the from-scratch neural classifier."""

    token_to_id: dict[str, int]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_frequency: int = 1,
    ) -> "Vocabulary":
        """Build a vocabulary from training text only."""

        if min_frequency <= 0:
            raise ValueError(
                "min_frequency must be greater than zero."
            )

        counter: Counter[str] = Counter()

        for text in texts:
            counter.update(
                simple_tokenize(text)
            )

        token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        for token in sorted(counter):
            if counter[token] >= min_frequency:
                token_to_id[token] = len(
                    token_to_id
                )

        return cls(
            token_to_id=token_to_id
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[
            self.pad_token
        ]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[
            self.unk_token
        ]

    def __len__(self) -> int:
        return len(
            self.token_to_id
        )

    def encode(
        self,
        text: str,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """Encode text into fixed-length token ids and an attention mask."""

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero."
            )

        tokens = simple_tokenize(
            text
        )[:max_length]

        token_ids = [
            self.token_to_id.get(
                token,
                self.unk_id,
            )
            for token in tokens
        ]

        attention_mask = [
            1
            for _ in token_ids
        ]

        padding_needed = (
            max_length
            -
            len(token_ids)
        )

        token_ids.extend(
            [self.pad_id]
            * padding_needed
        )

        attention_mask.extend(
            [0]
            * padding_needed
        )

        return (
            token_ids,
            attention_mask,
        )

    def to_dict(self) -> dict[str, int]:
        return dict(
            self.token_to_id
        )

    @classmethod
    def from_dict(
        cls,
        token_to_id: dict[str, int],
    ) -> "Vocabulary":
        return cls(
            token_to_id={
                str(token): int(index)
                for token, index
                in token_to_id.items()
            }
        )
