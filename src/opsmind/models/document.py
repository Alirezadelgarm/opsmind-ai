"""Typed domain models used by the document pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


FileType = Literal["txt", "md"]


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata known at load time."""

    source_path: Path
    file_type: FileType
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Document:
    """A document immediately after loading from disk."""

    title: str
    content: str
    metadata: DocumentMetadata


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """A validated and normalized document."""

    title: str
    content: str
    metadata: DocumentMetadata
    word_count: int
    character_count: int


@dataclass(frozen=True, slots=True)
class ProcessingFailure:
    """Represents one file that could not be processed."""

    source_path: Path
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """Final batch result: successes and failures."""

    processed: tuple[ProcessedDocument, ...]
    failures: tuple[ProcessingFailure, ...]

    @property
    def total_files(self) -> int:
        return len(self.processed) + len(self.failures)

    @property
    def successful_files(self) -> int:
        return len(self.processed)

    @property
    def failed_files(self) -> int:
        return len(self.failures)
