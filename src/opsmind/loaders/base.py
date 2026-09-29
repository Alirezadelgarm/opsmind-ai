"""Loader interface."""

from pathlib import Path
from typing import Protocol

from opsmind.models import Document


class DocumentLoader(Protocol):
    """Structural interface implemented by document loaders."""

    extensions: frozenset[str]

    def load(self, path: Path) -> Document:
        """Load a path into a Document."""
        ...
