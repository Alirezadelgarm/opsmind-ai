"""Select a loader based on file extension."""

from __future__ import annotations

from pathlib import Path

from opsmind.exceptions import UnsupportedFileTypeError
from opsmind.loaders.base import DocumentLoader


class LoaderRegistry:
    """Registry that maps extensions such as '.txt' to loader objects."""

    def __init__(self, loaders: list[DocumentLoader]) -> None:
        self._loaders: dict[str, DocumentLoader] = {}

        for loader in loaders:
            for extension in loader.extensions:
                normalized = extension.lower()

                if normalized in self._loaders:
                    raise ValueError(
                        f"Duplicate loader registration for {normalized!r}."
                    )

                self._loaders[normalized] = loader

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._loaders)

    def get_loader(self, path: Path) -> DocumentLoader:
        extension = path.suffix.lower()

        try:
            return self._loaders[extension]
        except KeyError as exc:
            raise UnsupportedFileTypeError(
                f"Unsupported file type {extension!r}: {path}"
            ) from exc
