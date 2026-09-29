"""Lazy file discovery for the ingestion pipeline."""

from collections.abc import Iterator
from pathlib import Path


def iter_supported_files(
    directory: Path,
    supported_extensions: frozenset[str],
) -> Iterator[Path]:
    """Yield supported files recursively without building a large list."""

    if not directory.exists():
        return

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() in supported_extensions:
            yield path
