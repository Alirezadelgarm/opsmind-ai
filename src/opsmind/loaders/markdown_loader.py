"""Markdown document loader."""

from pathlib import Path

from opsmind.loaders.common import read_utf8_text
from opsmind.models import Document, DocumentMetadata


class MarkdownLoader:
    """Load .md documents and use the first H1 as title when available."""

    extensions = frozenset({".md"})

    def load(self, path: Path) -> Document:
        content = read_utf8_text(path)

        metadata = DocumentMetadata(
            source_path=path,
            file_type="md",
            size_bytes=path.stat().st_size,
        )

        return Document(
            title=self._extract_title(content, fallback=path.stem),
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                if title:
                    return title

        return fallback
