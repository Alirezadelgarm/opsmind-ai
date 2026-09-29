"""Plain-text document loader."""

from pathlib import Path

from opsmind.loaders.common import read_utf8_text
from opsmind.models import Document, DocumentMetadata


class TextLoader:
    """Load .txt documents."""

    extensions = frozenset({".txt"})

    def load(self, path: Path) -> Document:
        content = read_utf8_text(path)

        metadata = DocumentMetadata(
            source_path=path,
            file_type="txt",
            size_bytes=path.stat().st_size,
        )

        return Document(
            title=path.stem,
            content=content,
            metadata=metadata,
        )
