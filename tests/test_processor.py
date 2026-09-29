from pathlib import Path

import pytest

from opsmind.exceptions import DocumentValidationError
from opsmind.models import Document, DocumentMetadata
from opsmind.processing import DocumentProcessor


def make_document(content: str) -> Document:
    return Document(
        title="  Example   Document ",
        content=content,
        metadata=DocumentMetadata(
            source_path=Path("example.txt"),
            file_type="txt",
            size_bytes=len(content.encode("utf-8")),
        ),
    )


def test_processor_returns_normalized_document() -> None:
    result = DocumentProcessor().process(
        make_document("  AI   Engineering is useful  ")
    )

    assert result.title == "Example Document"
    assert result.content == "AI Engineering is useful"
    assert result.word_count == 4


def test_processor_rejects_empty_content() -> None:
    with pytest.raises(DocumentValidationError):
        DocumentProcessor().process(make_document("   \n\n   "))
