from pathlib import Path

from opsmind.models import (
    DocumentMetadata,
    ProcessedDocument,
)
from opsmind.retrieval.chunking import chunk_document


def _document(text: str) -> ProcessedDocument:
    return ProcessedDocument(
        title="Runbook",
        content=text,
        metadata=DocumentMetadata(
            source_path=Path("runbook.md"),
            file_type="md",
            size_bytes=len(text),
        ),
        word_count=len(text.split()),
        character_count=len(text),
    )


def test_chunk_document_preserves_overlap_and_metadata() -> None:
    document = _document(
        "one two three four five six seven eight"
    )

    chunks = chunk_document(
        document,
        chunk_size=4,
        chunk_overlap=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven eight",
    ]
    assert chunks[0].source_path == Path("runbook.md")
    assert chunks[0].chunk_id != chunks[1].chunk_id
