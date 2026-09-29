"""Transform loaded Documents into validated ProcessedDocuments."""

from opsmind.exceptions import DocumentValidationError
from opsmind.models import Document, ProcessedDocument
from opsmind.processing.cleaner import clean_text, count_words, normalize_title


class DocumentProcessor:
    """Core Stage 1 business logic for document normalization."""

    def process(self, document: Document) -> ProcessedDocument:
        title = normalize_title(document.title)
        content = clean_text(document.content)

        if not title:
            raise DocumentValidationError(
                f"Document title is empty: {document.metadata.source_path}"
            )

        if not content:
            raise DocumentValidationError(
                f"Document content is empty: {document.metadata.source_path}"
            )

        return ProcessedDocument(
            title=title,
            content=content,
            metadata=document.metadata,
            word_count=count_words(content),
            character_count=len(content),
        )
