"""Batch orchestration for Stage 1 document processing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from opsmind.discovery import iter_supported_files
from opsmind.exceptions import OpsMindError
from opsmind.loaders import LoaderRegistry
from opsmind.models import (
    PipelineReport,
    ProcessingFailure,
    ProcessedDocument,
)
from opsmind.processing import DocumentProcessor


logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Coordinate discovery, loading, processing, and failure isolation."""

    def __init__(
        self,
        registry: LoaderRegistry,
        processor: DocumentProcessor,
        max_concurrency: int,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero.")

        self._registry = registry
        self._processor = processor
        self._max_concurrency = max_concurrency

    def _process_path(self, path: Path) -> ProcessedDocument:
        loader = self._registry.get_loader(path)
        document = loader.load(path)
        return self._processor.process(document)

    async def _process_path_async(
        self,
        path: Path,
        semaphore: asyncio.Semaphore,
    ) -> ProcessedDocument:
        async with semaphore:
            # Stage 1 loaders are synchronous file-I/O code.
            # to_thread prevents that blocking I/O from occupying the event loop.
            return await asyncio.to_thread(self._process_path, path)

    async def process_directory(self, directory: Path) -> PipelineReport:
        paths = tuple(
            iter_supported_files(
                directory,
                self._registry.supported_extensions,
            )
        )

        if not paths:
            logger.warning(
                "No supported documents found in %s",
                directory,
            )
            return PipelineReport(processed=(), failures=())

        logger.info(
            "Discovered %d supported document(s) in %s",
            len(paths),
            directory,
        )

        semaphore = asyncio.Semaphore(self._max_concurrency)

        results = await asyncio.gather(
            *[
                self._safe_process_path(path, semaphore)
                for path in paths
            ]
        )

        processed: list[ProcessedDocument] = []
        failures: list[ProcessingFailure] = []

        for result in results:
            if isinstance(result, ProcessedDocument):
                processed.append(result)
            else:
                failures.append(result)

        return PipelineReport(
            processed=tuple(processed),
            failures=tuple(failures),
        )

    async def _safe_process_path(
        self,
        path: Path,
        semaphore: asyncio.Semaphore,
    ) -> ProcessedDocument | ProcessingFailure:
        try:
            result = await self._process_path_async(path, semaphore)

            logger.info(
                "Processed %s (%d words)",
                path,
                result.word_count,
            )

            return result

        except OpsMindError as exc:
            logger.error(
                "Failed to process %s: %s",
                path,
                exc,
            )

            return ProcessingFailure(
                source_path=path,
                error_type=type(exc).__name__,
                message=str(exc),
            )

        except Exception as exc:
            # Unexpected programming/system failures are logged with traceback.
            logger.exception(
                "Unexpected error while processing %s",
                path,
            )

            return ProcessingFailure(
                source_path=path,
                error_type=type(exc).__name__,
                message=str(exc),
            )
