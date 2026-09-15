import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.chunk_store import ChunkStore
from pipeline.chunker import Chunker, ParagraphChunker
from pipeline.document_store import DocumentStore
from pipeline.embedder import CachingEmbedder, Embedder, OpenRouterEmbedder
from pipeline.extractor import clean_text, extract_text_from_pdf
from pipeline.vector_store import PgVectorStore, VectorStore
from sources.base import RawDocument
from sources.registry import get_source

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    documents_fetched: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        document_store: Optional[DocumentStore] = None,
        chunk_store: Optional[ChunkStore] = None,
    ):
        self._chunker = chunker or ParagraphChunker()
        self._embedder = embedder or CachingEmbedder(OpenRouterEmbedder())
        self._document_store = document_store or DocumentStore()
        self._chunk_store = chunk_store or ChunkStore()
        self._vector_store = vector_store or PgVectorStore()

    def run(
        self,
        source: str,
        filters: Optional[dict] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> IngestionResult:
        result = IngestionResult()

        documents = get_source(source).fetch(**(filters or {}))
        result.documents_fetched = len(documents)
        logger.info("fetched documents", extra={"source": source, "count": len(documents)})

        for position, document in enumerate(documents, start=1):
            try:
                if self._document_store.exists(document.id):
                    result.documents_skipped += 1
                    logger.info("skipped already ingested document", extra={"document_id": document.id})
                    continue

                result.chunks_created += self._ingest_document(document, source)
                result.documents_ingested += 1
            except Exception as error:
                result.errors.append(f"{document.id}: {error}")
                logger.error("failed to ingest document", extra={"document_id": document.id, "error": str(error)})
            finally:
                logger.info("progress", extra={"processed": position, "total": len(documents)})
                if on_progress:
                    on_progress(position, len(documents))

        logger.info(
            "ingestion finished",
            extra={
                "documents_fetched": result.documents_fetched,
                "documents_ingested": result.documents_ingested,
                "documents_skipped": result.documents_skipped,
                "chunks_created": result.chunks_created,
                "errors": len(result.errors),
            },
        )

        return result

    def _ingest_document(self, document: RawDocument, source: str) -> int:
        if document.pdf_path and not document.text:
            document.text = clean_text(extract_text_from_pdf(document.pdf_path))

        self._document_store.save(document, source_name=source)

        chunks = self._chunker.split(document)
        self._chunk_store.save(chunks)
        logger.info("created chunks", extra={"document_id": document.id, "count": len(chunks)})

        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, vectors)

        return len(chunks)
