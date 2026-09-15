from typing import Callable, Optional

from pipeline.ingestion import IngestionPipeline, IngestionResult


def run(
    source: str,
    filters: Optional[dict] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> IngestionResult:
    return IngestionPipeline().run(source, filters, on_progress=on_progress)
