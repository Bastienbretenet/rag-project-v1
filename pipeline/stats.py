from dataclasses import dataclass

import psycopg

from config import get_settings

COUNT_QUERIES = {
    "documents_count": "SELECT COUNT(*) FROM documents",
    "chunks_count": "SELECT COUNT(*) FROM chunks",
    "embeddings_count": "SELECT COUNT(*) FROM chunk_vectors",
}


@dataclass
class PipelineStats:
    documents_count: int = 0
    chunks_count: int = 0
    embeddings_count: int = 0


def get_pipeline_stats() -> PipelineStats:
    counts = {}
    database_url = get_settings().database_url

    with psycopg.connect(database_url, autocommit=True) as connection:
        for field_name, query in COUNT_QUERIES.items():
            try:
                counts[field_name] = connection.execute(query).fetchone()[0]
            except psycopg.errors.UndefinedTable:
                connection.rollback()
                counts[field_name] = 0

    return PipelineStats(**counts)
