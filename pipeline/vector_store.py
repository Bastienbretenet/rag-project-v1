from abc import ABC, abstractmethod
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from config import get_settings
from pipeline.chunker import Chunk
from pipeline.embedder import Vector

CREATE_EXTENSION_QUERY = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_TABLE_QUERY_TEMPLATE = """
    CREATE TABLE IF NOT EXISTS chunk_vectors (
        chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
        document_id TEXT NOT NULL REFERENCES documents(id),
        embedding vector({dimension}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

CREATE_INDEX_QUERY = """
    CREATE INDEX IF NOT EXISTS chunk_vectors_embedding_hnsw_idx
    ON chunk_vectors USING hnsw (embedding vector_cosine_ops)
"""

UPSERT_QUERY = """
    INSERT INTO chunk_vectors (chunk_id, document_id, embedding)
    VALUES (%s, %s, %s)
    ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
"""

SEARCH_QUERY_TEMPLATE = """
    SELECT c.id, c.document_id, c.text, d.domain, d.date, d.title, d.url,
           cv.embedding <=> %s::vector AS distance
    FROM chunk_vectors cv
    JOIN chunks c ON c.id = cv.chunk_id
    JOIN documents d ON d.id = cv.document_id
    {where_clause}
    ORDER BY distance ASC
    LIMIT %s
"""


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    distance: float
    domain: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: Vector, top_k: int = 5, filters: Optional[dict] = None) -> list[SearchResult]:
        ...


class PgVectorStore(VectorStore):
    def __init__(self):
        settings = get_settings()
        self._database_url = settings.database_url
        self._dimension = settings.vector_dimension
        self._create_table_if_missing()

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        with self._connect() as connection:
            for chunk, vector in zip(chunks, vectors):
                connection.execute(UPSERT_QUERY, (chunk.id, chunk.document_id, vector))

    def search(self, query_vector: Vector, top_k: int = 5, filters: Optional[dict] = None) -> list[SearchResult]:
        where_clause, where_params = self._build_where_clause(filters or {})
        query = SEARCH_QUERY_TEMPLATE.format(where_clause=where_clause)
        params = [query_vector, *where_params, top_k]

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [self._to_search_result(row) for row in rows]

    def _build_where_clause(self, filters: dict) -> tuple[str, list]:
        conditions = []
        params: list = []

        domain = filters.get("domain")
        if domain:
            conditions.append("d.domain = %s")
            params.append(domain)

        date_start = filters.get("date_start")
        if date_start:
            conditions.append("d.date >= %s")
            params.append(date_start)

        date_end = filters.get("date_end")
        if date_end:
            conditions.append("d.date <= %s")
            params.append(date_end)

        if not conditions:
            return "", params

        return "WHERE " + " AND ".join(conditions), params

    def _to_search_result(self, row: tuple) -> SearchResult:
        chunk_id, document_id, text, domain, date, title, url, distance = row
        return SearchResult(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            distance=distance,
            domain=domain,
            date=date,
            title=title,
            url=url,
        )

    def _create_table_if_missing(self) -> None:
        with psycopg.connect(self._database_url, autocommit=True) as connection:
            connection.execute(CREATE_EXTENSION_QUERY)
            connection.execute(CREATE_TABLE_QUERY_TEMPLATE.format(dimension=self._dimension))
            connection.execute(CREATE_INDEX_QUERY)

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self._database_url, autocommit=True)
        register_vector(connection)
        return connection
