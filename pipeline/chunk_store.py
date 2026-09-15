import psycopg

from config import get_settings
from pipeline.chunker import Chunk

CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id),
        position INTEGER NOT NULL,
        text TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

INSERT_CHUNK_QUERY = """
    INSERT INTO chunks (id, document_id, position, text)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""


class ChunkStore:
    def __init__(self):
        self._database_url = get_settings().database_url
        self._create_table_if_missing()

    def save(self, chunks: list[Chunk]) -> None:
        with self._connect() as connection:
            for chunk in chunks:
                connection.execute(
                    INSERT_CHUNK_QUERY,
                    (chunk.id, chunk.document_id, chunk.position, chunk.text),
                )

    def _create_table_if_missing(self) -> None:
        with self._connect() as connection:
            connection.execute(CREATE_TABLE_QUERY)

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, autocommit=True)
