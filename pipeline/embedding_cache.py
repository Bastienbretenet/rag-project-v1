import psycopg
from pgvector.psycopg import register_vector

from config import get_settings

CREATE_EXTENSION_QUERY = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_TABLE_QUERY_TEMPLATE = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        text_hash TEXT PRIMARY KEY,
        embedding vector({dimension}) NOT NULL,
        model TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

SELECT_MANY_QUERY = "SELECT text_hash, embedding FROM embedding_cache WHERE text_hash = ANY(%s)"

INSERT_QUERY = """
    INSERT INTO embedding_cache (text_hash, embedding, model)
    VALUES (%s, %s, %s)
    ON CONFLICT (text_hash) DO NOTHING
"""


class EmbeddingCache:
    def __init__(self):
        settings = get_settings()
        self._database_url = settings.database_url
        self._dimension = settings.vector_dimension
        self._model = settings.embedding_model
        self._create_table_if_missing()

    def get_many(self, text_hashes: list[str]) -> dict[str, list[float]]:
        with self._connect() as connection:
            rows = connection.execute(SELECT_MANY_QUERY, (text_hashes,)).fetchall()
        return {row[0]: row[1].to_list() for row in rows}

    def save_many(self, entries: dict[str, list[float]]) -> None:
        with self._connect() as connection:
            for text_hash, embedding in entries.items():
                connection.execute(INSERT_QUERY, (text_hash, embedding, self._model))

    def _create_table_if_missing(self) -> None:
        with psycopg.connect(self._database_url, autocommit=True) as connection:
            connection.execute(CREATE_EXTENSION_QUERY)
            connection.execute(CREATE_TABLE_QUERY_TEMPLATE.format(dimension=self._dimension))

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self._database_url, autocommit=True)
        register_vector(connection)
        return connection
