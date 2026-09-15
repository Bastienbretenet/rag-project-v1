import psycopg

from config import get_settings
from sources.base import RawDocument

CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        source_name TEXT NOT NULL,
        text TEXT,
        pdf_path TEXT,
        title TEXT NOT NULL,
        date TEXT,
        url TEXT,
        domain TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

INSERT_DOCUMENT_QUERY = """
    INSERT INTO documents (id, source_name, text, pdf_path, title, date, url, domain)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        source_name = EXCLUDED.source_name,
        text = EXCLUDED.text,
        pdf_path = EXCLUDED.pdf_path,
        title = EXCLUDED.title,
        date = EXCLUDED.date,
        url = EXCLUDED.url,
        domain = EXCLUDED.domain
"""


class DocumentStore:
    def __init__(self):
        self._database_url = get_settings().database_url
        self._create_table_if_missing()

    def exists(self, document_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE id = %s", (document_id,)
            ).fetchone()
        return row is not None

    def save(self, document: RawDocument, source_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                INSERT_DOCUMENT_QUERY,
                (
                    document.id,
                    source_name,
                    document.text,
                    document.pdf_path,
                    document.metadata.title,
                    document.metadata.date,
                    document.metadata.url,
                    document.metadata.domain,
                ),
            )

    def _create_table_if_missing(self) -> None:
        with self._connect() as connection:
            connection.execute(CREATE_TABLE_QUERY)

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, autocommit=True)
