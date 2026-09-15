import psycopg

from config import get_settings

COUNT_DOCUMENTS_QUERY = "SELECT COUNT(*) FROM documents"

LIST_DOCUMENTS_QUERY = """
    SELECT id, title, date, domain, url, source_name
    FROM documents
    ORDER BY date DESC NULLS LAST
    LIMIT %s OFFSET %s
"""

COUNT_CHUNKS_QUERY = "SELECT COUNT(*) FROM chunks WHERE document_id = %s"

LIST_CHUNKS_QUERY = """
    SELECT id, position, text
    FROM chunks
    WHERE document_id = %s
    ORDER BY position ASC
    LIMIT %s OFFSET %s
"""


def count_documents() -> int:
    return _fetch_count(COUNT_DOCUMENTS_QUERY)


def list_documents(offset: int, limit: int) -> list[dict]:
    with _connect() as connection:
        try:
            rows = connection.execute(LIST_DOCUMENTS_QUERY, (limit, offset)).fetchall()
        except psycopg.errors.UndefinedTable:
            return []

    return [
        {
            "id": row[0],
            "title": row[1],
            "date": row[2],
            "domain": row[3],
            "url": row[4],
            "source_name": row[5],
        }
        for row in rows
    ]


def count_chunks(document_id: str) -> int:
    return _fetch_count(COUNT_CHUNKS_QUERY, (document_id,))


def list_chunks(document_id: str, offset: int, limit: int) -> list[dict]:
    with _connect() as connection:
        try:
            rows = connection.execute(LIST_CHUNKS_QUERY, (document_id, limit, offset)).fetchall()
        except psycopg.errors.UndefinedTable:
            return []

    return [{"id": row[0], "position": row[1], "text": row[2]} for row in rows]


def _fetch_count(query: str, params: tuple = ()) -> int:
    with _connect() as connection:
        try:
            return connection.execute(query, params).fetchone()[0]
        except psycopg.errors.UndefinedTable:
            return 0


def _connect() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)
