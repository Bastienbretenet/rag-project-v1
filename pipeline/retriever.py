from typing import Optional

from config import get_settings
from pipeline.embedder import Embedder
from pipeline.vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: Optional[int] = None):
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k or get_settings().top_k

    def retrieve(self, question: str, filters: Optional[dict] = None) -> list[SearchResult]:
        query_vector = self._embedder.embed([question])[0]
        return self._vector_store.search(query_vector, top_k=self._top_k, filters=filters)
