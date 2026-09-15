import hashlib
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

from config import get_settings
from pipeline.embedding_cache import EmbeddingCache

Vector = list[float]

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
DEFAULT_BATCH_SIZE = 100


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        ...


class OpenRouterEmbedder(Embedder):
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        settings = get_settings()
        self._api_key = settings.open_router_api.get_secret_value()
        self._model = settings.embedding_model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[Vector]:
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    OPENROUTER_EMBEDDINGS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": batch},
                    timeout=30,
                )
                response.raise_for_status()
                return [[float(value) for value in item["embedding"]] for item in response.json()["data"]]
            except requests.RequestException as error:
                last_error = error
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise RuntimeError(f"Embedding batch failed after {MAX_RETRIES} attempts") from last_error


class CachingEmbedder(Embedder):
    def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None):
        self._embedder = embedder
        self._cache = cache or EmbeddingCache()

    def embed(self, texts: list[str]) -> list[Vector]:
        text_hashes = [hash_text(text) for text in texts]
        cached_vectors = self._cache.get_many(text_hashes)

        missing_indices = [i for i, text_hash in enumerate(text_hashes) if text_hash not in cached_vectors]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_vectors = self._embedder.embed(missing_texts)
            new_entries = {text_hashes[i]: vector for i, vector in zip(missing_indices, new_vectors)}
            self._cache.save_many(new_entries)
            cached_vectors.update(new_entries)

        return [cached_vectors[text_hash] for text_hash in text_hashes]
