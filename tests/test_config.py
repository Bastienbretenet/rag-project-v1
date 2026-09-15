import pytest
from pydantic import ValidationError

from config import Settings


def test_missing_database_url_raises_error(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_default_values_are_applied():
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:pass@localhost:5432/db",
        open_router_api="fake-key",
        google_api_key="fake-key",
    )

    assert settings.vector_dimension == 1536
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.embedding_provider == "open-router"
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert settings.llm_provider == "google"
    assert settings.llm_model == "google/gemini-3.8-flash"
    assert settings.top_k == 5
    assert settings.environment == "dev"
    assert settings.log_level == "INFO"


def test_llm_provider_without_matching_api_key_raises_error():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url="postgresql://user:pass@localhost:5432/db",
            open_router_api="fake-key",
            llm_provider="anthropic",
        )
