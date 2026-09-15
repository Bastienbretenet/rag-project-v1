from functools import lru_cache
from typing import Literal, Optional

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Clés API / secrets externes
    open_router_api: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None

    # Connexion base de données
    database_url: str
    vector_dimension: int = 1536

    # Paramètres du pipeline
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_provider: Literal["open-router", "local"] = "open-router"
    embedding_model: str = "openai/text-embedding-3-small"
    llm_provider: Literal["google", "anthropic", "openai"] = "google"
    llm_model: str = "google/gemini-3.8-flash"
    top_k: int = 5

    # Paramètres d'environnement
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_provider_api_keys(self) -> "Settings":
        if self.embedding_provider == "open-router" and self.open_router_api is None:
            raise ValueError(
                "embedding_provider='open-router' requires open_router_api to be set"
            )

        llm_provider_to_key = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
        }
        required_key = llm_provider_to_key[self.llm_provider]
        if required_key is None:
            raise ValueError(
                f"llm_provider='{self.llm_provider}' requires "
                f"{self.llm_provider}_api_key to be set"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
