from abc import ABC, abstractmethod

import requests

from config import get_settings

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OpenRouterLLMProvider(LLMProvider):
    def __init__(self):
        settings = get_settings()
        self._api_key = settings.open_router_api.get_secret_value()
        self._model = settings.llm_model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
