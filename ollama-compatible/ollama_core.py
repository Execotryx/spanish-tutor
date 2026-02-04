from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
else:
    ChatCompletion = Any  # type: ignore[misc,assignment]
    ChatCompletionMessageParam = Any  # type: ignore[misc,assignment]

T = TypeVar("T")


class OllamaConfig:
    """
    Loads Ollama configuration from environment variables and exposes
    validated read-only accessors.
    """

    API_KEY_ENV: str = "OLLAMA_API_KEY"
    BASE_URL_ENV: str = "OLLAMA_BASE_URL"
    MODEL_ENV: str = "OLLAMA_MODEL"
    DEFAULT_BASE_URL: str = "http://localhost:11434/v1"
    DEFAULT_API_KEY: str = "ollama"
    
    def __init__(self) -> None:
        load_dotenv()
        model_name: str = os.getenv(self.MODEL_ENV, "").strip()
        base_url: str = os.getenv(self.BASE_URL_ENV, "").strip() or self.DEFAULT_BASE_URL
        api_key: str = os.getenv(self.API_KEY_ENV, "").strip() or self.DEFAULT_API_KEY
        if not model_name:
            raise ValueError(f"Missing or empty environment variable: {self.MODEL_ENV}")
        if not base_url:
            raise ValueError(f"Missing or empty environment variable: {self.BASE_URL_ENV}")
        self.__api_key: str = api_key
        self.__base_url: str = base_url
        self.__model_name: str = model_name

    @property
    def api_key(self) -> str:
        return self.__api_key

    @property
    def base_url(self) -> str:
        return self.__base_url

    @property
    def model_name(self) -> str:
        return self.__model_name


class OllamaChatCompletionsCore(ABC, Generic[T]):
    """
    Base class for Ollama operations using the Chat Completions API.
    Subclasses implement response conversion via `handle_response()`.
    """

    def __init__(
        self,
        config: OllamaConfig,
        system_prompt: str,
        client: OpenAI | None = None,
    ) -> None:
        self.__config: OllamaConfig = config
        self.__client: OpenAI = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.__system_prompt: str = system_prompt
        self.__messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    @property
    def _config(self) -> OllamaConfig:
        """Protected, read-only access to the Ollama configuration for subclasses."""
        return self.__config

    @property
    def _client(self) -> OpenAI:
        """Protected, read-only access to the OpenAI client for subclasses."""
        return self.__client

    @property
    def _system_prompt(self) -> str:
        """Protected, read-only access to the system prompt for subclasses."""
        return self.__system_prompt

    @property
    def _messages(self) -> list[ChatCompletionMessageParam]:
        """Protected access to the chat history used for requests."""
        return self.__messages

    def create_completion(
        self,
        input_text: str | None = None,
        **kwargs: Any,
    ) -> T:
        if input_text is None:
            raise ValueError("input_text must be provided.")
        if input_text is not None:
            self.__messages.append({"role": "user", "content": input_text})

        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": self.__messages,
        }
        payload.update(kwargs)

        response: ChatCompletion = self._client.chat.completions.create(**payload)
        assistant_message: str = self._extract_assistant_content(response)
        if assistant_message.strip():
            self.__messages.append({"role": "assistant", "content": assistant_message})
        return self.handle_response(response)

    @staticmethod
    def _extract_assistant_content(response: ChatCompletion) -> str:
        content: str | None = None
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError):
            content = None
        return content or ""

    @abstractmethod
    def handle_response(self, response: ChatCompletion) -> T:
        """
        Convert a raw Chat Completions response into the generic output type `T`.
        """
        raise NotImplementedError
