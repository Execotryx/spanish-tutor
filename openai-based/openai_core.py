from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.responses import Response
else:
    Response = Any  # type: ignore[misc,assignment]

T = TypeVar("T")


class OpenAIConfig:
    """
    Loads OpenAI configuration from environment variables and exposes
    validated read-only accessors.
    """

    api_key_env: str = "OPENAI_API_KEY"
    model_env: str = "OPENAI_MODEL"

    def __init__(self) -> None:
        load_dotenv()
        api_key: str = os.getenv(self.api_key_env, "").strip()
        model_name: str = os.getenv(self.model_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing or empty environment variable: {self.api_key_env}")
        if not model_name:
            raise ValueError(f"Missing or empty environment variable: {self.model_env}")
        self.__api_key: str = api_key
        self.__model_name: str = model_name

    @property
    def api_key(self) -> str:
        return self.__api_key

    @property
    def model_name(self) -> str:
        return self.__model_name


class OpenAICore(ABC, Generic[T]):
    """
    Base class for OpenAI operations using the Responses API.
    Subclasses implement response conversion via `handle_response()`.
    """

    def __init__(
        self,
        config: OpenAIConfig,
        system_prompt: str,
        client: OpenAI | None = None,
    ) -> None:
        self.__config: OpenAIConfig = config
        self.__client: OpenAI = client or OpenAI(api_key=config.api_key)
        self.__previous_response_id: str | None = None
        self.__system_prompt: str = system_prompt

    @property
    def _config(self) -> OpenAIConfig:
        """Read-only OpenAI configuration accessor for subclasses."""
        return self.__config

    @property
    def _client(self) -> OpenAI:
        """Read-only OpenAI client accessor for subclasses."""
        return self.__client

    @property
    def _previous_response_id(self) -> str | None:
        """Read-only previous response ID accessor for subclasses."""
        return self.__previous_response_id

    @property
    def _system_prompt(self) -> str:
        """Read-only system prompt accessor for subclasses."""
        return self.__system_prompt

    def create_response(self, input_text: str, **kwargs: Any) -> T:
        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "input": input_text,
            "instructions": self._system_prompt,
        }
        if self._previous_response_id is not None:
            payload["previous_response_id"] = self._previous_response_id
        payload.update(kwargs)
        response: Response = self._client.responses.create(**payload)
        if response.id:
            self.__previous_response_id = response.id
        return self.handle_response(response)

    @abstractmethod
    def handle_response(self, response: Response) -> T:
        """
        Convert a raw OpenAI response into the generic output type `T`.
        """
        raise NotImplementedError
