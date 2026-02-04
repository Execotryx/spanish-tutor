from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

from ollama_core import OllamaChatCompletionsCore, OllamaConfig

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


class SpanishAITutor(OllamaChatCompletionsCore[str]):
    """Spanish tutor implementation using the Ollama Chat Completions API."""

    def __init__(
        self,
        config: OllamaConfig,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__(
            config=config,
            system_prompt=(
                "You are a patient, encouraging Spanish tutor. "
                "Respond in Spanish, explain mistakes clearly, and provide brief examples."
            ),
            client=client,
        )

    def handle_response(self, response: ChatCompletion) -> str:
        """Extract the tutor reply text from a Chat Completions result."""
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""
