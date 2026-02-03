from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

from openai_core import OpenAIConfig, OpenAICore

if TYPE_CHECKING:
    from openai.types.responses import Response


class SpanishAITutor(OpenAICore[str]):
    """Spanish tutor implementation using the OpenAI Responses API."""

    def __init__(
        self,
        config: OpenAIConfig,
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

    def handle_response(self, response: Response) -> str:
        """Extract the tutor reply text from a Responses API result."""
        return response.output_text or ""
