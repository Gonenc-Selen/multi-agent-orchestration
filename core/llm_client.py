from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from core.config import settings

if TYPE_CHECKING:
    from google import genai as genai_module

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0

# Approximate Gemini 2.5 Pro pricing (per 1M tokens)
_INPUT_COST_PER_M = 1.25
_OUTPUT_COST_PER_M = 10.0


class LLMClient:
    """Vertex AI Gemini wrapper. genai.Client is lazily initialized on first call."""

    def __init__(self) -> None:
        self._client: genai_module.Client | None = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def _get_client(self) -> genai_module.Client:
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
            )
        return self._client

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Call Gemini with structured output. Retries up to 3 times with backoff.

        Raises:
            ValueError: if all retries fail.
        """
        from google.genai import types

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.6,
                        response_mime_type="application/json",
                        response_schema=schema,
                        max_output_tokens=8192,
                        seed=settings.random_seed,
                    ),
                )
                if response.usage_metadata:
                    self.total_prompt_tokens += (
                        response.usage_metadata.prompt_token_count or 0
                    )
                    self.total_completion_tokens += (
                        response.usage_metadata.candidates_token_count or 0
                    )
                parsed = response.parsed
                if parsed is None:
                    finish = (
                        response.candidates[0].finish_reason
                        if response.candidates
                        else "unknown"
                    )
                    raise ValueError(f"response.parsed is None (finish_reason={finish})")
                return parsed  # type: ignore[return-value]

            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY_S * (2**attempt)
                    logger.warning(
                        "LLM attempt %d/%d failed: %s. Retrying in %.1fs.",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise ValueError(
            f"LLM call failed after {_MAX_RETRIES} retries: {last_error}"
        )

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.total_prompt_tokens / 1_000_000 * _INPUT_COST_PER_M
            + self.total_completion_tokens / 1_000_000 * _OUTPUT_COST_PER_M
        )


llm_client = LLMClient()
