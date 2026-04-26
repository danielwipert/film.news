"""OpenRouter HTTP client.

Implements the LLMProvider abstraction (spec §2.3): a thin async client with a
single `complete(system, user, model, ...)` method that returns the model's
text response. Retries once on transient failures (5xx, 429, timeouts, network
errors) per spec §18.1.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 60.0
RETRY_BACKOFF_SECS = 2.0
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

# OpenRouter recommends sending these so usage shows up in their dashboard.
HTTP_REFERER = "https://github.com/danielwipert/film.news"
APP_TITLE = "DAN-FILM"


class LLMError(RuntimeError):
    """Raised when an LLM call fails non-recoverably."""


class OpenRouterProvider:
    """Async OpenRouter chat-completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY not set")
        self.timeout = timeout
        self._client = client  # injected for tests; otherwise we make one per call

    @property
    def name(self) -> str:
        return "openrouter"

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat-completion and return the assistant's text content.

        Retries once on transient HTTP statuses (5xx, 429) and on network errors.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last_err: str | None = None
        for attempt in (1, 2):
            try:
                response = await self._post(headers, body)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_err = f"network: {e}"
                log.warning("openrouter network error (attempt %d/2): %s", attempt, e)
                if attempt == 1:
                    await asyncio.sleep(RETRY_BACKOFF_SECS)
                    continue
                raise LLMError(f"OpenRouter request failed: {last_err}") from e

            if response.status_code in TRANSIENT_STATUSES:
                last_err = f"{response.status_code}: {response.text[:200]}"
                log.warning("openrouter transient %s (attempt %d/2)", response.status_code, attempt)
                if attempt == 1:
                    await asyncio.sleep(RETRY_BACKOFF_SECS)
                    continue
                raise LLMError(f"OpenRouter transient failure after retry: {last_err}")

            if response.status_code != 200:
                raise LLMError(f"OpenRouter {response.status_code}: {response.text[:300]}")

            try:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                raise LLMError(f"OpenRouter response shape unexpected: {response.text[:300]}") from e

        # Unreachable: both branches above either return or raise.
        raise LLMError(f"OpenRouter request failed: {last_err}")

    async def _post(self, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(OPENROUTER_URL, headers=headers, json=body, timeout=self.timeout)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(OPENROUTER_URL, headers=headers, json=body)
