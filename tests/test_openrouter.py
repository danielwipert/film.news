"""Tests for the OpenRouter LLM client."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dan.llm import openrouter
from dan.llm.openrouter import LLMError, OpenRouterProvider


def _resp(status: int, payload: dict | None = None, text: str = "") -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text or (str(payload) if payload else "")
    return r


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _fake_sleep(_):
        return None
    monkeypatch.setattr(openrouter.asyncio, "sleep", _fake_sleep)


def _provider_with_client(*responses_or_excs):
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=list(responses_or_excs))
    return OpenRouterProvider(api_key="test-key", client=client), client


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(api_key=None)


def test_happy_path_returns_content():
    payload = {"choices": [{"message": {"content": "hello world"}}]}
    provider, client = _provider_with_client(_resp(200, payload))
    out = asyncio.run(provider.complete(system="s", user="u", model="m"))
    assert out == "hello world"
    assert client.post.await_count == 1
    sent = client.post.await_args.kwargs
    assert sent["json"]["model"] == "m"
    assert sent["json"]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert "response_format" not in sent["json"]
    assert sent["headers"]["Authorization"] == "Bearer test-key"


def test_json_mode_sets_response_format():
    payload = {"choices": [{"message": {"content": "{}"}}]}
    provider, client = _provider_with_client(_resp(200, payload))
    asyncio.run(provider.complete(system="s", user="u", model="m", json_mode=True))
    body = client.post.await_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}


def test_5xx_retries_once_then_succeeds():
    payload = {"choices": [{"message": {"content": "ok"}}]}
    provider, client = _provider_with_client(_resp(503, text="busy"), _resp(200, payload))
    out = asyncio.run(provider.complete(system="s", user="u", model="m"))
    assert out == "ok"
    assert client.post.await_count == 2


def test_persistent_5xx_raises():
    provider, _ = _provider_with_client(_resp(500, text="boom"), _resp(500, text="still boom"))
    with pytest.raises(LLMError, match="500"):
        asyncio.run(provider.complete(system="s", user="u", model="m"))


def test_4xx_raises_immediately_without_retry():
    provider, client = _provider_with_client(_resp(401, text="bad key"))
    with pytest.raises(LLMError, match="401"):
        asyncio.run(provider.complete(system="s", user="u", model="m"))
    assert client.post.await_count == 1


def test_429_retries_once():
    payload = {"choices": [{"message": {"content": "rl ok"}}]}
    provider, client = _provider_with_client(_resp(429, text="slow down"), _resp(200, payload))
    out = asyncio.run(provider.complete(system="s", user="u", model="m"))
    assert out == "rl ok"
    assert client.post.await_count == 2


def test_network_error_retries():
    payload = {"choices": [{"message": {"content": "nope-then-ok"}}]}
    provider, _ = _provider_with_client(httpx.ConnectError("nope"), _resp(200, payload))
    out = asyncio.run(provider.complete(system="s", user="u", model="m"))
    assert out == "nope-then-ok"


def test_persistent_network_error_raises():
    provider, _ = _provider_with_client(
        httpx.ConnectError("nope"), httpx.ConnectError("still nope"),
    )
    with pytest.raises(LLMError, match="network"):
        asyncio.run(provider.complete(system="s", user="u", model="m"))


def test_unexpected_response_shape_raises():
    provider, _ = _provider_with_client(_resp(200, {"unexpected": "shape"}))
    with pytest.raises(LLMError, match="shape unexpected"):
        asyncio.run(provider.complete(system="s", user="u", model="m"))
