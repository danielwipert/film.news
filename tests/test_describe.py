"""Tests for Stage 9.1 — dan.llm.describe."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dan.llm import describe
from dan.llm.describe import DescribeError
from dan.llm.openrouter import LLMError


# ---------- _strip_ssml ----------

def test_strip_ssml_removes_tags_and_collapses_whitespace():
    ssml = (
        "<speak><voice name='x'>"
        "Good morning.   <break time='500ms'/>  Here's the news."
        "</voice></speak>"
    )
    assert describe._strip_ssml(ssml) == "Good morning. Here's the news."


def test_strip_ssml_returns_empty_for_tags_only():
    assert describe._strip_ssml("<speak><voice/></speak>") == ""


# ---------- _validate_description ----------

def test_validate_description_happy_path():
    text = "Today on the brief, two festival headlines and a casting note. The full rundown is inside."
    out = describe._validate_description(text)
    assert out == text


def test_validate_description_strips_outer_quotes():
    text = '"This is a wrapped description that the model returned in quotes for some reason. We strip them."'
    out = describe._validate_description(text)
    assert not out.startswith('"') and not out.endswith('"')
    assert out.startswith("This is")


def test_validate_description_strips_curly_quotes():
    text = "“Curly quotes also need stripping. They show up when models autoformat strings.”"
    out = describe._validate_description(text)
    assert not out.startswith("“") and not out.endswith("”")


def test_validate_description_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        describe._validate_description("")
    with pytest.raises(ValueError, match="empty"):
        describe._validate_description("   ")


def test_validate_description_rejects_too_short():
    with pytest.raises(ValueError, match="too short"):
        describe._validate_description("Short.")


def test_validate_description_rejects_too_long():
    text = "A" * (describe.DESCRIPTION_MAX_CHARS + 1) + "."
    with pytest.raises(ValueError, match="too long"):
        describe._validate_description(text)


def test_validate_description_rejects_missing_terminator():
    text = "This is forty plus characters but it has no period at the end of any sentence so it fails"
    with pytest.raises(ValueError, match="terminator"):
        describe._validate_description(text)


# ---------- _describe_async retry behavior ----------

def _provider(*completions_or_excs) -> MagicMock:
    p = MagicMock()
    p.complete = AsyncMock(side_effect=list(completions_or_excs))
    return p


def test_describe_async_succeeds_first_try():
    import asyncio
    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider(good)
    out = asyncio.run(describe._describe_async(p, "model", "system", "script"))
    assert out == good
    assert p.complete.await_count == 1


def test_describe_async_retries_after_validation_failure():
    import asyncio
    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider("too short", good)
    out = asyncio.run(describe._describe_async(p, "model", "system", "script"))
    assert out == good
    assert p.complete.await_count == 2


def test_describe_async_retries_after_llm_error():
    import asyncio
    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider(LLMError("transient"), good)
    out = asyncio.run(describe._describe_async(p, "model", "system", "script"))
    assert out == good


def test_describe_async_raises_after_two_validation_failures():
    import asyncio
    p = _provider("too short", "also too short")
    with pytest.raises(DescribeError, match="after retry"):
        asyncio.run(describe._describe_async(p, "model", "system", "script"))


def test_describe_async_second_prompt_includes_failure_reason():
    import asyncio
    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider("nope", good)
    asyncio.run(describe._describe_async(p, "model", "BASE", "script"))
    # Second call's system prompt mentions the previous failure reason
    second_kwargs = p.complete.await_args_list[1].kwargs
    assert "BASE" in second_kwargs["system"]
    assert "previous response was not usable" in second_kwargs["system"]


# ---------- describe (orchestration) ----------

def test_describe_writes_output_and_returns_path(monkeypatch, tmp_path):
    monkeypatch.setattr(describe, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(describe.config, "models", lambda: {"describe": "test/model"})
    monkeypatch.setattr(describe, "read_text", lambda p: (
        "<speak>Good morning. Here's the brief. More inside.</speak>"
        if str(p).endswith("04_script.txt")
        else "PROMPT"
    ))

    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider(good)
    out = describe.describe(provider=p)

    assert out == tmp_path / "09_description.txt"
    assert out.read_text(encoding="utf-8").strip() == good
    p.complete.assert_awaited_once()


def test_describe_raises_on_empty_script(monkeypatch, tmp_path):
    monkeypatch.setattr(describe, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(describe.config, "models", lambda: {"describe": "test/model"})
    monkeypatch.setattr(describe, "read_text", lambda p: (
        "<speak><voice/></speak>"
        if str(p).endswith("04_script.txt")
        else "PROMPT"
    ))
    with pytest.raises(DescribeError, match="no spoken text"):
        describe.describe(provider=_provider())


def test_describe_passes_clean_text_to_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(describe, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(describe.config, "models", lambda: {"describe": "test/model"})
    monkeypatch.setattr(describe, "read_text", lambda p: (
        "<speak><voice name='x'>Hello world. <break time='500ms'/> Goodbye world.</voice></speak>"
        if str(p).endswith("04_script.txt")
        else "PROMPT"
    ))

    good = "Today's brief covers two festival stories and a major casting decision. Full details inside."
    p = _provider(good)
    describe.describe(provider=p)

    user = p.complete.await_args.kwargs["user"]
    assert "<speak>" not in user and "<break" not in user
    assert "Hello world." in user and "Goodbye world." in user
