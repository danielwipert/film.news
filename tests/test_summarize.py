"""Tests for Stage 3 — dan.llm.summarize."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dan.llm import summarize
from dan.llm.openrouter import LLMError


# ---------- _validate_summary ----------

def _good_summary(words: int = 100) -> dict:
    """Build a valid summary payload of the given word count."""
    return {
        "summary": "word " * words,
        "key_facts": ["Director: Jane Doe", "Date: 2026-04-25", "Budget: $40M"],
        "source_url": "https://x/1",
    }


def test_validate_summary_happy_path():
    out = summarize._validate_summary(_good_summary(100), source_url="https://x/1")
    assert out["summary"].startswith("word")
    assert len(out["key_facts"]) == 3
    assert out["source_url"] == "https://x/1"


def test_validate_summary_rejects_non_dict():
    with pytest.raises(ValueError, match="JSON object"):
        summarize._validate_summary([1, 2], source_url="u")


def test_validate_summary_rejects_empty_summary():
    payload = _good_summary()
    payload["summary"] = "   "
    with pytest.raises(ValueError, match="empty summary"):
        summarize._validate_summary(payload, source_url="u")


def test_validate_summary_rejects_unreasonable_word_count():
    payload = _good_summary(words=10)
    with pytest.raises(ValueError, match="word count"):
        summarize._validate_summary(payload, source_url="u")
    payload = _good_summary(words=300)
    with pytest.raises(ValueError, match="word count"):
        summarize._validate_summary(payload, source_url="u")


def test_validate_summary_clamps_key_facts_to_five():
    payload = _good_summary()
    payload["key_facts"] = [f"fact{i}" for i in range(8)]
    out = summarize._validate_summary(payload, source_url="u")
    assert len(out["key_facts"]) == 5


def test_validate_summary_rejects_when_no_usable_key_facts():
    payload = _good_summary()
    payload["key_facts"] = ["", "   "]
    with pytest.raises(ValueError, match="key_facts"):
        summarize._validate_summary(payload, source_url="u")


def test_validate_summary_uses_default_source_url():
    payload = _good_summary()
    payload["source_url"] = ""
    out = summarize._validate_summary(payload, source_url="https://default")
    assert out["source_url"] == "https://default"


# ---------- _summarize_article ----------

def _provider(*completions_or_excs):
    p = MagicMock()
    p.complete = AsyncMock(side_effect=list(completions_or_excs))
    return p


def _article(id="a1") -> dict:
    return {
        "id": id, "title": "Some Title",
        "url": "https://x/1",
        "body": "This is the article body. " * 50,
    }


def test_summarize_article_happy_path():
    provider = _provider(json.dumps(_good_summary()))
    out = asyncio.run(summarize._summarize_article(provider, "m", _article(), "system"))
    assert out is not None
    assert out["summary"].startswith("word")
    assert provider.complete.await_count == 1


def test_summarize_article_retries_then_succeeds():
    provider = _provider("not json", json.dumps(_good_summary()))
    out = asyncio.run(summarize._summarize_article(provider, "m", _article(), "system"))
    assert out is not None
    assert provider.complete.await_count == 2


def test_summarize_article_returns_none_after_persistent_failure():
    provider = _provider("not json", "still not json")
    out = asyncio.run(summarize._summarize_article(provider, "m", _article(), "system"))
    assert out is None
    assert provider.complete.await_count == 2


def test_summarize_article_returns_none_on_persistent_llm_error():
    provider = _provider(LLMError("502"), LLMError("502 again"))
    out = asyncio.run(summarize._summarize_article(provider, "m", _article(), "system"))
    assert out is None


# ---------- summarize() pipeline entry ----------

def test_summarize_handles_empty_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize, "log_dir", lambda d=None: tmp_path)
    (tmp_path / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": []}), encoding="utf-8"
    )
    (tmp_path / "02_ranked.json").write_text(
        json.dumps({"meta": {}, "selected": [], "rejected": []}), encoding="utf-8"
    )
    out_path = summarize.summarize()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["count"] == 0
    assert data["items"] == []


def test_summarize_end_to_end_with_mocked_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(summarize.config, "models", lambda: {"summarize": "test/model"})

    articles = [
        {"id": f"film/2026/apr/26/story{i}", "title": f"Title {i}",
         "url": f"https://x/{i}", "body": "body " * 200}
        for i in range(3)
    ]
    (tmp_path / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": articles}), encoding="utf-8"
    )
    selected = [
        {"id": articles[0]["id"], "rank": 1, "scores": {}, "rationale": ""},
        {"id": articles[1]["id"], "rank": 2, "scores": {}, "rationale": ""},
        {"id": articles[2]["id"], "rank": 3, "scores": {}, "rationale": ""},
    ]
    (tmp_path / "02_ranked.json").write_text(
        json.dumps({"meta": {}, "selected": selected, "rejected": []}), encoding="utf-8"
    )

    fake = MagicMock()
    fake.complete = AsyncMock(return_value=json.dumps(_good_summary()))

    out_path = summarize.summarize(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert data["meta"]["model"] == "test/model"
    assert data["meta"]["count"] == 3
    assert len(data["items"]) == 3
    # Per spec §6.5 schema
    for item in data["items"]:
        assert set(item.keys()) >= {"id", "rank", "title", "summary", "key_facts", "source_url"}
    # Items preserved in rank order
    assert [it["rank"] for it in data["items"]] == [1, 2, 3]


def test_summarize_drops_articles_missing_in_raw(monkeypatch, tmp_path):
    """If 02_ranked.json references an id that 01_raw_articles.json doesn't have, skip it."""
    monkeypatch.setattr(summarize, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(summarize.config, "models", lambda: {"summarize": "test/model"})

    (tmp_path / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": [
            {"id": "real/article", "title": "Real", "url": "https://x/r", "body": "body " * 200}
        ]}), encoding="utf-8"
    )
    (tmp_path / "02_ranked.json").write_text(
        json.dumps({"meta": {}, "selected": [
            {"id": "real/article", "rank": 1, "scores": {}, "rationale": ""},
            {"id": "ghost/article", "rank": 2, "scores": {}, "rationale": ""},
        ], "rejected": []}), encoding="utf-8"
    )

    fake = MagicMock()
    fake.complete = AsyncMock(return_value=json.dumps(_good_summary()))

    out_path = summarize.summarize(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["count"] == 1
    assert data["items"][0]["id"] == "real/article"


def test_summarize_drops_items_that_fail_validation(monkeypatch, tmp_path):
    """Items where both LLM attempts return garbage are dropped from the output."""
    monkeypatch.setattr(summarize, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(summarize.config, "models", lambda: {"summarize": "test/model"})

    articles = [
        {"id": "a", "title": "A", "url": "https://x/a", "body": "body " * 200},
        {"id": "b", "title": "B", "url": "https://x/b", "body": "body " * 200},
    ]
    (tmp_path / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": articles}), encoding="utf-8"
    )
    (tmp_path / "02_ranked.json").write_text(
        json.dumps({"meta": {}, "selected": [
            {"id": "a", "rank": 1, "scores": {}, "rationale": ""},
            {"id": "b", "rank": 2, "scores": {}, "rationale": ""},
        ], "rejected": []}), encoding="utf-8"
    )

    # First article: clean response. Second article: two garbage responses.
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        json.dumps(_good_summary()),  # a, attempt 1
        "garbage1",                   # b, attempt 1
        "garbage2",                   # b, attempt 2
    ])
    out_path = summarize.summarize(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["count"] == 1
    assert data["items"][0]["id"] == "a"
