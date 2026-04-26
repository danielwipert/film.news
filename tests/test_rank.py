"""Tests for Stage 2 — dan.llm.rank."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dan.llm import rank
from dan.llm.openrouter import LLMError


# ---------- _validate_score ----------

def test_validate_score_happy_path():
    out = rank._validate_score(
        {"newsworthiness": 5, "audibility": 4, "freshness": 3,
         "category": "news", "rationale": "Big deal."},
        article_id="a1",
    )
    assert out == {
        "id": "a1", "newsworthiness": 5, "audibility": 4, "freshness": 3,
        "category": "news", "rationale": "Big deal.",
    }


def test_validate_score_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        rank._validate_score(
            {"newsworthiness": 7, "audibility": 4, "freshness": 3, "category": "news"},
            article_id="a1",
        )


def test_validate_score_rejects_unknown_category():
    with pytest.raises(ValueError, match="category"):
        rank._validate_score(
            {"newsworthiness": 3, "audibility": 3, "freshness": 3, "category": "gossip"},
            article_id="a1",
        )


def test_validate_score_rejects_non_dict():
    with pytest.raises(ValueError, match="JSON object"):
        rank._validate_score([1, 2, 3], article_id="a1")


# ---------- _score_article ----------

def _provider(*completions_or_excs):
    p = MagicMock()
    p.complete = AsyncMock(side_effect=list(completions_or_excs))
    return p


def _article(id="a1", title="Some Title"):
    return {
        "id": id, "title": title, "trail": "trail", "byline": "Byline",
        "wordcount": 200, "published_at": "2026-04-26T08:00:00Z",
        "url": "https://x", "body": "body " * 100,
    }


def test_score_article_happy_path():
    provider = _provider(json.dumps({
        "newsworthiness": 5, "audibility": 4, "freshness": 5,
        "category": "news", "rationale": "Big.",
    }))
    out = asyncio.run(rank._score_article(provider, "m", _article("a1"), "system"))
    assert out["id"] == "a1"
    assert out["newsworthiness"] == 5
    assert out["category"] == "news"
    assert provider.complete.await_count == 1


def test_score_article_retries_then_fallback():
    # First call returns malformed JSON; second returns invalid shape; both fail.
    provider = _provider("not-json", json.dumps({"foo": "bar"}))
    out = asyncio.run(rank._score_article(provider, "m", _article("a1"), "system"))
    assert out["newsworthiness"] == 3
    assert out["audibility"] == 3
    assert out["freshness"] == 3
    assert "fallback" in out["rationale"]
    assert provider.complete.await_count == 2


def test_score_article_recovers_on_second_attempt():
    provider = _provider(
        "not json",
        json.dumps({"newsworthiness": 4, "audibility": 4, "freshness": 4,
                    "category": "review", "rationale": "Solid."}),
    )
    out = asyncio.run(rank._score_article(provider, "m", _article("a1"), "system"))
    assert out["category"] == "review"
    assert out["newsworthiness"] == 4


def test_score_article_falls_back_on_persistent_llm_error():
    provider = _provider(LLMError("502"), LLMError("502 again"))
    out = asyncio.run(rank._score_article(provider, "m", _article("a1"), "system"))
    assert out["newsworthiness"] == 3


# ---------- _significant_tokens ----------

def test_significant_tokens_strips_stopwords_and_short():
    toks = rank._significant_tokens("Dune: Part Three wraps after a 14-month shoot")
    assert "dune" in toks
    assert "part" in toks      # exactly 4 chars, kept
    assert "three" in toks
    assert "wraps" in toks
    assert "shoot" in toks
    assert "after" not in toks  # stopword
    assert "a" not in toks      # too short


def test_significant_tokens_handles_empty():
    assert rank._significant_tokens("") == set()
    assert rank._significant_tokens(None) == set()


# ---------- _select_with_variety ----------

def _score(id, n, a, f, cat="news", rat=""):
    return {"id": id, "newsworthiness": n, "audibility": a, "freshness": f,
            "category": cat, "rationale": rat}


def _unique_title(i: int) -> str:
    """Build a title with three significant tokens, all unique to index i.

    Letter-only embedding keeps the tokens above the 4-char threshold and
    distinct from every other index, so the same-film heuristic stays quiet.
    """
    a, b, c = chr(97 + i), chr(97 + i + 1), chr(97 + i + 2)
    return f"alpha{a}headline beta{b}story gamma{c}content"


def test_select_keeps_top_when_no_variety_issues():
    scored = [_score(f"a{i}", 5, 5 - (i % 3), 5, "news") for i in range(12)]
    titles = {f"a{i}": _unique_title(i) for i in range(12)}
    selected, _ = rank._select_with_variety(scored, titles)
    assert 6 <= len(selected) <= rank.MAX_FINAL_COUNT
    # All-news input -> may have been swapped to include a non-news item if one
    # existed; here all are news, so no swap happens. Selected stays all news.


def test_select_swaps_in_non_news_when_all_top_are_news():
    # 12 news + 1 lower-scoring review. After category-dominance prune and
    # refill, candidates are still all news; _ensure_non_news swaps the review in.
    scored = [_score(f"a{i:02d}", 5, 5, 5, "news") for i in range(12)]
    scored.append(_score("rev99", 1, 1, 1, "review"))
    titles = {s["id"]: _unique_title(i) for i, s in enumerate(scored)}
    selected, rejected = rank._select_with_variety(scored, titles)
    assert any(s["category"] != "news" for s in selected), "non-news should be swapped in"
    assert any("variety" in r["reason"] for r in rejected)


def test_select_prunes_same_film_anchor():
    scored = [
        _score("dune1", 5, 5, 5),
        _score("dune2", 5, 5, 4),
        _score("dune3", 5, 4, 4),
        _score("other1", 4, 4, 4, "review"),
        _score("other2", 4, 4, 3, "interview"),
        _score("other3", 4, 3, 3),
        _score("other4", 3, 3, 3),
    ]
    titles = {
        "dune1": "Dune Part Three wraps shoot",
        "dune2": "Villeneuve talks Dune sequel premiere",
        "dune3": "Cast reflects on Dune production",
        "other1": "Cannes Festival opens surprise",
        "other2": "Profile reveals new ambition",
        "other3": "Office report weekend",
        "other4": "Studio greenlight project",
    }
    selected, rejected = rank._select_with_variety(scored, titles)
    dune_selected = [s for s in selected if s["id"].startswith("dune")]
    assert len(dune_selected) == 1, "only the highest-scoring dune item should survive"
    assert dune_selected[0]["id"] == "dune1"
    assert any("dune" in r["reason"].lower() for r in rejected)


def test_select_prunes_category_dominance():
    # 6 reviews + 3 news; top-8 candidates = 6 reviews + 2 news; 6 reviews >= 5
    # triggers prune until reviews < 5; refill pulls in news from the rest.
    scored = [_score(f"r{i}", 5, 5, 5, "review") for i in range(6)]
    scored += [_score(f"n{i}", 4, 4, 4, "news") for i in range(3)]
    titles = {s["id"]: _unique_title(i) for i, s in enumerate(scored)}
    selected, rejected = rank._select_with_variety(scored, titles)
    review_count = sum(1 for s in selected if s["category"] == "review")
    assert review_count < rank.SAME_CATEGORY_LIMIT, "category dominance should have been pruned"
    assert any("category" in r["reason"] for r in rejected)


def test_select_handles_empty_input():
    selected, rejected = rank._select_with_variety([], {})
    assert selected == []
    assert rejected == []


def test_select_falls_back_when_variety_empties_list():
    # Pathological: 6 articles all share a title anchor so same-film prune
    # collapses them to 1. Spec §5.5: fall back to top-N-by-score regardless.
    scored = [_score(f"a{i}", 5, 5, 5) for i in range(6)]
    titles = {s["id"]: f"shared anchor token specific{chr(97+i)}id" for i, s in enumerate(scored)}
    selected, _ = rank._select_with_variety(scored, titles)
    assert len(selected) >= min(rank.MIN_FINAL_COUNT, len(scored))


# ---------- rank() pipeline entry ----------

def test_rank_handles_zero_articles(monkeypatch, tmp_path):
    # Set up a logs/today dir with an empty 01_raw_articles.json.
    monkeypatch.setattr(rank, "log_dir", lambda d=None: tmp_path)
    (tmp_path / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": []}), encoding="utf-8"
    )
    out_path = rank.rank()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["output_count"] == 0
    assert data["selected"] == []


def test_rank_end_to_end_with_mocked_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(rank, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(rank.config, "models", lambda: {"rank": "test/model"})

    articles = [
        {"id": f"film/2026/apr/26/story{i}", "title": f"Distinct headline alpha {i} beta gamma delta",
         "trail": "trail", "byline": "B", "wordcount": 200,
         "published_at": "2026-04-26T08:00:00Z", "url": "u", "body": "body " * 100}
        for i in range(8)
    ]
    raw = {"meta": {}, "articles": articles}
    (tmp_path / "01_raw_articles.json").write_text(json.dumps(raw), encoding="utf-8")

    # Mocked provider returns a deterministic score for every article.
    def _payload(article_id, idx):
        return json.dumps({
            "newsworthiness": 5 - (idx % 3),
            "audibility": 4,
            "freshness": 5,
            "category": "news" if idx % 2 == 0 else "review",
            "rationale": f"because {idx}",
        })

    call_count = {"n": 0}

    async def _fake_complete(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        # Pull the article id out of the user prompt for round-trip into JSON output.
        article_id = kwargs["user"].split("\n", 1)[0].split(": ", 1)[1]
        return _payload(article_id, idx)

    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(side_effect=_fake_complete)

    out_path = rank.rank(provider=fake_provider)
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert data["meta"]["model"] == "test/model"
    assert data["meta"]["input_count"] == 8
    assert 6 <= data["meta"]["output_count"] <= 10
    # Selected items have the spec §5.4 shape.
    for item in data["selected"]:
        assert set(item.keys()) >= {"id", "rank", "scores", "rationale"}
        assert set(item["scores"].keys()) >= {"newsworthiness", "audibility", "freshness"}
    # Ranks contiguous 1..N
    ranks = [item["rank"] for item in data["selected"]]
    assert ranks == list(range(1, len(ranks) + 1))
