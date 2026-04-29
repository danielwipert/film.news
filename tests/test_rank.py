"""Tests for Stage 2 — dan.llm.rank."""
from __future__ import annotations

import asyncio
import json
from datetime import date
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


def test_select_includes_non_news_when_top_are_all_news():
    """Spec §5.2.2: include at least one non-news item when one is available.

    Regardless of which code path delivers it (category-aware refill or the
    explicit ensure-non-news swap), the outcome is the same.
    """
    scored = [_score(f"a{i:02d}", 5, 5, 5, "news") for i in range(12)]
    scored.append(_score("rev99", 1, 1, 1, "review"))
    titles = {s["id"]: _unique_title(i) for i, s in enumerate(scored)}
    selected, _ = rank._select_with_variety(scored, titles)
    assert any(s["category"] != "news" for s in selected), "non-news must appear in selected"


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


def test_select_dominance_refill_does_not_re_trigger_dominance():
    """After dominance pruning, refill must skip the dominant category.

    Regression: with 12 high-scoring news + 5 lower reviews, naive refill picked
    the next 4 news after dropping the bottom 4 — re-establishing dominance.
    """
    scored = [_score(f"news{i:02d}", 5, 5, 5, "news") for i in range(12)]
    scored += [_score(f"rev{i:02d}", 4, 3, 3, "review") for i in range(5)]
    titles = {s["id"]: _unique_title(i) for i, s in enumerate(scored)}
    selected, _ = rank._select_with_variety(scored, titles)
    news_count = sum(1 for s in selected if s["category"] == "news")
    review_count = sum(1 for s in selected if s["category"] == "review")
    assert news_count < rank.SAME_CATEGORY_LIMIT, f"news count {news_count} re-triggered dominance"
    assert review_count >= 1, "expected variety from non-dominant category"


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


# ---------- cross-episode dedup ----------

def _date_keyed_log_dir(tmp_path):
    """log_dir mock that maps each date to its own subdir so prior-day reads
    work in tests."""
    def _impl(d=None):
        if d is None:
            d = date(2026, 4, 29)
        p = tmp_path / d.isoformat()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return _impl


def _write_ranked(day_dir: Path, ids: list[str]) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "02_ranked.json").write_text(
        json.dumps({"selected": [{"id": i} for i in ids]}), encoding="utf-8",
    )


def test_load_recent_aired_ids_aggregates_across_lookback_window(
    monkeypatch, tmp_path,
):
    """Reads 02_ranked.json from each of the last `lookback_days` and unions
    the selected ids."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    today = date(2026, 4, 29)
    _write_ranked(tmp_path / "2026-04-28", ["a", "b"])
    _write_ranked(tmp_path / "2026-04-27", ["c"])
    _write_ranked(tmp_path / "2026-04-26", ["d", "e"])

    aired = rank._load_recent_aired_ids(today, lookback_days=3)
    assert aired == {"a", "b", "c", "d", "e"}


def test_load_recent_aired_ids_respects_lookback_horizon(monkeypatch, tmp_path):
    """Days outside the lookback window are not consulted, even if they exist."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    today = date(2026, 4, 29)
    _write_ranked(tmp_path / "2026-04-28", ["recent"])
    _write_ranked(tmp_path / "2026-04-20", ["ancient"])

    aired = rank._load_recent_aired_ids(today, lookback_days=3)
    assert aired == {"recent"}
    assert "ancient" not in aired


def test_load_recent_aired_ids_skips_missing_directories(monkeypatch, tmp_path):
    """Early days of the show have no prior episodes — return what we find."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    today = date(2026, 4, 29)
    _write_ranked(tmp_path / "2026-04-28", ["only"])
    # No directory for 2026-04-27 or 2026-04-26.

    aired = rank._load_recent_aired_ids(today, lookback_days=3)
    assert aired == {"only"}


def test_load_recent_aired_ids_skips_corrupt_json(monkeypatch, tmp_path, caplog):
    """An unreadable prior file logs a warning but doesn't break the run."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    today = date(2026, 4, 29)
    bad = tmp_path / "2026-04-28"
    bad.mkdir(parents=True)
    (bad / "02_ranked.json").write_text("{not json", encoding="utf-8")

    aired = rank._load_recent_aired_ids(today, lookback_days=3)
    assert aired == set()


def test_filter_recently_aired_drops_aired_articles():
    articles = [{"id": str(i)} for i in range(10)]
    out = rank._filter_recently_aired(articles, {"3", "7"})
    assert [a["id"] for a in out] == ["0", "1", "2", "4", "5", "6", "8", "9"]


def test_filter_recently_aired_returns_input_when_aired_empty():
    articles = [{"id": "a"}]
    out = rank._filter_recently_aired(articles, set())
    assert out is articles


def test_filter_recently_aired_skips_when_pool_would_drop_below_min():
    """Quiet day: if dedup would leave fewer than MIN_FINAL_COUNT articles,
    accept some repetition rather than ship a thin brief."""
    articles = [{"id": str(i)} for i in range(7)]
    aired = {"0", "1", "2", "3"}  # would leave 3, below MIN_FINAL_COUNT=6
    out = rank._filter_recently_aired(articles, aired)
    assert out == articles


def test_filter_recently_aired_applies_when_pool_stays_at_min():
    """Boundary: dedup is applied when the post-dedup pool is exactly
    MIN_FINAL_COUNT."""
    articles = [{"id": str(i)} for i in range(8)]
    aired = {"0", "1"}  # leaves 6, equal to MIN_FINAL_COUNT
    out = rank._filter_recently_aired(articles, aired)
    assert [a["id"] for a in out] == ["2", "3", "4", "5", "6", "7"]


def test_rank_filters_articles_aired_in_prior_episode(monkeypatch, tmp_path):
    """End-to-end: rank() reads prior days' 02_ranked.json and excludes
    articles already used."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    monkeypatch.setattr(rank, "today_utc", lambda: date(2026, 4, 29))
    monkeypatch.setattr(rank.config, "models", lambda: {"rank": "test/model"})

    # Yesterday: an article was already aired.
    _write_ranked(tmp_path / "2026-04-28", ["film/repeat"])

    # Today's pool: the repeat plus 7 fresh articles.
    today_dir = tmp_path / "2026-04-29"
    today_dir.mkdir(parents=True, exist_ok=True)
    articles = [
        {"id": "film/repeat", "title": "Repeat headline alpha beta gamma delta",
         "trail": "t", "byline": "B", "wordcount": 200,
         "published_at": "2026-04-28T08:00:00Z", "url": "u", "body": "body " * 100},
    ] + [
        {"id": f"film/fresh{i}",
         "title": f"Fresh headline epsilon{i} zeta eta theta iota",
         "trail": "t", "byline": "B", "wordcount": 200,
         "published_at": "2026-04-29T08:00:00Z", "url": "u", "body": "body " * 100}
        for i in range(7)
    ]
    (today_dir / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": articles}), encoding="utf-8",
    )

    async def _fake_complete(**kwargs):
        return json.dumps({
            "newsworthiness": 5, "audibility": 4, "freshness": 5,
            "category": "news", "rationale": "x",
        })
    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(side_effect=_fake_complete)

    out_path = rank.rank(provider=fake_provider)
    data = json.loads(out_path.read_text(encoding="utf-8"))

    selected_ids = [s["id"] for s in data["selected"]]
    assert "film/repeat" not in selected_ids
    # input_count reflects the post-dedup pool fed to the LLM.
    assert data["meta"]["input_count"] == 7
    # The LLM was only called for the 7 surviving articles, not the 8 raw.
    assert fake_provider.complete.await_count == 7


def test_rank_skips_dedup_on_quiet_day(monkeypatch, tmp_path):
    """If applying dedup would shrink the pool below MIN_FINAL_COUNT, the
    repeat is kept and the LLM scores all articles including the repeat."""
    monkeypatch.setattr(rank, "log_dir", _date_keyed_log_dir(tmp_path))
    monkeypatch.setattr(rank, "today_utc", lambda: date(2026, 4, 29))
    monkeypatch.setattr(rank.config, "models", lambda: {"rank": "test/model"})

    # Yesterday: 4 articles aired.
    _write_ranked(tmp_path / "2026-04-28",
                  ["film/a", "film/b", "film/c", "film/d"])

    # Today: only 7 articles, 4 of which were already aired. Dedup would
    # leave 3 — below MIN_FINAL_COUNT — so dedup must be skipped.
    today_dir = tmp_path / "2026-04-29"
    today_dir.mkdir(parents=True, exist_ok=True)
    repeat_ids = ["film/a", "film/b", "film/c", "film/d"]
    fresh_ids = ["film/e", "film/f", "film/g"]
    articles = [
        {"id": i, "title": f"Headline {i} alpha beta gamma delta",
         "trail": "t", "byline": "B", "wordcount": 200,
         "published_at": "x", "url": "u", "body": "body " * 100}
        for i in repeat_ids + fresh_ids
    ]
    (today_dir / "01_raw_articles.json").write_text(
        json.dumps({"meta": {}, "articles": articles}), encoding="utf-8",
    )

    async def _fake_complete(**kwargs):
        return json.dumps({
            "newsworthiness": 5, "audibility": 4, "freshness": 5,
            "category": "news", "rationale": "x",
        })
    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(side_effect=_fake_complete)

    rank.rank(provider=fake_provider)
    # All 7 articles scored — dedup was skipped because the pool was too thin.
    assert fake_provider.complete.await_count == 7


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
