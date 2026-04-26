"""Tests for Stage 1 — dan.sources.guardian."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests as _requests

from dan.sources import guardian
from dan.sources.guardian import (
    FetchError,
    _normalize,
    _passes_filters,
    _request_with_retries,
    fetch,
)


# ---------- _normalize ----------

def test_normalize_happy_path():
    raw = {
        "id": "film/2026/apr/26/dune-three",
        "webTitle": "Dune Three (webTitle)",
        "webPublicationDate": "2026-04-26T08:30:00Z",
        "webUrl": "https://www.theguardian.com/film/2026/apr/26/dune-three",
        "fields": {
            "headline": "Dune: Part Three wraps after 14-month shoot",
            "trailText": "  Villeneuve calls it ambitious.  ",
            "bodyText": "Director Denis Villeneuve has confirmed... " * 30,
            "byline": "  Andrew Pulver  ",
            "wordcount": "412",
        },
    }
    n = _normalize(raw)
    assert n is not None
    assert n["id"] == "film/2026/apr/26/dune-three"
    assert n["title"] == "Dune: Part Three wraps after 14-month shoot"  # headline beats webTitle
    assert n["trail"] == "Villeneuve calls it ambitious."
    assert n["body"].startswith("Director Denis Villeneuve")
    assert n["byline"] == "Andrew Pulver"
    assert n["wordcount"] == 412
    assert n["published_at"] == "2026-04-26T08:30:00Z"
    assert n["url"] == "https://www.theguardian.com/film/2026/apr/26/dune-three"


def test_normalize_returns_none_when_body_missing():
    assert _normalize({"id": "x", "fields": {"bodyText": ""}}) is None
    assert _normalize({"id": "x", "fields": {"bodyText": "   "}}) is None
    assert _normalize({"id": "x", "fields": {}}) is None
    assert _normalize({"id": "x"}) is None


def test_normalize_falls_back_to_webtitle_when_no_headline():
    raw = {
        "id": "x",
        "webTitle": "Fallback Title",
        "fields": {"bodyText": "body text", "wordcount": "10"},
    }
    assert _normalize(raw)["title"] == "Fallback Title"


def test_normalize_handles_missing_or_bad_wordcount():
    assert _normalize({"id": "x", "fields": {"bodyText": "b"}})["wordcount"] == 0
    assert _normalize({"id": "x", "fields": {"bodyText": "b", "wordcount": "nope"}})["wordcount"] == 0
    assert _normalize({"id": "x", "fields": {"bodyText": "b", "wordcount": None}})["wordcount"] == 0


# ---------- _passes_filters ----------

def _article(**overrides):
    base = {
        "id": "x", "title": "A reasonable headline", "body": "body",
        "byline": "A. Reporter", "wordcount": 200, "trail": "",
        "published_at": "", "url": "",
    }
    base.update(overrides)
    return base


def test_filter_drops_short_wordcount():
    assert _passes_filters(_article(wordcount=99)) is False
    assert _passes_filters(_article(wordcount=100)) is True


def test_filter_drops_bylineless_live_blog():
    # Guardian's live-blog convention: capitalized "Live" + no byline.
    assert _passes_filters(_article(byline="", title="Oscars 2024 - Live updates")) is False


def test_filter_keeps_lowercase_live_action_remake():
    # Spec-fidelity: case-sensitive 'Live' to avoid false-positive on live-action films.
    assert _passes_filters(_article(byline="", title="Snow White live-action remake bombs")) is True


def test_filter_keeps_live_blog_when_byline_present():
    assert _passes_filters(_article(byline="Reporter", title="Cannes Live")) is True


def test_filter_keeps_normal_article():
    assert _passes_filters(_article()) is True


# ---------- _request_with_retries ----------

def _resp(status_code, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.ok = 200 <= status_code < 300
    r.json.return_value = json_data if json_data is not None else {}
    r.text = text
    return r


def test_retries_on_5xx_then_succeeds():
    responses = [_resp(503), _resp(500), _resp(200, {"hello": "world"})]
    with patch.object(guardian.requests, "get", side_effect=responses) as m_get, \
         patch.object(guardian.time, "sleep") as m_sleep:
        result = _request_with_retries({"q": "x"})
    assert result == {"hello": "world"}
    assert m_get.call_count == 3
    # Backoffs from RETRY_BACKOFFS = (0, 1, 2, 4); first iteration skips sleep.
    assert [c.args[0] for c in m_sleep.call_args_list] == [1, 2]


def test_persistent_5xx_raises_with_status_in_message():
    with patch.object(guardian.requests, "get", return_value=_resp(500, text="oops")), \
         patch.object(guardian.time, "sleep"):
        with pytest.raises(FetchError, match="500"):
            _request_with_retries({})


def test_4xx_raises_immediately_without_retry():
    m_get = MagicMock(return_value=_resp(401, text="bad key"))
    with patch.object(guardian.requests, "get", m_get), \
         patch.object(guardian.time, "sleep"):
        with pytest.raises(FetchError, match="401"):
            _request_with_retries({})
    assert m_get.call_count == 1


def test_429_sleeps_60s_then_retries_once():
    # Per spec §4.4: 429 -> sleep 60s, retry once.
    responses = [_resp(429), _resp(200, {"ok": True})]
    with patch.object(guardian.requests, "get", side_effect=responses) as m_get, \
         patch.object(guardian.time, "sleep") as m_sleep:
        result = _request_with_retries({})
    assert result == {"ok": True}
    assert m_get.call_count == 2
    assert guardian.RATE_LIMIT_SLEEP in [c.args[0] for c in m_sleep.call_args_list]


def test_network_error_retries_with_backoff():
    responses = [_requests.Timeout("timeout"), _resp(200, {"ok": True})]
    with patch.object(guardian.requests, "get", side_effect=responses), \
         patch.object(guardian.time, "sleep"):
        result = _request_with_retries({})
    assert result == {"ok": True}


def test_all_attempts_network_error_raises():
    err = _requests.ConnectionError("nope")
    with patch.object(guardian.requests, "get", side_effect=[err, err, err, err]), \
         patch.object(guardian.time, "sleep"):
        with pytest.raises(FetchError, match="network"):
            _request_with_retries({})


# ---------- fetch ----------

def test_fetch_requires_api_key(monkeypatch):
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    with pytest.raises(FetchError, match="GUARDIAN_API_KEY"):
        fetch()


def test_fetch_writes_normalized_json_with_filters_applied(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")
    monkeypatch.setattr(guardian, "log_dir", lambda d=None: tmp_path)

    payload = {
        "response": {
            "status": "ok",
            "results": [
                {  # keep
                    "id": "film/2026/apr/26/keep",
                    "webPublicationDate": "2026-04-26T08:00:00Z",
                    "webUrl": "https://x/keep",
                    "fields": {
                        "headline": "Keep this one",
                        "bodyText": "body text " * 200,
                        "byline": "Reporter",
                        "wordcount": "300",
                    },
                },
                {  # drop: too short
                    "id": "film/2026/apr/26/short",
                    "fields": {"bodyText": "body", "wordcount": "10"},
                },
                {  # drop: missing body
                    "id": "film/2026/apr/26/nobody",
                    "fields": {"bodyText": ""},
                },
                {  # drop: bylineless live blog
                    "id": "film/2026/apr/26/live",
                    "fields": {
                        "headline": "Oscars - Live updates",
                        "bodyText": "body text " * 200,
                        "byline": "",
                        "wordcount": "300",
                    },
                },
            ],
        }
    }

    with patch.object(guardian.requests, "get", return_value=_resp(200, payload)), \
         patch.object(guardian.time, "sleep"):
        out_path = fetch()

    assert out_path.name == "01_raw_articles.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert data["meta"]["section"] == "film"
    assert data["meta"]["count"] == 1
    assert data["meta"]["from_date"]  # ISO date string present
    assert data["meta"]["fetched_at"].endswith("Z")  # UTC ISO
    assert len(data["articles"]) == 1
    assert data["articles"][0]["id"].endswith("keep")


def test_fetch_raises_when_response_not_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")
    monkeypatch.setattr(guardian, "log_dir", lambda d=None: tmp_path)
    payload = {"response": {"status": "error", "message": "bad key"}}
    with patch.object(guardian.requests, "get", return_value=_resp(200, payload)), \
         patch.object(guardian.time, "sleep"):
        with pytest.raises(FetchError, match="not ok"):
            fetch()


def test_fetch_continues_with_zero_articles(monkeypatch, tmp_path):
    """Per spec §4.5: zero articles is not a hard fail; writer handles a quiet day."""
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")
    monkeypatch.setattr(guardian, "log_dir", lambda d=None: tmp_path)
    payload = {"response": {"status": "ok", "results": []}}
    with patch.object(guardian.requests, "get", return_value=_resp(200, payload)), \
         patch.object(guardian.time, "sleep"):
        out_path = fetch()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["count"] == 0
    assert data["articles"] == []
