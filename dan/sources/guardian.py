"""Stage 1 — Fetch Guardian film-section articles -> 01_raw_articles.json.

Spec: §4. Pulls the last 24h of /search results from the Guardian Open Platform,
normalizes them to our article shape, applies content filters, and writes the
canonical Stage-1 artifact for downstream stages.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from dan.io import write_json
from dan.paths import log_dir, today_utc

log = logging.getLogger(__name__)

API_URL = "https://content.guardianapis.com/search"
SECTION = "film"
PAGE_SIZE = 50
SHOW_FIELDS = "headline,trailText,bodyText,byline,wordcount,publication"
TIMEOUT_SECS = 30
USER_AGENT = "DAN-FILM/0.1 (+daily film news brief)"

MIN_WORDCOUNT = 100
MIN_VALID_ARTICLES = 5

RETRY_BACKOFFS = (0, 1, 2, 4)  # initial attempt + 3 retries on 5xx
RATE_LIMIT_SLEEP = 60


class FetchError(RuntimeError):
    """Stage 1 hard failure — auth, persistent 5xx, or unparseable response."""


def _request_with_retries(params: dict[str, Any]) -> dict[str, Any]:
    """GET the search endpoint with the spec's retry policy.

    - 5xx: up to 3 retries with 1s/2s/4s backoff.
    - 429: sleep 60s and retry once before falling through to the next attempt.
    """
    headers = {"User-Agent": USER_AGENT}
    last_status: int | None = None
    last_body: str = ""

    for attempt, wait in enumerate(RETRY_BACKOFFS, start=1):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(API_URL, params=params, timeout=TIMEOUT_SECS, headers=headers)
        except requests.RequestException as e:
            log.warning("guardian network error (attempt %d): %s", attempt, e)
            continue

        if r.status_code == 429:
            log.warning("guardian 429 rate-limited; sleeping %ds", RATE_LIMIT_SLEEP)
            time.sleep(RATE_LIMIT_SLEEP)
            try:
                r = requests.get(API_URL, params=params, timeout=TIMEOUT_SECS, headers=headers)
            except requests.RequestException as e:
                log.warning("guardian network error after 429 sleep: %s", e)
                continue

        if 500 <= r.status_code < 600:
            log.warning("guardian %d (attempt %d)", r.status_code, attempt)
            last_status, last_body = r.status_code, r.text[:200]
            continue

        if not r.ok:
            raise FetchError(f"Guardian API returned {r.status_code}: {r.text[:200]}")

        return r.json()

    if last_status is not None:
        raise FetchError(f"Guardian {last_status} after retries: {last_body}")
    raise FetchError("Guardian request failed: network errors on all attempts")


def _normalize(result: dict[str, Any]) -> dict[str, Any] | None:
    """Map a Guardian /search result to our article shape; None if body is missing."""
    fields = result.get("fields") or {}
    body = (fields.get("bodyText") or "").strip()
    if not body:
        return None
    try:
        wordcount = int(fields.get("wordcount") or 0)
    except (TypeError, ValueError):
        wordcount = 0
    return {
        "id": result.get("id", ""),
        "title": fields.get("headline") or result.get("webTitle", ""),
        "trail": (fields.get("trailText") or "").strip(),
        "body": body,
        "byline": (fields.get("byline") or "").strip(),
        "wordcount": wordcount,
        "published_at": result.get("webPublicationDate", ""),
        "url": result.get("webUrl", ""),
    }


def _passes_filters(a: dict[str, Any]) -> bool:
    """§4.4 content filters: drop short bodies and bylineless live blogs.

    The 'Live' check is case-sensitive on purpose: Guardian's live-blog headlines
    use capitalized 'Live' ("Oscars 2024 - Live updates"), while ordinary articles
    about "live-action" remakes use lowercase. Lowercasing the title would
    false-positive on those.
    """
    if a["wordcount"] < MIN_WORDCOUNT:
        return False
    if not a["byline"] and "Live" in a["title"]:
        return False
    return True


def fetch(d: date | None = None, *, days_back: int = 1) -> Path:
    """Fetch articles published since (d - days_back) and write 01_raw_articles.json.

    Production uses days_back=1 per spec §4.2. Higher values are intended for
    local testing on quiet days when one day of Guardian film coverage is too
    thin to exercise the rank/variety logic.

    Returns the output path. Raises FetchError on auth/API failure.
    """
    api_key = os.environ.get("GUARDIAN_API_KEY")
    if not api_key:
        raise FetchError("GUARDIAN_API_KEY not set")
    if days_back < 1:
        raise ValueError(f"days_back must be >= 1, got {days_back}")

    if d is None:
        d = today_utc()
    from_date = (d - timedelta(days=days_back)).isoformat()

    params = {
        "section": SECTION,
        "from-date": from_date,
        "order-by": "newest",
        "page-size": PAGE_SIZE,
        "show-fields": SHOW_FIELDS,
        "api-key": api_key,
    }
    log.info("fetching guardian /search section=%s from=%s", SECTION, from_date)
    payload = _request_with_retries(params)

    response = payload.get("response") or {}
    if response.get("status") != "ok":
        raise FetchError(
            f"Guardian response not ok: status={response.get('status')!r} "
            f"message={response.get('message')!r}"
        )

    raw_results = response.get("results") or []

    normalized: list[dict[str, Any]] = []
    dropped_no_body = 0
    dropped_filtered = 0
    for r in raw_results:
        norm = _normalize(r)
        if norm is None:
            dropped_no_body += 1
            continue
        if not _passes_filters(norm):
            dropped_filtered += 1
            continue
        normalized.append(norm)

    log.info(
        "guardian returned=%d normalized=%d dropped_no_body=%d dropped_filtered=%d",
        len(raw_results), len(normalized), dropped_no_body, dropped_filtered,
    )

    if 0 < len(normalized) < MIN_VALID_ARTICLES:
        # §4.5: with <5 valid articles we still continue (writer handles a quiet day),
        # but this is worth flagging so an operator can spot a degraded source.
        log.warning("only %d valid articles after filtering (spec wants >=5)", len(normalized))

    out = {
        "meta": {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "from_date": from_date,
            "section": SECTION,
            "count": len(normalized),
        },
        "articles": normalized,
    }

    out_path = log_dir(d) / "01_raw_articles.json"
    write_json(out_path, out)
    log.info("wrote %d articles -> %s", len(normalized), out_path)
    return out_path
