"""Stage 3 — Summarize.

Spec §6: per-article wire-service summary, run in parallel via asyncio.gather.
Reads 02_ranked.json + 01_raw_articles.json (for full body text), writes
03_summaries.json with summary + key_facts per selected item.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from dan import config
from dan.io import read_json, read_text, write_json
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.paths import ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

# Prompt asks for 80-120 words; validate with slack so we don't retry on
# minor over/undershoot. We do retry if the response is clearly broken.
SUMMARY_MIN_WORDS = 30
SUMMARY_MAX_WORDS = 250
KEY_FACTS_MAX = 5  # clamp to spec ceiling
BODY_MAX_CHARS = 8000  # cap very long articles to keep prompts cheap

PROMPT_PATH = ROOT / "dan" / "prompts" / "summarize.txt"


def _build_user_prompt(article: dict[str, Any]) -> str:
    body = (article.get("body") or "")[:BODY_MAX_CHARS]
    return (
        f"Article title: {article.get('title', '')}\n"
        f"Source URL: {article.get('url', '')}\n\n"
        f"Article body:\n{body}"
    )


def _validate_summary(data: Any, *, source_url: str) -> dict[str, Any]:
    """Coerce / validate the LLM response; raise ValueError on bad shape."""
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")

    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("missing or empty summary")
    word_count = len(summary.split())
    if not SUMMARY_MIN_WORDS <= word_count <= SUMMARY_MAX_WORDS:
        raise ValueError(f"summary word count {word_count} outside {SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS}")

    facts_raw = data.get("key_facts", [])
    if not isinstance(facts_raw, list):
        raise ValueError("key_facts is not a list")
    facts = [str(f).strip() for f in facts_raw if str(f).strip()][:KEY_FACTS_MAX]
    if not facts:
        raise ValueError("no usable key_facts")

    return {
        "summary": summary,
        "key_facts": facts,
        "source_url": str(data.get("source_url") or source_url),
    }


async def _summarize_article(
    provider: OpenRouterProvider,
    model: str,
    article: dict[str, Any],
    base_system: str,
) -> dict[str, Any] | None:
    """Return parsed summary fields, or None if both attempts fail."""
    user = _build_user_prompt(article)
    article_id = article.get("id", "")

    for attempt in (1, 2):
        system = base_system
        if attempt == 2:
            system = (
                base_system
                + "\n\nThe previous response was not valid. "
                + "Output strictly the JSON specified above and nothing else."
            )
        try:
            text = await provider.complete(
                system=system, user=user, model=model,
                json_mode=True, temperature=0.2,
            )
            return _validate_summary(json.loads(text), source_url=article.get("url", ""))
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("summarize parse error for %s (attempt %d): %s", article_id, attempt, e)
        except LLMError as e:
            log.warning("summarize LLM error for %s (attempt %d): %s", article_id, attempt, e)

    log.warning("summarize: dropping %s after parse failures", article_id)
    return None


async def _summarize_async(
    selected: list[dict[str, Any]],
    articles_by_id: dict[str, dict[str, Any]],
    model: str,
    provider: OpenRouterProvider,
    base_system: str,
) -> list[dict[str, Any]]:
    """Run one LLM call per selected item in parallel; emit items in rank order."""
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in selected:
        article = articles_by_id.get(item["id"])
        if not article:
            log.warning("summarize: no raw article for %s; skipping", item["id"])
            continue
        pairs.append((item, article))

    summaries = await asyncio.gather(
        *(_summarize_article(provider, model, article, base_system) for _, article in pairs)
    )

    out_items: list[dict[str, Any]] = []
    for (item, article), summary in zip(pairs, summaries):
        if summary is None:
            continue
        out_items.append({
            "id": item["id"],
            "rank": item["rank"],
            "title": article.get("title", ""),
            "summary": summary["summary"],
            "key_facts": summary["key_facts"],
            "source_url": summary["source_url"],
        })
    out_items.sort(key=lambda x: x["rank"])
    return out_items


def summarize(d: date | None = None, *, provider: OpenRouterProvider | None = None) -> Path:
    """Read 01_raw_articles.json + 02_ranked.json, write 03_summaries.json. Returns path."""
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    raw = read_json(day_dir / "01_raw_articles.json")
    ranked = read_json(day_dir / "02_ranked.json")

    articles_by_id = {a["id"]: a for a in (raw.get("articles") or [])}
    selected = ranked.get("selected") or []

    if not selected:
        log.warning("summarize: no selected items in 02_ranked.json; emitting empty")
        out = {"meta": {"model": "(skipped — no selected items)", "count": 0}, "items": []}
        out_path = day_dir / "03_summaries.json"
        write_json(out_path, out)
        return out_path

    model = config.models()["summarize"]
    base_system = read_text(PROMPT_PATH)
    if provider is None:
        provider = OpenRouterProvider()

    log.info("summarize: %d items with %s", len(selected), model)
    items = asyncio.run(_summarize_async(selected, articles_by_id, model, provider, base_system))
    dropped = len(selected) - len(items)
    if dropped:
        log.warning("summarize: dropped %d item(s) after parse failures", dropped)
    log.info("summarize: produced %d summaries", len(items))

    out = {
        "meta": {"model": model, "count": len(items)},
        "items": items,
    }
    out_path = day_dir / "03_summaries.json"
    write_json(out_path, out)
    log.info("wrote summaries -> %s", out_path)
    return out_path
