"""Stage 2 — Rank.

Spec §5: per-article LLM scoring (newsworthiness/audibility/freshness/category)
followed by a pure-Python sort + variety filter. Reads 01_raw_articles.json,
writes 02_ranked.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dan import config
from dan.io import read_json, read_text, write_json
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.paths import ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

CATEGORIES = ("news", "review", "interview", "opinion", "other")
TOP_CANDIDATES = 8
SAME_CATEGORY_LIMIT = 5     # §5.2.2: 5+ sharing a category triggers prune
SAME_FILM_LIMIT = 3         # §5.2.2: 3+ about the same film triggers prune
MIN_FINAL_COUNT = 6         # §5.2.2: final list 6-10 items
MAX_FINAL_COUNT = 10
VARIETY_FALLBACK_THRESHOLD = 5  # §5.5: <5 surviving variety -> fall back to top-N
DEDUP_LOOKBACK_DAYS = 3     # exclude articles selected in the last N days' episodes

PROMPT_PATH = ROOT / "dan" / "prompts" / "rank_score.txt"

# Tokens we ignore when looking for "same film" anchors in titles.
_STOPWORDS = frozenset({
    "the", "and", "but", "for", "with", "from", "into", "onto", "after",
    "before", "this", "that", "these", "those", "their", "they", "them",
    "his", "her", "its", "your", "ours", "what", "when", "where", "why",
    "how", "than", "then", "also", "just", "more", "most", "much", "many",
    "some", "such", "only", "even", "very", "still", "yet", "as", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "will", "would", "could", "should", "can", "may", "might", "must",
    "review", "reviews", "interview", "film", "films", "movie", "movies",
    "star", "stars", "says", "said", "calls", "called", "new", "best",
    "first", "last", "next", "out", "over", "back", "down", "again",
    "year", "years", "day", "days", "week", "weeks",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_MIN_TOKEN_LEN = 4


# ---------- LLM scoring ----------

def _build_user_prompt(article: dict[str, Any]) -> str:
    body_excerpt = (article.get("body") or "")[:2000]
    return (
        f"Article id: {article.get('id', '')}\n"
        f"Title: {article.get('title', '')}\n"
        f"Trail: {article.get('trail', '')}\n"
        f"Byline: {article.get('byline', '')}\n"
        f"Published: {article.get('published_at', '')}\n"
        f"Wordcount: {article.get('wordcount', 0)}\n\n"
        f"Body excerpt (first 2000 chars):\n{body_excerpt}"
    )


def _validate_score(data: Any, article_id: str) -> dict[str, Any]:
    """Coerce / validate a parsed JSON score; raise ValueError on bad shape."""
    if not isinstance(data, dict):
        raise ValueError("score is not a JSON object")
    out: dict[str, Any] = {"id": article_id}
    for k in ("newsworthiness", "audibility", "freshness"):
        if k not in data:
            raise ValueError(f"missing score: {k}")
        v = int(data[k])
        if not 1 <= v <= 5:
            raise ValueError(f"{k}={v} out of range 1-5")
        out[k] = v
    cat = data.get("category", "other")
    if cat not in CATEGORIES:
        raise ValueError(f"unknown category: {cat!r}")
    out["category"] = cat
    out["rationale"] = str(data.get("rationale", "")).strip()
    return out


def _default_score(article_id: str, reason: str) -> dict[str, Any]:
    """§5.5 fallback: middle scores when parsing fails twice."""
    return {
        "id": article_id,
        "newsworthiness": 3,
        "audibility": 3,
        "freshness": 3,
        "category": "other",
        "rationale": f"(default 3/3/3 fallback: {reason})",
    }


async def _score_article(
    provider: OpenRouterProvider,
    model: str,
    article: dict[str, Any],
    base_system: str,
) -> dict[str, Any]:
    """LLM-score one article. §5.5: retry once with a stricter prompt; then default."""
    user = _build_user_prompt(article)
    article_id = article.get("id", "")

    for attempt in (1, 2):
        system = base_system
        if attempt == 2:
            system = (
                base_system
                + "\n\nThe previous response was not valid JSON. "
                + "Respond with ONLY the JSON object specified above."
            )
        try:
            text = await provider.complete(
                system=system, user=user, model=model,
                json_mode=True, temperature=0.2,
            )
            return _validate_score(json.loads(text), article_id)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("score parse error for %s (attempt %d): %s", article_id, attempt, e)
        except LLMError as e:
            log.warning("score LLM error for %s (attempt %d): %s", article_id, attempt, e)

    return _default_score(article_id, "LLM/JSON failure")


# ---------- variety filter ----------

def _total(score: dict[str, Any]) -> int:
    return score["newsworthiness"] + score["audibility"] + score["freshness"]


def _significant_tokens(title: str) -> set[str]:
    """Return lowercase tokens >=4 chars that aren't stopwords."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(title or "")
        if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _STOPWORDS
    }


def _sort_key(score: dict[str, Any]) -> tuple[int, int, str]:
    # Sort by total desc, then newsworthiness desc, then id asc for stability.
    return (-_total(score), -score["newsworthiness"], score["id"])


def _select_with_variety(
    scored: list[dict[str, Any]],
    titles_by_id: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (selected_in_score_order, rejected_with_reasons) per spec §5.2.2.

    Implementation order:
      1. Take top TOP_CANDIDATES by total score.
      2. Same-film prune: if SAME_FILM_LIMIT+ candidates share any significant
         title token, keep the highest-scoring; reject the rest.
      3. Category-dominance prune: if SAME_CATEGORY_LIMIT+ share a category,
         drop low-scorers in that category.
      4. Refill from the next tier (excluding rejected ids).
      5. Ensure at least one non-news item if available in the wider pool.
      6. §5.5: if fewer than VARIETY_FALLBACK_THRESHOLD survive, fall back to top-N.
      7. Clamp to MAX_FINAL_COUNT.
    """
    all_sorted = sorted(scored, key=_sort_key)
    if not all_sorted:
        return [], []

    candidates = list(all_sorted[:TOP_CANDIDATES])
    rejected: list[dict[str, Any]] = []
    rejected_ids: set[str] = set()

    candidates = _prune_same_film(candidates, titles_by_id, rejected, rejected_ids)
    candidates = _prune_category_dominance(candidates, rejected, rejected_ids)
    # Refill after pruning, but respect the category cap so we don't refill the
    # same dominant category back in (which would re-trigger dominance).
    candidates = _refill(candidates, all_sorted, TOP_CANDIDATES,
                         exclude_ids=rejected_ids, respect_category_cap=True)
    candidates = _ensure_non_news(candidates, all_sorted, rejected, rejected_ids)

    # §5.5 fallback if variety prune left us thin.
    if len(candidates) < VARIETY_FALLBACK_THRESHOLD and len(all_sorted) >= VARIETY_FALLBACK_THRESHOLD:
        log.warning(
            "only %d candidates survived variety filter; falling back to top-%d by score",
            len(candidates), MIN_FINAL_COUNT,
        )
        candidates = list(all_sorted[:max(MIN_FINAL_COUNT, len(candidates))])
        kept_ids = {s["id"] for s in candidates}
        rejected = [{"id": s["id"], "reason": "lower total score"}
                    for s in all_sorted if s["id"] not in kept_ids]

    if len(candidates) > MAX_FINAL_COUNT:
        kept_list = sorted(candidates, key=_sort_key)[:MAX_FINAL_COUNT]
        kept_ids = {s["id"] for s in kept_list}
        for s in candidates:
            if s["id"] not in kept_ids:
                rejected.append({"id": s["id"], "reason": "exceeds max final count"})
        candidates = kept_list

    selected = sorted(candidates, key=_sort_key)
    # De-dupe rejected by id, last-write-wins.
    seen: dict[str, dict[str, Any]] = {}
    for r in rejected:
        seen[r["id"]] = r
    return selected, list(seen.values())


def _refill(
    candidates: list[dict[str, Any]],
    all_sorted: list[dict[str, Any]],
    target: int,
    *,
    exclude_ids: set[str],
    respect_category_cap: bool = False,
) -> list[dict[str, Any]]:
    """Top up candidates from the next tier, skipping rejected ids.

    If respect_category_cap is True, skip items whose category is already at
    SAME_CATEGORY_LIMIT - 1 in the current selection — so refill never pushes
    a category back over the dominance threshold we just pruned to.
    """
    out = list(candidates)
    selected = {s["id"] for s in out}
    counts = Counter(s["category"] for s in out)
    for s in all_sorted:
        if len(out) >= target:
            break
        if s["id"] in selected or s["id"] in exclude_ids:
            continue
        if respect_category_cap and counts[s["category"]] + 1 >= SAME_CATEGORY_LIMIT:
            continue
        out.append(s)
        selected.add(s["id"])
        counts[s["category"]] += 1
    return out


def _prune_same_film(
    candidates: list[dict[str, Any]],
    titles_by_id: dict[str, str],
    rejected: list[dict[str, Any]],
    rejected_ids: set[str],
) -> list[dict[str, Any]]:
    """If SAME_FILM_LIMIT+ candidates share any significant token, keep the highest-scoring.

    Iterates: after pruning a group, recompute counts in case other tokens still
    cluster. Tiebreak among equally-shared tokens picks the longest (more specific).
    """
    out = list(candidates)
    while True:
        token_to_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in out:
            for t in _significant_tokens(titles_by_id.get(s["id"], "")):
                token_to_group[t].append(s)
        if not token_to_group:
            break
        best_token = max(token_to_group, key=lambda t: (len(token_to_group[t]), len(t)))
        group = token_to_group[best_token]
        if len(group) < SAME_FILM_LIMIT:
            break
        keeper = sorted(group, key=_sort_key)[0]
        for s in group:
            if s["id"] == keeper["id"]:
                continue
            out = [c for c in out if c["id"] != s["id"]]
            rejected.append({"id": s["id"], "reason": f"duplicate film anchor ({best_token})"})
            rejected_ids.add(s["id"])
    return out


def _prune_category_dominance(
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    rejected_ids: set[str],
) -> list[dict[str, Any]]:
    out = list(candidates)
    counts = Counter(s["category"] for s in out)
    while any(c >= SAME_CATEGORY_LIMIT for c in counts.values()):
        dominant = max(counts, key=lambda c: counts[c])
        in_cat = sorted([s for s in out if s["category"] == dominant], key=_sort_key)
        loser = in_cat[-1]  # _sort_key sorts best-first, so [-1] is the worst in-category
        out = [s for s in out if s["id"] != loser["id"]]
        rejected.append({"id": loser["id"], "reason": f"category dominance ({dominant})"})
        rejected_ids.add(loser["id"])
        counts[dominant] -= 1
    return out


def _ensure_non_news(
    candidates: list[dict[str, Any]],
    all_sorted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    rejected_ids: set[str],
) -> list[dict[str, Any]]:
    if any(s["category"] != "news" for s in candidates):
        return candidates

    selected_ids = {s["id"] for s in candidates}
    fallback = next(
        (s for s in all_sorted
         if s["id"] not in selected_ids and s["id"] not in rejected_ids and s["category"] != "news"),
        None,
    )
    if fallback is None:
        return candidates  # genuinely no variety available in the pool

    # Drop the lowest-scoring news item, swap in the fallback.
    by_score = sorted(candidates, key=_sort_key)
    swapped_out = by_score[-1]
    new_candidates = [s for s in candidates if s["id"] != swapped_out["id"]] + [fallback]
    rejected.append({"id": swapped_out["id"], "reason": "swapped for tonal variety"})
    rejected_ids.add(swapped_out["id"])
    return new_candidates


# ---------- cross-episode dedup ----------

def _load_recent_aired_ids(
    d: date, lookback_days: int = DEDUP_LOOKBACK_DAYS,
) -> set[str]:
    """Collect article ids selected in the prior `lookback_days` 02_ranked.json
    files. Missing day directories or unreadable JSON are skipped silently —
    early days of the show simply have nothing to dedup against."""
    aired: set[str] = set()
    for k in range(1, lookback_days + 1):
        prior_dir = log_dir(d - timedelta(days=k))
        ranked_path = prior_dir / "02_ranked.json"
        if not ranked_path.exists():
            continue
        try:
            data = read_json(ranked_path)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("dedup: could not read %s: %s", ranked_path, e)
            continue
        for sel in (data.get("selected") or []):
            sid = sel.get("id")
            if sid:
                aired.add(sid)
    return aired


def _filter_recently_aired(
    articles: list[dict[str, Any]], aired: set[str],
) -> list[dict[str, Any]]:
    """Drop articles whose id was selected in a prior episode within the lookback
    window. Skipped entirely if applying it would leave fewer than
    MIN_FINAL_COUNT articles — on a quiet day we'd rather accept some
    repetition than ship a thin brief."""
    if not aired:
        return articles
    deduped = [a for a in articles if a.get("id") not in aired]
    dropped = len(articles) - len(deduped)
    if dropped == 0:
        return articles
    if len(deduped) < MIN_FINAL_COUNT:
        log.warning(
            "dedup: %d previously-aired article(s) found, but applying the "
            "filter would leave %d in the pool (min %d) — skipping for this run",
            dropped, len(deduped), MIN_FINAL_COUNT,
        )
        return articles
    log.info(
        "dedup: dropped %d previously-aired article(s) (%d-day lookback)",
        dropped, DEDUP_LOOKBACK_DAYS,
    )
    return deduped


# ---------- pipeline entry ----------

def _format_selected(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape selected to match spec §5.4 schema: id/rank/scores/rationale/category."""
    return [
        {
            "id": s["id"],
            "rank": i + 1,
            "scores": {
                "newsworthiness": s["newsworthiness"],
                "audibility": s["audibility"],
                "freshness": s["freshness"],
                "total": _total(s),
            },
            "category": s["category"],
            "rationale": s["rationale"],
        }
        for i, s in enumerate(selected)
    ]


async def _rank_async(
    articles: list[dict[str, Any]],
    model: str,
    provider: OpenRouterProvider,
    base_system: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = await asyncio.gather(
        *(_score_article(provider, model, a, base_system) for a in articles)
    )
    titles_by_id = {a["id"]: a.get("title", "") for a in articles}
    return _select_with_variety(list(scored), titles_by_id)


def rank(d: date | None = None, *, provider: OpenRouterProvider | None = None) -> Path:
    """Read 01_raw_articles.json, score every article, write 02_ranked.json. Returns path."""
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    raw = read_json(day_dir / "01_raw_articles.json")
    articles = raw.get("articles") or []

    # Cross-episode dedup: drop articles already aired in the last few days.
    aired = _load_recent_aired_ids(d)
    articles = _filter_recently_aired(articles, aired)

    if not articles:
        log.warning("rank: zero articles in 01_raw_articles.json; emitting empty selection")
        out = {
            "meta": {"model": "(skipped — no articles)", "input_count": 0, "output_count": 0},
            "selected": [],
            "rejected": [],
        }
        out_path = day_dir / "02_ranked.json"
        write_json(out_path, out)
        return out_path

    model = config.models()["rank"]
    base_system = read_text(PROMPT_PATH)
    if provider is None:
        provider = OpenRouterProvider()

    log.info("rank: scoring %d articles with %s", len(articles), model)
    selected, rejected = asyncio.run(_rank_async(articles, model, provider, base_system))
    log.info("rank: selected %d, rejected %d", len(selected), len(rejected))

    out = {
        "meta": {
            "model": model,
            "input_count": len(articles),
            "output_count": len(selected),
        },
        "selected": _format_selected(selected),
        "rejected": rejected,
    }

    out_path = day_dir / "02_ranked.json"
    write_json(out_path, out)
    log.info("wrote ranked -> %s", out_path)
    return out_path
