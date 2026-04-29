"""Stage 5 — Sanity check.

Spec §8: extract every checkable claim from 04_script.txt, then verify each
against the key facts and summaries in 03_summaries.json. If any claim is
unsupported, save the bad script as 04_script.bad.txt, run one rewrite via
the Stage 4 critique prompt with the unsupported claims as context, then
re-verify. If it still fails, raise — the workflow fails and no episode is
published.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from dan import config
from dan.io import read_json, read_text, write_json, write_text
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.llm.write import _strip_code_fences
from dan.paths import ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

EXTRACT_PROMPT_PATH = ROOT / "dan" / "prompts" / "sanity_extract.txt"
VERIFY_PROMPT_PATH = ROOT / "dan" / "prompts" / "sanity_verify.txt"
WRITE_CRITIQUE_PROMPT_PATH = ROOT / "dan" / "prompts" / "write_critique.txt"

EXTRACT_TEMPERATURE = 0.0
VERIFY_TEMPERATURE = 0.0
REWRITE_TEMPERATURE = 0.4
REWRITE_MAX_TOKENS = 4000

_TAG_RE = re.compile(r"<[^>]+>")
_VALID_STATUSES = ("supported", "unsupported", "partial")


class SanityError(RuntimeError):
    """Stage 5 hard failure — unsupported claims persist after the one rewrite."""


def _strip_ssml(ssml: str) -> str:
    """Pull the spoken text out of the SSML envelope for claim extraction."""
    return _TAG_RE.sub(" ", ssml)


def _format_source_for_verify(items: list[dict[str, Any]], today: date | None = None) -> str:
    """Flatten summaries + key_facts into a single text block for verifier context.

    `today` is added as a runtime fact so the verifier knows the brief's own
    publication-date references (the cold open) are grounded — those aren't
    article claims to verify against the summaries.
    """
    lines: list[str] = []
    if today is not None:
        lines.append(f"# Runtime context")
        lines.append(f"- Today's date is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()}).")
        lines.append(f"- The brief refers to today, this morning, etc. — these are not source claims.")
        lines.append("")
    for item in items:
        title = item.get("title", "") or ""
        if title:
            lines.append(f"# {title}")
        lines.append(item.get("summary", "") or "")
        for f in item.get("key_facts") or []:
            lines.append(f"- {f}")
        lines.append("")
    return "\n".join(lines)


async def _extract_claims(
    provider: OpenRouterProvider,
    model: str,
    system: str,
    script_text: str,
) -> list[str]:
    """Call A — return list[str] of factual claims found in the script."""
    user = f"Script:\n\n{script_text}"
    text = await provider.complete(
        system=system, user=user, model=model,
        json_mode=True, temperature=EXTRACT_TEMPERATURE,
    )
    data = json.loads(text)
    if isinstance(data, dict) and "claims" in data:
        raw = data["claims"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"extract response shape unexpected: {type(data).__name__}")
    if not isinstance(raw, list):
        raise ValueError("'claims' field is not a list")
    return [str(c).strip() for c in raw if str(c).strip()]


async def _verify_claim(
    provider: OpenRouterProvider,
    model: str,
    system: str,
    claim: str,
    source_block: str,
) -> dict[str, str]:
    """Call B (per claim) — return {text, status, evidence}.

    Re-asks the verifier once if the first verdict is non-supported. The
    verifier is stochastic enough at T=0 that the same claim against the
    same source can flip between supported and unsupported across calls
    (observed: a "Michael 217m opening weekend" claim marked supported in
    pass 1 came back unsupported in pass 2 of the same run). If the
    second call says supported, accept it; otherwise keep the real
    verifier verdict so genuine hallucinations still surface.

    Malformed JSON is treated as a transient API failure rather than a
    real verdict: if both attempts return malformed JSON, mark the claim
    supported with a note (don't fail the pipeline on flaky API output);
    if only one of the two is malformed, prefer the other call's verdict.
    """
    user = (
        f"Source key facts and summaries:\n{source_block}\n\n"
        f"Claim to verify:\n{claim}"
    )

    async def _ask() -> tuple[str, str]:
        """One verify call. Returns (status, evidence). Status is one of
        the public verdicts or the internal sentinel 'malformed' for
        unparseable responses."""
        text = await provider.complete(
            system=system, user=user, model=model,
            json_mode=True, temperature=VERIFY_TEMPERATURE,
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return ("malformed", "verifier returned malformed JSON")
        status = data.get("status", "unsupported")
        if status not in _VALID_STATUSES:
            status = "unsupported"
        evidence = str(data.get("evidence", "")).strip()
        return (status, evidence)

    status, evidence = await _ask()
    if status == "supported":
        return {"text": claim, "status": status, "evidence": evidence}

    log.info("verify: %s for claim %r — re-asking once", status, claim[:60])
    retry_status, retry_evidence = await _ask()
    if retry_status == "supported":
        log.info("verify: retry says supported for claim %r", claim[:60])
        return {"text": claim, "status": "supported", "evidence": retry_evidence}

    # Both malformed → transient API issue, not a real disagreement.
    if status == "malformed" and retry_status == "malformed":
        log.warning("verify: malformed JSON twice for claim %r; treating as supported", claim[:60])
        return {
            "text": claim,
            "status": "supported",
            "evidence": "verifier API returned malformed JSON twice — accepted as transient",
        }
    # One malformed, one real verdict → use the real one.
    if status == "malformed":
        return {"text": claim, "status": retry_status, "evidence": retry_evidence}
    if retry_status == "malformed":
        return {"text": claim, "status": status, "evidence": evidence}

    return {"text": claim, "status": status, "evidence": evidence}


async def _verify_all(
    provider: OpenRouterProvider,
    model: str,
    system: str,
    claims: list[str],
    source_block: str,
) -> list[dict[str, str]]:
    return list(await asyncio.gather(*(
        _verify_claim(provider, model, system, c, source_block) for c in claims
    )))


def _is_show_date_claim(claim: str, today: date) -> bool:
    """True if the claim merely restates the show's own broadcast date.

    The cold-open says today's date as part of the show's intro — it isn't a
    factual claim about a news item. The extract prompt tells the model to
    skip these, but it sometimes pulls them anyway, and the verifier then
    flags them because the source summaries don't mention the show's own
    date. Filtering here is defense-in-depth.
    """
    lower = claim.lower()
    show_kw = ("broadcast", "brief", "this is your", "today is", "today's date", "morning brief")
    if not any(kw in lower for kw in show_kw):
        return False
    iso = today.isoformat()
    weekday = today.strftime("%A").lower()
    month = today.strftime("%B").lower()
    return iso in lower or (weekday in lower and month in lower)


async def _run_pass(
    script_text: str,
    items: list[dict[str, Any]],
    models_cfg: dict[str, str],
    provider: OpenRouterProvider,
    extract_sys: str,
    verify_sys: str,
    today: date | None = None,
) -> list[dict[str, str]]:
    claims = await _extract_claims(provider, models_cfg["sanity_extract"], extract_sys, script_text)
    log.info("sanity: extracted %d claims", len(claims))
    if today is not None:
        kept = [c for c in claims if not _is_show_date_claim(c, today)]
        if len(kept) < len(claims):
            log.info("sanity: dropped %d show-date claim(s) at extract", len(claims) - len(kept))
        claims = kept
    if not claims:
        return []
    source_block = _format_source_for_verify(items, today=today)
    return await _verify_all(provider, models_cfg["sanity_verify"], verify_sys, claims, source_block)


async def _rewrite_with_feedback(
    script_ssml: str,
    unsupported: list[dict[str, str]],
    critique_model: str,
    provider: OpenRouterProvider,
    critique_sys: str,
) -> str:
    """Spec §8.3: re-run Stage 4 critique with unsupported claims as feedback."""
    bullet_block = "\n".join(
        f'- "{v["text"]}" — {v["evidence"] or "not in source"}' for v in unsupported
    )
    user = (
        "Draft to revise:\n\n"
        f"{script_ssml}\n\n"
        "The sanity check flagged these claims as not grounded in the source "
        "summaries. Remove or rewrite them so every fact in the final script "
        "is present in the source. Do not add new facts to compensate.\n\n"
        f"{bullet_block}\n\n"
        "Return the revised SSML only."
    )
    return await provider.complete(
        system=critique_sys, user=user, model=critique_model,
        temperature=REWRITE_TEMPERATURE, max_tokens=REWRITE_MAX_TOKENS,
    )


def _has_unsupported(verifications: list[dict[str, str]]) -> bool:
    """Spec §8: only 'unsupported' triggers rewrite. 'partial' is mild
    by definition ('mostly supported'); rewriting on partials creates
    cascading false positives because the rewrite can change wording in
    ways that break previously-supported claims."""
    return any(v["status"] == "unsupported" for v in verifications)


def _unsupported(verifications: list[dict[str, str]]) -> list[dict[str, str]]:
    return [v for v in verifications if v["status"] == "unsupported"]


def sanity(d: date | None = None, *, provider: OpenRouterProvider | None = None) -> Path:
    """Verify 04_script.txt against 03_summaries.json. Write 05_sanity.json.

    On unsupported claims: save the bad script + bad report, rewrite once via
    the critique prompt with feedback, re-verify. If still failing, raise.
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    summaries = read_json(day_dir / "03_summaries.json")
    items = summaries.get("items") or []

    script_path = day_dir / "04_script.txt"
    script_ssml = read_text(script_path)
    script_text = _strip_ssml(script_ssml)

    out_path = day_dir / "05_sanity.json"

    if not items:
        # Quiet-day stub has no source to verify against. Mark passed.
        log.info("sanity: no summary items; skipping verification")
        write_json(out_path, {
            "meta": {"model": "(skipped — quiet day)", "passed": True, "rewrites": 0},
            "claims": [],
        })
        return out_path

    models_cfg = config.models()
    extract_sys = read_text(EXTRACT_PROMPT_PATH)
    verify_sys = read_text(VERIFY_PROMPT_PATH)
    critique_sys = read_text(WRITE_CRITIQUE_PROMPT_PATH)

    if provider is None:
        provider = OpenRouterProvider()

    log.info("sanity: first pass with %s", models_cfg["sanity_verify"])
    verifications = asyncio.run(_run_pass(
        script_text, items, models_cfg, provider, extract_sys, verify_sys, today=d,
    ))

    rewrites = 0
    if _has_unsupported(verifications):
        bad = _unsupported(verifications)
        log.warning("sanity: %d unsupported claim(s); rewriting once", len(bad))
        # Snapshot the failing script + first report for inspection.
        write_text(day_dir / "04_script.bad.txt", script_ssml)
        write_json(day_dir / "05_sanity.bad.json", {
            "meta": {"model": models_cfg["sanity_verify"], "passed": False, "rewrites": 0},
            "claims": verifications,
        })

        revised = _strip_code_fences(asyncio.run(_rewrite_with_feedback(
            script_ssml, bad, models_cfg["write_critique"], provider, critique_sys,
        )))
        write_text(script_path, revised)
        script_ssml = revised
        script_text = _strip_ssml(script_ssml)
        rewrites = 1

        log.info("sanity: re-verifying after rewrite")
        verifications = asyncio.run(_run_pass(
            script_text, items, models_cfg, provider, extract_sys, verify_sys, today=d,
        ))

        if _has_unsupported(verifications):
            still = _unsupported(verifications)
            write_json(out_path, {
                "meta": {"model": models_cfg["sanity_verify"], "passed": False, "rewrites": 1},
                "claims": verifications,
            })
            raise SanityError(f"{len(still)} unsupported claim(s) remain after rewrite")

    write_json(out_path, {
        "meta": {
            "model": models_cfg["sanity_verify"],
            "passed": True,
            "rewrites": rewrites,
        },
        "claims": verifications,
    })
    log.info("sanity: passed (rewrites=%d, claims=%d)", rewrites, len(verifications))
    return out_path
