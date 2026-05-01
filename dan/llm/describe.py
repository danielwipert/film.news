"""Stage 9.1 — RSS episode description.

Spec §12.6: a single cheap LLM call turns 04_script.txt into a 2-3 sentence
plain-prose description that goes in the RSS <item><description>. This is
what listeners see under the episode title in Apple Podcasts.

Reads 04_script.txt + config/models.yaml (`describe`), writes
09_description.txt. Strips SSML before sending so the model never sees
<voice>/<break>/<express-as> tags.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from pathlib import Path

from dan import config
from dan.io import read_text, write_text
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.paths import ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

PROMPT_PATH = ROOT / "dan" / "prompts" / "describe.txt"

# Spec §12.6: under 300 chars total. Keep the lower bound loose — anything
# that survived the period-terminator check below is at least one full
# sentence, which is the real signal that we got something usable.
DESCRIPTION_MIN_CHARS = 40
DESCRIPTION_MAX_CHARS = 300

DESCRIBE_TEMPERATURE = 0.3
DESCRIBE_MAX_TOKENS = 200

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TERMINATOR_RE = re.compile(r"[.!?]")
# Split on sentence terminators, keeping the terminator with the preceding
# sentence so we can rejoin without losing punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Strip a wrapping pair of straight or curly quotes if the model returns
# the description as a single quoted string.
_OUTER_QUOTES_RE = re.compile(r'^[\s"“”\']+|[\s"“”\']+$')


class DescribeError(RuntimeError):
    """Stage 9.1 hard failure — both attempts produced an unusable description."""


def _truncate_to_fit(text: str, limit: int) -> str:
    """Drop trailing sentences until the text fits under `limit` chars.

    The describe model (flash-lite) sometimes overshoots 300 chars by a small
    margin even on retry. Rather than fail the whole pipeline on metadata,
    cut whole sentences from the end. If even the first sentence is too long,
    hard-cut at a word boundary and append an ellipsis.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    while sentences:
        candidate = " ".join(sentences).strip()
        if len(candidate) <= limit:
            return candidate
        sentences.pop()
    # First sentence alone exceeds the limit — hard-cut at a word boundary.
    head = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:- ")
    return head + "…"


def _strip_ssml(ssml: str) -> str:
    """Pull spoken text out of the SSML envelope and collapse whitespace.

    Mirrors dan.llm.sanity._strip_ssml's regex approach: SSML is well-formed
    XML by Stage 6, but a regex strip is cheaper here and we don't need
    structural information — just the prose.
    """
    text = _TAG_RE.sub(" ", ssml)
    return _WS_RE.sub(" ", text).strip()


def _validate_description(text: str) -> str:
    """Coerce / validate the model output. Raise ValueError on bad shape."""
    cleaned = _OUTER_QUOTES_RE.sub("", text or "").strip()
    if not cleaned:
        raise ValueError("empty description")
    if len(cleaned) < DESCRIPTION_MIN_CHARS:
        raise ValueError(f"description too short ({len(cleaned)} chars)")
    if len(cleaned) > DESCRIPTION_MAX_CHARS:
        raise ValueError(f"description too long ({len(cleaned)} chars, max {DESCRIPTION_MAX_CHARS})")
    if not _TERMINATOR_RE.search(cleaned):
        raise ValueError("description has no sentence terminator")
    return cleaned


async def _describe_async(
    provider: OpenRouterProvider,
    model: str,
    base_system: str,
    script_text: str,
) -> str:
    """Run the LLM call with one retry on validation/LLM error."""
    user = f"Episode script:\n{script_text}\n\nWrite the description."
    last_err: str | None = None
    last_overlong: str | None = None  # candidate that only failed the length check

    for attempt in (1, 2):
        system = base_system
        if attempt == 2:
            system = (
                base_system
                + f"\n\nThe previous response was not usable: {last_err}. "
                + f"Output ONLY the description text, 2-3 sentences, "
                + f"under {DESCRIPTION_MAX_CHARS} characters. "
                + "No preamble, no quotes, no markdown."
            )
        try:
            text = await provider.complete(
                system=system, user=user, model=model,
                temperature=DESCRIBE_TEMPERATURE,
                max_tokens=DESCRIBE_MAX_TOKENS,
            )
            return _validate_description(text)
        except ValueError as e:
            last_err = str(e)
            log.warning("describe validation failed (attempt %d/2): %s", attempt, e)
            # Stash the cleaned-but-overlong candidate so we can salvage it
            # if the second attempt also fails.
            cleaned = _OUTER_QUOTES_RE.sub("", text or "").strip()
            if cleaned and len(cleaned) > DESCRIPTION_MAX_CHARS and _TERMINATOR_RE.search(cleaned):
                last_overlong = cleaned
        except LLMError as e:
            last_err = str(e)
            log.warning("describe LLM error (attempt %d/2): %s", attempt, e)

    if last_overlong is not None:
        salvaged = _truncate_to_fit(last_overlong, DESCRIPTION_MAX_CHARS)
        if len(salvaged) >= DESCRIPTION_MIN_CHARS and _TERMINATOR_RE.search(salvaged):
            log.warning(
                "describe: both attempts overran %d chars; truncated to %d chars",
                DESCRIPTION_MAX_CHARS, len(salvaged),
            )
            return salvaged

    raise DescribeError(f"describe failed after retry: {last_err}")


def describe(d: date | None = None, *, provider: OpenRouterProvider | None = None) -> Path:
    """Read 04_script.txt, write 09_description.txt. Returns the output path.

    Idempotent across reruns: the existing description (if any) is overwritten.
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    ssml = read_text(day_dir / "04_script.txt")
    script_text = _strip_ssml(ssml)
    if not script_text:
        raise DescribeError("04_script.txt produced no spoken text after SSML strip")

    model = config.models()["describe"]
    base_system = read_text(PROMPT_PATH)
    if provider is None:
        provider = OpenRouterProvider()

    log.info("describe: %d chars of script -> %s", len(script_text), model)
    description = asyncio.run(_describe_async(provider, model, base_system, script_text))

    out_path = day_dir / "09_description.txt"
    write_text(out_path, description + "\n")
    log.info("describe: wrote %s (%d chars)", out_path.name, len(description))
    return out_path
