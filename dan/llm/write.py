"""Stage 4 — Write (the personality stage).

Spec §7: two LLM calls — draft + critique — that produce a single SSML
broadcast script in the DAN house voice. Reads 03_summaries.json and
config/voice.yaml, writes 04_script.txt.
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
from dan.io import read_json, read_text, write_text
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.paths import ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

# Spec §7.6 target band; §7.9 says ±200 outside that band logs a warning
# but the stage still proceeds (a second critique pass risks an infinite loop).
SCRIPT_HARD_MIN_WORDS = 1000
SCRIPT_HARD_MAX_WORDS = 2000

DRAFT_TEMPERATURE = 0.7
CRITIQUE_TEMPERATURE = 0.4
SCRIPT_MAX_TOKENS = 4000  # comfortably above the 1800-word ceiling

DRAFT_PROMPT_PATH = ROOT / "dan" / "prompts" / "write_draft.txt"
CRITIQUE_PROMPT_PATH = ROOT / "dan" / "prompts" / "write_critique.txt"
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"


def _format_date_for_speech(d: date) -> str:
    """Return a date string the writer can drop into the cold open verbatim."""
    return d.strftime("%A, %B %d, %Y").replace(" 0", " ")


def _summaries_for_prompt(items: list[dict[str, Any]]) -> str:
    """Compact JSON of just the fields the writer needs."""
    minimal = [
        {
            "rank": it["rank"],
            "title": it.get("title", ""),
            "summary": it["summary"],
            "key_facts": it["key_facts"],
        }
        for it in items
    ]
    return json.dumps(minimal, indent=2, ensure_ascii=False)


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap output in ```xml ... ``` even when told not to."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


_TAG_RE = re.compile(r"<[^>]+>")


def _word_count(ssml: str) -> int:
    """Count words in the spoken text, ignoring SSML/XML tags."""
    return len(_TAG_RE.sub(" ", ssml).split())


def _quiet_day_script(today_str: str, voice_name: str) -> str:
    """Hand-written stub for the (rare) zero-summary day.

    Matches the SSML shape the writer prompt now produces — no
    `<mstts:express-as>` wrapper, since DragonHD voices reject it.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0"\n'
        '       xmlns="http://www.w3.org/2001/10/synthesis"\n'
        '       xml:lang="en-US">\n'
        f'  <voice name="{voice_name}">\n'
        f'    Good morning. This is your daily film news brief for {today_str}. '
        '    <break time="300ms"/>'
        '    It\'s a quiet day on the wires — nothing major from the studios '
        'or the festival circuit overnight. We\'ll be back tomorrow with a fuller report. '
        '    <break time="500ms"/>'
        '    That\'s your film news for today. We\'ll see you tomorrow.\n'
        '  </voice>\n'
        '</speak>\n'
    )


async def _write_async(
    items: list[dict[str, Any]],
    today_str: str,
    voice_name: str,
    models_cfg: dict[str, str],
    provider: OpenRouterProvider,
    draft_system: str,
    critique_system: str,
) -> str:
    """Run draft -> critique. Returns the final SSML string."""
    summaries_block = _summaries_for_prompt(items)
    draft_user = (
        f"Today's date: {today_str}\n"
        f"Voice name: {voice_name}\n\n"
        f"Stories (ranked, highest score first):\n"
        f"{summaries_block}\n\n"
        f"Write the full SSML script now."
    )
    log.info("write: draft pass with %s (%d stories)", models_cfg["write_draft"], len(items))
    draft = _strip_code_fences(await provider.complete(
        system=draft_system,
        user=draft_user,
        model=models_cfg["write_draft"],
        temperature=DRAFT_TEMPERATURE,
        max_tokens=SCRIPT_MAX_TOKENS,
    ))
    log.info("write: draft is %d words", _word_count(draft))

    # The critique system prompt instructs the model to "expand using texture
    # from the summaries" if the draft is short — that instruction is only
    # actionable if the summaries are in the message. Without them, the
    # critique can only rephrase what's already in the draft.
    critique_user = (
        "Source summaries the draft was built from (for fact-checking and "
        "for restoring texture if the draft is short):\n\n"
        f"{summaries_block}\n\n"
        f"Draft to revise:\n\n{draft}\n\nReturn the revised SSML."
    )
    log.info("write: critique pass with %s", models_cfg["write_critique"])
    final = _strip_code_fences(await provider.complete(
        system=critique_system,
        user=critique_user,
        model=models_cfg["write_critique"],
        temperature=CRITIQUE_TEMPERATURE,
        max_tokens=SCRIPT_MAX_TOKENS,
    ))
    final_words = _word_count(final)
    log.info("write: final script is %d words", final_words)

    if not (SCRIPT_HARD_MIN_WORDS <= final_words <= SCRIPT_HARD_MAX_WORDS):
        # §7.9: warn but proceed — a second critique pass risks looping.
        log.warning(
            "write: final word count %d outside %d-%d (proceeding anyway per spec §7.9)",
            final_words, SCRIPT_HARD_MIN_WORDS, SCRIPT_HARD_MAX_WORDS,
        )
    return final


def write(d: date | None = None, *, provider: OpenRouterProvider | None = None) -> Path:
    """Read 03_summaries.json, run draft + critique, write 04_script.txt."""
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    summaries = read_json(day_dir / "03_summaries.json")
    items = summaries.get("items") or []

    voice_cfg = config.voice() or {}
    voice_name = voice_cfg.get("voice", DEFAULT_VOICE)
    today_str = _format_date_for_speech(d)

    if not items:
        log.warning("write: no summary items; emitting quiet-day stub")
        ssml = _quiet_day_script(today_str, voice_name)
    else:
        models_cfg = config.models()
        draft_system = read_text(DRAFT_PROMPT_PATH)
        critique_system = read_text(CRITIQUE_PROMPT_PATH)
        if provider is None:
            provider = OpenRouterProvider()
        ssml = asyncio.run(_write_async(
            items, today_str, voice_name, models_cfg, provider, draft_system, critique_system,
        ))

    out_path = day_dir / "04_script.txt"
    write_text(out_path, ssml)
    log.info("wrote script -> %s", out_path)
    return out_path
