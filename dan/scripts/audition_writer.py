"""Stage 4 writer audition: run draft + critique on several candidate models
against a known 03_summaries.json, then report length / structure stats so a
human can compare output quality side-by-side.

Run:
  python -m dan.scripts.audition_writer
  python -m dan.scripts.audition_writer --date 2026-04-28
  python -m dan.scripts.audition_writer --models anthropic/claude-sonnet-4.6,deepseek/deepseek-chat-v3.1

Each `--models` entry is either a single OR slug (used for both passes,
matching config/models.yaml) or a `draft+critique` pair like
`qwen/qwen3-235b-a22b-2507+anthropic/claude-sonnet-4.6`. Pairs let you
audition the hybrid-cost setup: cheap model drafts, stronger model
critiques and re-inflates length.

Outputs to logs/_auditions/<input-date>_writer/:
  <slug>__draft.txt   — first-pass SSML
  <slug>__final.txt   — post-critique SSML
  report.txt          — word counts + segment counts side-by-side
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from dan import config
from dan.io import read_json, read_text, write_text
from dan.llm.openrouter import LLMError, OpenRouterProvider
from dan.llm.write import (
    CRITIQUE_PROMPT_PATH,
    DRAFT_PROMPT_PATH,
    SCRIPT_HARD_MAX_WORDS,
    SCRIPT_HARD_MIN_WORDS,
    _format_date_for_speech,
    _strip_code_fences,
    _summaries_for_prompt,
    _word_count,
)
from dan.paths import LOGS_ROOT, ROOT, log_dir, today_utc

load_dotenv(ROOT / ".env", override=False)
log = logging.getLogger(__name__)

# Sonnet baseline plus the two Chinese OSS frontier models worth comparing.
# Slugs are OpenRouter's; override with --models on the CLI if any are stale.
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4.6",                                       # baseline
    "qwen/qwen3-235b-a22b-2507+anthropic/claude-sonnet-4.6",             # hybrid: Qwen draft, Sonnet critique
    "deepseek/deepseek-v3.2+anthropic/claude-sonnet-4.6",                # hybrid: DeepSeek draft, Sonnet critique
]

DRAFT_TEMPERATURE = 0.7
CRITIQUE_TEMPERATURE = 0.4
SCRIPT_MAX_TOKENS = 4000

_BREAK_500_RE = re.compile(r'<break\s+time="500ms"\s*/>')


def _slug_one(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(".", "_")


def _slug(draft_model: str, critique_model: str) -> str:
    """Filesystem-safe slug for either a single model or a draft+critique pair."""
    if draft_model == critique_model:
        return _slug_one(draft_model)
    return f"{_slug_one(draft_model)}_PLUS_{_slug_one(critique_model)}"


def _label(draft_model: str, critique_model: str) -> str:
    """Human-readable label for reports."""
    if draft_model == critique_model:
        return draft_model
    return f"{draft_model} -> {critique_model}"


def _parse_pair(spec: str) -> tuple[str, str]:
    """Parse 'draft+critique' or a single model id. Returns (draft, critique)."""
    if "+" in spec:
        draft, critique = spec.split("+", 1)
        return draft.strip(), critique.strip()
    return spec.strip(), spec.strip()


def _segment_count(ssml: str) -> int:
    """Approximate segment count: 500ms breaks separate major segments."""
    return len(_BREAK_500_RE.findall(ssml))


async def _run_one(
    *,
    draft_model: str,
    critique_model: str,
    draft_user: str,
    summaries_block: str,
    draft_system: str,
    critique_system: str,
    provider: OpenRouterProvider,
) -> tuple[str, str]:
    """Run draft on `draft_model`, critique on `critique_model`. Returns (draft, final).

    The critique user-message includes the source summaries so the critique
    prompt's "expand using texture from the summaries" instruction is
    actionable. This matters most when draft_model != critique_model: a
    cheap drafter often compresses away texture, leaving the same-model
    critique nothing to re-inflate from.
    """
    label = _label(draft_model, critique_model)
    log.info("[%s] draft pass on %s", label, draft_model)
    draft_raw = await provider.complete(
        system=draft_system,
        user=draft_user,
        model=draft_model,
        temperature=DRAFT_TEMPERATURE,
        max_tokens=SCRIPT_MAX_TOKENS,
    )
    if not draft_raw:
        raise LLMError(f"draft model {draft_model} returned empty/null content")
    draft = _strip_code_fences(draft_raw)
    log.info("[%s] draft = %d words", label, _word_count(draft))

    critique_user = (
        "Source summaries the draft was built from (for fact-checking and "
        "for restoring texture if the draft is short):\n\n"
        f"{summaries_block}\n\n"
        f"Draft to revise:\n\n{draft}\n\nReturn the revised SSML."
    )
    log.info("[%s] critique pass on %s", label, critique_model)
    final_raw = await provider.complete(
        system=critique_system,
        user=critique_user,
        model=critique_model,
        temperature=CRITIQUE_TEMPERATURE,
        max_tokens=SCRIPT_MAX_TOKENS,
    )
    if not final_raw:
        raise LLMError(f"critique model {critique_model} returned empty/null content")
    final = _strip_code_fences(final_raw)
    log.info("[%s] final = %d words", label, _word_count(final))
    return draft, final


def _format_report_row(label: str, draft: str | None, final: str | None, err: str | None) -> str:
    if err is not None:
        return f"  {label:<70s}  FAILED: {err}"
    assert draft is not None and final is not None
    dw = _word_count(draft)
    fw = _word_count(final)
    fs = _segment_count(final)
    target_lo, target_hi = SCRIPT_HARD_MIN_WORDS, SCRIPT_HARD_MAX_WORDS
    in_band = target_lo <= fw <= target_hi
    band_marker = "OK " if in_band else "OUT"
    return (
        f"  {label:<70s}  draft={dw:>4d}w  final={fw:>4d}w  "
        f"segments={fs:>2d}  [{band_marker} target {target_lo}-{target_hi}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dan.scripts.audition_writer")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="ISO date of logs/<DATE>/03_summaries.json to use as input. "
             "Defaults to the most recent log dir with summaries.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated OpenRouter model slugs. Default: {','.join(DEFAULT_MODELS)}",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    input_date = args.date or _latest_summaries_date()
    if input_date is None:
        log.error("audition_writer: no logs/<date>/03_summaries.json found; pass --date")
        return 1

    summaries_path = LOGS_ROOT / input_date.isoformat() / "03_summaries.json"
    if not summaries_path.exists():
        log.error("audition_writer: %s does not exist", summaries_path)
        return 1

    summaries = read_json(summaries_path)
    items = summaries.get("items") or []
    if not items:
        log.error("audition_writer: %s has no items — cannot audition on a quiet day", summaries_path)
        return 1

    voice_cfg = config.voice() or {}
    voice_name = voice_cfg.get("voice", "en-US-Davis:DragonHDLatestNeural")
    today_str = _format_date_for_speech(input_date)
    draft_system = read_text(DRAFT_PROMPT_PATH)
    critique_system = read_text(CRITIQUE_PROMPT_PATH)
    summaries_block = _summaries_for_prompt(items)

    draft_user = (
        f"Today's date: {today_str}\n"
        f"Voice name: {voice_name}\n\n"
        f"Stories (ranked, highest score first):\n"
        f"{summaries_block}\n\n"
        f"Write the full SSML script now."
    )

    out_dir = LOGS_ROOT / "_auditions" / f"{input_date.isoformat()}_writer"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("audition_writer: input=%s items=%d -> %s", summaries_path, len(items), out_dir)

    pairs = [_parse_pair(m) for m in args.models.split(",") if m.strip()]
    provider = OpenRouterProvider()

    results: list[tuple[str, str, str | None, str | None, str | None]] = []
    for draft_model, critique_model in pairs:
        label = _label(draft_model, critique_model)
        try:
            draft, final = asyncio.run(_run_one(
                draft_model=draft_model,
                critique_model=critique_model,
                draft_user=draft_user,
                summaries_block=summaries_block,
                draft_system=draft_system,
                critique_system=critique_system,
                provider=provider,
            ))
        except LLMError as e:
            log.warning("[%s] FAILED: %s", label, e)
            results.append((draft_model, critique_model, None, None, str(e)))
            continue

        slug = _slug(draft_model, critique_model)
        write_text(out_dir / f"{slug}__draft.txt", draft)
        write_text(out_dir / f"{slug}__final.txt", final)
        results.append((draft_model, critique_model, draft, final, None))

    report_lines = [
        f"Stage 4 writer audition — input: {summaries_path} ({len(items)} stories)",
        f"Target band: {SCRIPT_HARD_MIN_WORDS}-{SCRIPT_HARD_MAX_WORDS} words "
        f"(8-10 min spoken)",
        "",
    ]
    for draft_model, critique_model, draft, final, err in results:
        report_lines.append(_format_report_row(
            _label(draft_model, critique_model), draft, final, err,
        ))
    report_lines.append("")
    report_lines.append("Per-model files:")
    for draft_model, critique_model, _, final, err in results:
        if err is None:
            slug = _slug(draft_model, critique_model)
            report_lines.append(f"  {out_dir / (slug + '__final.txt')}")
    report = "\n".join(report_lines) + "\n"

    write_text(out_dir / "report.txt", report)
    print()
    print(report)
    return 0


def _latest_summaries_date() -> date | None:
    """Pick the most recent logs/YYYY-MM-DD/ that has a non-empty 03_summaries.json."""
    if not LOGS_ROOT.exists():
        return None
    candidates: list[date] = []
    for child in LOGS_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            d = date.fromisoformat(child.name)
        except ValueError:
            continue
        if (child / "03_summaries.json").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates)


if __name__ == "__main__":
    sys.exit(main())
