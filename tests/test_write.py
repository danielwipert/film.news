"""Tests for Stage 4 — dan.llm.write."""
from __future__ import annotations

import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from lxml import etree

from dan.llm import write as write_stage


# ---------- helpers ----------

def _ssml_shell(body: str, voice: str = "en-US-AndrewMultilingualNeural") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'  <voice name="{voice}">\n'
        '    <mstts:express-as style="newscast">\n'
        f'      {body}\n'
        '    </mstts:express-as>\n'
        '  </voice>\n'
        '</speak>'
    )


def _provider(*completions):
    p = MagicMock()
    p.complete = AsyncMock(side_effect=list(completions))
    return p


def _items(n: int = 3) -> list[dict]:
    return [
        {"rank": i + 1, "id": f"film/2026/apr/26/story{i}",
         "title": f"Story {i} headline",
         "summary": "word " * 100,
         "key_facts": [f"fact{i}-1", f"fact{i}-2", f"fact{i}-3"],
         "source_url": f"https://x/{i}"}
        for i in range(n)
    ]


# ---------- pure helpers ----------

def test_format_date_for_speech_strips_leading_zero():
    out = write_stage._format_date_for_speech(date(2026, 4, 6))
    # "Monday, April 6, 2026" — single-digit day with no leading zero
    assert "April 6," in out
    assert "April 06" not in out


def test_summaries_for_prompt_strips_to_minimal_fields():
    items = [{"rank": 1, "id": "x", "title": "T", "summary": "S",
              "key_facts": ["f1"], "source_url": "u", "extra": "ignored"}]
    out_json = write_stage._summaries_for_prompt(items)
    parsed = json.loads(out_json)
    assert parsed[0] == {"rank": 1, "title": "T", "summary": "S", "key_facts": ["f1"]}


def test_strip_code_fences_xml():
    s = "```xml\n<speak>hi</speak>\n```"
    assert write_stage._strip_code_fences(s) == "<speak>hi</speak>"


def test_strip_code_fences_plain_backticks():
    s = "```\n<speak>hi</speak>\n```"
    assert write_stage._strip_code_fences(s) == "<speak>hi</speak>"


def test_strip_code_fences_no_fences_returned_as_is():
    assert write_stage._strip_code_fences("<speak>hi</speak>") == "<speak>hi</speak>"


def test_word_count_strips_xml_tags():
    ssml = '<speak><voice name="x">Good morning <break time="500ms"/>everyone today</voice></speak>'
    assert write_stage._word_count(ssml) == 4  # Good morning everyone today


# ---------- _write_async ----------

def test_write_async_runs_draft_then_critique():
    draft = _ssml_shell("draft body " * 200)  # ~400 words
    final = _ssml_shell("final body " * 220)  # ~440 words; under min, but proceed warn
    provider = _provider(draft, final)
    out = asyncio.run(write_stage._write_async(
        items=_items(3),
        today_str="Sunday, April 26, 2026",
        voice_name="en-US-AndrewMultilingualNeural",
        models_cfg={"write_draft": "anthropic/claude-haiku-4.5",
                    "write_critique": "anthropic/claude-haiku-4.5"},
        provider=provider,
        draft_system="DRAFT_SYS",
        critique_system="CRITIQUE_SYS",
    ))
    assert out == final
    assert provider.complete.await_count == 2

    draft_call, critique_call = provider.complete.await_args_list
    assert draft_call.kwargs["system"] == "DRAFT_SYS"
    assert "Sunday, April 26, 2026" in draft_call.kwargs["user"]
    assert "Stories" in draft_call.kwargs["user"]
    assert draft_call.kwargs["model"] == "anthropic/claude-haiku-4.5"
    assert draft_call.kwargs["temperature"] == write_stage.DRAFT_TEMPERATURE

    assert critique_call.kwargs["system"] == "CRITIQUE_SYS"
    assert draft in critique_call.kwargs["user"]
    # Summaries must be in the critique message so the prompt's
    # "expand using texture from the summaries" instruction is actionable.
    assert "Story 0 headline" in critique_call.kwargs["user"]
    assert "fact0-1" in critique_call.kwargs["user"]
    assert critique_call.kwargs["temperature"] == write_stage.CRITIQUE_TEMPERATURE


def test_write_async_strips_fenced_output():
    draft_with_fences = "```xml\n" + _ssml_shell("draft " * 300) + "\n```"
    final_with_fences = "```\n" + _ssml_shell("final " * 1300) + "\n```"
    provider = _provider(draft_with_fences, final_with_fences)
    out = asyncio.run(write_stage._write_async(
        items=_items(3),
        today_str="Sun",
        voice_name="v",
        models_cfg={"write_draft": "m", "write_critique": "m"},
        provider=provider,
        draft_system="ds",
        critique_system="cs",
    ))
    assert not out.startswith("```")
    assert out.startswith("<?xml")


# ---------- write() pipeline entry ----------

def test_write_handles_empty_items_with_quiet_day_stub(monkeypatch, tmp_path):
    monkeypatch.setattr(write_stage, "log_dir", lambda d=None: tmp_path)
    (tmp_path / "03_summaries.json").write_text(
        json.dumps({"meta": {}, "items": []}), encoding="utf-8"
    )
    out_path = write_stage.write()
    content = out_path.read_text(encoding="utf-8")
    assert "<speak" in content
    assert "quiet day" in content.lower()
    assert "We'll see you tomorrow" in content
    # Quiet-day stub must still parse as XML
    etree.fromstring(content.encode("utf-8"))


def test_write_end_to_end_with_mocked_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(write_stage, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(write_stage.config, "models", lambda: {
        "write_draft": "anthropic/claude-haiku-4.5",
        "write_critique": "anthropic/claude-haiku-4.5",
    })
    monkeypatch.setattr(write_stage.config, "voice", lambda: {"voice": "en-US-Test"})

    (tmp_path / "03_summaries.json").write_text(
        json.dumps({"meta": {}, "items": _items(8)}), encoding="utf-8"
    )

    draft_ssml = _ssml_shell("draft " * 1400, voice="en-US-Test")
    final_ssml = _ssml_shell("final " * 1500, voice="en-US-Test")
    fake = _provider(draft_ssml, final_ssml)

    out_path = write_stage.write(provider=fake)
    content = out_path.read_text(encoding="utf-8")
    assert content == final_ssml
    # Resulting file is well-formed SSML
    root = etree.fromstring(content.encode("utf-8"))
    assert root.tag.endswith("speak")
    voice_el = root[0]
    assert voice_el.tag.endswith("voice")
    assert voice_el.get("name") == "en-US-Test"


def test_write_logs_warning_on_out_of_range_length(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(write_stage, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(write_stage.config, "models", lambda: {
        "write_draft": "m", "write_critique": "m",
    })
    monkeypatch.setattr(write_stage.config, "voice", lambda: {"voice": "v"})
    (tmp_path / "03_summaries.json").write_text(
        json.dumps({"meta": {}, "items": _items(2)}), encoding="utf-8"
    )
    short_draft = _ssml_shell("a")           # ~1 word
    short_final = _ssml_shell("a b c")       # ~3 words
    fake = _provider(short_draft, short_final)
    with caplog.at_level("WARNING"):
        write_stage.write(provider=fake)
    assert any("outside" in rec.message for rec in caplog.records)
