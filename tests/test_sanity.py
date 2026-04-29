"""Tests for Stage 5 — dan.llm.sanity."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from dan.llm import sanity
from dan.llm.sanity import SanityError


def _provider(*completions):
    p = MagicMock()
    p.complete = AsyncMock(side_effect=list(completions))
    return p


def _items() -> list[dict]:
    return [
        {"rank": 1, "id": "x1", "title": "Story 1",
         "summary": "Director Jane Doe wrapped a fourteen-month shoot.",
         "key_facts": ["Director: Jane Doe", "Shoot: 14 months"],
         "source_url": "https://x/1"},
    ]


# ---------- _strip_ssml ----------

def test_strip_ssml_removes_tags_keeps_text():
    s = '<speak><voice name="x">Hello <break time="500ms"/>world</voice></speak>'
    out = sanity._strip_ssml(s)
    assert "Hello" in out and "world" in out
    assert "<" not in out


# ---------- _format_source_for_verify ----------

def test_format_source_includes_titles_summaries_and_facts():
    items = [
        {"title": "T1", "summary": "S1", "key_facts": ["f1", "f2"]},
        {"title": "T2", "summary": "S2", "key_facts": ["f3"]},
    ]
    out = sanity._format_source_for_verify(items)
    assert "# T1" in out and "# T2" in out
    assert "S1" in out and "S2" in out
    assert "- f1" in out and "- f3" in out


# ---------- _extract_claims ----------

def test_extract_claims_handles_object_with_claims_key():
    payload = json.dumps({"claims": ["A directed B", "C earned $100M"]})
    p = _provider(payload)
    out = asyncio.run(sanity._extract_claims(p, "m", "sys", "script text"))
    assert out == ["A directed B", "C earned $100M"]


def test_extract_claims_handles_bare_list():
    payload = json.dumps(["one", "two"])
    p = _provider(payload)
    out = asyncio.run(sanity._extract_claims(p, "m", "sys", "script text"))
    assert out == ["one", "two"]


def test_extract_claims_drops_empty_strings():
    payload = json.dumps({"claims": ["one", "", "  ", "two"]})
    p = _provider(payload)
    out = asyncio.run(sanity._extract_claims(p, "m", "sys", "script text"))
    assert out == ["one", "two"]


def test_extract_claims_raises_on_unexpected_shape():
    p = _provider(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError):
        asyncio.run(sanity._extract_claims(p, "m", "sys", "script text"))


# ---------- _verify_claim ----------

def test_verify_claim_happy_path():
    p = _provider(json.dumps({"status": "supported", "evidence": "key_fact: X"}))
    out = asyncio.run(sanity._verify_claim(p, "m", "sys", "claim X", "source"))
    assert out == {"text": "claim X", "status": "supported", "evidence": "key_fact: X"}


def test_verify_claim_unknown_status_falls_back_to_unsupported():
    p = _provider(json.dumps({"status": "maybe", "evidence": "uncertain"}))
    out = asyncio.run(sanity._verify_claim(p, "m", "sys", "c", "src"))
    assert out["status"] == "unsupported"


def test_verify_claim_malformed_json_falls_back_to_unsupported_after_retry():
    """Two malformed responses in a row → defaults to unsupported."""
    p = _provider("not json", "still not json")
    out = asyncio.run(sanity._verify_claim(p, "m", "sys", "c", "src"))
    assert out["status"] == "unsupported"
    assert "malformed" in out["evidence"]
    assert p.complete.await_count == 2


def test_verify_claim_retries_on_malformed_json():
    """One malformed response then a valid one → uses the valid one."""
    p = _provider("not json", json.dumps({"status": "supported", "evidence": "ok"}))
    out = asyncio.run(sanity._verify_claim(p, "m", "sys", "c", "src"))
    assert out["status"] == "supported"
    assert out["evidence"] == "ok"
    assert p.complete.await_count == 2


# ---------- sanity() pipeline entry ----------

def _setup_inputs(tmp_path, items=None, script="<speak>hi</speak>"):
    items = items if items is not None else _items()
    (tmp_path / "03_summaries.json").write_text(
        json.dumps({"meta": {}, "items": items}), encoding="utf-8"
    )
    (tmp_path / "04_script.txt").write_text(script, encoding="utf-8")


def _patch_config(monkeypatch):
    monkeypatch.setattr(sanity.config, "models", lambda: {
        "sanity_extract": "test/extract",
        "sanity_verify": "test/verify",
        "write_critique": "test/critique",
    })


def test_sanity_skips_when_no_items(monkeypatch, tmp_path):
    monkeypatch.setattr(sanity, "log_dir", lambda d=None: tmp_path)
    _setup_inputs(tmp_path, items=[])
    out_path = sanity.sanity()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["passed"] is True
    assert data["meta"]["rewrites"] == 0
    assert data["claims"] == []


def test_sanity_passes_when_all_claims_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(sanity, "log_dir", lambda d=None: tmp_path)
    _patch_config(monkeypatch)
    _setup_inputs(tmp_path)

    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        json.dumps({"claims": ["claim A", "claim B"]}),
        json.dumps({"status": "supported", "evidence": "fact A"}),
        json.dumps({"status": "supported", "evidence": "fact B"}),
    ])
    out_path = sanity.sanity(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["passed"] is True
    assert data["meta"]["rewrites"] == 0
    assert len(data["claims"]) == 2
    assert all(c["status"] == "supported" for c in data["claims"])


def test_sanity_rewrites_then_passes_on_second_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(sanity, "log_dir", lambda d=None: tmp_path)
    _patch_config(monkeypatch)
    original_script = "<speak>original with bad fact</speak>"
    _setup_inputs(tmp_path, script=original_script)

    revised_script = "<speak>revised clean</speak>"
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        # Pass 1: extract -> 2 claims
        json.dumps({"claims": ["good claim", "hallucinated claim"]}),
        # Pass 1: verify
        json.dumps({"status": "supported", "evidence": "ok"}),
        json.dumps({"status": "unsupported", "evidence": "not in source"}),
        # Rewrite (Stage 4 critique)
        revised_script,
        # Pass 2: extract
        json.dumps({"claims": ["good claim only"]}),
        # Pass 2: verify
        json.dumps({"status": "supported", "evidence": "ok"}),
    ])

    out_path = sanity.sanity(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["passed"] is True
    assert data["meta"]["rewrites"] == 1
    # Bad-script + bad-report snapshots saved on first failure
    assert (tmp_path / "04_script.bad.txt").exists()
    assert (tmp_path / "04_script.bad.txt").read_text(encoding="utf-8") == original_script
    assert (tmp_path / "05_sanity.bad.json").exists()
    # 04_script.txt was overwritten with the revised version
    assert (tmp_path / "04_script.txt").read_text(encoding="utf-8") == revised_script


def test_sanity_raises_when_rewrite_still_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(sanity, "log_dir", lambda d=None: tmp_path)
    _patch_config(monkeypatch)
    _setup_inputs(tmp_path)

    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        # Pass 1
        json.dumps({"claims": ["claim 1"]}),
        json.dumps({"status": "unsupported", "evidence": "no"}),
        # Rewrite
        "<speak>revised but still broken</speak>",
        # Pass 2
        json.dumps({"claims": ["claim 1"]}),
        json.dumps({"status": "unsupported", "evidence": "still no"}),
    ])

    with pytest.raises(SanityError, match="1 unsupported"):
        sanity.sanity(provider=fake)
    # Final report is written even on failure so debug info is preserved
    final = json.loads((tmp_path / "05_sanity.json").read_text(encoding="utf-8"))
    assert final["meta"]["passed"] is False
    assert final["meta"]["rewrites"] == 1


def test_sanity_partial_status_is_treated_as_failure(monkeypatch, tmp_path):
    """A 'partial' verification status should trigger the rewrite path
    just like 'unsupported' — partial means the script added a detail not
    in the source."""
    monkeypatch.setattr(sanity, "log_dir", lambda d=None: tmp_path)
    _patch_config(monkeypatch)
    _setup_inputs(tmp_path)

    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=[
        json.dumps({"claims": ["one fact"]}),
        json.dumps({"status": "partial", "evidence": "added a detail"}),
        # Rewrite
        "<speak>tightened</speak>",
        # Pass 2
        json.dumps({"claims": ["one tight fact"]}),
        json.dumps({"status": "supported", "evidence": "ok"}),
    ])
    out_path = sanity.sanity(provider=fake)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["meta"]["rewrites"] == 1
    assert data["meta"]["passed"] is True
