"""Tests for Stage 7 — dan.audio.tts."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dan.audio import tts
from dan.audio.tts import AzureTTSProvider, TTSError


# ---------- AzureTTSProvider init validation ----------

def test_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    with pytest.raises(TTSError, match="AZURE_SPEECH_KEY"):
        AzureTTSProvider(region="eastus")


def test_provider_raises_without_region(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    with pytest.raises(TTSError, match="AZURE_SPEECH_REGION"):
        AzureTTSProvider(key="k")


def test_provider_name_and_voice():
    p = AzureTTSProvider(key="k", region="eastus", voice="en-US-Test")
    assert p.name == "azure:eastus"
    assert p.voice == "en-US-Test"


# ---------- AzureTTSProvider.synthesize via mocked SDK ----------

# Sentinel constants we'll use as mocked enum values; identity comparison is
# what the production code does (`result.reason == speechsdk.ResultReason.X`),
# so reusing the same object on both sides makes mocks behave correctly.
_COMPLETED = object()
_CANCELED = object()


def _install_mock_sdk(monkeypatch, results: list) -> MagicMock:
    """Patch dan.audio.tts.speechsdk with a controlled mock.

    results is a list of strings/exceptions describing each successive call:
      - "success"   -> result with reason=Completed, audio_data=AUDIO_BYTES
      - "cancel"    -> result with reason=Canceled, cancellation_details set
      - Exception   -> raised by the .get() call (simulates SDK bubbling)
    """
    AUDIO_BYTES = b"\x00\x01MP3DATA"

    sdk = MagicMock()
    sdk.ResultReason.SynthesizingAudioCompleted = _COMPLETED
    sdk.ResultReason.Canceled = _CANCELED

    futures = []
    for r in results:
        future = MagicMock()
        if isinstance(r, Exception):
            future.get.side_effect = r
        elif r == "success":
            ok = MagicMock()
            ok.reason = _COMPLETED
            ok.audio_data = AUDIO_BYTES
            future.get.return_value = ok
        elif r == "cancel":
            cn = MagicMock()
            cn.reason = _CANCELED
            cn.cancellation_details.reason = "RateLimitExceeded"
            cn.cancellation_details.error_details = "rate limited"
            future.get.return_value = cn
        else:
            raise ValueError(f"unknown result kind: {r!r}")
        futures.append(future)

    synthesizer = MagicMock()
    synthesizer.speak_ssml_async.side_effect = futures
    sdk.SpeechSynthesizer.return_value = synthesizer
    monkeypatch.setattr(tts, "speechsdk", sdk)
    # Don't actually sleep during retry tests
    monkeypatch.setattr(tts.time, "sleep", lambda *_: None)
    return sdk


def test_synthesize_success(monkeypatch):
    _install_mock_sdk(monkeypatch, ["success"])
    p = AzureTTSProvider(key="k", region="r")
    out = p.synthesize("<speak/>")
    assert out == b"\x00\x01MP3DATA"


def test_synthesize_retries_once_then_succeeds(monkeypatch):
    sdk = _install_mock_sdk(monkeypatch, ["cancel", "success"])
    p = AzureTTSProvider(key="k", region="r")
    out = p.synthesize("<speak/>")
    assert out == b"\x00\x01MP3DATA"
    # Called the synthesizer twice (once cancel, once success)
    synthesizer = sdk.SpeechSynthesizer.return_value
    assert synthesizer.speak_ssml_async.call_count == 2


def test_synthesize_persistent_cancel_raises_after_one_retry(monkeypatch):
    _install_mock_sdk(monkeypatch, ["cancel", "cancel"])
    p = AzureTTSProvider(key="k", region="r")
    with pytest.raises(TTSError, match="after retry"):
        p.synthesize("<speak/>")


def test_synthesize_handles_sdk_exception_then_succeeds(monkeypatch):
    _install_mock_sdk(monkeypatch, [RuntimeError("network down"), "success"])
    p = AzureTTSProvider(key="k", region="r")
    out = p.synthesize("<speak/>")
    assert out == b"\x00\x01MP3DATA"


def test_synthesize_persistent_sdk_exception_raises(monkeypatch):
    _install_mock_sdk(monkeypatch, [RuntimeError("down"), RuntimeError("still down")])
    p = AzureTTSProvider(key="k", region="r")
    with pytest.raises(TTSError, match="after retry"):
        p.synthesize("<speak/>")


# ---------- synthesize_chunks ----------

def _fake_provider(synthesize_results: list[bytes]) -> MagicMock:
    p = MagicMock()
    p.name = "fake:test"
    p.voice = "test-voice"
    p.synthesize = MagicMock(side_effect=list(synthesize_results))
    return p


def test_synthesize_chunks_loops_and_writes_audio_and_char_count(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "log_dir", lambda d=None: tmp_path)
    chunks_dir = tmp_path / "06_chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_01.xml").write_text("<speak>one</speak>", encoding="utf-8")
    (chunks_dir / "chunk_02.xml").write_text("<speak>two longer</speak>", encoding="utf-8")

    fake = _fake_provider([b"AUDIO_ONE", b"AUDIO_TWO"])

    out_dir = tts.synthesize_chunks(provider=fake)
    assert out_dir == tmp_path / "07_audio_chunks"
    assert (out_dir / "chunk_01.mp3").read_bytes() == b"AUDIO_ONE"
    assert (out_dir / "chunk_02.mp3").read_bytes() == b"AUDIO_TWO"

    expected_chars = len("<speak>one</speak>") + len("<speak>two longer</speak>")
    assert (tmp_path / "tts_chars.txt").read_text(encoding="utf-8").strip() == str(expected_chars)


def test_synthesize_chunks_raises_if_no_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "log_dir", lambda d=None: tmp_path)
    (tmp_path / "06_chunks").mkdir()
    with pytest.raises(TTSError, match="no chunks"):
        tts.synthesize_chunks(provider=_fake_provider([]))


def test_synthesize_chunks_clears_stale_audio_from_prior_run(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "log_dir", lambda d=None: tmp_path)
    chunks_dir = tmp_path / "06_chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_01.xml").write_text("<speak/>", encoding="utf-8")
    audio_dir = tmp_path / "07_audio_chunks"
    audio_dir.mkdir()
    stale = audio_dir / "chunk_99.mp3"
    stale.write_bytes(b"old")

    tts.synthesize_chunks(provider=_fake_provider([b"NEW"]))
    assert not stale.exists()
    assert (audio_dir / "chunk_01.mp3").read_bytes() == b"NEW"


def test_synthesize_chunks_runs_chunks_in_sorted_order(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "log_dir", lambda d=None: tmp_path)
    chunks_dir = tmp_path / "06_chunks"
    chunks_dir.mkdir()
    # Create out of order; sorted glob should still feed them 01, 02, 03
    (chunks_dir / "chunk_03.xml").write_text("<speak>three</speak>", encoding="utf-8")
    (chunks_dir / "chunk_01.xml").write_text("<speak>one</speak>", encoding="utf-8")
    (chunks_dir / "chunk_02.xml").write_text("<speak>two</speak>", encoding="utf-8")

    seen = []
    fake = MagicMock()
    fake.name = "f"; fake.voice = "v"
    def _capture(ssml):
        seen.append(ssml)
        return b"X"
    fake.synthesize = _capture

    tts.synthesize_chunks(provider=fake)
    assert seen == ["<speak>one</speak>", "<speak>two</speak>", "<speak>three</speak>"]


# ---------- synthesize_one ----------

def test_synthesize_one_writes_to_sibling_audio_dir(tmp_path):
    chunks_dir = tmp_path / "06_chunks"
    chunks_dir.mkdir()
    chunk = chunks_dir / "chunk_01.xml"
    chunk.write_text("<speak>hi</speak>", encoding="utf-8")

    fake = _fake_provider([b"AUDIOBYTES"])
    out = tts.synthesize_one(chunk, provider=fake)
    assert out == tmp_path / "07_audio_chunks" / "chunk_01.mp3"
    assert out.read_bytes() == b"AUDIOBYTES"
