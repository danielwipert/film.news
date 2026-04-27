"""Stage 7 — TTS.

Spec §10: TTSProvider Protocol + Azure AI Speech implementation. Synthesizes
each SSML chunk from 06_chunks/ to an MP3 in 07_audio_chunks/. Logs the
total character count per run for free-tier monitoring (spec §10.6).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Protocol

import azure.cognitiveservices.speech as speechsdk

from dan import config
from dan.io import read_text, write_text
from dan.paths import log_dir, today_utc

log = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
RETRY_BACKOFF_SECS = 2.0


class TTSError(RuntimeError):
    """Stage 7 hard failure — auth, persistent service error, malformed SSML."""


class TTSProvider(Protocol):
    """Spec §10.2 abstraction so we can swap providers later."""

    def synthesize(self, ssml: str) -> bytes:
        """Return MP3 bytes for the given SSML, or raise TTSError."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def voice(self) -> str: ...


class AzureTTSProvider:
    """Azure AI Speech implementation of TTSProvider.

    Reads AZURE_SPEECH_KEY and AZURE_SPEECH_REGION from environment unless
    passed in directly. Synthesizes to Audio24Khz48KBitRateMonoMp3 per spec
    §10.3. Retries once with a short backoff on any synthesis failure; the
    second failure is a hard error per spec §18.
    """

    def __init__(
        self,
        key: str | None = None,
        region: str | None = None,
        voice: str = DEFAULT_VOICE,
    ) -> None:
        self._key = key if key is not None else os.environ.get("AZURE_SPEECH_KEY")
        self._region = region if region is not None else os.environ.get("AZURE_SPEECH_REGION")
        self._voice = voice
        if not self._key:
            raise TTSError("AZURE_SPEECH_KEY not set")
        if not self._region:
            raise TTSError("AZURE_SPEECH_REGION not set")

    @property
    def name(self) -> str:
        return f"azure:{self._region}"

    @property
    def voice(self) -> str:
        return self._voice

    def _make_synthesizer(self) -> speechsdk.SpeechSynthesizer:
        speech_config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        # audio_config=None: don't write to a device or file; we capture bytes from the result.
        return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    def synthesize(self, ssml: str) -> bytes:
        last_err: str | None = None
        for attempt in (1, 2):
            try:
                synthesizer = self._make_synthesizer()
                result = synthesizer.speak_ssml_async(ssml).get()
            except Exception as e:  # noqa: BLE001 — any SDK exception treated as transient
                last_err = f"SDK exception: {e}"
                log.warning("tts: SDK exception (attempt %d/2): %s", attempt, e)
                if attempt == 1:
                    time.sleep(RETRY_BACKOFF_SECS)
                continue

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return bytes(result.audio_data)

            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                last_err = f"canceled: {details.reason} ({details.error_details})"
            else:
                last_err = f"unexpected result reason: {result.reason}"

            log.warning("tts: synthesis failed (attempt %d/2): %s", attempt, last_err)
            if attempt == 1:
                time.sleep(RETRY_BACKOFF_SECS)

        raise TTSError(f"Azure TTS failed after retry: {last_err}")


def _provider_from_config() -> AzureTTSProvider:
    voice_cfg = config.voice() or {}
    voice = voice_cfg.get("voice", DEFAULT_VOICE)
    return AzureTTSProvider(voice=voice)


def synthesize_chunks(d: date | None = None, *, provider: TTSProvider | None = None) -> Path:
    """Synthesize every chunk in 06_chunks/ to MP3 in 07_audio_chunks/.

    Returns the audio_chunks directory. Raises TTSError if there are no
    chunks to synthesize. Per-chunk char count is summed and written to
    tts_chars.txt for monthly free-tier monitoring (§10.6).
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    chunks_dir = day_dir / "06_chunks"
    audio_dir = day_dir / "07_audio_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for old in audio_dir.glob("chunk_*.mp3"):
        old.unlink()

    chunk_files = sorted(chunks_dir.glob("chunk_*.xml"))
    if not chunk_files:
        raise TTSError(f"no chunks found in {chunks_dir}")

    if provider is None:
        provider = _provider_from_config()

    log.info("tts: synthesizing %d chunk(s) via %s (voice=%s)",
             len(chunk_files), provider.name, provider.voice)

    total_chars = 0
    for chunk_file in chunk_files:
        ssml = read_text(chunk_file)
        char_count = len(ssml)
        log.info("tts: %s -> MP3 (%d chars)", chunk_file.name, char_count)
        audio_bytes = provider.synthesize(ssml)
        out_path = audio_dir / f"{chunk_file.stem}.mp3"
        out_path.write_bytes(audio_bytes)
        log.info("tts: wrote %s (%d bytes)", out_path.name, len(audio_bytes))
        total_chars += char_count

    write_text(day_dir / "tts_chars.txt", f"{total_chars}\n")
    log.info("tts: synthesized %d chunk(s), %d total chars", len(chunk_files), total_chars)
    return audio_dir


def synthesize_one(chunk_path: Path, *, provider: TTSProvider | None = None) -> Path:
    """Synthesize a single chunk to its sibling MP3. Useful for voice auditions
    and re-running a single failed chunk without redoing the rest."""
    if provider is None:
        provider = _provider_from_config()
    ssml = read_text(chunk_path)
    log.info("tts: %s -> MP3 (%d chars, voice=%s)",
             chunk_path.name, len(ssml), provider.voice)
    audio_bytes = provider.synthesize(ssml)
    out_path = chunk_path.parent.parent / "07_audio_chunks" / f"{chunk_path.stem}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    log.info("tts: wrote %s (%d bytes)", out_path.name, len(audio_bytes))
    return out_path
