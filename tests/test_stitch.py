"""Tests for Stage 8 — dan.audio.stitch."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mutagen.id3 import ID3

from dan.audio import stitch
from dan.audio.stitch import StitchError


# ---------- _write_concat_list ----------

def test_write_concat_list_uses_absolute_paths_and_quotes(tmp_path):
    a = tmp_path / "chunk_01.mp3"; a.write_bytes(b"")
    b = tmp_path / "chunk_02.mp3"; b.write_bytes(b"")
    list_path = tmp_path / "concat_list.txt"
    stitch._write_concat_list(list_path, [a, b])
    text = list_path.read_text(encoding="utf-8")
    assert text == f"file '{a.resolve()}'\nfile '{b.resolve()}'\n"


def test_write_concat_list_escapes_single_quote(tmp_path):
    # Simulate a file path containing a literal single quote by passing a
    # Path object whose resolve()-stringification has one. We can't easily
    # create such a file on Windows, so check the escape rule on a fake path.
    fake = Path("/weird/it's/chunk_01.mp3")
    list_path = tmp_path / "concat_list.txt"
    stitch._write_concat_list(list_path, [fake])
    text = list_path.read_text(encoding="utf-8")
    # ffmpeg's escape rule: ' -> '\''
    assert r"it'\''s" in text


# ---------- _episode_number ----------

def test_episode_number_first_run_is_one(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    assert stitch._episode_number(date(2026, 4, 27)) == 1


def test_episode_number_increments_with_prior_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    for d in ("2026-04-25", "2026-04-26"):
        day = tmp_path / d
        day.mkdir()
        (day / "08_episode.mp3").write_bytes(b"x")
    assert stitch._episode_number(date(2026, 4, 27)) == 3


def test_episode_number_idempotent_for_existing_date(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    for d in ("2026-04-25", "2026-04-26", "2026-04-27"):
        day = tmp_path / d
        day.mkdir()
        (day / "08_episode.mp3").write_bytes(b"x")
    # Today already exists -> still numbered 3, not 4
    assert stitch._episode_number(date(2026, 4, 27)) == 3


def test_episode_number_orders_by_date_not_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    # Create out of chronological order
    for d in ("2026-05-01", "2026-04-15"):
        day = tmp_path / d
        day.mkdir()
        (day / "08_episode.mp3").write_bytes(b"x")
    assert stitch._episode_number(date(2026, 4, 20)) == 2


# ---------- _tag_episode ----------

def test_tag_episode_writes_expected_id3_frames(tmp_path):
    # mutagen will happily write an ID3v2 header into any file, even if the
    # rest isn't valid MPEG. That's enough to round-trip the tag fields.
    mp3 = tmp_path / "ep.mp3"
    mp3.write_bytes(b"\x00" * 256)
    stitch._tag_episode(mp3, date(2026, 4, 27), 5)
    tags = ID3(mp3)
    assert str(tags["TIT2"]) == "DAN Film Brief — 2026-04-27"
    assert str(tags["TPE1"]) == "Daily Audio News"
    assert str(tags["TALB"]) == "Daily Audio News — Film"
    assert str(tags["TDRC"]) == "2026"
    assert str(tags["TCON"]) == "Podcast"
    assert str(tags["TRCK"]) == "5"


# ---------- stitch (orchestration) ----------

def _seed_chunks(day_dir: Path, n: int = 2) -> Path:
    audio_dir = day_dir / "07_audio_chunks"
    audio_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        (audio_dir / f"chunk_{i:02d}.mp3").write_bytes(b"\x00" * 64)
    return audio_dir


def test_stitch_raises_if_no_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "log_dir", lambda d=None: tmp_path)
    (tmp_path / "07_audio_chunks").mkdir()
    with pytest.raises(StitchError, match="no audio chunks"):
        stitch.stitch()


def test_stitch_raises_if_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "log_dir", lambda d=None: tmp_path)
    _seed_chunks(tmp_path)
    monkeypatch.setattr(stitch.shutil, "which", lambda _name: None)
    with pytest.raises(StitchError, match="ffmpeg not found"):
        stitch.stitch()


def test_stitch_invokes_ffmpeg_twice_and_tags(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(stitch, "today_utc", lambda: date(2026, 4, 27))
    monkeypatch.setattr(stitch.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    _seed_chunks(tmp_path, n=2)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # Simulate ffmpeg's loudnorm pass producing the output file. The
        # second call's last argument is the output path (08_episode.mp3).
        if len(calls) == 2:
            Path(cmd[-1]).write_bytes(b"\x00" * 256)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    monkeypatch.setattr(stitch.subprocess, "run", fake_run)

    out = stitch.stitch()
    assert out == tmp_path / "08_episode.mp3"
    assert out.exists()
    assert len(calls) == 2

    # Pass 1: concat demuxer with -c copy
    pass1 = calls[0]
    assert "-f" in pass1 and "concat" in pass1
    assert "-c" in pass1 and "copy" in pass1
    # Pass 2: loudnorm + sample rate + bitrate -> 08_episode.mp3
    pass2 = calls[1]
    assert stitch.LOUDNORM_FILTER in pass2
    assert stitch.SAMPLE_RATE in pass2
    assert stitch.BITRATE in pass2
    assert pass2[-1].endswith("08_episode.mp3")

    # ID3 was actually written
    tags = ID3(out)
    assert str(tags["TIT2"]) == "DAN Film Brief — 2026-04-27"
    assert str(tags["TRCK"]) == "1"


def test_stitch_propagates_ffmpeg_error(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(stitch.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    _seed_chunks(tmp_path)

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "boom"
        return result

    monkeypatch.setattr(stitch.subprocess, "run", fake_run)
    with pytest.raises(StitchError, match=r"ffmpeg failed.*boom"):
        stitch.stitch()


def test_stitch_overwrites_stale_episode(monkeypatch, tmp_path):
    monkeypatch.setattr(stitch, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(stitch, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(stitch, "today_utc", lambda: date(2026, 4, 27))
    monkeypatch.setattr(stitch.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    _seed_chunks(tmp_path)

    stale = tmp_path / "08_episode.mp3"
    stale.write_bytes(b"OLD" * 100)

    def fake_run(cmd, **kwargs):
        if "loudnorm=I=-16:TP=-1.5:LRA=11" in cmd:
            Path(cmd[-1]).write_bytes(b"\x00" * 256)
        r = MagicMock(); r.returncode = 0; r.stderr = ""; return r

    monkeypatch.setattr(stitch.subprocess, "run", fake_run)
    out = stitch.stitch()
    assert out.read_bytes().startswith(b"ID3") or len(out.read_bytes()) == 256 + len(b"ID3...")
    # Specifically: the stale "OLD..." prefix is gone.
    assert not out.read_bytes().startswith(b"OLD")
