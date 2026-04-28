"""Stage 8 — Concat chunks, loudnorm, ID3 tags -> 08_episode.mp3.

Spec §11: ffmpeg concat demuxer joins the per-chunk MP3s losslessly, then
a second ffmpeg pass applies EBU R128 loudnorm (-16 LUFS / -1.5 dBTP /
LRA 11) so every day's episode lands at the same perceived loudness.
mutagen writes ID3 tags so podcast apps display title/artist/album/track.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError, TALB, TCON, TDRC, TIT2, TPE1, TRCK

from dan.paths import LOGS_ROOT, log_dir, today_utc

log = logging.getLogger(__name__)

FFMPEG = "ffmpeg"

# Loudness target (spec §11.3).
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
SAMPLE_RATE = "44100"
BITRATE = "96k"

ID3_ARTIST = "Daily Audio News"
ID3_ALBUM = "Daily Audio News — Film"
ID3_GENRE = "Podcast"


class StitchError(RuntimeError):
    """Stage 8 hard failure — missing ffmpeg, missing chunks, ffmpeg or tagging error."""


def _write_concat_list(path: Path, chunk_files: list[Path]) -> None:
    """Write the ffmpeg concat demuxer's list file.

    Each line is `file '<absolute path>'`. Single quotes inside the path
    are escaped per ffmpeg's concat-demuxer rules (`'` -> `'\\''`). Paths
    are absolute so the list works regardless of the cwd we hand ffmpeg.
    """
    lines = []
    for f in chunk_files:
        p = str(f.resolve()).replace("'", r"'\''")
        lines.append(f"file '{p}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", *args]
    log.debug("stitch: running %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise StitchError(f"ffmpeg not found on PATH: {e}") from e
    if result.returncode != 0:
        raise StitchError(
            f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def _episode_number(d: date) -> int:
    """1-based ordinal of `d` among all dated runs that have a stitched episode.

    Counts every existing `logs/YYYY-MM-DD/08_episode.mp3` plus today's date,
    deduped and sorted. Idempotent across re-runs: rerunning Stage 8 on the
    same date keeps the same episode number, and gaps in the calendar do
    not skip numbers.
    """
    dates = {p.parent.name for p in LOGS_ROOT.glob("*/08_episode.mp3")}
    dates.add(d.isoformat())
    return sorted(dates).index(d.isoformat()) + 1


def _tag_episode(path: Path, d: date, episode_number: int) -> None:
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags["TIT2"] = TIT2(encoding=3, text=f"DAN Film Brief — {d.isoformat()}")
    tags["TPE1"] = TPE1(encoding=3, text=ID3_ARTIST)
    tags["TALB"] = TALB(encoding=3, text=ID3_ALBUM)
    tags["TDRC"] = TDRC(encoding=3, text=str(d.year))
    tags["TCON"] = TCON(encoding=3, text=ID3_GENRE)
    tags["TRCK"] = TRCK(encoding=3, text=str(episode_number))
    tags.save(path)


def stitch(d: date | None = None) -> Path:
    """Concat 07_audio_chunks/*.mp3 -> 08_episode.mp3 with loudnorm + ID3.

    Returns the path to 08_episode.mp3. Raises StitchError if chunks are
    missing, ffmpeg isn't on PATH, or any ffmpeg call exits non-zero.
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)
    audio_dir = day_dir / "07_audio_chunks"
    chunk_files = sorted(audio_dir.glob("chunk_*.mp3"))
    if not chunk_files:
        raise StitchError(f"no audio chunks found in {audio_dir}")
    if shutil.which(FFMPEG) is None:
        raise StitchError(f"ffmpeg not found on PATH ({FFMPEG!r})")

    out_path = day_dir / "08_episode.mp3"
    if out_path.exists():
        out_path.unlink()

    log.info("stitch: concatenating %d chunk(s)", len(chunk_files))
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        concat_list = td_path / "concat_list.txt"
        joined = td_path / "joined.mp3"
        _write_concat_list(concat_list, chunk_files)

        # Pass 1: lossless concat via the demuxer. Requires identical codec /
        # sample rate / channels across inputs — Azure's per-chunk output is
        # uniform per voice, so this holds.
        _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(joined),
        ])

        # Pass 2: loudnorm + re-encode at a fixed sample rate/bitrate so every
        # day's episode lands at the same perceived loudness and quality.
        log.info("stitch: applying loudnorm")
        _run_ffmpeg([
            "-i", str(joined),
            "-af", LOUDNORM_FILTER,
            "-ar", SAMPLE_RATE,
            "-b:a", BITRATE,
            str(out_path),
        ])

    episode_no = _episode_number(d)
    log.info("stitch: writing ID3 tags (episode %d)", episode_no)
    _tag_episode(out_path, d, episode_no)

    size = out_path.stat().st_size
    log.info("stitch: wrote %s (%d bytes, episode %d)", out_path.name, size, episode_no)
    return out_path
