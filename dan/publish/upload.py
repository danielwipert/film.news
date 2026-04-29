"""Stage 9.2 — Upload episode MP3 to object storage, HEAD-verify reachable.

Spec §12.4: take logs/YYYY-MM-DD/08_episode.mp3, upload to R2 at the keyed
path `episodes/YYYY/MM/dan-film-YYYY-MM-DD.mp3` with `Content-Type:
audio/mpeg`, then HEAD-request the resulting public URL to confirm a
listener could actually fetch it. Writes 09_upload.json with the URL and
byte length so rss.py has the data it needs for the <enclosure> element.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from dan.io import write_json
from dan.paths import log_dir, today_utc
from dan.publish.store import ObjectStore, ObjectStoreError, R2ObjectStore

log = logging.getLogger(__name__)

CONTENT_TYPE = "audio/mpeg"
HEAD_TIMEOUT_SECS = 15.0
# A non-default User-Agent — Cloudflare's WAF in front of pub-*.r2.dev
# returns error 1010 for `Python-urllib/X.Y` and similar bot-flavored UAs.
# Real podcast clients always send something realistic, so we do too.
HEAD_USER_AGENT = "Mozilla/5.0 (compatible; dan-pipeline/1.0)"


class UploadError(RuntimeError):
    """Stage 9.2 hard failure — missing episode, upload error, or HEAD failed."""


def episode_key(d: date) -> str:
    """Build the spec §12.2 storage key for date `d`."""
    return f"episodes/{d:%Y}/{d:%m}/dan-film-{d.isoformat()}.mp3"


def _head_verify(url: str) -> int:
    """HEAD `url`; return content-length on success, raise UploadError otherwise."""
    try:
        resp = httpx.head(
            url,
            headers={"User-Agent": HEAD_USER_AGENT},
            timeout=HEAD_TIMEOUT_SECS,
            follow_redirects=True,
        )
    except httpx.HTTPError as e:
        raise UploadError(f"HEAD {url} network error: {e}") from e
    if resp.status_code != 200:
        raise UploadError(f"HEAD {url} returned HTTP {resp.status_code}")
    cl = resp.headers.get("Content-Length")
    return int(cl) if cl and cl.isdigit() else -1


def upload(d: date | None = None, *, store: ObjectStore | None = None) -> Path:
    """Upload today's 08_episode.mp3 to storage; write 09_upload.json.

    Returns the path to 09_upload.json. Raises UploadError if the episode
    file is missing, the upload fails, or the post-upload HEAD doesn't
    return 200. Idempotent: re-runs overwrite the storage object and the
    local manifest.
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    episode_path = day_dir / "08_episode.mp3"
    if not episode_path.exists():
        raise UploadError(f"missing episode file: {episode_path}")

    if store is None:
        store = R2ObjectStore()

    data = episode_path.read_bytes()
    size = len(data)
    key = episode_key(d)
    log.info("upload: %s (%d bytes) -> %s/%s", episode_path.name, size, store.name, key)

    try:
        store.put(key, data, CONTENT_TYPE)
    except ObjectStoreError as e:
        raise UploadError(f"put failed: {e}") from e

    url = store.url_for(key)
    log.info("upload: HEAD-verifying %s", url)
    served_size = _head_verify(url)
    if served_size != -1 and served_size != size:
        # Server reported a different content-length than we uploaded — refuse
        # to publish a feed that points at a truncated/corrupted object.
        raise UploadError(
            f"HEAD content-length {served_size} != uploaded size {size} for {url}"
        )

    manifest = {
        "url": url,
        "key": key,
        "size": size,
        "content_type": CONTENT_TYPE,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = day_dir / "09_upload.json"
    write_json(out_path, manifest)
    log.info("upload: wrote %s", out_path.name)
    return out_path
