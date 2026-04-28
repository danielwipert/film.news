"""Tests for Stage 9.2 — dan.publish.upload."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from dan.publish import upload
from dan.publish.store import ObjectStoreError
from dan.publish.upload import UploadError


# ---------- _episode_key ----------

def test_episode_key_zero_pads_month():
    assert upload._episode_key(date(2026, 4, 27)) == (
        "episodes/2026/04/dan-film-2026-04-27.mp3"
    )


def test_episode_key_double_digit_month():
    assert upload._episode_key(date(2026, 12, 1)) == (
        "episodes/2026/12/dan-film-2026-12-01.mp3"
    )


# ---------- _head_verify ----------

def _fake_response(status: int, content_length: str | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"Content-Length": content_length} if content_length else {}
    return r


def test_head_verify_returns_content_length(monkeypatch):
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200, "12345"))
    assert upload._head_verify("https://x/y") == 12345


def test_head_verify_returns_minus_one_when_no_content_length(monkeypatch):
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200))
    assert upload._head_verify("https://x/y") == -1


def test_head_verify_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(403))
    with pytest.raises(UploadError, match="HTTP 403"):
        upload._head_verify("https://x/y")


def test_head_verify_raises_on_network_error(monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectError("dns down")
    monkeypatch.setattr(upload.httpx, "head", boom)
    with pytest.raises(UploadError, match="network error"):
        upload._head_verify("https://x/y")


def test_head_verify_sends_browser_ua(monkeypatch):
    seen = {}
    def capture(url, **kw):
        seen["headers"] = kw.get("headers", {})
        return _fake_response(200, "1")
    monkeypatch.setattr(upload.httpx, "head", capture)
    upload._head_verify("https://x/y")
    assert "User-Agent" in seen["headers"]
    # Must NOT be the default python-httpx UA (Cloudflare WAF blocks it)
    assert "python" not in seen["headers"]["User-Agent"].lower()


# ---------- upload (orchestration) ----------

def _fake_store(url: str = "https://pub-xyz.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3") -> MagicMock:
    s = MagicMock()
    s.name = "fake:test"
    s.url_for.return_value = url
    return s


def _seed_episode(tmp_path, content: bytes = b"\x00" * 1024) -> None:
    (tmp_path / "08_episode.mp3").write_bytes(content)


def test_upload_happy_path_writes_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200, "1024"))
    _seed_episode(tmp_path, content=b"\xff" * 1024)

    store = _fake_store()
    out = upload.upload(store=store)

    assert out == tmp_path / "09_upload.json"
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["url"] == "https://pub-xyz.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3"
    assert manifest["key"] == "episodes/2026/04/dan-film-2026-04-26.mp3"
    assert manifest["size"] == 1024
    assert manifest["content_type"] == "audio/mpeg"
    assert "uploaded_at" in manifest


def test_upload_calls_store_put_with_audio_mpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200, "1024"))
    payload = b"\xff" * 1024
    _seed_episode(tmp_path, content=payload)

    store = _fake_store()
    upload.upload(store=store)

    store.put.assert_called_once_with(
        "episodes/2026/04/dan-film-2026-04-26.mp3", payload, "audio/mpeg",
    )


def test_upload_raises_on_missing_episode(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    with pytest.raises(UploadError, match="missing episode file"):
        upload.upload(store=_fake_store())


def test_upload_wraps_store_put_error(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    _seed_episode(tmp_path)
    store = _fake_store()
    store.put.side_effect = ObjectStoreError("denied")
    with pytest.raises(UploadError, match=r"put failed.*denied"):
        upload.upload(store=store)


def test_upload_raises_on_head_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(404))
    _seed_episode(tmp_path)
    with pytest.raises(UploadError, match="HTTP 404"):
        upload.upload(store=_fake_store())


def test_upload_raises_on_size_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload, "today_utc", lambda: date(2026, 4, 26))
    # We upload 1024 bytes, but Cloudflare reports 999 — refuse to publish.
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200, "999"))
    _seed_episode(tmp_path, content=b"\xff" * 1024)
    with pytest.raises(UploadError, match=r"content-length 999 != uploaded size 1024"):
        upload.upload(store=_fake_store())


def test_upload_uses_explicit_date(monkeypatch, tmp_path):
    monkeypatch.setattr(upload, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(upload.httpx, "head", lambda *a, **kw: _fake_response(200, "1"))
    (tmp_path / "08_episode.mp3").write_bytes(b"\xff")
    store = _fake_store()
    upload.upload(date(2026, 1, 5), store=store)
    args, _ = store.put.call_args
    assert args[0] == "episodes/2026/01/dan-film-2026-01-05.mp3"
