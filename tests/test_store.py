"""Tests for dan.publish.store — ObjectStore protocol + R2 impl.

Uses a MagicMock'd boto3 client throughout; no real network calls.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from dan.publish import store
from dan.publish.store import ObjectStoreError, R2ObjectStore


def _client_error(code: str, op: str = "GetObject") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


def _store(client: MagicMock | None = None, **overrides) -> R2ObjectStore:
    """Build an R2ObjectStore with valid defaults; override per-test."""
    kwargs = dict(
        account_id="acct123",
        access_key="ak",
        secret_key="sk",
        bucket="dan-podcast",
        public_base_url="https://pub-xyz.r2.dev",
        client=client if client is not None else MagicMock(),
    )
    kwargs.update(overrides)
    return R2ObjectStore(**kwargs)


# ---------- init / env validation ----------

def test_init_raises_when_required_env_var_missing(monkeypatch):
    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                "R2_BUCKET", "R2_PUBLIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ObjectStoreError, match="R2_ACCOUNT_ID"):
        R2ObjectStore(client=MagicMock())


def test_init_validates_each_var_individually(monkeypatch):
    monkeypatch.delenv("R2_PUBLIC_BASE_URL", raising=False)
    with pytest.raises(ObjectStoreError, match="R2_PUBLIC_BASE_URL"):
        R2ObjectStore(
            account_id="a", access_key="b", secret_key="c", bucket="d",
            client=MagicMock(),
        )


def test_init_strips_trailing_slash_from_public_url():
    s = _store(public_base_url="https://pub-xyz.r2.dev/")
    assert s.url_for("foo.mp3") == "https://pub-xyz.r2.dev/foo.mp3"


def test_init_builds_boto3_client_with_r2_endpoint(monkeypatch):
    fake_client = MagicMock()
    captured = {}

    def fake_boto3_client(service, **kwargs):
        captured["service"] = service
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(store.boto3, "client", fake_boto3_client)
    R2ObjectStore(
        account_id="abc123def", access_key="K", secret_key="S",
        bucket="dan-podcast", public_base_url="https://pub-x.r2.dev",
    )
    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://abc123def.r2.cloudflarestorage.com"
    assert captured["aws_access_key_id"] == "K"
    assert captured["aws_secret_access_key"] == "S"
    assert captured["region_name"] == "auto"


def test_name_and_bucket_properties():
    s = _store()
    assert s.name == "r2:dan-podcast"
    assert s.bucket == "dan-podcast"


# ---------- put ----------

def test_put_calls_put_object_with_expected_args():
    client = MagicMock()
    s = _store(client=client)
    s.put("episodes/2026/04/foo.mp3", b"AUDIO", "audio/mpeg")
    client.put_object.assert_called_once_with(
        Bucket="dan-podcast",
        Key="episodes/2026/04/foo.mp3",
        Body=b"AUDIO",
        ContentType="audio/mpeg",
    )


def test_put_wraps_client_error():
    client = MagicMock()
    client.put_object.side_effect = _client_error("AccessDenied", "PutObject")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="put 'k' failed"):
        s.put("k", b"x", "text/plain")


def test_put_wraps_botocore_error():
    client = MagicMock()
    client.put_object.side_effect = EndpointConnectionError(endpoint_url="https://x")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="put 'k' failed"):
        s.put("k", b"x", "text/plain")


# ---------- get ----------

def test_get_returns_bytes():
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(b"HELLO")}
    s = _store(client=client)
    assert s.get("k") == b"HELLO"
    client.get_object.assert_called_once_with(Bucket="dan-podcast", Key="k")


def test_get_returns_none_for_missing_key():
    client = MagicMock()
    client.get_object.side_effect = _client_error("NoSuchKey")
    s = _store(client=client)
    assert s.get("missing") is None


def test_get_returns_none_for_404_alias():
    client = MagicMock()
    client.get_object.side_effect = _client_error("404")
    s = _store(client=client)
    assert s.get("missing") is None


def test_get_raises_on_other_client_error():
    client = MagicMock()
    client.get_object.side_effect = _client_error("AccessDenied")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="get 'k' failed"):
        s.get("k")


def test_get_raises_on_botocore_error():
    client = MagicMock()
    client.get_object.side_effect = EndpointConnectionError(endpoint_url="https://x")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="get 'k' failed"):
        s.get("k")


# ---------- url_for ----------

def test_url_for_simple_key():
    assert _store().url_for("foo.mp3") == "https://pub-xyz.r2.dev/foo.mp3"


def test_url_for_nested_key():
    assert _store().url_for("episodes/2026/04/dan-film-2026-04-26.mp3") == (
        "https://pub-xyz.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3"
    )


def test_url_for_strips_leading_slash():
    assert _store().url_for("/feed.xml") == "https://pub-xyz.r2.dev/feed.xml"


# ---------- exists ----------

def test_exists_returns_true_on_head_success():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 1}
    s = _store(client=client)
    assert s.exists("k") is True
    client.head_object.assert_called_once_with(Bucket="dan-podcast", Key="k")


def test_exists_returns_false_on_404():
    client = MagicMock()
    client.head_object.side_effect = _client_error("404", "HeadObject")
    s = _store(client=client)
    assert s.exists("missing") is False


def test_exists_returns_false_on_not_found():
    client = MagicMock()
    client.head_object.side_effect = _client_error("NotFound", "HeadObject")
    s = _store(client=client)
    assert s.exists("missing") is False


def test_exists_raises_on_other_client_error():
    client = MagicMock()
    client.head_object.side_effect = _client_error("AccessDenied", "HeadObject")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="head 'k' failed"):
        s.exists("k")


def test_exists_raises_on_botocore_error():
    client = MagicMock()
    client.head_object.side_effect = EndpointConnectionError(endpoint_url="https://x")
    s = _store(client=client)
    with pytest.raises(ObjectStoreError, match="head 'k' failed"):
        s.exists("k")
