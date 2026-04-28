"""ObjectStore protocol + Cloudflare R2 implementation.

Spec §12.3: a thin abstraction over object storage so the rest of the
publish stage doesn't depend on a specific backend. The R2 implementation
uses boto3 against R2's S3-compatible endpoint, with credentials and the
bucket name read from the environment.

Adds `get()` to the protocol beyond the spec's listing — rss.py needs it
to fetch the prior feed.xml on each run, and going through the public
r2.dev URL would mean fighting Cloudflare's WAF on every poll.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger(__name__)


class ObjectStoreError(RuntimeError):
    """Stage 9 hard failure — auth, bucket, or network problem talking to storage."""


class ObjectStore(Protocol):
    """Storage abstraction. Swap the impl to migrate to S3, GCS, etc."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def url_for(self, key: str) -> str: ...
    def exists(self, key: str) -> bool: ...


class R2ObjectStore:
    """Cloudflare R2 implementation. Reads R2_* env vars at construction.

    `R2_PUBLIC_BASE_URL` is the public r2.dev or custom-domain base for
    objects (no trailing slash needed — we strip one). The boto3 endpoint
    is derived as `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`.
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        public_base_url: str | None = None,
        client: object | None = None,
    ) -> None:
        self._account_id = account_id if account_id is not None else os.environ.get("R2_ACCOUNT_ID")
        self._access_key = access_key if access_key is not None else os.environ.get("R2_ACCESS_KEY_ID")
        self._secret_key = secret_key if secret_key is not None else os.environ.get("R2_SECRET_ACCESS_KEY")
        self._bucket = bucket if bucket is not None else os.environ.get("R2_BUCKET")
        raw_url = public_base_url if public_base_url is not None else os.environ.get("R2_PUBLIC_BASE_URL")
        self._public_base_url = (raw_url or "").rstrip("/")

        for name, val in (
            ("R2_ACCOUNT_ID", self._account_id),
            ("R2_ACCESS_KEY_ID", self._access_key),
            ("R2_SECRET_ACCESS_KEY", self._secret_key),
            ("R2_BUCKET", self._bucket),
            ("R2_PUBLIC_BASE_URL", self._public_base_url),
        ):
            if not val:
                raise ObjectStoreError(f"{name} not set")

        self._client = client if client is not None else boto3.client(
            "s3",
            endpoint_url=f"https://{self._account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name="auto",
        )

    @property
    def name(self) -> str:
        return f"r2:{self._bucket}"

    @property
    def bucket(self) -> str:
        return self._bucket

    def put(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (ClientError, BotoCoreError) as e:
            raise ObjectStoreError(f"put {key!r} failed: {e}") from e
        log.info("store: put %s/%s (%d bytes, %s)", self._bucket, key, len(data), content_type)

    def get(self, key: str) -> bytes | None:
        """Return the object's bytes, or None if the key does not exist."""
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return None
            raise ObjectStoreError(f"get {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise ObjectStoreError(f"get {key!r} failed: {e}") from e

    def url_for(self, key: str) -> str:
        """Build the public HTTPS URL for `key`. Strips a leading slash so
        callers can pass either `episodes/...` or `/episodes/...`."""
        return f"{self._public_base_url}/{key.lstrip('/')}"

    def exists(self, key: str) -> bool:
        """True if the key exists in the bucket. Uses HEAD via boto3 (auth);
        does NOT go through the public URL."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            # head_object returns "404"/"NotFound" rather than "NoSuchKey".
            if code in ("NoSuchKey", "404", "NotFound"):
                return False
            raise ObjectStoreError(f"head {key!r} failed: {e}") from e
        except BotoCoreError as e:
            raise ObjectStoreError(f"head {key!r} failed: {e}") from e
