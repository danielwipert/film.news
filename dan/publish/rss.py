"""Stage 9.3 — feed.xml generation/update.

Spec §12.5: download the current feed.xml from R2 if it exists; otherwise
scaffold a fresh one from config/show.yaml. Append today's <item>, sort
by pubDate desc, cap at 50 entries, re-upload as application/rss+xml.

feedgen handles serialization (with the iTunes namespace via the
'podcast' extension) but it has no parsing API, so prior feed.xml is
parsed with lxml and replayed into a fresh FeedGenerator. Same-date
re-runs are idempotent: any prior <item> with the same guid as today's
episode is dropped before append.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from feedgen.feed import FeedGenerator
from lxml import etree
from mutagen.mp3 import MP3

from dan.config import show as load_show
from dan.io import read_json
from dan.paths import log_dir, today_utc
from dan.publish.store import ObjectStore, ObjectStoreError, R2ObjectStore
from dan.publish.upload import episode_key

log = logging.getLogger(__name__)

FEED_KEY = "feed.xml"
FEED_CONTENT_TYPE = "application/rss+xml"
ENCLOSURE_TYPE = "audio/mpeg"
EPISODE_PREFIX = "episodes/"
# Rolling retention. The feed cap and the storage prune are the SAME number —
# they must move together, otherwise the feed points at deleted MP3s and Apple
# Podcasts / Spotify will show 404s for old episodes.
RETENTION_COUNT = 7

# iTunes namespace used to read prior <itunes:duration> when re-parsing feed.xml.
NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


class RSSError(RuntimeError):
    """Stage 9.3 hard failure — bad config, missing inputs, parse/serialize error."""


def _format_duration(seconds: float) -> str:
    """Convert float seconds -> 'HH:MM:SS' for itunes:duration."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _episode_pubdate(d: date) -> datetime:
    """Episode pubDate: midnight UTC of the run date. feedgen RFC-822-formats it."""
    return datetime.combine(d, time(0, 0, 0), tzinfo=timezone.utc)


def _today_entry(d: date, day_dir: Path) -> dict[str, Any]:
    """Build today's <item> data from on-disk Stage 8/9.1/9.2 artifacts."""
    upload_path = day_dir / "09_upload.json"
    if not upload_path.exists():
        raise RSSError(f"missing {upload_path} (run upload first)")
    desc_path = day_dir / "09_description.txt"
    if not desc_path.exists():
        raise RSSError(f"missing {desc_path} (run describe first)")
    episode_path = day_dir / "08_episode.mp3"
    if not episode_path.exists():
        raise RSSError(f"missing {episode_path} (run stitch first)")

    manifest = read_json(upload_path)
    description = desc_path.read_text(encoding="utf-8").strip()

    try:
        mp3 = MP3(episode_path)
        duration = _format_duration(mp3.info.length)
    except Exception as e:
        raise RSSError(f"could not read MP3 duration from {episode_path}: {e}") from e

    return {
        "title": f"DAN Film Brief — {d.isoformat()}",
        "description": description,
        "url": manifest["url"],
        "size": int(manifest["size"]),
        "pubDate": _episode_pubdate(d),
        "guid": manifest["url"],
        "duration": duration,
    }


def _parse_rfc822(s: str) -> datetime:
    dt = parsedate_to_datetime(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_existing_feed(xml: bytes) -> list[dict[str, Any]]:
    """Pull <item> entries out of a prior feed.xml. Returns dicts in the same
    shape as `_today_entry()`. Items missing required fields are skipped with
    a warning rather than raising — one corrupt entry shouldn't poison the feed."""
    try:
        root = etree.fromstring(xml)
    except etree.XMLSyntaxError as e:
        raise RSSError(f"prior feed.xml is malformed: {e}") from e

    entries: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        pubdate_str = (item.findtext("pubDate") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        enclosure = item.find("enclosure")
        duration = (item.findtext("itunes:duration", namespaces=NS) or "").strip()

        if enclosure is None or not pubdate_str or not guid:
            log.warning("rss: dropping prior <item> missing required fields (guid=%r)", guid)
            continue
        try:
            pub = _parse_rfc822(pubdate_str)
            size = int(enclosure.get("length") or 0)
        except (TypeError, ValueError) as e:
            log.warning("rss: dropping prior <item> with bad pubDate/length (guid=%r): %s", guid, e)
            continue

        entries.append({
            "title": title,
            "description": description,
            "url": enclosure.get("url"),
            "size": size,
            "pubDate": pub,
            "guid": guid,
            "duration": duration,
        })
    return entries


def _build_feed(show_cfg: dict[str, Any], entries: list[dict[str, Any]]) -> bytes:
    """Render the RSS XML from channel config + entry list. Newest first."""
    title = (show_cfg.get("title") or "").strip()
    description = (show_cfg.get("description") or "").strip()
    author = (show_cfg.get("author") or "").strip()
    artwork = (show_cfg.get("artwork_url") or "").strip()
    category = (show_cfg.get("category") or "News").strip()
    language = (show_cfg.get("language") or "en-us").strip()
    explicit = "yes" if show_cfg.get("explicit") else "no"

    missing = [name for name, val in
               (("title", title), ("description", description),
                ("author", author), ("artwork_url", artwork))
               if not val]
    if missing:
        raise RSSError(f"show.yaml missing required field(s): {', '.join(missing)}")

    fg = FeedGenerator()
    fg.load_extension("podcast")

    fg.title(title)
    fg.description(description)
    # Spec §12.7: <link> can be anything stable — reuse artwork URL.
    fg.link(href=artwork, rel="alternate")
    fg.language(language)
    fg.lastBuildDate(datetime.now(timezone.utc))

    fg.podcast.itunes_author(author)
    fg.podcast.itunes_image(artwork)
    fg.podcast.itunes_category(category)
    fg.podcast.itunes_explicit(explicit)

    for e in entries:
        # order='append': we already sorted entries newest-first; without this
        # feedgen prepends each, reversing the order in the rendered feed.
        fe = fg.add_entry(order="append")
        fe.title(e["title"])
        fe.description(e["description"])
        # permalink=False: guid is the episode URL but it's not meant as a web link.
        fe.guid(e["guid"], permalink=False)
        fe.pubDate(e["pubDate"])
        fe.enclosure(e["url"], str(e["size"]), ENCLOSURE_TYPE)
        if e.get("duration"):
            fe.podcast.itunes_duration(e["duration"])

    return fg.rss_str(pretty=True)


def _kept_episode_keys(entries: list[dict[str, Any]]) -> set[str]:
    """Storage keys that must remain in R2 — derived from feed entries' pubDates.

    Each entry's pubDate gives us the episode date; episode_key() gives the
    canonical storage key for that date (spec §12.2). We don't parse the
    enclosure URL because the URL format depends on the public base, while the
    key format is stable across backends."""
    keys: set[str] = set()
    for e in entries:
        pub = e.get("pubDate")
        if pub is None:
            continue
        d = pub.date() if hasattr(pub, "date") else pub
        keys.add(episode_key(d))
    return keys


def _prune_orphan_episodes(store: ObjectStore, keep_keys: set[str]) -> int:
    """Delete MP3s under EPISODE_PREFIX that aren't referenced by the feed.

    Soft-fails: a single delete error is logged but doesn't abort the run —
    orphan storage gets retried next run. Failure to list the prefix at all
    is also non-fatal: the feed has already been published, retention is a
    storage-cost concern not a correctness one."""
    try:
        all_keys = store.list_prefix(EPISODE_PREFIX)
    except ObjectStoreError as e:
        log.warning("retention: could not list %s: %s", EPISODE_PREFIX, e)
        return 0
    deleted = 0
    for key in all_keys:
        if key in keep_keys:
            continue
        try:
            store.delete(key)
            deleted += 1
        except ObjectStoreError as e:
            log.warning("retention: could not delete %s: %s", key, e)
    return deleted


def update_feed(d: date | None = None, *, store: ObjectStore | None = None) -> Path:
    """Stage 9.3: download prior feed.xml, append today's item, re-upload.

    Returns the local audit-copy path (logs/YYYY-MM-DD/09_feed.xml). The
    canonical feed lives at object-storage key `feed.xml`. Idempotent
    across same-date re-runs (today's item is deduped by guid).
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)
    if store is None:
        store = R2ObjectStore()

    today_entry = _today_entry(d, day_dir)

    try:
        prior = store.get(FEED_KEY)
    except ObjectStoreError as e:
        raise RSSError(f"failed to fetch prior {FEED_KEY}: {e}") from e

    if prior is not None:
        entries = _parse_existing_feed(prior)
        log.info("rss: parsed %d prior entries from %s", len(entries), FEED_KEY)
    else:
        entries = []
        log.info("rss: no prior feed.xml; scaffolding fresh feed")

    # Same-date re-run: drop any prior item that shares today's guid.
    entries = [e for e in entries if e["guid"] != today_entry["guid"]]
    entries.append(today_entry)

    entries.sort(key=lambda e: e["pubDate"], reverse=True)
    entries = entries[:RETENTION_COUNT]

    show_cfg = load_show()
    xml_bytes = _build_feed(show_cfg, entries)

    try:
        store.put(FEED_KEY, xml_bytes, FEED_CONTENT_TYPE)
    except ObjectStoreError as e:
        raise RSSError(f"failed to upload {FEED_KEY}: {e}") from e

    out_path = day_dir / "09_feed.xml"
    out_path.write_bytes(xml_bytes)
    log.info("rss: wrote %s (%d entries) and uploaded to %s/%s",
             out_path.name, len(entries), store.name, FEED_KEY)

    # Prune R2 to match the feed: anything under episodes/ that isn't
    # referenced by one of the surviving entries gets deleted, so listeners'
    # apps never hit a 404.
    keep_keys = _kept_episode_keys(entries)
    deleted = _prune_orphan_episodes(store, keep_keys)
    if deleted:
        log.info("retention: pruned %d orphan episode(s) from %s",
                 deleted, store.name)
    return out_path
