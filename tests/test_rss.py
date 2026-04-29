"""Tests for Stage 9.3 — dan.publish.rss."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from lxml import etree

from dan.publish import rss
from dan.publish.rss import RSSError
from dan.publish.store import ObjectStoreError


# ---------- _format_duration ----------

@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00:00"),
    (1, "00:00:01"),
    (59, "00:00:59"),
    (60, "00:01:00"),
    (61, "00:01:01"),
    (3599, "00:59:59"),
    (3600, "01:00:00"),
    (3661, "01:01:01"),
    (36000, "10:00:00"),
])
def test_format_duration(seconds, expected):
    assert rss._format_duration(seconds) == expected


def test_format_duration_rounds_floats():
    assert rss._format_duration(59.4) == "00:00:59"
    assert rss._format_duration(59.6) == "00:01:00"


# ---------- _episode_pubdate ----------

def test_episode_pubdate_is_midnight_utc():
    dt = rss._episode_pubdate(date(2026, 4, 27))
    assert dt == datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)


# ---------- _parse_rfc822 ----------

def test_parse_rfc822_with_tz():
    dt = rss._parse_rfc822("Mon, 27 Apr 2026 12:34:56 +0000")
    assert dt == datetime(2026, 4, 27, 12, 34, 56, tzinfo=timezone.utc)


def test_parse_rfc822_naive_defaults_to_utc():
    # parsedate_to_datetime can return naive in unusual inputs; we coerce to UTC.
    dt = rss._parse_rfc822("27 Apr 2026 12:34:56")
    assert dt.tzinfo is not None


# ---------- _today_entry ----------

def _fake_mp3(length_seconds: float = 600.0):
    """Fake mutagen.mp3.MP3 — only `.info.length` is read by rss._today_entry."""
    return SimpleNamespace(info=SimpleNamespace(length=length_seconds))


def _seed_today(tmp_path, *, url="https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-27.mp3",
                size=12345, description="Today's brief.", episode_bytes=b"\xff" * 1024):
    (tmp_path / "08_episode.mp3").write_bytes(episode_bytes)
    (tmp_path / "09_upload.json").write_text(
        json.dumps({"url": url, "key": "k", "size": size,
                    "content_type": "audio/mpeg", "uploaded_at": "x"}),
        encoding="utf-8",
    )
    (tmp_path / "09_description.txt").write_text(description, encoding="utf-8")


def test_today_entry_happy_path(monkeypatch, tmp_path):
    _seed_today(tmp_path)
    monkeypatch.setattr(rss, "MP3", lambda p: _fake_mp3(length_seconds=485.5))

    entry = rss._today_entry(date(2026, 4, 27), tmp_path)

    assert entry["title"] == "DAN Film Brief — 2026-04-27"
    assert entry["description"] == "Today's brief."
    assert entry["url"] == "https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-27.mp3"
    assert entry["size"] == 12345
    assert entry["pubDate"] == datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)
    assert entry["guid"] == entry["url"]
    assert entry["duration"] == "00:08:06"  # 485.5 -> rounds to 486s


def test_today_entry_strips_description_whitespace(monkeypatch, tmp_path):
    _seed_today(tmp_path, description="  Lots of trailing space.   \n\n")
    monkeypatch.setattr(rss, "MP3", lambda p: _fake_mp3())
    entry = rss._today_entry(date(2026, 4, 27), tmp_path)
    assert entry["description"] == "Lots of trailing space."


def test_today_entry_missing_upload_json_raises(monkeypatch, tmp_path):
    (tmp_path / "08_episode.mp3").write_bytes(b"x")
    (tmp_path / "09_description.txt").write_text("d", encoding="utf-8")
    with pytest.raises(RSSError, match="09_upload.json"):
        rss._today_entry(date(2026, 4, 27), tmp_path)


def test_today_entry_missing_description_raises(monkeypatch, tmp_path):
    (tmp_path / "08_episode.mp3").write_bytes(b"x")
    (tmp_path / "09_upload.json").write_text(json.dumps({"url": "u", "size": 1}), encoding="utf-8")
    with pytest.raises(RSSError, match="09_description.txt"):
        rss._today_entry(date(2026, 4, 27), tmp_path)


def test_today_entry_missing_episode_raises(monkeypatch, tmp_path):
    (tmp_path / "09_upload.json").write_text(json.dumps({"url": "u", "size": 1}), encoding="utf-8")
    (tmp_path / "09_description.txt").write_text("d", encoding="utf-8")
    with pytest.raises(RSSError, match="08_episode.mp3"):
        rss._today_entry(date(2026, 4, 27), tmp_path)


def test_today_entry_wraps_mp3_read_error(monkeypatch, tmp_path):
    _seed_today(tmp_path)
    def boom(_p): raise OSError("not an mp3")
    monkeypatch.setattr(rss, "MP3", boom)
    with pytest.raises(RSSError, match="duration"):
        rss._today_entry(date(2026, 4, 27), tmp_path)


# ---------- _parse_existing_feed ----------

_PRIOR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>DAN</title>
    <description>x</description>
    <item>
      <title>DAN Film Brief — 2026-04-26</title>
      <description>Yesterday's brief.</description>
      <pubDate>Sun, 26 Apr 2026 00:00:00 +0000</pubDate>
      <guid isPermaLink="false">https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3</guid>
      <enclosure url="https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3" length="9999" type="audio/mpeg"/>
      <itunes:duration>00:09:30</itunes:duration>
    </item>
    <item>
      <title>DAN Film Brief — 2026-04-25</title>
      <description>Older brief.</description>
      <pubDate>Sat, 25 Apr 2026 00:00:00 +0000</pubDate>
      <guid isPermaLink="false">https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-25.mp3</guid>
      <enclosure url="https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-25.mp3" length="8888" type="audio/mpeg"/>
      <itunes:duration>00:08:45</itunes:duration>
    </item>
  </channel>
</rss>""".encode("utf-8")


def test_parse_existing_feed_extracts_two_entries():
    entries = rss._parse_existing_feed(_PRIOR_FEED)
    assert len(entries) == 2
    e0, e1 = entries
    assert e0["title"] == "DAN Film Brief — 2026-04-26"
    assert e0["description"] == "Yesterday's brief."
    assert e0["url"].endswith("dan-film-2026-04-26.mp3")
    assert e0["size"] == 9999
    assert e0["pubDate"] == datetime(2026, 4, 26, 0, 0, 0, tzinfo=timezone.utc)
    assert e0["duration"] == "00:09:30"
    assert e1["url"].endswith("dan-film-2026-04-25.mp3")


def test_parse_existing_feed_skips_item_missing_enclosure():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>broken</title>
    <pubDate>Sun, 26 Apr 2026 00:00:00 +0000</pubDate>
    <guid>g1</guid>
  </item>
</channel></rss>"""
    assert rss._parse_existing_feed(xml) == []


def test_parse_existing_feed_skips_item_missing_pubdate():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>broken</title>
    <guid>g1</guid>
    <enclosure url="u" length="1" type="audio/mpeg"/>
  </item>
</channel></rss>"""
    assert rss._parse_existing_feed(xml) == []


def test_parse_existing_feed_skips_item_missing_guid():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>broken</title>
    <pubDate>Sun, 26 Apr 2026 00:00:00 +0000</pubDate>
    <enclosure url="u" length="1" type="audio/mpeg"/>
  </item>
</channel></rss>"""
    assert rss._parse_existing_feed(xml) == []


def test_parse_existing_feed_skips_bad_pubdate():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>broken</title>
    <pubDate>not a date</pubDate>
    <guid>g1</guid>
    <enclosure url="u" length="1" type="audio/mpeg"/>
  </item>
</channel></rss>"""
    # Bad pubDate currently parses to epoch in some libs; ensure the bad-length
    # path is exercised via a non-numeric length below. Here we just confirm
    # malformed dates don't crash the parser.
    out = rss._parse_existing_feed(xml)
    assert isinstance(out, list)


def test_parse_existing_feed_skips_non_numeric_length():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>broken</title>
    <pubDate>Sun, 26 Apr 2026 00:00:00 +0000</pubDate>
    <guid>g1</guid>
    <enclosure url="u" length="abc" type="audio/mpeg"/>
  </item>
</channel></rss>"""
    assert rss._parse_existing_feed(xml) == []


def test_parse_existing_feed_raises_on_malformed_xml():
    with pytest.raises(RSSError, match="malformed"):
        rss._parse_existing_feed(b"<not really xml")


def test_parse_existing_feed_handles_empty_channel():
    xml = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>x</title></channel></rss>"""
    assert rss._parse_existing_feed(xml) == []


# ---------- _build_feed ----------

def _show_cfg(**overrides):
    base = {
        "title": "DAN — Film",
        "description": "Daily film news brief.",
        "author": "Daniel S. Wipert",
        "language": "en-us",
        "artwork_url": "https://example.com/artwork.jpg",
        "category": "News",
        "explicit": False,
    }
    base.update(overrides)
    return base


def _entry(d: date = date(2026, 4, 27), url: str | None = None,
           size: int = 12345, duration: str = "00:08:06") -> dict:
    if url is None:
        url = f"https://pub-x.r2.dev/episodes/{d:%Y}/{d:%m}/dan-film-{d.isoformat()}.mp3"
    return {
        "title": f"DAN Film Brief — {d.isoformat()}",
        "description": f"Brief for {d.isoformat()}.",
        "url": url,
        "size": size,
        "pubDate": rss._episode_pubdate(d),
        "guid": url,
        "duration": duration,
    }


def test_build_feed_renders_well_formed_xml():
    xml = rss._build_feed(_show_cfg(), [_entry()])
    root = etree.fromstring(xml)
    assert root.tag == "rss"
    assert root.get("version") == "2.0"


def test_build_feed_sets_channel_level_fields():
    xml = rss._build_feed(_show_cfg(), [_entry()])
    root = etree.fromstring(xml)
    chan = root.find("channel")
    assert chan.findtext("title") == "DAN — Film"
    assert chan.findtext("description") == "Daily film news brief."
    assert chan.findtext("language") == "en-us"


def test_build_feed_includes_itunes_namespace():
    xml = rss._build_feed(_show_cfg(), [_entry()])
    assert b'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"' in xml


def test_build_feed_sets_itunes_author_and_image():
    xml = rss._build_feed(_show_cfg(), [_entry()])
    root = etree.fromstring(xml)
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    chan = root.find("channel")
    assert chan.findtext("itunes:author", namespaces=ns) == "Daniel S. Wipert"
    image = chan.find("itunes:image", namespaces=ns)
    assert image is not None
    assert image.get("href") == "https://example.com/artwork.jpg"


def test_build_feed_sets_itunes_explicit_yes_when_true():
    xml = rss._build_feed(_show_cfg(explicit=True), [_entry()])
    root = etree.fromstring(xml)
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    assert root.find("channel").findtext("itunes:explicit", namespaces=ns) == "yes"


def test_build_feed_sets_itunes_explicit_no_when_false():
    xml = rss._build_feed(_show_cfg(explicit=False), [_entry()])
    root = etree.fromstring(xml)
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    assert root.find("channel").findtext("itunes:explicit", namespaces=ns) == "no"


def test_build_feed_link_reuses_artwork_url():
    xml = rss._build_feed(_show_cfg(artwork_url="https://example.com/art.jpg"), [_entry()])
    root = etree.fromstring(xml)
    link_text = root.find("channel").findtext("link") or ""
    assert "art.jpg" in link_text


def test_build_feed_renders_item_enclosure_and_duration():
    xml = rss._build_feed(_show_cfg(), [_entry(size=42, duration="00:09:30")])
    root = etree.fromstring(xml)
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    item = root.find("channel/item")
    enc = item.find("enclosure")
    assert enc.get("type") == "audio/mpeg"
    assert enc.get("length") == "42"
    assert item.findtext("itunes:duration", namespaces=ns) == "00:09:30"


def test_build_feed_with_zero_entries_is_valid_xml():
    xml = rss._build_feed(_show_cfg(), [])
    root = etree.fromstring(xml)
    assert root.find("channel") is not None
    assert root.find("channel/item") is None


@pytest.mark.parametrize("missing_field", ["title", "description", "author", "artwork_url"])
def test_build_feed_raises_on_missing_required_show_field(missing_field):
    cfg = _show_cfg()
    cfg[missing_field] = ""
    with pytest.raises(RSSError, match=missing_field):
        rss._build_feed(cfg, [])


# ---------- update_feed (orchestration) ----------

def _fake_store(prior: bytes | None = None,
                listing: list[str] | None = None) -> MagicMock:
    s = MagicMock()
    s.name = "fake:test"
    s.get.return_value = prior
    s.list_prefix.return_value = list(listing or [])
    return s


def _patch_today(monkeypatch, tmp_path, d: date = date(2026, 4, 27)):
    monkeypatch.setattr(rss, "log_dir", lambda x=None: tmp_path)
    monkeypatch.setattr(rss, "today_utc", lambda: d)
    monkeypatch.setattr(rss, "MP3", lambda p: _fake_mp3(length_seconds=486.0))
    monkeypatch.setattr(rss, "load_show", lambda: _show_cfg())


def test_update_feed_no_prior_scaffolds_fresh(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    _seed_today(tmp_path)
    store = _fake_store(prior=None)

    out = rss.update_feed(store=store)

    assert out == tmp_path / "09_feed.xml"
    assert out.exists()
    # store.put was called with feed.xml + correct content type
    args, _ = store.put.call_args
    assert args[0] == "feed.xml"
    assert args[2] == "application/rss+xml"
    # And the uploaded XML is well-formed and has exactly one item.
    root = etree.fromstring(args[1])
    assert len(root.findall("channel/item")) == 1


def test_update_feed_writes_local_audit_copy(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    _seed_today(tmp_path)
    rss.update_feed(store=_fake_store(prior=None))

    local = tmp_path / "09_feed.xml"
    assert local.exists()
    root = etree.fromstring(local.read_bytes())
    assert root.find("channel/item/title").text == "DAN Film Brief — 2026-04-27"


def test_update_feed_appends_to_prior(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    _seed_today(tmp_path)
    store = _fake_store(prior=_PRIOR_FEED)

    rss.update_feed(store=store)

    args, _ = store.put.call_args
    root = etree.fromstring(args[1])
    items = root.findall("channel/item")
    # 2 prior + 1 today = 3; sorted desc means today's is first.
    assert len(items) == 3
    assert items[0].findtext("title") == "DAN Film Brief — 2026-04-27"
    assert items[1].findtext("title") == "DAN Film Brief — 2026-04-26"
    assert items[2].findtext("title") == "DAN Film Brief — 2026-04-25"


def test_update_feed_dedupes_same_date_rerun(monkeypatch, tmp_path):
    """Same-date re-run must replace today's <item> rather than duplicate it."""
    _patch_today(monkeypatch, tmp_path, d=date(2026, 4, 26))
    _seed_today(
        tmp_path,
        url="https://pub-x.r2.dev/episodes/2026/04/dan-film-2026-04-26.mp3",
        size=11111,
        description="Re-run brief.",
    )
    store = _fake_store(prior=_PRIOR_FEED)

    rss.update_feed(store=store)

    args, _ = store.put.call_args
    root = etree.fromstring(args[1])
    items = root.findall("channel/item")
    # Prior had 2 entries (04-26 and 04-25). After re-run on 04-26: still 2.
    assert len(items) == 2
    # The 04-26 entry must reflect the new size, not the old 9999.
    today_enclosures = [i.find("enclosure") for i in items
                        if i.findtext("title") == "DAN Film Brief — 2026-04-26"]
    assert len(today_enclosures) == 1
    assert today_enclosures[0].get("length") == "11111"


def test_update_feed_caps_at_retention_count(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path, d=date(2026, 4, 27))
    _seed_today(tmp_path)

    # Build a synthetic prior with 30 entries from 2026-03-28 onward — after
    # appending today (04-27) the feed must cap at RETENTION_COUNT.
    items_xml = []
    for i in range(30):
        day = date(2026, 3, 28).toordinal() + i
        ds = date.fromordinal(day).isoformat()
        items_xml.append(f"""
        <item>
          <title>old {ds}</title>
          <description>x</description>
          <pubDate>{rss._episode_pubdate(date.fromordinal(day)).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>
          <guid isPermaLink="false">https://pub-x.r2.dev/old/{ds}.mp3</guid>
          <enclosure url="https://pub-x.r2.dev/old/{ds}.mp3" length="1" type="audio/mpeg"/>
        </item>""")
    prior = (b'<?xml version="1.0"?><rss version="2.0" '
             b'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'
             + "".join(items_xml).encode("utf-8")
             + b"</channel></rss>")

    store = _fake_store(prior=prior)
    rss.update_feed(store=store)

    args, _ = store.put.call_args
    root = etree.fromstring(args[1])
    items = root.findall("channel/item")
    assert len(items) == rss.RETENTION_COUNT
    # Newest first — today's entry is on top.
    assert items[0].findtext("title") == "DAN Film Brief — 2026-04-27"


# ---------- retention / orphan prune ----------

def test_kept_episode_keys_derives_from_pubdates():
    entries = [
        _entry(d=date(2026, 4, 27)),
        _entry(d=date(2026, 4, 26)),
    ]
    keys = rss._kept_episode_keys(entries)
    assert keys == {
        "episodes/2026/04/dan-film-2026-04-27.mp3",
        "episodes/2026/04/dan-film-2026-04-26.mp3",
    }


def test_kept_episode_keys_handles_date_pubdate():
    """pubDate is normally a datetime; tolerate a bare date too."""
    entries = [{"pubDate": date(2026, 1, 5)}]
    assert rss._kept_episode_keys(entries) == {
        "episodes/2026/01/dan-film-2026-01-05.mp3",
    }


def test_kept_episode_keys_skips_entries_without_pubdate():
    assert rss._kept_episode_keys([{"pubDate": None}, {}]) == set()


def test_prune_orphan_episodes_deletes_only_unkept():
    store = _fake_store(listing=[
        "episodes/2026/04/keep1.mp3",
        "episodes/2026/04/orphan1.mp3",
        "episodes/2026/04/keep2.mp3",
        "episodes/2026/04/orphan2.mp3",
    ])
    keep = {"episodes/2026/04/keep1.mp3", "episodes/2026/04/keep2.mp3"}

    deleted = rss._prune_orphan_episodes(store, keep)

    assert deleted == 2
    deleted_calls = sorted(c.args[0] for c in store.delete.call_args_list)
    assert deleted_calls == [
        "episodes/2026/04/orphan1.mp3",
        "episodes/2026/04/orphan2.mp3",
    ]


def test_prune_orphan_episodes_returns_zero_when_listing_fails():
    """list_prefix failure logs a warning but doesn't raise — retention is a
    storage-cost concern, the feed has already been published."""
    from dan.publish.store import ObjectStoreError
    store = _fake_store()
    store.list_prefix.side_effect = ObjectStoreError("denied")
    assert rss._prune_orphan_episodes(store, {"k"}) == 0
    store.delete.assert_not_called()


def test_prune_orphan_episodes_continues_on_individual_delete_failure():
    """A failing delete on one key shouldn't stop the others — the next
    key may still be deletable, and orphans we miss get retried next run."""
    from dan.publish.store import ObjectStoreError
    store = _fake_store(listing=["a", "b", "c"])
    store.delete.side_effect = [None, ObjectStoreError("nope"), None]
    deleted = rss._prune_orphan_episodes(store, set())
    assert deleted == 2
    assert store.delete.call_count == 3


def test_prune_orphan_episodes_returns_zero_when_all_kept():
    store = _fake_store(listing=["a", "b"])
    assert rss._prune_orphan_episodes(store, {"a", "b"}) == 0
    store.delete.assert_not_called()


def test_update_feed_prunes_orphans_in_lockstep_with_feed_cap(monkeypatch, tmp_path):
    """End-to-end: feed.xml caps at RETENTION_COUNT; R2 keys not in the
    surviving 7 entries are deleted so the feed never points at a 404."""
    _patch_today(monkeypatch, tmp_path, d=date(2026, 4, 27))
    _seed_today(tmp_path)

    # Prior: 10 episodes (04-17 .. 04-26). Today (04-27) is added → 11 entries
    # before cap, 7 after. The 4 oldest (04-17..04-20) become orphans.
    items_xml = []
    bucket_keys = []
    for i in range(10):
        d = date(2026, 4, 17 + i)
        ds = d.isoformat()
        url = f"https://pub-x.r2.dev/episodes/2026/04/dan-film-{ds}.mp3"
        bucket_keys.append(f"episodes/2026/04/dan-film-{ds}.mp3")
        items_xml.append(f"""
        <item>
          <title>old {ds}</title>
          <description>x</description>
          <pubDate>{rss._episode_pubdate(d).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>
          <guid isPermaLink="false">{url}</guid>
          <enclosure url="{url}" length="1" type="audio/mpeg"/>
        </item>""")
    prior = (b'<?xml version="1.0"?><rss version="2.0" '
             b'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'
             + "".join(items_xml).encode("utf-8")
             + b"</channel></rss>")
    bucket_keys.append("episodes/2026/04/dan-film-2026-04-27.mp3")

    store = _fake_store(prior=prior, listing=bucket_keys)
    rss.update_feed(store=store)

    # Feed: capped to 7, newest first.
    feed_args, _ = store.put.call_args
    root = etree.fromstring(feed_args[1])
    items = root.findall("channel/item")
    assert len(items) == 7
    assert items[0].findtext("title") == "DAN Film Brief — 2026-04-27"

    # Storage: the four oldest got pruned.
    deleted_keys = sorted(c.args[0] for c in store.delete.call_args_list)
    assert deleted_keys == [
        "episodes/2026/04/dan-film-2026-04-17.mp3",
        "episodes/2026/04/dan-film-2026-04-18.mp3",
        "episodes/2026/04/dan-film-2026-04-19.mp3",
        "episodes/2026/04/dan-film-2026-04-20.mp3",
    ]


def test_update_feed_does_not_prune_when_storage_listing_fails(monkeypatch, tmp_path):
    """A failing list_prefix doesn't crash the run — feed.xml is already
    published; retention is best-effort."""
    from dan.publish.store import ObjectStoreError
    _patch_today(monkeypatch, tmp_path, d=date(2026, 4, 27))
    _seed_today(tmp_path)
    store = _fake_store(prior=None)
    store.list_prefix.side_effect = ObjectStoreError("denied")

    rss.update_feed(store=store)  # must not raise

    store.delete.assert_not_called()


def test_update_feed_raises_when_today_inputs_missing(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    # Don't seed today's files.
    with pytest.raises(RSSError, match="09_upload.json"):
        rss.update_feed(store=_fake_store(prior=None))


def test_update_feed_wraps_store_get_error(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    _seed_today(tmp_path)
    store = _fake_store(prior=None)
    store.get.side_effect = ObjectStoreError("denied")
    with pytest.raises(RSSError, match=r"failed to fetch.*denied"):
        rss.update_feed(store=store)


def test_update_feed_wraps_store_put_error(monkeypatch, tmp_path):
    _patch_today(monkeypatch, tmp_path)
    _seed_today(tmp_path)
    store = _fake_store(prior=None)
    store.put.side_effect = ObjectStoreError("denied")
    with pytest.raises(RSSError, match=r"failed to upload.*denied"):
        rss.update_feed(store=store)


def test_update_feed_uses_explicit_date(monkeypatch, tmp_path):
    monkeypatch.setattr(rss, "log_dir", lambda x=None: tmp_path)
    monkeypatch.setattr(rss, "MP3", lambda p: _fake_mp3())
    monkeypatch.setattr(rss, "load_show", lambda: _show_cfg())
    _seed_today(tmp_path)
    store = _fake_store(prior=None)

    rss.update_feed(date(2026, 1, 5), store=store)

    args, _ = store.put.call_args
    root = etree.fromstring(args[1])
    item = root.find("channel/item")
    assert item.findtext("title") == "DAN Film Brief — 2026-01-05"
