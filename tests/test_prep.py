"""Tests for Stage 6 — dan.audio.prep."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from lxml import etree

from dan.audio import prep


SSML_NS = "http://www.w3.org/2001/10/synthesis"
MSTTS_NS = "http://www.w3.org/2001/mstts"


def _wrap(body: str, voice: str = "en-US-AndrewMultilingualNeural") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'  <voice name="{voice}">\n'
        '    <mstts:express-as style="newscast">\n'
        f'      {body}\n'
        '    </mstts:express-as>\n'
        '  </voice>\n'
        '</speak>'
    )


# ---------- apply_pronunciations ----------

def test_apply_pronunciations_no_overrides_is_noop():
    s = _wrap("Hello Villeneuve world.")
    assert prep.apply_pronunciations(s, []) == s
    assert prep.apply_pronunciations(s, None or []) == s


def test_apply_pronunciations_simple_sub():
    overrides = [{"match": "Villeneuve",
                  "replacement": '<sub alias="vil-NUHV">Villeneuve</sub>'}]
    out = prep.apply_pronunciations("Director Villeneuve announced.", overrides)
    assert '<sub alias="vil-NUHV">Villeneuve</sub>' in out


def test_apply_pronunciations_longest_match_first():
    """Longer matches must apply before shorter ones to avoid partial replacement."""
    overrides = [
        {"match": "Ronan", "replacement": "<sub alias='RONE-an'>Ronan</sub>"},
        {"match": "Saoirse Ronan",
         "replacement": "<sub alias='SUR-shuh RONE-an'>Saoirse Ronan</sub>"},
    ]
    out = prep.apply_pronunciations("Saoirse Ronan won an Oscar.", overrides)
    assert "SUR-shuh RONE-an" in out
    # The bare-Ronan replacement must NOT also have fired inside the longer alias
    # (its replacement <sub> contains "Ronan" but the longer rule already consumed it).
    assert out.count("<sub") == 1


def test_apply_pronunciations_whole_word_only():
    """A match for 'Bay' should not trigger inside 'Bayfield' or 'Baye'."""
    overrides = [{"match": "Bay", "replacement": "<sub alias='BAY'>Bay</sub>"}]
    out = prep.apply_pronunciations("Bayfield is in Baye, near Bay City.", overrides)
    assert "Bayfield" in out  # untouched
    assert "Baye" in out      # untouched
    assert "<sub alias='BAY'>Bay</sub> City" in out


def test_apply_pronunciations_skips_blank_overrides():
    overrides = [{"match": "", "replacement": "x"}, {"match": "x", "replacement": ""}]
    out = prep.apply_pronunciations("hello", overrides)
    assert out == "hello"


# ---------- validate_ssml ----------

def test_validate_ssml_happy():
    root = prep.validate_ssml(_wrap("Good morning."))
    assert root.tag.endswith("}speak")


def test_validate_ssml_rejects_malformed_xml():
    with pytest.raises(prep.SsmlError, match="not well-formed"):
        prep.validate_ssml("<speak><voice></speak>")


def test_validate_ssml_rejects_wrong_root():
    with pytest.raises(prep.SsmlError, match="root element"):
        prep.validate_ssml('<?xml version="1.0"?><other xmlns="x"/>')


def test_validate_ssml_accepts_no_mstts_when_no_mstts_elements():
    """DragonHD path: no express-as means we don't need the mstts namespace."""
    ok = (
        '<?xml version="1.0"?>'
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<voice name="en-US-Davis:DragonHDLatestNeural">Hello.</voice>'
        '</speak>'
    )
    root = prep.validate_ssml(ok)
    assert root is not None


def test_validate_ssml_rejects_mstts_element_without_namespace():
    """An mstts:* element with no namespace declaration is malformed."""
    bad = (
        '<?xml version="1.0"?>'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="en-US">'
        '<voice name="x"><mstts:express-as style="newscast">Hi.</mstts:express-as></voice>'
        '</speak>'
    )
    # Sanity check: the well-formed mstts version validates.
    root = prep.validate_ssml(bad)
    assert root is not None


def test_validate_ssml_rejects_missing_voice_name():
    bad = (
        '<?xml version="1.0"?>'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="en-US"><voice/></speak>'
    )
    with pytest.raises(prep.SsmlError, match="name"):
        prep.validate_ssml(bad)


def test_validate_ssml_rejects_multiple_voices():
    bad = _wrap("hi").replace("</voice>", "</voice><voice name='other'/>", 1)
    with pytest.raises(prep.SsmlError, match="exactly one"):
        prep.validate_ssml(bad)


# ---------- _is_long_break ----------

def test_long_break_thresholds():
    def br(time):
        return etree.fromstring(f'<break xmlns="{SSML_NS}" time="{time}"/>')
    assert prep._is_long_break(br("500ms")) is True
    assert prep._is_long_break(br("400ms")) is True
    assert prep._is_long_break(br("250ms")) is False
    assert prep._is_long_break(br("1s")) is True
    assert prep._is_long_break(br("0.3s")) is False


# ---------- chunk_ssml ----------

def test_chunk_ssml_single_chunk_when_small():
    body = (
        "Good morning. <break time='500ms'/> "
        "Top story today. <break time='250ms'/> Some details. "
        "<break time='500ms'/> Sign off."
    )
    chunks = prep.chunk_ssml(_wrap(body), target_chars=10000)
    assert len(chunks) == 1
    prep.validate_ssml(chunks[0])  # well-formed


def test_chunk_ssml_splits_only_at_long_breaks():
    body = (
        "Story one with detail.<break time='250ms'/>more detail."
        "<break time='500ms'/>Story two."
        "<break time='500ms'/>Story three."
    )
    chunks = prep.chunk_ssml(_wrap(body), target_chars=20)  # tiny target -> force splits
    assert len(chunks) >= 2
    for c in chunks:
        # Each chunk independently validates
        prep.validate_ssml(c)
    # Short break (250ms) must NOT have produced a split — text on either side
    # of it should still appear together in some chunk.
    full = " ".join(chunks)
    assert "Story one with detail." in full
    assert "more detail." in full


def test_chunk_ssml_preserves_text_across_chunks():
    body = (
        "Alpha alpha alpha." "<break time='500ms'/>"
        "Bravo bravo bravo." "<break time='500ms'/>"
        "Charlie charlie charlie."
    )
    chunks = prep.chunk_ssml(_wrap(body), target_chars=20)
    full = " ".join(chunks)
    for token in ("Alpha", "Bravo", "Charlie"):
        assert token in full


def test_chunk_ssml_each_chunk_has_voice_and_style():
    body = "One.<break time='500ms'/>Two.<break time='500ms'/>Three."
    chunks = prep.chunk_ssml(_wrap(body, voice="en-US-Test"), target_chars=6)
    assert len(chunks) >= 2
    for c in chunks:
        root = etree.fromstring(c.encode("utf-8"))
        voice = root.find(prep._qn("voice"))
        assert voice.get("name") == "en-US-Test"
        ea = voice.find(prep._qn("express-as", MSTTS_NS))
        assert ea.get("style") == "newscast"


def test_chunk_ssml_handles_inline_emphasis_inside_segment():
    body = (
        'Top story <emphasis level="moderate">major</emphasis> reveal.'
        '<break time="500ms"/>'
        'Second story.'
    )
    chunks = prep.chunk_ssml(_wrap(body), target_chars=10000)
    assert len(chunks) == 1
    assert "<emphasis" in chunks[0]
    prep.validate_ssml(chunks[0])


def _wrap_no_style(body: str, voice: str = "en-US-Davis:DragonHDLatestNeural") -> str:
    """Wrap body in a <speak><voice>...</voice></speak> with no express-as.
    This is the shape DragonHD voices require — express-as is rejected."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">\n'
        f'  <voice name="{voice}">\n'
        f'    {body}\n'
        '  </voice>\n'
        '</speak>'
    )


def test_chunk_ssml_handles_no_express_as():
    """DragonHD path: SSML with no <mstts:express-as> chunks fine and the
    chunks themselves come back without an express-as wrapper."""
    body = (
        "Story one.<break time='500ms'/>"
        "Story two.<break time='500ms'/>"
        "Story three."
    )
    chunks = prep.chunk_ssml(_wrap_no_style(body), target_chars=10)
    assert len(chunks) >= 2
    for c in chunks:
        root = etree.fromstring(c.encode("utf-8"))
        # Voice present, but NO express-as wrapper.
        voice = root.find(prep._qn("voice"))
        assert voice is not None
        assert voice.find(prep._qn("express-as", MSTTS_NS)) is None
        prep.validate_ssml(c)


def test_chunk_ssml_no_style_chunks_omit_mstts_namespace():
    """Cleanliness: chunks built without express-as shouldn't declare an
    unused mstts namespace."""
    body = "One.<break time='500ms'/>Two."
    chunks = prep.chunk_ssml(_wrap_no_style(body), target_chars=2)
    for c in chunks:
        assert "xmlns:mstts" not in c


def test_chunk_ssml_does_not_duplicate_text_around_short_breaks():
    """Regression: etree.tostring(child) includes the child's tail by default.
    Without with_tail=False, the tail text would be appended twice and the
    spoken paragraph after a short break would repeat."""
    body = (
        "Alpha paragraph one."
        '<break time="250ms"/>'
        "Bravo paragraph two."
        '<break time="500ms"/>'
        "Charlie next story."
    )
    chunks = prep.chunk_ssml(_wrap(body), target_chars=10000)
    joined = "\n".join(chunks)
    assert joined.count("Alpha paragraph one.") == 1
    assert joined.count("Bravo paragraph two.") == 1
    assert joined.count("Charlie next story.") == 1


# ---------- prep() pipeline entry ----------

def test_prep_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(prep.config, "pronunciations", lambda: [])
    monkeypatch.setattr(prep.config, "voice", lambda: {"voice": "en-US-Test"})

    body = "One." + "<break time='500ms'/>" + "Two." + "<break time='500ms'/>" + "Three."
    (tmp_path / "04_script.txt").write_text(_wrap(body, voice="en-US-Test"), encoding="utf-8")

    out_path = prep.prep()
    assert out_path.name == "06_ssml.xml"
    assert out_path.exists()

    chunks_dir = tmp_path / "06_chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.xml"))
    assert len(chunk_files) >= 1
    for cf in chunk_files:
        prep.validate_ssml(cf.read_text(encoding="utf-8"))


def test_prep_applies_pronunciations(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(prep.config, "pronunciations", lambda: [
        {"match": "Villeneuve", "replacement": '<sub alias="vil-NUHV">Villeneuve</sub>'}
    ])
    monkeypatch.setattr(prep.config, "voice", lambda: {"voice": "en-US-Test"})

    body = "Director Villeneuve announced today.<break time='500ms'/>End."
    (tmp_path / "04_script.txt").write_text(_wrap(body, voice="en-US-Test"), encoding="utf-8")

    out_path = prep.prep()
    content = out_path.read_text(encoding="utf-8")
    assert 'alias="vil-NUHV"' in content


def test_prep_clears_stale_chunks_from_prior_run(monkeypatch, tmp_path):
    """Stage 6 owns the chunks dir; previous-run files must not pollute the new run."""
    monkeypatch.setattr(prep, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(prep.config, "pronunciations", lambda: [])
    monkeypatch.setattr(prep.config, "voice", lambda: {"voice": "v"})
    chunks_dir = tmp_path / "06_chunks"
    chunks_dir.mkdir()
    stale = chunks_dir / "chunk_99.xml"
    stale.write_text("stale", encoding="utf-8")

    body = "Just one segment."
    (tmp_path / "04_script.txt").write_text(_wrap(body, voice="v"), encoding="utf-8")

    prep.prep()
    assert not stale.exists()
    assert (chunks_dir / "chunk_01.xml").exists()


def test_prep_overrides_voice_from_config(monkeypatch, tmp_path):
    """Regression: voice.yaml must override the voice baked into the source
    SSML, otherwise Azure honors the SSML <voice> tag and ignores the SDK
    config — meaning a voice change in voice.yaml has no effect until we
    rewrite the chunks."""
    monkeypatch.setattr(prep, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(prep.config, "pronunciations", lambda: [])
    monkeypatch.setattr(prep.config, "voice", lambda: {"voice": "en-US-DavisNeural"})

    body = "One.<break time='500ms'/>Two."
    # Source has Andrew baked in (e.g. from when the writer ran)
    src = _wrap(body, voice="en-US-AndrewMultilingualNeural")
    (tmp_path / "04_script.txt").write_text(src, encoding="utf-8")

    out_path = prep.prep()
    full = out_path.read_text(encoding="utf-8")
    assert 'name="en-US-DavisNeural"' in full
    assert "Andrew" not in full

    for chunk_file in sorted((tmp_path / "06_chunks").glob("chunk_*.xml")):
        c = chunk_file.read_text(encoding="utf-8")
        assert 'name="en-US-DavisNeural"' in c
        assert "Andrew" not in c


def test_chunk_ssml_voice_name_override():
    body = "One.<break time='500ms'/>Two."
    src = _wrap(body, voice="en-US-AndrewMultilingualNeural")
    chunks = prep.chunk_ssml(src, voice_name="en-US-DavisNeural")
    for c in chunks:
        root = etree.fromstring(c.encode("utf-8"))
        voice = root.find(prep._qn("voice"))
        assert voice.get("name") == "en-US-DavisNeural"


def test_prep_raises_on_invalid_ssml_input(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, "log_dir", lambda d=None: tmp_path)
    monkeypatch.setattr(prep.config, "pronunciations", lambda: [])
    monkeypatch.setattr(prep.config, "voice", lambda: {"voice": "v"})
    (tmp_path / "04_script.txt").write_text("<not-ssml/>", encoding="utf-8")
    with pytest.raises(prep.SsmlError):
        prep.prep()
