"""Stage 6 — Audio prep.

Spec §9: take the validated script SSML, apply pronunciation overrides,
validate against the Azure SSML schema, and chunk into TTS-sized requests.
Reads 04_script.txt + config/pronunciations.yaml + config/voice.yaml,
writes 06_ssml.xml + 06_chunks/chunk_NN.xml.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from lxml import etree

from dan import config
from dan.io import read_text, write_text
from dan.paths import log_dir, today_utc

log = logging.getLogger(__name__)

SSML_NS = "http://www.w3.org/2001/10/synthesis"
MSTTS_NS = "http://www.w3.org/2001/mstts"
XML_NS = "http://www.w3.org/XML/1998/namespace"

DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
DEFAULT_STYLE = "newscast"

# §9.4 chunking: aim for 2000-4000 chars; max around 4500 for safety.
TARGET_CHUNK_CHARS = 3500
LONG_BREAK_THRESHOLD_MS = 400  # breaks at or above this mark are inter-segment

_LONG_BREAK_RE = re.compile(r"^(\d+)\s*ms$|^(\d+(?:\.\d+)?)\s*s$")


class SsmlError(RuntimeError):
    """Stage 6 hard failure — SSML doesn't parse, doesn't match Azure shape, or chunks bad."""


def _qn(tag: str, ns: str = SSML_NS) -> str:
    return f"{{{ns}}}{tag}"


# ---------- pronunciation pass ----------

def apply_pronunciations(ssml: str, overrides: list[dict[str, str]]) -> str:
    """Apply pronunciation overrides as whole-word text substitution.

    All overrides are applied in a single regex pass with alternation,
    longest match first. This prevents the "double-substitution" trap where
    a 'Ronan' rule would re-match inside text the 'Saoirse Ronan' rule had
    already replaced — `re.sub` doesn't rescan its own output, so the
    longer rule consumes the bytes before the shorter one ever sees them.

    Word-boundary uses lookarounds (`(?<!\\w)` / `(?!\\w)`) instead of
    `\\b` so multi-word names with apostrophes or hyphens don't get
    bisected.
    """
    if not overrides:
        return ssml
    valid = [(o["match"], o["replacement"]) for o in overrides
             if o.get("match") and o.get("replacement")]
    if not valid:
        return ssml
    valid.sort(key=lambda mr: -len(mr[0]))
    repl_map = dict(valid)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(m) for m, _ in valid) + r")(?!\w)"
    )
    return pattern.sub(lambda m: repl_map[m.group(1)], ssml)


# ---------- validation ----------

def validate_ssml(ssml: str) -> etree._Element:
    """Parse SSML and check shape against Azure expectations (spec §9.3).

    Returns the root element on success. Raises SsmlError on any failure.
    """
    try:
        root = etree.fromstring(ssml.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        raise SsmlError(f"SSML is not well-formed XML: {e}") from e

    if root.tag != _qn("speak"):
        raise SsmlError(f"root element is {root.tag!r}, expected speak in SSML namespace")

    ns_uris = set(root.nsmap.values())
    if SSML_NS not in ns_uris:
        raise SsmlError(f"missing SSML namespace ({SSML_NS})")

    if root.get(_qn("lang", XML_NS)) != "en-US":
        raise SsmlError("missing or wrong xml:lang on <speak> (expected en-US)")

    voices = root.findall(_qn("voice"))
    if len(voices) != 1:
        raise SsmlError(f"expected exactly one <voice>, found {len(voices)}")
    if not voices[0].get("name"):
        raise SsmlError("<voice> missing name attribute")

    # The mstts namespace is only required if the document actually uses it
    # (e.g. <mstts:express-as>). DragonHD voices skip express-as entirely
    # and have no reason to declare mstts.
    has_mstts_element = any(
        el.tag.startswith(f"{{{MSTTS_NS}}}") for el in root.iter()
    )
    if has_mstts_element and MSTTS_NS not in ns_uris:
        raise SsmlError(f"missing mstts namespace ({MSTTS_NS}) but <mstts:*> elements present")

    return root


# ---------- chunking ----------

def _is_long_break(el: etree._Element) -> bool:
    if el.tag != _qn("break"):
        return False
    time_attr = (el.get("time") or "").strip()
    m = _LONG_BREAK_RE.match(time_attr)
    if not m:
        return False
    if m.group(1) is not None:  # ms
        return int(m.group(1)) >= LONG_BREAK_THRESHOLD_MS
    return float(m.group(2)) >= LONG_BREAK_THRESHOLD_MS / 1000.0


def _escape_xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_NS_DECL_RE = re.compile(
    r'\s+xmlns(?::\w+)?="(?:'
    + re.escape(SSML_NS) + r'|' + re.escape(MSTTS_NS)
    + r')"'
)


def _serialize_inline(element: etree._Element) -> str:
    """Render an inline element only — never its tail.

    `etree.tostring` includes the element's `.tail` text by default, so without
    `with_tail=False` we'd emit the tail here AND again when the caller
    explicitly appends `child.tail` — duplicating the spoken paragraph.
    Strip our SSML/mstts namespace declarations too; the chunk root declares
    them.
    """
    s = etree.tostring(element, encoding="unicode", with_tail=False)
    return _NS_DECL_RE.sub("", s)


def _split_segments(express_as: etree._Element) -> list[str]:
    """Walk the express-as body and split at long-break boundaries.

    Returns a list of segment inner-XML strings (no surrounding tags).
    Short breaks (<400ms) and other in-line elements stay inside their
    segment.
    """
    segments: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        s = "".join(buf).strip()
        if s:
            segments.append(s)
        buf.clear()

    if express_as.text and express_as.text.strip():
        buf.append(_escape_xml_text(express_as.text))

    for child in express_as:
        if _is_long_break(child):
            flush()
        else:
            buf.append(_serialize_inline(child))
        if child.tail and child.tail.strip():
            buf.append(_escape_xml_text(child.tail))

    flush()
    return segments


def _group_into_chunks(segments: list[str], target_chars: int = TARGET_CHUNK_CHARS) -> list[list[str]]:
    """Greedy: pack segments into the current chunk until adding the next
    would exceed target_chars. A single oversized segment becomes its own
    chunk by itself."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for seg in segments:
        seg_size = len(seg)
        if current and current_size + seg_size > target_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(seg)
        current_size += seg_size
    if current:
        chunks.append(current)
    return chunks


def _build_chunk_doc(segments: list[str], voice_name: str, style: str | None) -> str:
    """Wrap a list of segment inner-XML strings into a complete <speak> document.

    When `style` is None the chunk has no `<mstts:express-as>` wrapper and the
    body sits directly inside `<voice>` — required for DragonHD voices, which
    silently fall back to neutral if express-as is present.
    """
    body = '\n      <break time="500ms"/>\n      '.join(segments)
    if style is not None:
        inner = (
            f'    <mstts:express-as style="{style}">\n'
            f'      {body}\n'
            f'    </mstts:express-as>\n'
        )
        ns_decl = '\n       xmlns:mstts="http://www.w3.org/2001/mstts"'
    else:
        inner = f'    {body}\n'
        ns_decl = ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0"\n'
        '       xmlns="http://www.w3.org/2001/10/synthesis"'
        f'{ns_decl}\n'
        '       xml:lang="en-US">\n'
        f'  <voice name="{voice_name}">\n'
        f'{inner}'
        '  </voice>\n'
        '</speak>'
    )


def chunk_ssml(
    ssml: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    voice_name: str | None = None,
) -> list[str]:
    """Split a validated SSML doc into one or more <speak> chunks.

    Splits at long-break boundaries inside the body element — either
    `<mstts:express-as>` (legacy / styled-voice path) or `<voice>` itself
    (DragonHD path; HD voices reject express-as and infer prosody from
    text). Each returned chunk is a complete, independently valid SSML
    document that preserves the input's express-as wrapper iff present.

    If `voice_name` is given, it overrides the voice in the source SSML —
    Azure's TTS honors the SSML <voice> tag over any SDK-level config, so
    the chunks must carry the voice we actually want to synthesize.
    """
    root = validate_ssml(ssml)
    voice_el = root.find(_qn("voice"))
    chunk_voice = voice_name if voice_name is not None else voice_el.get("name", DEFAULT_VOICE)
    express_as = voice_el.find(_qn("express-as", MSTTS_NS))
    if express_as is not None:
        body_el = express_as
        style = express_as.get("style", DEFAULT_STYLE)
    else:
        body_el = voice_el
        style = None

    segments = _split_segments(body_el)
    if not segments:
        raise SsmlError("voice body produced zero segments")

    grouped = _group_into_chunks(segments, target_chars=target_chars)
    return [_build_chunk_doc(group, chunk_voice, style) for group in grouped]


def _set_voice_in_ssml(ssml: str, voice_name: str) -> str:
    """Rewrite the `name` attribute on the single <voice> element."""
    root = etree.fromstring(ssml.encode("utf-8"))
    voice_el = root.find(_qn("voice"))
    if voice_el is None:
        raise SsmlError("no <voice> element to retag")
    voice_el.set("name", voice_name)
    body = etree.tostring(root, encoding="unicode")
    if not body.lstrip().startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body


# ---------- pipeline entry ----------

def prep(d: date | None = None) -> Path:
    """Prepare 04_script.txt for TTS: pronunciations, validate, chunk.

    Returns the path to 06_ssml.xml (the full prepped SSML doc). The chunks
    are written alongside in 06_chunks/.
    """
    if d is None:
        d = today_utc()
    day_dir = log_dir(d)

    script = read_text(day_dir / "04_script.txt")
    overrides = config.pronunciations()
    voice_cfg = config.voice() or {}
    voice_name = voice_cfg.get("voice", DEFAULT_VOICE)

    log.info("prep: applying %d pronunciation override(s)", len(overrides))
    prepped = apply_pronunciations(script, overrides)

    log.info("prep: validating SSML")
    validate_ssml(prepped)

    # Apply the configured voice to both the full prepped doc and the chunks.
    # Azure's TTS honors the SSML <voice> tag over any SDK config, so changing
    # voice.yaml only takes effect if we actually rewrite the SSML here.
    log.info("prep: setting voice to %s", voice_name)
    prepped = _set_voice_in_ssml(prepped, voice_name)

    ssml_path = day_dir / "06_ssml.xml"
    write_text(ssml_path, prepped)

    chunks_dir = day_dir / "06_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale chunks from a prior run so chunk numbering stays canonical.
    for old in chunks_dir.glob("chunk_*.xml"):
        old.unlink()

    chunks = chunk_ssml(prepped, voice_name=voice_name)
    for i, chunk_xml in enumerate(chunks, start=1):
        validate_ssml(chunk_xml)  # belt + suspenders: each chunk must independently parse
        chunk_path = chunks_dir / f"chunk_{i:02d}.xml"
        write_text(chunk_path, chunk_xml)

    sizes = [len(c) for c in chunks]
    log.info(
        "prep: wrote 06_ssml.xml (%d chars) and %d chunk(s) sized %s",
        len(prepped), len(chunks), sizes,
    )
    return ssml_path
