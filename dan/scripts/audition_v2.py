"""v2 audition focused on reducing the 'AI-like' quality of Davis chat.

Tests three things against each other:
  A: Davis + chat at styledegree=0.9 (intensity dialed back)
  B: A + mid-clause <break>s (humans pause at clause boundaries, robots don't)
  C: Davis DragonHDLatestNeural (LLM-backed HD voice, no express-as)

Run: `python -m dan.scripts.audition_v2`. MP3s go to
`logs/_auditions/YYYY-MM-DD_v2/`.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from dan.audio.tts import AzureTTSProvider, TTSError
from dan.paths import LOGS_ROOT, ROOT, today_utc

load_dotenv(ROOT / ".env", override=False)

log = logging.getLogger(__name__)

# Same content as v1 (cold open, hard news, transition with one emphasis,
# sign-off) so v1 and v2 MP3s are directly comparable.
BODY_PLAIN = """\
Good morning. This is your daily film news brief for Sunday, April twenty-sixth, twenty twenty-six. Here's what's playing today.

<break time="500ms"/>

The film world is mourning the loss of Nathalie Baye, one of France's most accomplished actors. Baye died on Friday evening at her home in Paris. She was seventy-seven years old.

<break time="250ms"/>

In other film news, a debate is playing out in Hollywood about the role of <emphasis level="moderate">artificial intelligence</emphasis> in filmmaking. Some of the industry's most respected names are embracing the technology.

<break time="500ms"/>

That's your film news for today. We'll see you tomorrow.
"""

# Same script but with mid-clause breaks at natural breath/clause boundaries.
# Targeting 80-120ms per the production guide — long enough to be heard,
# short enough not to feel staged.
BODY_WITH_BREAKS = """\
Good morning. <break time="120ms"/> This is your daily film news brief for Sunday, April twenty-sixth, <break time="100ms"/> twenty twenty-six. Here's what's playing today.

<break time="500ms"/>

The film world is mourning the loss of Nathalie Baye, <break time="100ms"/> one of France's most accomplished actors. Baye died on Friday evening <break time="100ms"/> at her home in Paris. She was seventy-seven years old.

<break time="250ms"/>

In other film news, <break time="120ms"/> a debate is playing out in Hollywood <break time="100ms"/> about the role of <emphasis level="moderate">artificial intelligence</emphasis> in filmmaking. Some of the industry's most respected names <break time="100ms"/> are embracing the technology.

<break time="500ms"/>

That's your film news for today. <break time="200ms"/> We'll see you tomorrow.
"""


@dataclass(frozen=True)
class Candidate:
    slug: str
    voice: str
    style: str | None
    styledegree: str | None
    body: str


CANDIDATES: list[Candidate] = [
    Candidate(
        slug="A__davis__chat__sd09",
        voice="en-US-DavisNeural",
        style="chat",
        styledegree="0.9",
        body=BODY_PLAIN,
    ),
    Candidate(
        slug="B__davis__chat__sd09__breaks",
        voice="en-US-DavisNeural",
        style="chat",
        styledegree="0.9",
        body=BODY_WITH_BREAKS,
    ),
    Candidate(
        slug="C__davis_dragonhd",
        voice="en-US-Davis:DragonHDLatestNeural",
        style=None,            # HD voices don't accept express-as
        styledegree=None,
        body=BODY_PLAIN,
    ),
]


def _build_ssml(c: Candidate) -> str:
    """Wrap the body in <speak> + <voice> [+ <express-as>] per candidate."""
    inner = c.body
    if c.style is not None:
        sd = f' styledegree="{c.styledegree}"' if c.styledegree else ""
        inner = (
            f'    <mstts:express-as style="{c.style}"{sd}>\n'
            f'      {c.body}\n'
            f'    </mstts:express-as>\n'
        )
    else:
        inner = f'    {c.body}\n'
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0"\n'
        '       xmlns="http://www.w3.org/2001/10/synthesis"\n'
        '       xmlns:mstts="http://www.w3.org/2001/mstts"\n'
        '       xml:lang="en-US">\n'
        f'  <voice name="{c.voice}">\n'
        f'{inner}'
        '  </voice>\n'
        '</speak>'
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = LOGS_ROOT / "_auditions" / f"{today_utc().isoformat()}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = AzureTTSProvider()
    log.info("audition v2: provider=%s, %d candidate(s)", provider.name, len(CANDIDATES))

    results: list[tuple[Candidate, Path | None, str | None]] = []
    for c in CANDIDATES:
        out_path = out_dir / f"{c.slug}.mp3"
        ssml = _build_ssml(c)
        log.info("audition v2: rendering %s", c.slug)
        try:
            audio = provider.synthesize(ssml)
        except TTSError as e:
            log.warning("audition v2: %s FAILED: %s", c.slug, e)
            results.append((c, None, str(e)))
            continue
        out_path.write_bytes(audio)
        log.info("audition v2: wrote %s (%d bytes)", out_path.name, len(audio))
        results.append((c, out_path, None))

    print()
    print(f"v2 auditions in {out_dir}:")
    for c, p, err in results:
        if p:
            print(f"  OK   {c.slug}.mp3  voice={c.voice} style={c.style} sd={c.styledegree}")
        else:
            print(f"  FAIL {c.slug}  ({err})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
