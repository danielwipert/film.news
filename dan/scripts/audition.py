"""Voice/style audition: synthesize the same representative body through
multiple voice/style combinations so a human can A/B-listen.

Run: `python -m dan.scripts.audition` (auto-loads .env). MP3s are written to
`logs/_auditions/YYYY-MM-DD/<voice>__<style>__sd<N>.mp3`.

The SSML <voice> tag overrides any SDK-level voice, so a single
AzureTTSProvider handles all candidates — we just vary the SSML body.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from dan.audio.tts import AzureTTSProvider
from dan.paths import LOGS_ROOT, ROOT, today_utc

load_dotenv(ROOT / ".env", override=False)

log = logging.getLogger(__name__)

# Representative body covering the registers a real episode hits: cold open,
# somber hard news, signpost into a different topic, sign-off. Keeps each
# audition ~30-40s so A/B listening stays focused.
AUDITION_BODY = """\
Good morning. This is your daily film news brief for Sunday, April twenty-sixth, twenty twenty-six. Here's what's playing today.

<break time="500ms"/>

The film world is mourning the loss of Nathalie Baye, one of France's most accomplished actors. Baye died on Friday evening at her home in Paris. She was seventy-seven years old.

<break time="250ms"/>

In other film news, a debate is playing out in Hollywood about the role of <emphasis level="moderate">artificial intelligence</emphasis> in filmmaking. Some of the industry's most respected names are embracing the technology.

<break time="500ms"/>

That's your film news for today. We'll see you tomorrow.
"""


@dataclass(frozen=True)
class Candidate:
    voice: str
    style: str
    styledegree: str = "1.5"

    @property
    def slug(self) -> str:
        v = self.voice.removeprefix("en-US-").removesuffix("Neural")
        sd = self.styledegree.replace(".", "")
        return f"{v.lower()}__{self.style}__sd{sd}"


CANDIDATES: list[Candidate] = [
    Candidate("en-US-DavisNeural",  "friendly"),         # locked voice + working style
    Candidate("en-US-DavisNeural",  "chat"),             # same voice, warmer
    Candidate("en-US-GuyNeural",    "newscast"),         # native newsreader
    Candidate("en-US-TonyNeural",   "friendly"),         # alt male timbre
    Candidate("en-US-AriaNeural",   "newscast-formal"),  # female reference
]


def _build_ssml(c: Candidate) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0"\n'
        '       xmlns="http://www.w3.org/2001/10/synthesis"\n'
        '       xmlns:mstts="http://www.w3.org/2001/mstts"\n'
        '       xml:lang="en-US">\n'
        f'  <voice name="{c.voice}">\n'
        f'    <mstts:express-as style="{c.style}" styledegree="{c.styledegree}">\n'
        f'      {AUDITION_BODY}\n'
        '    </mstts:express-as>\n'
        '  </voice>\n'
        '</speak>'
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = LOGS_ROOT / "_auditions" / today_utc().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = AzureTTSProvider()  # voice is overridden per-call by the SSML
    log.info("audition: provider=%s, %d candidate(s)", provider.name, len(CANDIDATES))

    for c in CANDIDATES:
        out_path = out_dir / f"{c.slug}.mp3"
        ssml = _build_ssml(c)
        log.info("audition: %s + %s @ %s -> %s",
                 c.voice, c.style, c.styledegree, out_path.name)
        audio = provider.synthesize(ssml)
        out_path.write_bytes(audio)
        log.info("audition: wrote %s (%d bytes)", out_path.name, len(audio))

    print()
    print(f"Auditions written to {out_dir}:")
    for c in CANDIDATES:
        print(f"  {c.slug}.mp3  ({c.voice} + style={c.style} @ styledegree={c.styledegree})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
