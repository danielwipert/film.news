"""Top-level orchestrator + CLI. Runs stages 1-9 in sequence.

Currently wired stages: 1 (fetch), 2 (rank). Subsequent stages added as they land.
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from dan.audio import prep as prep_stage, stitch as stitch_stage, tts as tts_stage
from dan.llm import describe as describe_stage, rank, sanity, summarize, write as write_stage
from dan.paths import ROOT
from dan.publish import upload as upload_stage
from dan.sources import guardian

# Auto-load .env for local dev. override=False so GitHub Actions secrets
# (which arrive as real env vars) always win.
load_dotenv(ROOT / ".env", override=False)

STAGES = ("fetch", "rank", "summarize", "write", "sanity", "prep", "tts",
          "stitch", "describe", "upload")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dan.pipeline")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--only",
        choices=STAGES,
        help="Run a single stage instead of the full pipeline.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Fetch window in days (default 1 per spec; raise for testing on quiet days).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.only == "fetch":
        guardian.fetch(days_back=args.days_back)
    elif args.only == "rank":
        rank.rank()
    elif args.only == "summarize":
        summarize.summarize()
    elif args.only == "write":
        write_stage.write()
    elif args.only == "sanity":
        sanity.sanity()
    elif args.only == "prep":
        prep_stage.prep()
    elif args.only == "tts":
        tts_stage.synthesize_chunks()
    elif args.only == "stitch":
        stitch_stage.stitch()
    elif args.only == "describe":
        describe_stage.describe()
    elif args.only == "upload":
        upload_stage.upload()
    else:
        guardian.fetch(days_back=args.days_back)
        rank.rank()
        summarize.summarize()
        write_stage.write()
        sanity.sanity()
        prep_stage.prep()
        tts_stage.synthesize_chunks()
        stitch_stage.stitch()
        describe_stage.describe()
        upload_stage.upload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
