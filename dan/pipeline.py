"""Top-level orchestrator + CLI. Runs stages 1-9 in sequence.

Currently wired stages: 1 (fetch), 2 (rank). Subsequent stages added as they land.
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from dan.llm import rank, sanity, summarize, write as write_stage
from dan.paths import ROOT
from dan.sources import guardian

# Auto-load .env for local dev. override=False so GitHub Actions secrets
# (which arrive as real env vars) always win.
load_dotenv(ROOT / ".env", override=False)

STAGES = ("fetch", "rank", "summarize", "write", "sanity")


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
    else:
        guardian.fetch(days_back=args.days_back)
        rank.rank()
        summarize.summarize()
        write_stage.write()
        sanity.sanity()
    return 0


if __name__ == "__main__":
    sys.exit(main())
