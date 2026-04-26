"""Top-level orchestrator + CLI. Runs stages 1-9 in sequence.

Currently wired stages: 1 (fetch). Subsequent stages will be added as they land.
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from dan.paths import ROOT
from dan.sources import guardian

# Auto-load .env for local dev. override=False so GitHub Actions secrets
# (which arrive as real env vars) always win.
load_dotenv(ROOT / ".env", override=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dan.pipeline")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    guardian.fetch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
