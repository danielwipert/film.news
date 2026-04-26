"""Top-level orchestrator + CLI. Runs stages 1-9 in sequence.

Currently wired stages: 1 (fetch). Subsequent stages will be added as they land.
"""
from __future__ import annotations

import argparse
import logging
import sys

from dan.sources import guardian


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
