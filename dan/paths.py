"""logs/YYYY-MM-DD/ helpers — the on-disk contract between stages."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = ROOT / "logs"


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def log_dir(d: date | None = None) -> Path:
    """Return logs/YYYY-MM-DD/ for `d` (default: today UTC), creating it if missing."""
    if d is None:
        d = today_utc()
    p = LOGS_ROOT / d.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p
