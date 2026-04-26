"""Loads YAML configs (show, models, voice, pronunciations) and validates schemas."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dan.paths import ROOT

CONFIG_DIR = ROOT / "config"


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def models() -> dict[str, str]:
    """Return per-stage model assignments from config/models.yaml."""
    data = _load(CONFIG_DIR / "models.yaml") or {}
    if not isinstance(data, dict):
        raise ValueError(f"models.yaml: expected mapping, got {type(data).__name__}")
    return {str(k): str(v) for k, v in data.items()}


def show() -> dict[str, Any]:
    """Show-level RSS metadata from config/show.yaml."""
    return _load(CONFIG_DIR / "show.yaml") or {}


def voice() -> dict[str, Any]:
    """TTS voice config from config/voice.yaml."""
    return _load(CONFIG_DIR / "voice.yaml") or {}


def pronunciations() -> list[dict[str, str]]:
    """Pronunciation override list from config/pronunciations.yaml."""
    data = _load(CONFIG_DIR / "pronunciations.yaml") or {}
    overrides = data.get("overrides", []) or []
    if not isinstance(overrides, list):
        raise ValueError("pronunciations.yaml: 'overrides' must be a list")
    return overrides
