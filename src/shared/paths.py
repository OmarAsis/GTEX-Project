"""Shared project paths used by the pipeline, application, and verifiers."""

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
DATA = PROJECT_ROOT / "data"
CONFIG = SRC_ROOT / "config"
PROCESSED = SRC_ROOT / "processed"
