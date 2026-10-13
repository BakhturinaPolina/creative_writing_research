"""Project root and shared data paths.

Every script imports ROOT from here so we do not hard-code absolute paths.
"""
from pathlib import Path

# This file lives at src/common/paths.py → three parents up is the repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"
