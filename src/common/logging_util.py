"""Simple console + file logging for every harvest script.

Why this exists: colleagues should see progress in the terminal and still have
a timestamped log file under logs/ when something fails.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from src.common.paths import LOGS


class RunLogger:
    """Prints timestamped lines to the terminal and appends them to a log file."""

    def __init__(self, script_name: str):
        # Make sure the logs folder exists (tracked via .gitkeep).
        LOGS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Safe file name: strip path bits if someone passes __file__.
        safe = Path(script_name).stem.replace(" ", "_")
        self.path = LOGS / f"{safe}_{stamp}.log"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.info(f"Log started for {safe} → {self.path}")

    def _stamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message: str) -> None:
        """Write one progress line to console and log file."""
        line = f"[{self._stamp()}] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def error(self, message: str) -> None:
        """Same as info, but also write to stderr for visibility."""
        line = f"[{self._stamp()}] ERROR: {message}"
        print(line, file=sys.stderr, flush=True)
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the log file handle."""
        self.info("Log finished.")
        self._fh.close()


def setup_logger(script_name: str) -> RunLogger:
    """Create a RunLogger for the calling script (pass __file__ or a short name)."""
    return RunLogger(script_name)
