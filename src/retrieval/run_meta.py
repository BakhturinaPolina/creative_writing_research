"""Shared run metadata helpers for retrieval baselines."""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.common.paths import DATA_PROCESSED, ROOT


def git_commit(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse"]
        args.append("--short" if short else "HEAD")
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    return h.hexdigest()


def peak_rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def software_versions() -> dict:
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("pandas", "yaml", "numpy"):
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            vers[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[pkg] = "missing"
    try:
        import rank_bm25  # noqa: F401

        vers["rank_bm25"] = getattr(rank_bm25, "__version__", "present")
    except Exception:
        vers["rank_bm25"] = "not_installed"
    return vers


def base_run_meta(**extra) -> dict:
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": git_commit(short=False),
        "git_commit_short": git_commit(short=True),
        "software": software_versions(),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "book_stats_hashes": {
            "en": file_sha256(DATA_PROCESSED / "en_book_stats.csv"),
            "de": file_sha256(DATA_PROCESSED / "de_book_stats.csv"),
        },
    }
    meta.update(extra)
    return meta


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
