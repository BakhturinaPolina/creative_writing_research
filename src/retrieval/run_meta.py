"""Shared run metadata helpers for retrieval baselines."""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common.paths import CONFIG, DATA_PROCESSED, ROOT

DEFAULT_CORPUS_SNAPSHOT_ID = "corpus_snapshot_analysis_v1_candidate_20261122"

# Config / code paths fingerprinted when the working tree is dirty.
_FINGERPRINT_PATHS = [
    CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml",
    CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml",
    CONFIG / "regex_concept_map_v03.yml",
    CONFIG / "regex_concept_map_v01.yml",
    ROOT / "src" / "retrieval",
]


def git_commit(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse"]
        args.append("--short" if short else "HEAD")
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def git_working_tree_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    return h.hexdigest()


def _hash_path(path: Path, h) -> None:
    if path.is_file():
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and "__pycache__" not in child.parts:
                h.update(str(child.relative_to(ROOT)).encode())
                h.update(child.read_bytes())


def code_config_fingerprint() -> str:
    """SHA-256 over retrieval code + lexicon/regex configs (dirty-tree attribution)."""
    h = hashlib.sha256()
    for p in _FINGERPRINT_PATHS:
        if p.exists():
            _hash_path(p, h)
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


def corpus_denominators() -> dict:
    out: dict = {}
    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"{lang}_book_stats.csv"
        if not path.is_file():
            out[lang] = {"available": False}
            continue
        try:
            import pandas as pd

            df = pd.read_csv(path, dtype=str).fillna("")
            df["in_analysis_set"] = (
                df["in_analysis_set"].astype(str).str.lower().isin(["true", "1"])
            )
            analysis = df[df["in_analysis_set"]]
            body = 0.0
            if "body_words" in analysis.columns:
                body = float(
                    pd.to_numeric(analysis["body_words"], errors="coerce").fillna(0).sum()
                )
            out[lang] = {
                "available": True,
                "n_books_analysis": int(len(analysis)),
                "n_books_total": int(len(df)),
                "body_words": body,
                "book_stats_sha256": file_sha256(path),
            }
        except Exception as exc:
            out[lang] = {"available": False, "error": str(exc)}
    return out


def base_run_meta(**extra) -> dict:
    dirty = git_working_tree_dirty()
    commit = git_commit(short=False)
    commit_short = git_commit(short=True)
    fingerprint = code_config_fingerprint()
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "git_commit_short": commit_short,
        "git_working_tree_dirty": dirty,
        "code_config_fingerprint": fingerprint,
        # When dirty, do not imply the recorded commit alone produced the output.
        "reproducibility_note": (
            "Working tree dirty; attribute via code_config_fingerprint, not git_commit alone."
            if dirty
            else "Clean working tree; git_commit identifies the code revision."
        ),
        "corpus_snapshot_id": DEFAULT_CORPUS_SNAPSHOT_ID,
        "software": software_versions(),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "corpus_denominators": corpus_denominators(),
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
