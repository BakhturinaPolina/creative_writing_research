"""Post-extract cleaner: residual hyphens + running headers on fulltext JSON.

Applies the shared rules in cleaning.py to every paragraph in
data/interim/fulltext/json/{lang}/*.json, records extraction.cleaning stats,
and updates n_words / chapter n_words. Does not change the manifest.

Usage:
  python -m src.fulltext.clean_fulltext --dry-run
  python -m src.fulltext.clean_fulltext --lang de
  python -m src.fulltext.clean_fulltext --lang all
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.fulltext.books import FULLTEXT_DIR, JSON_DIR, MANIFEST_PATH, word_count
from src.fulltext.cleaning import (
    DEFAULT_MAX_HEADER_WORDS,
    DEFAULT_MIN_REPEATS,
    RULES_VERSION,
    clean_paragraph,
)

STATS_PATH = FULLTEXT_DIR / "cleaning_stats.csv"


def clean_record(
    record: dict,
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_words: int = DEFAULT_MAX_HEADER_WORDS,
) -> tuple[dict, dict]:
    """Return (cleaned_record, book-level stats)."""
    from src.fulltext.cleaning import (
        paragraph_key,
        rejoin_cross_paragraph_hyphens,
        repeated_short_keys,
    )

    chapters_in = record.get("chapters") or []
    # Per chapter: paragraph clean, then cross-paragraph hyphen merges (ebook block splits).
    staged: list[dict] = []
    flat: list[str] = []
    hyphen_joins = 0
    cross_joins = 0
    for chapter in chapters_in:
        cleaned_paras: list[str] = []
        for p in chapter.get("paragraphs") or []:
            text, n = clean_paragraph(p)
            hyphen_joins += n
            if text:
                cleaned_paras.append(text)
        cleaned_paras, n_cross = rejoin_cross_paragraph_hyphens(cleaned_paras)
        cross_joins += n_cross
        staged.append({**chapter, "paragraphs": cleaned_paras})
        flat.extend(cleaned_paras)

    drop = repeated_short_keys(flat, min_repeats=min_repeats, max_words=max_words)
    headers_removed = 0
    chapters: list[dict] = []
    for chapter in staged:
        if drop:
            paras = []
            for p in chapter["paragraphs"]:
                if paragraph_key(p) in drop:
                    headers_removed += 1
                else:
                    paras.append(p)
        else:
            paras = list(chapter["paragraphs"])
        chapter = {**chapter, "paragraphs": paras, "n_words": sum(word_count(p) for p in paras)}
        chapters.append(chapter)

    n_words = sum(c["n_words"] for c in chapters)
    total_hyphen = hyphen_joins + cross_joins
    extraction = dict(record.get("extraction") or {})
    extraction["cleaning"] = {
        "rules_version": RULES_VERSION,
        "date": date.today().isoformat(),
        "hyphen_joins": total_hyphen,
        "hyphen_joins_within": hyphen_joins,
        "hyphen_joins_cross_paragraph": cross_joins,
        "headers_removed": headers_removed,
        "min_repeats": min_repeats,
        "max_header_words": max_words,
    }
    out = {**record, "chapters": chapters, "n_words": n_words, "extraction": extraction}
    stats = {
        "book_id": record.get("book_id", ""),
        "lang": record.get("lang", ""),
        "hyphen_joins": total_hyphen,
        "hyphen_joins_within": hyphen_joins,
        "hyphen_joins_cross_paragraph": cross_joins,
        "headers_removed": headers_removed,
        "n_words_before": int(record.get("n_words") or 0),
        "n_words_after": n_words,
        "rules_version": RULES_VERSION,
    }
    return out, stats


def clean_file(path: Path, *, dry_run: bool, min_repeats: int, max_words: int) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    cleaned, stats = clean_record(record, min_repeats=min_repeats, max_words=max_words)
    if not dry_run:
        path.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "other", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="compute stats only; do not write JSON")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-repeats", type=int, default=DEFAULT_MIN_REPEATS)
    parser.add_argument("--max-header-words", type=int, default=DEFAULT_MAX_HEADER_WORDS)
    parser.add_argument("--force", action="store_true",
                        help="re-clean books that already have cleaning.rules_version")
    args = parser.parse_args()
    log = setup_logger(__file__)

    langs = ["en", "de", "other"] if args.lang == "all" else [args.lang]
    paths: list[Path] = []
    for lang in langs:
        folder = JSON_DIR / lang
        if folder.is_dir():
            paths.extend(sorted(folder.glob("*.json")))
    if args.limit:
        paths = paths[: args.limit]

    results = []
    skipped = 0
    for i, path in enumerate(paths, 1):
        if not args.force and not args.dry_run:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                ver = ((existing.get("extraction") or {}).get("cleaning") or {}).get("rules_version")
                if ver == RULES_VERSION:
                    skipped += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        stats = clean_file(
            path, dry_run=args.dry_run,
            min_repeats=args.min_repeats, max_words=args.max_header_words,
        )
        results.append(stats)
        if i % 50 == 0 or i == len(paths):
            log.info(f"  {i}/{len(paths)} (cleaned {len(results)}, skipped {skipped})")

    df = pd.DataFrame(results)
    if not df.empty:
        if STATS_PATH.is_file() and not args.force and not args.dry_run:
            old = pd.read_csv(STATS_PATH, dtype={"book_id": str})
            df = pd.concat([old[~old["book_id"].isin(df["book_id"])], df], ignore_index=True)
        if not args.dry_run:
            df.to_csv(STATS_PATH, index=False, encoding="utf-8")
            log.info(f"wrote {STATS_PATH} ({len(df)} rows)")
        log.info(
            "hyphen_joins total=%s  headers_removed total=%s"
            % (int(df["hyphen_joins"].sum()), int(df["headers_removed"].sum()))
        )
        log.info("by lang:\n" + df.groupby("lang")[["hyphen_joins", "headers_removed"]].sum().to_string())
    else:
        log.info(f"nothing to clean (skipped already-clean={skipped})")

    # Optional: sync n_words for cleaned books into the manifest.
    if not args.dry_run and results and MANIFEST_PATH.is_file():
        manifest = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
        update = {r["book_id"]: str(r["n_words_after"]) for r in results}
        idx = manifest["book_id"].isin(update)
        manifest.loc[idx, "n_words"] = manifest.loc[idx, "book_id"].map(update)
        manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
        log.info(f"updated manifest n_words for {int(idx.sum())} books")

    log.close()


if __name__ == "__main__":
    main()
