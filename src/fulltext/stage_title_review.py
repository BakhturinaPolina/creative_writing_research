"""Move English books that still need a title/author check into staging folders.

Layout:
  data/raw/books/en/_title_review/{wrong_book|conflict|medium}/{Title} - {Author} [{md5}].{ext}

Books already placed by hand (not at their previous path), already overridden, or
missing on disk are skipped. After you rename a file and move it into
primary/review/comparison/negative, run:

  python -m src.fulltext.sync_raw_layout

Usage:
  python -m src.fulltext.stage_title_review           # apply
  python -m src.fulltext.stage_title_review --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.fulltext.books import (
    FULLTEXT_DIR,
    MANIFEST_PATH,
    RAW_BOOKS,
    TITLE_REVIEW_CONFIDENCE,
    TITLE_REVIEW_DIR,
    in_title_review,
    md5_of,
    raw_files_by_md5,
    role_from_path,
)

REVIEW_PATH = FULLTEXT_DIR / "title_author_review.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lang", default="en", help="language folder to stage (default: en)")
    args = parser.parse_args()
    log = setup_logger(__file__)

    review = pd.read_csv(REVIEW_PATH, dtype=str).fillna("")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("").set_index("book_id", drop=False)
    located = raw_files_by_md5()

    moved = already = left = missing = 0
    for row in review.to_dict(orient="records"):
        confidence = row["confidence"]
        if confidence not in TITLE_REVIEW_CONFIDENCE:
            continue
        if row["book_id"] not in manifest.index:
            log.error(f"not in manifest: {row['book_id']}")
            missing += 1
            continue
        mrow = manifest.loc[row["book_id"]]
        if isinstance(mrow, pd.DataFrame):
            mrow = mrow.iloc[0]
        if mrow["lang"] != args.lang:
            continue

        src = located.get(mrow["md5"])
        if src is None:
            log.error(f"missing file: {row['book_id']}")
            missing += 1
            continue
        if in_title_review(src):
            already += 1
            continue
        # Leave books you already sorted into comparison/negative where they are.
        folder_role = role_from_path(src)
        if folder_role in {"comparison", "negative"}:
            log.info(f"leave in {src.parent.name}/: {row['book_id']}")
            left += 1
            continue

        dest_dir = RAW_BOOKS / args.lang / TITLE_REVIEW_DIR / confidence
        dest = dest_dir / src.name
        if src.resolve() == dest.resolve():
            already += 1
            continue
        if dest.exists():
            log.error(f"destination exists, skip {row['book_id']}: {dest.name}")
            missing += 1
            continue

        log.info(f"{confidence}: {src.relative_to(RAW_BOOKS)} → {dest.relative_to(RAW_BOOKS)}")
        if args.dry_run:
            moved += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        if md5_of(dest) != mrow["md5"]:
            dest.rename(src)
            log.error(f"hash mismatch, restored: {row['book_id']}")
            missing += 1
            continue
        moved += 1
        # Drop empty parents left behind
        parent = src.parent
        while parent != RAW_BOOKS and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

        # Keep review CSV and manifest pointing at the new location
        review.loc[review["book_id"] == row["book_id"], "raw_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")
        manifest.at[row["book_id"], "raw_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")

    if not args.dry_run:
        review.to_csv(REVIEW_PATH, index=False, encoding="utf-8")
        manifest.reset_index(drop=True).to_csv(MANIFEST_PATH, index=False, encoding="utf-8")

    log.info(
        f"{'dry-run ' if args.dry_run else ''}done: {moved} {'to move' if args.dry_run else 'moved'}, "
        f"{already} already in _title_review, {left} left in comparison/negative, {missing} missing/failed"
    )
    log.info(f"after renaming, put each file in {args.lang}/primary|review|comparison|negative/ "
             f"then run: python -m src.fulltext.sync_raw_layout")
    log.close()


if __name__ == "__main__":
    main()
