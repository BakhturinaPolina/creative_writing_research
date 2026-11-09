"""Rename and sort raw books into browseable role folders.

Layout:
  data/raw/books/{lang}/{primary|review|comparison|negative}/{Title} - {Author} [{md5}].{ext}

`unreviewed` → review/. Filenames keep the full MD5 so inventory can still parse them.
Updates manifest.csv `raw_path`. Idempotent; verifies MD5 after each move.

Usage:
  python -m src.fulltext.reorganize_books           # apply
  python -m src.fulltext.reorganize_books --dry-run # print planned moves only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.fulltext.books import MANIFEST_PATH, RAW_BOOKS, md5_of, raw_book_path, role_folder


def display_title_author(row: pd.Series) -> tuple[str, str]:
    title = str(row.get("title") or row.get("file_title") or "").strip()
    author = str(row.get("author_name") or row.get("file_author") or "").strip()
    if title in {"", "nan"}:
        title = str(row.get("file_title") or "untitled").strip() or "untitled"
    if author in {"", "nan"}:
        author = str(row.get("file_author") or "").strip()
    return title, author


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print planned moves only")
    args = parser.parse_args()

    log = setup_logger(__file__)
    m = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    if m.empty:
        raise SystemExit(f"empty manifest: {MANIFEST_PATH}")

    moved = skipped = failed = 0
    new_paths: list[str] = []

    for _, row in m.iterrows():
        src = ROOT / row["raw_path"] if row["raw_path"] else None
        title, author = display_title_author(row)
        dest = raw_book_path(row["lang"], row["corpus_role"], title, author, row["md5"], row["format"])
        new_paths.append(str(dest.relative_to(ROOT)).replace("\\", "/"))

        if src is None or not src.is_file():
            log.error(f"missing source for {row['book_id']}: {row['raw_path']}")
            failed += 1
            continue
        if src.resolve() == dest.resolve():
            skipped += 1
            continue
        if dest.exists():
            log.error(f"destination exists, skip {row['book_id']}: {dest.name}")
            failed += 1
            continue

        log.info(f"{role_folder(row['corpus_role'])}: {src.name} → {dest.relative_to(RAW_BOOKS)}")
        if args.dry_run:
            moved += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        if md5_of(dest) != row["md5"]:
            dest.rename(src)
            log.error(f"hash mismatch, restored: {row['book_id']}")
            failed += 1
            continue
        moved += 1

        # Drop empty role-less parent dirs left behind (e.g. books/en/)
        parent = src.parent
        while parent != RAW_BOOKS and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    m["raw_path"] = new_paths
    if not args.dry_run:
        m.to_csv(MANIFEST_PATH, index=False)
        # Remove empty language leftovers that only held hash-named files
        for lang_dir in RAW_BOOKS.iterdir() if RAW_BOOKS.exists() else []:
            if lang_dir.is_dir() and not any(lang_dir.rglob("*")):
                lang_dir.rmdir()

    log.info(
        f"{'dry-run ' if args.dry_run else ''}done: {moved} moved, {skipped} already ok, {failed} failed; "
        f"manifest {'unchanged' if args.dry_run else 'updated'}"
    )


if __name__ == "__main__":
    main()
