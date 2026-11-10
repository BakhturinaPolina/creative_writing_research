"""Sync the manifest (and override CSVs) with the on-disk raw books tree.

Use after you move or rename files by hand under data/raw/books/:

  - updates manifest `raw_path` to where each file actually is
  - if the file sits in primary/review/comparison/negative, updates `corpus_role`
    and upserts config/book_role_overrides.csv so inventory/reorganize keep it there
  - if the filename title/author differ from the current display name, upserts
    config/book_metadata_overrides.csv so reorganize does not rename it back
  - files under `_title_review/` keep their previous corpus_role until you place them

Usage:
  python -m src.fulltext.sync_raw_layout           # apply
  python -m src.fulltext.sync_raw_layout --dry-run # report only
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, ROOT
from src.fulltext.books import (
    MANIFEST_PATH,
    METADATA_OVERRIDES,
    RAW_BOOKS,
    display_title_author,
    in_title_review,
    load_metadata_overrides,
    parse_filename,
    raw_files_by_md5,
    role_from_path,
    safe_book_filename,
)

ROLE_OVERRIDES = CONFIG / "book_role_overrides.csv"
ROLE_COLS = ["book_id", "corpus_role", "exclusion_reason", "corpus_type",
             "translation_status", "original_language", "reviewer", "note"]
META_COLS = ["book_id", "title", "author", "source", "confidence", "identity_changed", "reviewer", "note"]


def _upsert_csv(path, cols, records: dict[str, dict]) -> pd.DataFrame:
    existing = pd.read_csv(path, dtype=str).fillna("") if path.is_file() else pd.DataFrame(columns=cols)
    by_id = {r["book_id"]: r for r in existing.to_dict(orient="records")}
    by_id.update(records)
    out = pd.DataFrame(list(by_id.values()), columns=cols).fillna("").sort_values("book_id")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)

    m = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    located = raw_files_by_md5()
    meta = load_metadata_overrides()
    role_updates: dict[str, dict] = {}
    meta_updates: dict[str, dict] = {}
    path_updates = role_changes = name_changes = missing = staged = 0

    rows = m.to_dict(orient="records")
    for row in rows:
        actual = located.get(row["md5"])
        if actual is None:
            if row["raw_path"]:
                log.error(f"missing: {row['book_id']} ({row['raw_path']})")
                missing += 1
            continue

        new_path = str(actual.relative_to(ROOT)).replace("\\", "/")
        if new_path != row["raw_path"]:
            log.info(f"path: {row['book_id']}: {row['raw_path']} → {new_path}")
            row["raw_path"] = new_path
            path_updates += 1

        if in_title_review(actual):
            staged += 1
            continue

        folder_role = role_from_path(actual)
        if folder_role and folder_role != row["corpus_role"]:
            log.info(f"role: {row['book_id']}: {row['corpus_role']} → {folder_role} ({actual.parent.name}/)")
            row["corpus_role"] = folder_role
            row["role_source"] = "manual:folder"
            role_updates[row["book_id"]] = {
                "book_id": row["book_id"],
                "corpus_role": folder_role,
                "exclusion_reason": "",
                "corpus_type": row.get("corpus_type", ""),
                "translation_status": row.get("translation_status", ""),
                "original_language": row.get("original_language", ""),
                "reviewer": "manual",
                "note": f"hand-placed in {actual.parent.name}/",
            }
            role_changes += 1

        # Existing metadata overrides win (they may be richer than the filename).
        # Only learn a new name from disk when there is no override yet.
        if row["book_id"] not in meta and row["book_id"] not in meta_updates:
            parsed = parse_filename(actual.name)
            cur_title, cur_author = display_title_author(row, meta)
            file_title, file_author = parsed["file_title"], parsed["file_author"]
            expected_name = safe_book_filename(cur_title, cur_author, row["md5"], row["format"])
            if file_title and actual.name != expected_name:
                log.info(f"name: {row['book_id']}: {cur_title!r} / {cur_author!r} → {file_title!r} / {file_author!r}")
                meta_updates[row["book_id"]] = {
                    "book_id": row["book_id"],
                    "title": file_title,
                    "author": file_author,
                    "source": "filename",
                    "confidence": "manual",
                    "identity_changed": str(file_title.casefold() != cur_title.casefold()),
                    "reviewer": "manual",
                    "note": "hand-renamed; synced from filename",
                }
                name_changes += 1

    if not args.dry_run:
        pd.DataFrame(rows).reindex(columns=m.columns).to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
        if role_updates:
            roles = _upsert_csv(ROLE_OVERRIDES, ROLE_COLS, role_updates)
            roles.to_csv(ROLE_OVERRIDES, index=False, encoding="utf-8")
            log.info(f"wrote {ROLE_OVERRIDES.relative_to(ROOT)} ({len(roles)} rows)")
        if meta_updates:
            metas = _upsert_csv(METADATA_OVERRIDES, META_COLS, meta_updates)
            metas.to_csv(METADATA_OVERRIDES, index=False, encoding="utf-8")
            log.info(f"wrote {METADATA_OVERRIDES.relative_to(ROOT)} ({len(metas)} rows)")

    log.info(
        f"{'dry-run ' if args.dry_run else ''}done: {path_updates} paths, {role_changes} roles, "
        f"{name_changes} names, {staged} still in {RAW_BOOKS.name}/*/ _title_review/, {missing} missing"
    )
    log.close()


if __name__ == "__main__":
    main()
