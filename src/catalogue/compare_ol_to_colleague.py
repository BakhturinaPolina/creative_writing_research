"""Compare a new Open Library harvest to colleague lists and seed recall.

Reports:
- seed recall (base title before colon + author surname) for EN and DE
- share of new rows that have ISBN
- overlap of OLIDs with filtered_openlibrary lists
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.candidates.migrate_candidates import (
    author_surname,
    base_title_norm,
    extract_ol,
)
from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_RAW

SEED_PATH = DATA_INTERIM / "catalogue" / "seed_candidates.csv"
FILTERED_EN = DATA_INTERIM / "filtered_openlibrary" / "filtered_english_corp.csv"
FILTERED_DE = DATA_INTERIM / "filtered_openlibrary" / "filtered_german_corp.csv"
OL_INTERIM = DATA_INTERIM / "openlibrary"


def _cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def filtered_olids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, dtype=str, sep=";", encoding="utf-8").fillna("")
    ids = set()
    for link in frame.get("link", []):
        work, edition, _ = extract_ol(link)
        oid = work or edition
        if oid:
            ids.add(oid)
    return ids


def seed_recall(ol_frame: pd.DataFrame, seed_frame: pd.DataFrame, lang: str) -> tuple[int, int, list[str]]:
    """Match seed titles against the new OL harvest; return hits, total, miss ids."""
    index = {}
    for record in ol_frame.to_dict(orient="records"):
        key = (base_title_norm(record.get("title")), author_surname(record.get("author")))
        if key[0] and key[1] and key not in index:
            index[key] = _cell(record.get("ol_work_id") or record.get("ol_edition_id"))

    seeds = seed_frame[seed_frame["edition_language"] == lang]
    hits, misses = 0, []
    for record in seeds.to_dict(orient="records"):
        key = (base_title_norm(record.get("title")), author_surname(record.get("author")))
        if key[0] and key[1] and key in index:
            hits += 1
        else:
            misses.append(_cell(record.get("candidate_id")))
    return hits, len(seeds), misses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--en",
        type=Path,
        default=OL_INTERIM / "ol_search_en.csv",
        help="Path to new English OL harvest CSV",
    )
    parser.add_argument(
        "--de",
        type=Path,
        default=OL_INTERIM / "ol_search_de.csv",
        help="Path to new German OL harvest CSV",
    )
    args = parser.parse_args()

    log = setup_logger(__file__)
    try:
        if not SEED_PATH.is_file():
            raise SystemExit("Need seed_candidates.csv — run migrate_candidates first.")
        seed = pd.read_csv(SEED_PATH, dtype=str).fillna("")

        for lang, path, filtered_path in (
            ("en", args.en, FILTERED_EN),
            ("de", args.de, FILTERED_DE),
        ):
            log.info(f"=== {lang.upper()} ===")
            if not path.is_file():
                log.info(f"missing {path} — skip")
                continue
            frame = pd.read_csv(path, dtype=str).fillna("")
            log.info(f"rows: {len(frame)}")

            has_isbn = (
                (frame.get("isbn_10", pd.Series(dtype=str)).fillna("").str.strip() != "")
                | (frame.get("isbn_13", pd.Series(dtype=str)).fillna("").str.strip() != "")
            )
            n_isbn = int(has_isbn.sum()) if len(frame) else 0
            pct = (100.0 * n_isbn / len(frame)) if len(frame) else 0.0
            log.info(f"rows with ISBN: {n_isbn}/{len(frame)} ({pct:.1f}%)")

            hits, total, misses = seed_recall(frame, seed, lang)
            recall = hits / total if total else 0.0
            log.info(f"seed recall: {hits}/{total} = {recall:.3f}")
            log.info(f"seed misses (first 15): {', '.join(misses[:15])}")

            new_ids = set()
            for record in frame.to_dict(orient="records"):
                oid = _cell(record.get("ol_work_id")) or _cell(record.get("ol_edition_id"))
                if oid:
                    new_ids.add(oid)
            old_ids = filtered_olids(filtered_path)
            overlap = new_ids & old_ids
            log.info(
                f"OLID overlap with filtered list: {len(overlap)} "
                f"(new={len(new_ids)}, filtered={len(old_ids)})"
            )

        # Colleague raw sizes for context
        en_raw = DATA_RAW / "colleague_openlibrary" / "creative writing books scrabing_open library_english.csv"
        de_raw = DATA_RAW / "colleague_openlibrary" / "kreatives_schreiben_bücher_open library deutsch.csv"
        log.info(f"colleague raw paths exist: en={en_raw.is_file()}, de={de_raw.is_file()}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
