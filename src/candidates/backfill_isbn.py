"""Backfill ISBN-13/10 onto final_lists CSVs from ol_search / DNB / OpenSyllabus.

Run after (or independently of) build_final_lists. Prefer rebuilding the union
with `python -m src.candidates.build_final_lists`, which already applies the
same OLID→ISBN index. This CLI rewrites existing CSVs in place when you only
need ISBN enrichment.

Sources (in order, first hit wins per field):
1. Open Library harvest: data/interim/openlibrary/ol_search_{en,de}.csv (by OL id)
2. OpenSyllabus: opensyllabus_creative_writing_all.csv (by ISBN already present, or OL work id)
3. DNB title hits: dnb_hits_tit_*.csv (by normalized title+surname → isbn)

Usage:
  python -m src.candidates.backfill_isbn
  python -m src.candidates.backfill_isbn --targets union_all,final_list_en,final_list_de
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.candidates.build_final_lists import (
    OUT_DIR,
    build_ol_isbn_index,
    _first_isbn10,
    _first_isbn13,
    _pick_isbns,
    _title_author_key,
)
from src.candidates.migrate_candidates import _cell
from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn
from src.common.paths import DATA_INTERIM

DNB_DIR = DATA_INTERIM / "dnb"
OSP_PATH = DATA_INTERIM / "opensyllabus" / "opensyllabus_creative_writing_all.csv"
PIPE_SPLIT = re.compile(r"\s*\|\s*")

DEFAULT_TARGETS = [
    "union_all.csv",
    "final_list_en.csv",
    "final_list_de.csv",
    "review_en.csv",
    "review_de.csv",
]


def build_osp_ol_index(path: Path = OSP_PATH) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not path.is_file():
        return index
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    for record in frame.to_dict(orient="records"):
        isbn_13, isbn_10 = _pick_isbns("", "", _cell(record.get("isbns")))
        if not isbn_13 and not isbn_10:
            continue
        payload = {"isbn_13": isbn_13, "isbn_10": isbn_10}
        for ol_id in PIPE_SPLIT.split(_cell(record.get("ol_work_ids"))):
            ol_id = ol_id.strip()
            if ol_id and ol_id not in index:
                index[ol_id] = payload
    return index


def build_dnb_title_index(dnb_dir: Path = DNB_DIR) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in sorted(dnb_dir.glob("dnb_hits_tit_*.csv")):
        frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
        for record in frame.to_dict(orient="records"):
            isbn_13, isbn_10 = _pick_isbns("", "", _cell(record.get("isbn")))
            if not isbn_13 and not isbn_10:
                continue
            key = _title_author_key(record.get("title") or "", record.get("author") or "")
            title_part, _, surname_part = key.partition("|")
            if not title_part or not surname_part:
                continue
            if key not in index:
                index[key] = {"isbn_13": isbn_13, "isbn_10": isbn_10}
    return index


def enrich_frame(
    frame: pd.DataFrame,
    ol_index: dict,
    osp_index: dict,
    dnb_index: dict,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    frame = frame.copy().fillna("")
    filled = 0
    for idx, record in frame.iterrows():
        isbn_13 = _first_isbn13(record.get("isbn_13", ""))
        isbn_10 = _first_isbn10(record.get("isbn_10", ""))
        before = (isbn_13, isbn_10)

        if not isbn_13 or not isbn_10:
            for ol_id in (
                _cell(record.get("ol_work_id")),
                _cell(record.get("ol_edition_id")),
            ):
                if not ol_id:
                    continue
                for index in (ol_index, osp_index):
                    hit = index.get(ol_id)
                    if not hit:
                        continue
                    if not isbn_13 and hit.get("isbn_13"):
                        isbn_13 = hit["isbn_13"]
                    if not isbn_10 and hit.get("isbn_10"):
                        isbn_10 = hit["isbn_10"]
                if isbn_13 and isbn_10:
                    break

        if not isbn_13 or not isbn_10:
            key = _title_author_key(
                record.get("title") or "",
                record.get("author_name") or record.get("author") or "",
            )
            hit = dnb_index.get(key)
            if hit:
                if not isbn_13 and hit.get("isbn_13"):
                    isbn_13 = hit["isbn_13"]
                if not isbn_10 and hit.get("isbn_10"):
                    isbn_10 = hit["isbn_10"]

        if "isbn_13" in frame.columns:
            frame.at[idx, "isbn_13"] = isbn_13 or _cell(record.get("isbn_13"))
        if "isbn_10" in frame.columns:
            frame.at[idx, "isbn_10"] = isbn_10 or _cell(record.get("isbn_10"))
        after = (
            _first_isbn13(frame.at[idx, "isbn_13"]) if "isbn_13" in frame.columns else "",
            _first_isbn10(frame.at[idx, "isbn_10"]) if "isbn_10" in frame.columns else "",
        )
        if after != before and (after[0] or after[1]):
            if not before[0] and after[0] or not before[1] and after[1]:
                filled += 1
    return frame, filled


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated filenames under data/interim/final_lists/",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=OUT_DIR,
        help="Directory containing target CSVs",
    )
    args = parser.parse_args(argv)
    log = setup_logger(__file__)
    try:
        ol_index = build_ol_isbn_index()
        osp_index = build_osp_ol_index()
        dnb_index = build_dnb_title_index()
        log.info(
            f"indexes: ol={len(ol_index)} osp_ol={len(osp_index)} dnb_title={len(dnb_index)}"
        )

        total_filled = 0
        for name in [part.strip() for part in args.targets.split(",") if part.strip()]:
            path = args.dir / name
            if not path.is_file():
                log.info(f"skip missing {path}")
                continue
            frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
            enriched, filled = enrich_frame(frame, ol_index, osp_index, dnb_index)
            enriched.to_csv(path, index=False)
            total_filled += filled
            with_isbn = 0
            if "isbn_13" in enriched.columns:
                with_isbn = int(enriched["isbn_13"].map(lambda value: bool(normalize_isbn(str(value)) if value else None)).sum())
            log.info(f"{name}: filled {filled} rows; isbn_13 present on ~{with_isbn}/{len(enriched)}")
        log.info(f"done; rows newly filled across files: {total_filled}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
