"""Check seed bibliography titles in HeBIS (ISBN first, then title fallback).

Writes data/interim/hebis/hebis_seed_holdings.csv — one or more rows per seed that hit.
Frankfurt / local notes land in holding_note when HeBIS returns 852 fields.
"""
from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from src.candidates.migrate_candidates import base_title_raw
from src.catalogue.hebis import retrieve_all_hebis
from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn
from src.common.paths import DATA_INTERIM

SEED_PATH = DATA_INTERIM / "catalogue" / "seed_candidates.csv"
OUT_DIR = DATA_INTERIM / "hebis"


def _cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("de", "en", "both"), default="both")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N seed rows")
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    log = setup_logger(__file__)
    try:
        if not SEED_PATH.is_file():
            raise SystemExit(f"Missing {SEED_PATH} — run migrate_candidates first.")

        seeds = pd.read_csv(SEED_PATH, dtype=str).fillna("")
        langs = ("de", "en") if args.lang == "both" else (args.lang,)
        seeds = seeds[seeds["edition_language"].isin(langs)].copy()
        if args.limit is not None:
            seeds = seeds.head(args.limit)

        log.info(f"Checking {len(seeds)} seed titles in HeBIS (langs={langs})")
        all_hits = []
        found, missed = [], []

        for i, row in enumerate(seeds.to_dict(orient="records"), start=1):
            cid = _cell(row.get("candidate_id"))
            title = _cell(row.get("title"))
            isbn13 = normalize_isbn(_cell(row.get("isbn_13"))) or ""
            isbn10 = normalize_isbn(_cell(row.get("isbn_10"))) or ""
            isbn = isbn13 or isbn10

            if isbn:
                query = f"pica.isb={re.sub(r'[^0-9Xx]', '', isbn)}"
            else:
                base = base_title_raw(title)
                query = f'pica.tit="{base.replace(chr(34), "")}"'

            log.info(f"[{i}/{len(seeds)}] {cid}: {query}")
            hits = retrieve_all_hebis(query, max_pages=args.max_pages, pause=args.pause, log=log)
            if hits.empty and isbn:
                # Fallback: title search when ISBN misses.
                base = base_title_raw(title)
                query2 = f'pica.tit="{base.replace(chr(34), "")}"'
                log.info(f"  ISBN miss — trying title: {query2}")
                hits = retrieve_all_hebis(
                    query2, max_pages=args.max_pages, pause=args.pause, log=log
                )

            if hits.empty:
                missed.append(cid)
            else:
                found.append(cid)
                hits = hits.copy()
                hits["seed_candidate_id"] = cid
                hits["seed_title"] = title
                hits["seed_isbn"] = isbn
                all_hits.append(hits)
            time.sleep(args.pause)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "hebis_seed_holdings.csv"
        result = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
        result.to_csv(out, index=False)
        log.info(f"saved {out} ({len(result)} hit rows)")
        log.info(f"seeds with ≥1 HeBIS hit: {len(found)}/{len(seeds)}")
        log.info(f"seeds with 0 hits: {len(missed)}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
