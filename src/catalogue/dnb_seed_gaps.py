"""Look up seed titles that Open Library missed, using DNB title + author queries.

Reads data/interim/catalogue/seed_recall.csv (miss rows) and seed_candidates.csv.
Writes data/interim/dnb/dnb_seed_gaps_{lang}.csv plus a small summary in the log.

Use --loose to retry with ISBN, title-only, short title, and alternate surnames
(for seeds that the strict tit+per query missed).
"""
from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from src.candidates.migrate_candidates import AUTHOR_SPLIT_RE, author_surname, base_title_raw
from src.catalogue.dnb import retrieve_all_dnb
from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn
from src.common.paths import DATA_INTERIM

SEED_RECALL = DATA_INTERIM / "catalogue" / "seed_recall.csv"
SEED_CANDIDATES = DATA_INTERIM / "catalogue" / "seed_candidates.csv"
OUT_DIR = DATA_INTERIM / "dnb"


def _cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _cql_escape(text: str) -> str:
    """Escape double quotes inside a CQL string literal."""
    return text.replace('"', '\\"')


def _clean_title(title: str) -> str:
    """Drop subtitle markers and odd dashes so DNB phrase search matches better."""
    text = base_title_raw(title)
    # En/em dashes and similar → space
    text = re.sub(r"[\u2013\u2014\u2212–—−]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .;:")
    return text


def build_gap_query(title: str, author: str) -> str:
    """Build a narrow DNB query: title phrase + person (surname)."""
    base = _clean_title(title)
    surname = author_surname(author)
    parts = []
    if base:
        parts.append(f'tit="{_cql_escape(base)}"')
    if surname:
        parts.append(f'per="{_cql_escape(surname)}"')
    return " and ".join(parts)


def _surnames(author: str) -> list[str]:
    """All coauthor surnames (e.g. Bernays and Painter)."""
    seen = []
    for part in AUTHOR_SPLIT_RE.split(_cell(author)) or [_cell(author)]:
        part = part.strip()
        if not part:
            continue
        surname = author_surname(part)
        if surname and surname not in seen:
            seen.append(surname)
    return seen


def build_loose_queries(title: str, author: str, isbn_13: str = "", isbn_10: str = "") -> list[tuple[str, str]]:
    """Ordered (label, CQL) fallbacks from most precise to broader.

    1. ISBN-13 / ISBN-10
    2. Full cleaned title only
    3. Short title (first 2–4 words) + each surname
    4. Title only with first 3 words
    5. Each surname alone (last resort — noisy)
    """
    queries: list[tuple[str, str]] = []
    seen = set()

    def add(label: str, cql: str) -> None:
        cql = cql.strip()
        if not cql or cql in seen:
            return
        seen.add(cql)
        queries.append((label, cql))

    for isbn in (normalize_isbn(isbn_13), normalize_isbn(isbn_10)):
        if isbn:
            add(f"isbn:{isbn}", f"isb={isbn}")

    cleaned = _clean_title(title)
    # Ampersand / hyphen variants (seed: "Kinder- und …", DNB: "Kinder- & …")
    cleaned_amp = cleaned.replace(" und ", " & ") if " und " in cleaned else ""
    if cleaned:
        add("title_only", f'tit="{_cql_escape(cleaned)}"')
    if cleaned_amp and cleaned_amp != cleaned:
        add("title_amp", f'tit="{_cql_escape(cleaned_amp)}"')

    words = cleaned.split()
    surnames = _surnames(author)

    # Distinctive content words + author (before noisy short-title-only searches)
    stop = {
        "der", "die", "das", "und", "für", "für", "mit", "ein", "eine", "oder",
        "the", "and", "for", "a", "an", "of", "im", "in", "zu", "zum", "zur",
    }
    content = [w.strip(".,;:?!") for w in words if w.lower().strip(".,;:?!") not in stop and len(w) > 3]
    for surname in surnames:
        for word in content[:4]:
            add(
                f"per+word:{surname}+{word}",
                f'per="{_cql_escape(surname)}" and tit="{_cql_escape(word)}"',
            )

    for n in (4, 3, 2):
        if len(words) >= n:
            short = " ".join(words[:n])
            for surname in surnames:
                add(
                    f"short{n}+per:{surname}",
                    f'tit="{_cql_escape(short)}" and per="{_cql_escape(surname)}"',
                )
            # Title-only short queries are noisy — keep last
            add(f"short{n}_only", f'tit="{_cql_escape(short)}"')

    # Umlaut-free surname variants (ä→ae) — DNB sometimes indexes ASCII forms
    for surname in list(surnames):
        ascii_s = (
            surname.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        if ascii_s != surname and cleaned:
            add(
                f"title+ascii_per:{ascii_s}",
                f'tit="{_cql_escape(cleaned)}" and per="{_cql_escape(ascii_s)}"',
            )

    return queries


def _attach_seed(hits: pd.DataFrame, cid: str, title: str, author: str, query: str, strategy: str) -> pd.DataFrame:
    hits = hits.copy()
    hits["seed_candidate_id"] = cid
    hits["seed_title"] = title
    hits["seed_author"] = author
    hits["query_strategy"] = strategy
    # Keep the winning CQL visible even if retrieve_all already set source_query
    hits["gap_query"] = query
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("de", "en", "both"), default="de")
    parser.add_argument("--max-pages", type=int, default=1, help="Pages per query attempt (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N misses")
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help='Comma-separated seed IDs only, e.g. "DE-013,DE-015"',
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Try ISBN / title-only / short-title fallbacks until one hits",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Append new hits onto existing dnb_seed_gaps_{lang}.csv (drop duplicate IDN+seed)",
    )
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between queries")
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help='Optional filename suffix, e.g. "loose" → dnb_seed_gaps_de_loose.csv',
    )
    args = parser.parse_args()

    log = setup_logger(__file__)
    try:
        if not SEED_RECALL.is_file() or not SEED_CANDIDATES.is_file():
            raise SystemExit("Need seed_recall.csv and seed_candidates.csv — run migrate_candidates first.")

        recall = pd.read_csv(SEED_RECALL, dtype=str).fillna("")
        seeds = pd.read_csv(SEED_CANDIDATES, dtype=str).fillna("")
        seed_by_id = {_cell(r["candidate_id"]): r for r in seeds.to_dict(orient="records")}

        langs = ("de", "en") if args.lang == "both" else (args.lang,)
        if args.ids:
            wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
            # Prefer seed table rows so we still retry known hard misses
            id_rows = []
            for cid in wanted:
                seed = seed_by_id.get(cid)
                if not seed:
                    log.info(f"unknown seed id {cid} — skip")
                    continue
                id_rows.append(
                    {
                        "candidate_id": cid,
                        "edition_language": _cell(seed.get("edition_language")),
                        "author": _cell(seed.get("author")),
                        "title": _cell(seed.get("title")),
                        "match_status": "miss",
                    }
                )
            misses = pd.DataFrame(id_rows)
        else:
            misses = recall[
                (recall["match_status"] == "miss") & (recall["edition_language"].isin(langs))
            ].copy()
        if args.limit is not None:
            misses = misses.head(args.limit)

        mode = "loose" if args.loose else "strict tit+per"
        log.info(f"Seed DNB lookups: {len(misses)} rows (langs={langs}, mode={mode})")
        all_hits = []
        found_ids, not_found_ids = [], []
        winning = []

        for i, row in enumerate(misses.to_dict(orient="records"), start=1):
            cid = _cell(row["candidate_id"])
            seed = seed_by_id.get(cid, {})
            title = _cell(seed.get("title")) or _cell(row.get("title"))
            author = _cell(seed.get("author")) or _cell(row.get("author"))
            isbn_13 = _cell(seed.get("isbn_13"))
            isbn_10 = _cell(seed.get("isbn_10"))

            if args.loose:
                attempts = build_loose_queries(title, author, isbn_13, isbn_10)
            else:
                q = build_gap_query(title, author)
                attempts = [("tit+per", q)] if q else []

            if not attempts:
                log.info(f"[{i}/{len(misses)}] {cid}: no queries — skip")
                not_found_ids.append(cid)
                continue

            hit_frame = None
            used_label = used_query = ""
            for label, query in attempts:
                log.info(f"[{i}/{len(misses)}] {cid}: try {label} → {query}")
                hits = retrieve_all_dnb(query, max_pages=args.max_pages, pause=args.pause, log=log)
                time.sleep(args.pause)
                if not hits.empty:
                    hit_frame = _attach_seed(hits, cid, title, author, query, label)
                    used_label, used_query = label, query
                    break

            if hit_frame is None:
                not_found_ids.append(cid)
                log.info(f"[{i}/{len(misses)}] {cid}: still not found after {len(attempts)} strategies")
            else:
                found_ids.append(cid)
                winning.append({"seed_candidate_id": cid, "strategy": used_label, "query": used_query, "hits": len(hit_frame)})
                all_hits.append(hit_frame)
                log.info(f"[{i}/{len(misses)}] {cid}: FOUND via {used_label} ({len(hit_frame)} rows)")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        lang_tag = args.lang if args.lang != "both" else "en_de"
        suffix = f"_{args.out_suffix}" if args.out_suffix else ("_loose" if args.loose and args.ids else "")
        out = OUT_DIR / f"dnb_seed_gaps_{lang_tag}{suffix}.csv"
        result = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()

        if args.merge_existing:
            main_path = OUT_DIR / f"dnb_seed_gaps_{lang_tag}.csv"
            if main_path.is_file():
                old = pd.read_csv(main_path, dtype=str).fillna("")
                if not result.empty:
                    combined = pd.concat([old, result], ignore_index=True)
                else:
                    combined = old
                # Prefer keeping one row per (seed, DNB id)
                if "source_record_id" in combined.columns and "seed_candidate_id" in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=["seed_candidate_id", "source_record_id"], keep="last"
                    )
                combined.to_csv(main_path, index=False)
                log.info(f"merged into {main_path} ({len(combined)} rows)")
            result.to_csv(out, index=False)
        else:
            result.to_csv(out, index=False)

        log.info(f"saved {out} ({len(result)} hit rows)")
        log.info(f"seeds with ≥1 DNB hit: {len(found_ids)}/{len(misses)}")
        log.info(f"seeds with 0 hits: {len(not_found_ids)}")
        if winning:
            log.info("winning strategies:")
            for w in winning:
                log.info(f"  {w['seed_candidate_id']}: {w['strategy']} ({w['hits']} hits) — {w['query']}")
        if not_found_ids:
            log.info("not found: " + ", ".join(not_found_ids))
    finally:
        log.close()


if __name__ == "__main__":
    main()
