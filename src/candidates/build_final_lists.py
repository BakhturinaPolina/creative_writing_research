"""Merge filtered OpenLibrary lists with the seed corpus into final candidate lists.

Reads data/interim/filtered_openlibrary/ and data/interim/catalogue/seed_candidates.csv.
Writes data/interim/final_lists/final_list_{en,de}.csv and dedupe_log.csv.

Dedup key (per language): normalized base title (before colon) + author surname.
On clash, keep the seed row (richer ISBN/provenance).
"""
import pandas as pd

from src.candidates.migrate_candidates import (
    FEED_COLUMNS,
    author_display,
    author_surname,
    base_title_norm,
    base_title_raw,
    extract_ol,
    language_codes,
    parse_year,
    _cell,
)
from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn
from src.common.paths import DATA_INTERIM

FILTERED = DATA_INTERIM / "filtered_openlibrary"
SEED_PATH = DATA_INTERIM / "catalogue" / "seed_candidates.csv"
OUT_DIR = DATA_INTERIM / "final_lists"

DEDUPE_COLUMNS = [
    "edition_language",
    "dedupe_key",
    "kept_candidate_id",
    "dropped_candidate_id",
    "kept_source",
    "dropped_source",
    "title",
    "author_name",
]


def _empty_feed():
    return {column: "" for column in FEED_COLUMNS}


def _dedupe_key(title, author):
    return f"{base_title_norm(title)}|{author_surname(author)}"


def load_filtered(path, default_lang):
    """Load a filtered_* CSV (;, UTF-8). EN has language before year; DE has year before language."""
    frame = pd.read_csv(path, dtype=str, sep=";", encoding="utf-8").fillna("")
    rows = []
    seen_ol = {}
    for number, record in enumerate(frame.to_dict(orient="records"), start=1):
        title = _cell(record.get("title"))
        author = _cell(record.get("author"))
        year = _cell(record.get("year"))
        lang_raw = _cell(record.get("language"))
        link = _cell(record.get("link"))
        work_id, edition_id, _url = extract_ol(link)
        ol_id = work_id or edition_id
        if ol_id:
            if ol_id in seen_ol:
                continue
            seen_ol[ol_id] = True
            candidate_id = f"filtered_ol:{ol_id}"
        else:
            candidate_id = f"filtered_ol:row:{number}"

        codes = language_codes(lang_raw)
        lang = codes[0] if codes else default_lang
        if lang not in ("en", "de"):
            lang = default_lang

        row = _empty_feed()
        row["candidate_id"] = candidate_id
        row["title"] = base_title_raw(title)
        row["author_name"] = author_display(author)
        row["publication_year"] = parse_year(year)
        row["edition_language"] = lang
        row["_source"] = "filtered_openlibrary"
        row["_author_raw"] = author
        row["_title_raw"] = title
        rows.append(row)
    return rows


def load_seed(path):
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    rows = []
    for record in frame.to_dict(orient="records"):
        lang = _cell(record.get("edition_language"))
        if lang not in ("en", "de"):
            continue
        corpus_id = _cell(record.get("candidate_id")) or _cell(record.get("source_record_id"))
        title = _cell(record.get("title"))
        author = _cell(record.get("author"))
        row = _empty_feed()
        row["candidate_id"] = corpus_id if corpus_id.startswith("seed:") else f"seed:{corpus_id}"
        row["title"] = base_title_raw(title)
        row["author_name"] = author_display(author)
        row["publication_year"] = _cell(record.get("edition_year"))
        row["isbn_13"] = normalize_isbn(_cell(record.get("isbn_13"))) or ""
        row["isbn_10"] = normalize_isbn(_cell(record.get("isbn_10"))) or ""
        row["edition_language"] = lang
        row["_source"] = "seed_corpus"
        row["_author_raw"] = author
        row["_title_raw"] = title
        rows.append(row)
    return rows


def merge_and_dedupe(filtered_rows, seed_rows):
    """Keep seed on title+surname clash; log dropped filtered (or later seed) rows."""
    by_lang = {"en": [], "de": []}
    logs = []

    # Index seed first so it wins.
    for lang in ("en", "de"):
        index = {}  # dedupe_key -> row
        for row in seed_rows:
            if row["edition_language"] != lang:
                continue
            key = _dedupe_key(row["_title_raw"], row["_author_raw"])
            title_part, _, surname_part = key.partition("|")
            if not title_part or not surname_part:
                by_lang[lang].append(row)
                continue
            if key in index:
                kept = index[key]
                logs.append(
                    {
                        "edition_language": lang,
                        "dedupe_key": key,
                        "kept_candidate_id": kept["candidate_id"],
                        "dropped_candidate_id": row["candidate_id"],
                        "kept_source": kept["_source"],
                        "dropped_source": row["_source"],
                        "title": kept["title"],
                        "author_name": kept["author_name"],
                    }
                )
                continue
            index[key] = row
            by_lang[lang].append(row)

        for row in filtered_rows:
            if row["edition_language"] != lang:
                continue
            key = _dedupe_key(row["_title_raw"], row["_author_raw"])
            title_part, _, surname_part = key.partition("|")
            if not title_part or not surname_part:
                by_lang[lang].append(row)
                continue
            if key in index:
                kept = index[key]
                logs.append(
                    {
                        "edition_language": lang,
                        "dedupe_key": key,
                        "kept_candidate_id": kept["candidate_id"],
                        "dropped_candidate_id": row["candidate_id"],
                        "kept_source": kept["_source"],
                        "dropped_source": row["_source"],
                        "title": kept["title"],
                        "author_name": kept["author_name"],
                    }
                )
                continue
            index[key] = row
            by_lang[lang].append(row)

    return by_lang, logs


def _strip_working(rows):
    clean = []
    for row in rows:
        clean.append({column: row.get(column, "") for column in FEED_COLUMNS})
    return pd.DataFrame(clean, columns=FEED_COLUMNS)


def main():
    log = setup_logger(__file__)
    try:
        en_path = FILTERED / "filtered_english_corp.csv"
        de_path = FILTERED / "filtered_german_corp.csv"
        if not en_path.is_file() or not de_path.is_file():
            raise SystemExit(f"missing filtered CSVs under {FILTERED}")
        if not SEED_PATH.is_file():
            raise SystemExit(f"missing seed candidates: {SEED_PATH}")

        filtered = load_filtered(en_path, "en") + load_filtered(de_path, "de")
        seed = load_seed(SEED_PATH)
        by_lang, logs = merge_and_dedupe(filtered, seed)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        en_df = _strip_working(by_lang["en"])
        de_df = _strip_working(by_lang["de"])
        log_df = pd.DataFrame(logs, columns=DEDUPE_COLUMNS)

        en_out = OUT_DIR / "final_list_en.csv"
        de_out = OUT_DIR / "final_list_de.csv"
        log_out = OUT_DIR / "dedupe_log.csv"
        en_df.to_csv(en_out, index=False)
        de_df.to_csv(de_out, index=False)
        log_df.to_csv(log_out, index=False)

        seed_en = sum(1 for r in seed if r["edition_language"] == "en")
        seed_de = sum(1 for r in seed if r["edition_language"] == "de")
        filt_en = sum(1 for r in filtered if r["edition_language"] == "en")
        filt_de = sum(1 for r in filtered if r["edition_language"] == "de")
        log.info(f"filtered (after OLID dedupe): en {filt_en}, de {filt_de}")
        log.info(f"seed: en {seed_en}, de {seed_de}")
        log.info(f"final lists: en {len(en_df)} -> {en_out}")
        log.info(f"final lists: de {len(de_df)} -> {de_out}")
        log.info(f"dedupe drops: {len(log_df)} -> {log_out}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
