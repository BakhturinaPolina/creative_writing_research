"""Improved Open Library Search harvester (replaces colleague MiniMax scrapers for new runs).

Improvements vs data/raw/colleague_openlibrary/scripts/:
- requests rich fields= so ISBN / publisher / IA / ebook_access are filled
- EN: subject "creative writing" plus craft phrases from catalogue_queries.yml
- DE: German phrases + language=ger (no aggressive title-hint gate)
- records source_query on every row
- repairs work/edition ID confusion
- optional Work → Editions enrichment for screened primary/uncertain rows
- soft rate limit + raw JSON dump + timestamped logs
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date

import pandas as pd
import requests
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_RAW
from src.common.prescreen import load_screen_rules, prescreen

SEARCH_URL = "https://openlibrary.org/search.json"
WORKS_URL = "https://openlibrary.org"
RAW_DIR = DATA_RAW / "openlibrary"
OUT_DIR = DATA_INTERIM / "openlibrary"

# Fields we ask the Search API to return (colleague scripts omitted ISBN/publisher).
FIELDS = (
    "key,title,subtitle,author_name,first_publish_year,isbn,publisher,"
    "language,subject,ia,ebook_access,edition_count,edition_key,cover_edition_key"
)

OL_ID_RE = re.compile(r"(OL\d+)([WMwm])")
SEP = " | "


def load_queries(lang: str) -> list[tuple[str, dict]]:
    """Return (label, request_params) pairs for one language.

    EN starts with subject=creative writing, then free-text craft queries.
    DE uses free-text queries with language=ger.
    """
    with open(CONFIG / "catalogue_queries.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    specs: list[tuple[str, dict]] = []
    if lang == "en":
        specs.append(
            (
                "subject:creative writing",
                {"subject": "creative writing", "limit": 100},
            )
        )
        phrases = []
        for bucket in ("core", "craft", "pov_specific", "historical"):
            phrases.extend(cfg.get("english", {}).get(bucket) or [])
        # Avoid duplicating the subject phrase as a free-text query.
        for phrase in phrases:
            if phrase.strip().lower() == "creative writing":
                continue
            specs.append((f"q:{phrase}", {"q": phrase, "limit": 100}))
    else:
        phrases = []
        for bucket in ("core", "craft", "pov_specific", "historical"):
            phrases.extend(cfg.get("german", {}).get(bucket) or [])
        for phrase in phrases:
            specs.append(
                (
                    f"q:{phrase}|lang:ger",
                    {"q": phrase, "language": "ger", "limit": 100},
                )
            )
    return specs


def _join(values, limit=None):
    if not values:
        return ""
    if isinstance(values, str):
        return values
    items = [str(v) for v in values if v is not None and str(v).strip()]
    if limit is not None:
        items = items[:limit]
    return SEP.join(items)


def extract_ol_ids(key: str) -> tuple[str, str]:
    """Return (ol_work_id, ol_edition_id) from a Search API key or URL path."""
    key = (key or "").strip()
    match = OL_ID_RE.search(key)
    if not match:
        return "", ""
    ol_id = f"{match.group(1)}{match.group(2).upper()}"
    if ol_id.endswith("W"):
        return ol_id, ""
    return "", ol_id


def doc_to_row(doc: dict, source_query: str, retrieval_date: str, rules) -> dict:
    """Map one Search API document to a flat CSV row."""
    key = doc.get("key") or ""
    work_id, edition_id = extract_ol_ids(key)
    # Some docs put edition ids under /works/ by mistake — check edition_key too.
    if not work_id and not edition_id:
        for ek in doc.get("edition_key") or []:
            w, e = extract_ol_ids(ek)
            if e:
                edition_id = e
                break
            if w:
                work_id = w
                break

    # If key was /works/OL…M, treat as edition.
    if "/works/" in key and edition_id:
        work_id = ""

    title = doc.get("title") or ""
    subtitle = doc.get("subtitle") or ""
    status, rule_id = prescreen(title, subtitle, rules)

    isbns = doc.get("isbn") or []
    isbn_10 = [i for i in isbns if len(re.sub(r"[^0-9Xx]", "", str(i))) == 10]
    isbn_13 = [i for i in isbns if len(re.sub(r"[^0-9Xx]", "", str(i))) == 13]

    return {
        "source": "OpenLibrary",
        "source_query": source_query,
        "retrieval_date": retrieval_date,
        "ol_work_id": work_id,
        "ol_edition_id": edition_id,
        "key": key,
        "title": title,
        "subtitle": subtitle,
        "author": _join(doc.get("author_name")),
        "publication_year": doc.get("first_publish_year") or "",
        "isbn_10": _join(isbn_10, limit=5),
        "isbn_13": _join(isbn_13, limit=5),
        "publisher": _join(doc.get("publisher"), limit=5),
        "language": _join(doc.get("language"), limit=5),
        "subjects": _join(doc.get("subject"), limit=15),
        "ia_identifier": _join(doc.get("ia"), limit=5),
        "ebook_access": doc.get("ebook_access") or "",
        "edition_count": doc.get("edition_count") or "",
        "url": f"https://openlibrary.org{key}" if key else "",
        "screen_status": status,
        "screen_rule": rule_id,
    }


def search_pages(
    params: dict,
    source_query: str,
    max_pages: int | None,
    pause: float,
    retrieval_date: str,
    rules,
    log,
) -> list[dict]:
    """Fetch one query across pages; dump each JSON response under data/raw/openlibrary/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    page = 1
    slug = re.sub(r"\W+", "_", source_query).strip("_")[:50]

    while True:
        req = {**params, "page": page, "fields": FIELDS}
        log.info(f"OL search page {page}: {source_query}")
        try:
            response = requests.get(SEARCH_URL, params=req, timeout=60)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.error(f"request failed page {page}: {exc}")
            break

        raw_path = RAW_DIR / f"{slug}_{retrieval_date}_p{page:04d}.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.info(f"  saved raw JSON → {raw_path}")

        docs = data.get("docs") or []
        num_found = data.get("numFound", 0)
        log.info(f"  numFound={num_found}, docs on page={len(docs)}")
        if not docs:
            break

        for doc in docs:
            rows.append(doc_to_row(doc, source_query, retrieval_date, rules))

        page += 1
        if max_pages and page > max_pages:
            break
        # Stop if we have walked past numFound.
        if (page - 1) * params.get("limit", 100) >= num_found:
            break
        time.sleep(pause)

    return rows


def dedupe_keep_first(rows: list[dict]) -> list[dict]:
    """Keep the first row per Open Library work or edition key; preserve its source_query."""
    seen = set()
    out = []
    for row in rows:
        key = row.get("ol_work_id") or row.get("ol_edition_id") or row.get("key") or ""
        if not key:
            out.append(row)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def enrich_editions(rows: list[dict], pause: float, log, max_editions_pages: int = 2) -> list[dict]:
    """For likely_primary + uncertain works, pull editions.json to fill thin ISBN/publisher/IA."""
    enrichable = ("likely_primary", "uncertain")
    for i, row in enumerate(rows):
        if row.get("screen_status") not in enrichable:
            continue
        work_id = row.get("ol_work_id") or ""
        if not work_id:
            continue
        # Skip if we already have ISBN and publisher from search.
        if row.get("isbn_13") and row.get("publisher"):
            continue

        url = f"{WORKS_URL}/works/{work_id}/editions.json"
        log.info(f"enrich editions [{i+1}/{len(rows)}]: {work_id}")
        try:
            response = requests.get(url, params={"limit": 50}, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            log.error(f"  editions fetch failed: {exc}")
            time.sleep(pause)
            continue

        entries = data.get("entries") or []
        isbns_10, isbns_13, pubs, ias = [], [], [], []
        for ed in entries[: 50 * max_editions_pages]:
            for isbn in ed.get("isbn_10") or []:
                if isbn not in isbns_10:
                    isbns_10.append(isbn)
            for isbn in ed.get("isbn_13") or []:
                if isbn not in isbns_13:
                    isbns_13.append(isbn)
            for pub in ed.get("publishers") or []:
                if pub not in pubs:
                    pubs.append(pub)
            ocaid = ed.get("ocaid")
            if ocaid and ocaid not in ias:
                ias.append(ocaid)

        if not row.get("isbn_10") and isbns_10:
            row["isbn_10"] = _join(isbns_10, limit=5)
        if not row.get("isbn_13") and isbns_13:
            row["isbn_13"] = _join(isbns_13, limit=5)
        if not row.get("publisher") and pubs:
            row["publisher"] = _join(pubs, limit=5)
        if not row.get("ia_identifier") and ias:
            row["ia_identifier"] = _join(ias, limit=5)
        time.sleep(pause)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "de"), required=True)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages per query phrase (default 1 for smoke tests)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Only run the first N query specs (useful for smoke tests)",
    )
    parser.add_argument("--pause", type=float, default=0.75)
    parser.add_argument(
        "--enrich-editions",
        action="store_true",
        help="Call /works/.../editions.json for screened primary/uncertain rows missing ISBN",
    )
    args = parser.parse_args()

    log = setup_logger(__file__)
    try:
        rules = load_screen_rules()
        specs = load_queries(args.lang)
        if args.queries is not None:
            specs = specs[: args.queries]
        log.info(f"Open Library harvest lang={args.lang}, {len(specs)} queries, max_pages={args.max_pages}")

        retrieval_date = date.today().isoformat()
        all_rows: list[dict] = []
        for label, params in specs:
            all_rows.extend(
                search_pages(
                    params,
                    label,
                    args.max_pages,
                    args.pause,
                    retrieval_date,
                    rules,
                    log,
                )
            )

        before = len(all_rows)
        all_rows = dedupe_keep_first(all_rows)
        log.info(f"rows before dedupe={before}, after={len(all_rows)}")

        if args.enrich_editions:
            all_rows = enrich_editions(all_rows, args.pause, log)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"ol_search_{args.lang}.csv"
        frame = pd.DataFrame(all_rows)
        frame.to_csv(out, index=False)

        with_isbn = 0
        if not frame.empty:
            has10 = frame.get("isbn_10", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != ""
            has13 = frame.get("isbn_13", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != ""
            with_isbn = int((has10 | has13).sum())
        log.info(f"saved {out} ({len(frame)} rows, {with_isbn} with ISBN)")
        if not frame.empty and "screen_status" in frame.columns:
            log.info("screen counts:\n" + frame["screen_status"].value_counts().to_string())
    finally:
        log.close()


if __name__ == "__main__":
    main()
