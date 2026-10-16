"""Build true-union candidate lists for access checking and colleague review.

Ingests seed, filtered Open Library, catalogue prescreen, ol_search_*, DNB title
hits, and OpenSyllabus books. Writes under data/interim/final_lists/:

  union_all.csv          full deduped union (audit)
  final_list_{en,de}.csv searcher feeds (P0–P2, FEED_COLUMNS only)
  review_{en,de}.csv     colleague-facing (extra provenance columns)
  dedupe_log.csv         kept/dropped collisions

Dedup cascade (per language): ISBN-13 → OL work id → DNB id → title+surname.
On clash keep the richest row and merge source_tags.
"""
from __future__ import annotations

import re
from pathlib import Path

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
    split_isbns,
    _cell,
)
from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn
from src.common.paths import DATA_INTERIM
from src.common.prescreen import load_screen_rules, prescreen

FILTERED = DATA_INTERIM / "filtered_openlibrary"
CATALOGUE = DATA_INTERIM / "catalogue"
OL_DIR = DATA_INTERIM / "openlibrary"
DNB_DIR = DATA_INTERIM / "dnb"
OSP_DIR = DATA_INTERIM / "opensyllabus"
OUT_DIR = DATA_INTERIM / "final_lists"

SEED_PATH = CATALOGUE / "seed_candidates.csv"
PRESCREEN_PATHS = {
    "en": CATALOGUE / "prescreen_en.csv",
    "de": CATALOGUE / "prescreen_de.csv",
}
OL_SEARCH_PATHS = {
    "en": OL_DIR / "ol_search_en.csv",
    "de": OL_DIR / "ol_search_de.csv",
}
OSP_PATH = OSP_DIR / "opensyllabus_creative_writing_all.csv"

DNB_HIT_GLOB = "dnb_hits_tit_*.csv"
OL_SCREEN_KEEP = ("likely_primary", "uncertain")
SEARCH_PRIORITIES = ("P0", "P1", "P2")

UNION_EXTRA = [
    "source_tags",
    "screen_status",
    "screen_rule",
    "match_confidence",
    "download_priority",
    "ol_work_id",
    "ol_edition_id",
    "dnb_id",
    "subtitle",
]
UNION_COLUMNS = FEED_COLUMNS + UNION_EXTRA
REVIEW_COLUMNS = [
    "candidate_id",
    "title",
    "author_name",
    "publication_year",
    "isbn_13",
    "isbn_10",
    "edition_language",
    "source_tags",
    "screen_status",
    "screen_rule",
    "match_confidence",
    "download_priority",
    "ol_work_id",
    "dnb_id",
    "md5",
    "search_status",
    "found_format",
    "search_date",
]

DEDUPE_COLUMNS = [
    "edition_language",
    "dedupe_key",
    "dedupe_key_type",
    "kept_candidate_id",
    "dropped_candidate_id",
    "kept_source_tags",
    "dropped_source_tags",
    "title",
    "author_name",
]

# Higher score wins on clash. Tie-break: prefer ISBN, then lower candidate_id.
SOURCE_RICHNESS = {
    "seed": 100,
    "filtered_ol": 80,
    "opensyllabus": 70,
    "ol_search": 55,
    "dnb": 50,
    "prescreen": 40,
}

LANG_FROM_DNB = {"ger": "de", "deu": "de", "eng": "en", "en": "en", "de": "de"}
PIPE_SPLIT = re.compile(r"\s*\|\s*")


def _empty_union():
    return {column: "" for column in UNION_COLUMNS}


def _first_isbn13(value: str) -> str:
    for part in re.split(r"[;|\s]+", _cell(value)):
        isbn = normalize_isbn(part)
        if isbn and len(isbn) == 13:
            return isbn
    return ""


def _first_isbn10(value: str) -> str:
    for part in re.split(r"[;|\s]+", _cell(value)):
        isbn = normalize_isbn(part)
        if isbn and len(isbn) == 10:
            return isbn
    return ""


def _pick_isbns(isbn_13_raw: str, isbn_10_raw: str, combined: str = "") -> tuple[str, str]:
    isbn_13 = _first_isbn13(isbn_13_raw) or _first_isbn13(combined)
    isbn_10 = _first_isbn10(isbn_10_raw) or _first_isbn10(combined)
    if not isbn_13 and not isbn_10 and combined:
        isbn_10_joined, isbn_13_joined = split_isbns(combined.replace("|", ";"))
        isbn_13 = _first_isbn13(isbn_13_joined)
        isbn_10 = _first_isbn10(isbn_10_joined)
    return isbn_13, isbn_10


def _richness(row: dict) -> tuple:
    tags = set(PIPE_SPLIT.split(row.get("source_tags") or ""))
    score = max((SOURCE_RICHNESS.get(tag, 0) for tag in tags), default=0)
    has_isbn = 1 if row.get("isbn_13") or row.get("isbn_10") else 0
    return (score, has_isbn, row.get("candidate_id") or "")


def _merge_tags(*tag_strings: str) -> str:
    seen = []
    for text in tag_strings:
        for tag in PIPE_SPLIT.split(_cell(text)):
            if tag and tag not in seen:
                seen.append(tag)
    return "|".join(seen)


def _assign_priority(row: dict) -> str:
    tags = set(PIPE_SPLIT.split(row.get("source_tags") or ""))
    if "seed" in tags:
        return "P0"
    if "filtered_ol" in tags or "opensyllabus" in tags:
        return "P1"
    screen = _cell(row.get("screen_status"))
    if screen == "likely_primary":
        return "P2"
    return "P3"


def _match_confidence(row: dict) -> str:
    if row.get("isbn_13") or row.get("isbn_10"):
        return "isbn"
    if row.get("ol_work_id"):
        return "ol_id"
    if row.get("dnb_id"):
        return "dnb_id"
    title = base_title_norm(row.get("title") or "")
    surname = author_surname(row.get("author_name") or "")
    if title and surname:
        return "title_author"
    return "weak"


def _finalize_row(row: dict) -> dict:
    row["source_tags"] = _merge_tags(row.get("source_tags", ""))
    row["match_confidence"] = _match_confidence(row)
    row["download_priority"] = _assign_priority(row)
    return row


def _title_author_key(title: str, author: str) -> str:
    return f"{base_title_norm(title)}|{author_surname(author)}"


# ---------------------------------------------------------------------------
# ISBN index from ol_search (for backfill onto filtered / thin rows)
# ---------------------------------------------------------------------------


def build_ol_isbn_index(paths: dict[str, Path] | None = None) -> dict[str, dict]:
    """Map OL work/edition id → best isbn_13/isbn_10 from ol_search CSVs."""
    paths = paths or OL_SEARCH_PATHS
    index: dict[str, dict] = {}
    for path in paths.values():
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
        for record in frame.to_dict(orient="records"):
            isbn_13, isbn_10 = _pick_isbns(
                record.get("isbn_13", ""),
                record.get("isbn_10", ""),
            )
            if not isbn_13 and not isbn_10:
                continue
            payload = {"isbn_13": isbn_13, "isbn_10": isbn_10}
            for key_name in ("ol_work_id", "ol_edition_id"):
                ol_id = _cell(record.get(key_name))
                if not ol_id:
                    continue
                existing = index.get(ol_id)
                if existing is None or (isbn_13 and not existing.get("isbn_13")):
                    index[ol_id] = payload
    return index


def apply_isbn_backfill(row: dict, ol_isbn_index: dict[str, dict]) -> dict:
    if row.get("isbn_13") and row.get("isbn_10"):
        return row
    for ol_id in (row.get("ol_work_id"), row.get("ol_edition_id")):
        hit = ol_isbn_index.get(_cell(ol_id))
        if not hit:
            continue
        if not row.get("isbn_13") and hit.get("isbn_13"):
            row["isbn_13"] = hit["isbn_13"]
        if not row.get("isbn_10") and hit.get("isbn_10"):
            row["isbn_10"] = hit["isbn_10"]
        break
    return row


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_filtered(path: Path, default_lang: str, ol_isbn_index: dict) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, sep=";", encoding="utf-8").fillna("")
    rows = []
    seen_ol = {}
    rules = load_screen_rules()
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

        screen_status, screen_rule = prescreen(title, "", rules)
        row = _empty_union()
        row["candidate_id"] = candidate_id
        row["title"] = base_title_raw(title)
        row["author_name"] = author_display(author)
        row["publication_year"] = parse_year(year)
        row["edition_language"] = lang
        row["ol_work_id"] = work_id
        row["ol_edition_id"] = edition_id
        row["source_tags"] = "filtered_ol"
        row["screen_status"] = screen_status
        row["screen_rule"] = screen_rule
        row["_title_raw"] = title
        row["_author_raw"] = author
        apply_isbn_backfill(row, ol_isbn_index)
        rows.append(_finalize_row(row))
    return rows


def load_seed(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    rows = []
    for record in frame.to_dict(orient="records"):
        lang = _cell(record.get("edition_language"))
        if lang not in ("en", "de"):
            continue
        corpus_id = _cell(record.get("candidate_id")) or _cell(record.get("source_record_id"))
        title = _cell(record.get("title"))
        author = _cell(record.get("author"))
        row = _empty_union()
        row["candidate_id"] = corpus_id if corpus_id.startswith("seed:") else f"seed:{corpus_id}"
        row["title"] = base_title_raw(title)
        row["subtitle"] = _cell(record.get("subtitle"))
        row["author_name"] = author_display(author)
        row["publication_year"] = _cell(record.get("edition_year"))
        row["isbn_13"] = normalize_isbn(_cell(record.get("isbn_13"))) or ""
        row["isbn_10"] = normalize_isbn(_cell(record.get("isbn_10"))) or ""
        row["edition_language"] = lang
        row["ol_work_id"] = _cell(record.get("ol_work_id"))
        row["ol_edition_id"] = _cell(record.get("ol_edition_id"))
        row["dnb_id"] = _cell(record.get("dnb_id"))
        row["source_tags"] = "seed"
        row["screen_status"] = _cell(record.get("eligibility_status")) or "likely_primary"
        row["screen_rule"] = _cell(record.get("screen_rule"))
        row["_title_raw"] = title
        row["_author_raw"] = author
        rows.append(_finalize_row(row))
    return rows


def load_prescreen(path: Path, lang: str, ol_isbn_index: dict) -> list[dict]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    rules = load_screen_rules()
    rows = []
    for record in frame.to_dict(orient="records"):
        title = _cell(record.get("title"))
        author = _cell(record.get("author_name") or record.get("author"))
        screen_status, screen_rule = prescreen(title, "", rules)
        if screen_status == "likely_exclude":
            # Keep in union for audit; priority will be P3+ and search filter drops excludes.
            pass
        row = _empty_union()
        row["candidate_id"] = _cell(record.get("candidate_id")) or f"prescreen:{lang}:{len(rows)+1}"
        row["title"] = base_title_raw(title)
        row["author_name"] = author_display(author)
        row["publication_year"] = parse_year(record.get("publication_year"))
        row["isbn_13"], row["isbn_10"] = _pick_isbns(
            record.get("isbn_13", ""),
            record.get("isbn_10", ""),
        )
        row["edition_language"] = lang
        row["source_tags"] = "prescreen"
        row["screen_status"] = screen_status
        row["screen_rule"] = screen_rule
        # Recover OL id from candidate_id when present (colleague_openlibrary_*:OL…W).
        cid = row["candidate_id"]
        match = re.search(r"(OL\d+[WwMm])", cid)
        if match:
            ol_id = match.group(1).upper()
            if ol_id.endswith("W"):
                row["ol_work_id"] = ol_id
            else:
                row["ol_edition_id"] = ol_id
        row["_title_raw"] = title
        row["_author_raw"] = author
        apply_isbn_backfill(row, ol_isbn_index)
        rows.append(_finalize_row(row))
    return rows


def load_ol_search(path: Path, default_lang: str, ol_isbn_index: dict) -> list[dict]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    rows = []
    seen = set()
    for record in frame.to_dict(orient="records"):
        screen = _cell(record.get("screen_status"))
        if screen not in OL_SCREEN_KEEP:
            continue
        work_id = _cell(record.get("ol_work_id"))
        edition_id = _cell(record.get("ol_edition_id"))
        ol_key = work_id or edition_id or _cell(record.get("key"))
        if ol_key in seen:
            continue
        seen.add(ol_key)

        title = _cell(record.get("title"))
        subtitle = _cell(record.get("subtitle"))
        author = _cell(record.get("author"))
        codes = language_codes(record.get("language"))
        lang = codes[0] if codes else default_lang
        if lang not in ("en", "de"):
            # Keep if source file is language-scoped and eng/ger missing.
            if default_lang in ("en", "de") and (not codes or default_lang in codes):
                lang = default_lang
            else:
                lang = default_lang
        if lang not in ("en", "de"):
            continue

        isbn_13, isbn_10 = _pick_isbns(record.get("isbn_13", ""), record.get("isbn_10", ""))
        row = _empty_union()
        row["candidate_id"] = f"ol_search:{ol_key or len(rows)+1}"
        row["title"] = base_title_raw(title)
        row["subtitle"] = subtitle
        row["author_name"] = author_display(author)
        row["publication_year"] = parse_year(record.get("publication_year"))
        row["isbn_13"] = isbn_13
        row["isbn_10"] = isbn_10
        row["edition_language"] = lang
        row["ol_work_id"] = work_id
        row["ol_edition_id"] = edition_id
        row["source_tags"] = "ol_search"
        row["screen_status"] = screen
        row["screen_rule"] = _cell(record.get("screen_rule"))
        row["_title_raw"] = title
        row["_author_raw"] = author
        apply_isbn_backfill(row, ol_isbn_index)
        rows.append(_finalize_row(row))
    return rows


def load_dnb_hits(dnb_dir: Path = DNB_DIR) -> list[dict]:
    rules = load_screen_rules()
    rows = []
    seen_ids = set()
    for path in sorted(dnb_dir.glob(DNB_HIT_GLOB)):
        frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
        for record in frame.to_dict(orient="records"):
            dnb_id = _cell(record.get("source_record_id"))
            if dnb_id and dnb_id in seen_ids:
                continue
            if dnb_id:
                seen_ids.add(dnb_id)

            title = _cell(record.get("title"))
            author = _cell(record.get("author"))
            lang_raw = _cell(record.get("language")).lower()
            lang = LANG_FROM_DNB.get(lang_raw.split("|")[0].strip(), "")
            if lang not in ("en", "de"):
                # DNB creative-writing title harvests are predominantly German.
                lang = "de"

            isbn_13, isbn_10 = _pick_isbns("", "", _cell(record.get("isbn")))
            screen_status, screen_rule = prescreen(title, "", rules)
            row = _empty_union()
            row["candidate_id"] = f"dnb:{dnb_id}" if dnb_id else f"dnb:row:{path.stem}:{len(rows)+1}"
            row["title"] = base_title_raw(title)
            row["author_name"] = author_display(author)
            row["publication_year"] = parse_year(record.get("publication_year"))
            row["isbn_13"] = isbn_13
            row["isbn_10"] = isbn_10
            row["edition_language"] = lang
            row["dnb_id"] = dnb_id
            row["source_tags"] = "dnb"
            row["screen_status"] = screen_status
            row["screen_rule"] = screen_rule
            row["_title_raw"] = title
            row["_author_raw"] = author
            rows.append(_finalize_row(row))
    return rows


def load_opensyllabus(path: Path = OSP_PATH, ol_isbn_index: dict | None = None) -> list[dict]:
    ol_isbn_index = ol_isbn_index or {}
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    rules = load_screen_rules()
    rows = []
    for record in frame.to_dict(orient="records"):
        pub_type = _cell(record.get("publication_type")).lower()
        if pub_type and pub_type != "book":
            continue

        title = _cell(record.get("title"))
        subtitle = _cell(record.get("subtitle"))
        authors = _cell(record.get("authors"))
        # OSP uses "A | B"; take first author for display/surname.
        author = PIPE_SPLIT.split(authors)[0] if authors else ""
        osp_id = _cell(record.get("opensyllabus_id"))
        ol_works = [
            part.strip()
            for part in PIPE_SPLIT.split(_cell(record.get("ol_work_ids")))
            if part.strip()
        ]
        work_id = ol_works[0] if ol_works else ""

        isbn_13, isbn_10 = _pick_isbns("", "", _cell(record.get("isbns")))
        screen_status, screen_rule = prescreen(title, subtitle, rules)
        # OpenSyllabus creative-writing query is English-dominant.
        lang = "en"

        row = _empty_union()
        row["candidate_id"] = f"opensyllabus:{osp_id}" if osp_id else f"opensyllabus:row:{len(rows)+1}"
        row["title"] = base_title_raw(title)
        row["subtitle"] = subtitle
        row["author_name"] = author_display(author)
        row["publication_year"] = parse_year(record.get("year"))
        row["isbn_13"] = isbn_13
        row["isbn_10"] = isbn_10
        row["edition_language"] = lang
        row["ol_work_id"] = work_id
        row["source_tags"] = "opensyllabus"
        row["screen_status"] = screen_status
        row["screen_rule"] = screen_rule
        row["_title_raw"] = title
        row["_author_raw"] = author
        apply_isbn_backfill(row, ol_isbn_index)
        rows.append(_finalize_row(row))
    return rows


# ---------------------------------------------------------------------------
# Cascade dedupe
# ---------------------------------------------------------------------------


def _dedupe_keys_for_row(row: dict) -> list[tuple[str, str]]:
    """Return ordered (key_type, key) pairs for cascade matching."""
    keys = []
    isbn = _cell(row.get("isbn_13")) or ""
    if not isbn and row.get("isbn_10"):
        # Normalize ISBN-10 into a secondary key (not equivalent to ISBN-13 without conversion).
        isbn10 = _cell(row.get("isbn_10"))
        if isbn10:
            keys.append(("isbn10", isbn10))
    if isbn:
        keys.append(("isbn13", isbn))
    work = _cell(row.get("ol_work_id"))
    if work:
        keys.append(("ol_work", work))
    dnb = _cell(row.get("dnb_id"))
    if dnb:
        keys.append(("dnb", dnb))
    ta = _title_author_key(row.get("_title_raw") or row.get("title") or "", row.get("_author_raw") or row.get("author_name") or "")
    title_part, _, surname_part = ta.partition("|")
    if title_part and surname_part:
        keys.append(("title_author", ta))
    return keys


def merge_and_dedupe(all_rows: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Cascade-dedupe per language; merge source_tags on clash; keep richest row."""
    by_lang: dict[str, list[dict]] = {"en": [], "de": []}
    logs: list[dict] = []

    for lang in ("en", "de"):
        indexes: dict[str, dict[str, dict]] = {
            "isbn13": {},
            "isbn10": {},
            "ol_work": {},
            "dnb": {},
            "title_author": {},
        }
        kept_rows: list[dict] = []

        # Process richest first so weak rows attach to strong ones.
        lang_rows = [r for r in all_rows if r.get("edition_language") == lang]
        lang_rows.sort(key=_richness, reverse=True)

        for row in lang_rows:
            keys = _dedupe_keys_for_row(row)
            existing = None
            hit_type = ""
            hit_key = ""
            for key_type, key in keys:
                existing = indexes[key_type].get(key)
                if existing is not None:
                    hit_type, hit_key = key_type, key
                    break

            if existing is None:
                kept_rows.append(row)
                for key_type, key in keys:
                    indexes[key_type][key] = row
                continue

            # Clash: keep richer, merge tags onto kept.
            if _richness(row) > _richness(existing):
                # Promote incoming over existing (rare because we sort richest first).
                kept = row
                dropped = existing
                # Replace in kept_rows
                for i, candidate in enumerate(kept_rows):
                    if candidate is existing:
                        kept_rows[i] = kept
                        break
                # Re-index all keys for kept
                for key_type, key in _dedupe_keys_for_row(dropped):
                    if indexes[key_type].get(key) is dropped:
                        del indexes[key_type][key]
                for key_type, key in _dedupe_keys_for_row(kept):
                    indexes[key_type][key] = kept
            else:
                kept = existing
                dropped = row

            kept["source_tags"] = _merge_tags(kept.get("source_tags"), dropped.get("source_tags"))
            # Fill empty identifier fields from dropped.
            for field in (
                "isbn_13",
                "isbn_10",
                "ol_work_id",
                "ol_edition_id",
                "dnb_id",
                "subtitle",
                "publication_year",
                "md5",
                "search_status",
                "found_format",
                "search_date",
            ):
                if not kept.get(field) and dropped.get(field):
                    kept[field] = dropped[field]
            if (not kept.get("screen_status") or kept.get("screen_status") == "uncertain") and dropped.get(
                "screen_status"
            ) == "likely_primary":
                kept["screen_status"] = dropped["screen_status"]
                kept["screen_rule"] = dropped.get("screen_rule", "")
            _finalize_row(kept)

            # Ensure kept's keys are indexed (after ISBN fill).
            for key_type, key in _dedupe_keys_for_row(kept):
                indexes[key_type][key] = kept

            logs.append(
                {
                    "edition_language": lang,
                    "dedupe_key": hit_key,
                    "dedupe_key_type": hit_type,
                    "kept_candidate_id": kept["candidate_id"],
                    "dropped_candidate_id": dropped["candidate_id"],
                    "kept_source_tags": kept.get("source_tags", ""),
                    "dropped_source_tags": dropped.get("source_tags", ""),
                    "title": kept.get("title", ""),
                    "author_name": kept.get("author_name", ""),
                }
            )

        by_lang[lang] = kept_rows

    return by_lang, logs


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _to_union_frame(rows: list[dict]) -> pd.DataFrame:
    clean = []
    for row in rows:
        _finalize_row(row)
        clean.append({column: row.get(column, "") for column in UNION_COLUMNS})
    frame = pd.DataFrame(clean, columns=UNION_COLUMNS)
    if frame.empty:
        return frame
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_p"] = frame["download_priority"].map(lambda value: priority_order.get(value, 9))
    frame = frame.sort_values(
        ["_p", "edition_language", "title", "author_name"],
        kind="stable",
    ).drop(columns=["_p"])
    return frame.reset_index(drop=True)


def _search_frame(union: pd.DataFrame) -> pd.DataFrame:
    if union.empty:
        return pd.DataFrame(columns=FEED_COLUMNS)
    # P0 always searchable; other tiers drop likely_exclude.
    mask = union["download_priority"].isin(SEARCH_PRIORITIES) & union["edition_language"].isin(
        ["en", "de"]
    )
    mask &= (union["download_priority"] == "P0") | (union["screen_status"] != "likely_exclude")
    return union.loc[mask, FEED_COLUMNS].copy().reset_index(drop=True)


REVIEW_SAMPLE_PER_TIER = 100


def _review_frame(union: pd.DataFrame, sample_per_tier: int = REVIEW_SAMPLE_PER_TIER) -> pd.DataFrame:
    """Colleague slice: all P0+P1, plus a stratified sample of P2 and P3."""
    if union.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    # Always keep P0 (seed), even if a title pattern falsely marks likely_exclude.
    p0 = union[union["download_priority"] == "P0"].copy()
    usable = union[union["screen_status"] != "likely_exclude"].copy()
    parts = [
        p0,
        usable[usable["download_priority"] == "P1"],
    ]
    for tier in ("P2", "P3"):
        tier_rows = usable[usable["download_priority"] == tier]
        if tier_rows.empty:
            continue
        if len(tier_rows) <= sample_per_tier:
            parts.append(tier_rows)
        else:
            # Stable sample: sort then take evenly spaced rows.
            tier_rows = tier_rows.sort_values(
                ["title", "author_name", "candidate_id"],
                kind="stable",
            ).reset_index(drop=True)
            step = max(len(tier_rows) / sample_per_tier, 1.0)
            indexes = sorted({min(int(i * step), len(tier_rows) - 1) for i in range(sample_per_tier)})
            parts.append(tier_rows.iloc[indexes])
    frame = pd.concat(parts, ignore_index=True) if parts else usable.iloc[0:0]
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_p"] = frame["download_priority"].map(lambda value: priority_order.get(value, 9))
    frame = frame.sort_values(["_p", "title", "author_name"], kind="stable").drop(columns=["_p"])
    return frame[REVIEW_COLUMNS].reset_index(drop=True)


def collect_all_rows(ol_isbn_index: dict | None = None) -> list[dict]:
    ol_isbn_index = ol_isbn_index if ol_isbn_index is not None else build_ol_isbn_index()
    rows: list[dict] = []
    rows.extend(load_seed(SEED_PATH))
    en_filtered = FILTERED / "filtered_english_corp.csv"
    de_filtered = FILTERED / "filtered_german_corp.csv"
    if en_filtered.is_file():
        rows.extend(load_filtered(en_filtered, "en", ol_isbn_index))
    if de_filtered.is_file():
        rows.extend(load_filtered(de_filtered, "de", ol_isbn_index))
    for lang, path in PRESCREEN_PATHS.items():
        rows.extend(load_prescreen(path, lang, ol_isbn_index))
    for lang, path in OL_SEARCH_PATHS.items():
        rows.extend(load_ol_search(path, lang, ol_isbn_index))
    rows.extend(load_dnb_hits())
    rows.extend(load_opensyllabus(ol_isbn_index=ol_isbn_index))
    return rows


def main():
    log = setup_logger(__file__)
    try:
        if not SEED_PATH.is_file():
            raise SystemExit(f"missing seed candidates: {SEED_PATH}")

        log.info("Building OL ISBN index for backfill…")
        ol_isbn_index = build_ol_isbn_index()
        log.info(f"OL ISBN index size: {len(ol_isbn_index)}")

        all_rows = collect_all_rows(ol_isbn_index)
        log.info(f"raw union rows before dedupe: {len(all_rows)}")

        by_lang, logs = merge_and_dedupe(all_rows)
        union_rows = by_lang["en"] + by_lang["de"]
        union = _to_union_frame(union_rows)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        union_path = OUT_DIR / "union_all.csv"
        union.to_csv(union_path, index=False)

        for lang in ("en", "de"):
            part = union[union["edition_language"] == lang] if not union.empty else union
            search = _search_frame(part)
            review = _review_frame(part)
            search_path = OUT_DIR / f"final_list_{lang}.csv"
            review_path = OUT_DIR / f"review_{lang}.csv"
            search.to_csv(search_path, index=False)
            review.to_csv(review_path, index=False)
            log.info(
                f"{lang}: union {len(part)}, search(P0-P2) {len(search)}, "
                f"review {len(review)} -> {search_path.name} / {review_path.name}"
            )

        log_df = pd.DataFrame(logs, columns=DEDUPE_COLUMNS)
        log_path = OUT_DIR / "dedupe_log.csv"
        log_df.to_csv(log_path, index=False)

        if not union.empty:
            pcounts = union["download_priority"].value_counts().to_dict()
            scounts = union["screen_status"].value_counts().to_dict()
            isbn_n = int((union["isbn_13"].astype(str).str.len() > 0).sum())
            log.info(f"priority counts: {pcounts}")
            log.info(f"screen counts: {scounts}")
            log.info(f"rows with isbn_13: {isbn_n}/{len(union)}")
        log.info(f"dedupe drops: {len(log_df)} -> {log_path}")
        log.info(f"union_all: {len(union)} -> {union_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
