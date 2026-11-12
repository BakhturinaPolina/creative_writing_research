"""Build a detailed biblio CSV for English primary + comparison books.

One row per file under data/raw/books/en/{primary,comparison}. Joins interim
catalogue sources so time-based exploration can use edition year, first-work
year, and provenance side by side.

  python -m src.publication.build_en_biblio_timeline

Writes:
  data/processed/en_primary_comparison_biblio.csv
  data/processed/en_primary_comparison_biblio.meta.json
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.normalize import normalize_isbn, normalize_title
from src.common.paths import DATA_INTERIM, DATA_PROCESSED, ROOT

MANIFEST = DATA_INTERIM / "fulltext" / "manifest.csv"
UNION = DATA_INTERIM / "final_lists" / "union_all.csv"
SEED = DATA_INTERIM / "catalogue" / "seed_candidates.csv"
OL_CAND = DATA_INTERIM / "catalogue" / "openlibrary_candidates.csv"
OL_SEARCH = DATA_INTERIM / "openlibrary" / "ol_search_en.csv"
FILTERED = DATA_INTERIM / "filtered_openlibrary" / "filtered_english_corp.csv"
OSP = DATA_INTERIM / "opensyllabus" / "opensyllabus_creative_writing_all.csv"
STATS = DATA_PROCESSED / "en_book_stats.csv"

OUT_CSV = DATA_PROCESSED / "en_primary_comparison_biblio.csv"
OUT_META = DATA_PROCESSED / "en_primary_comparison_biblio.meta.json"

YEAR_MIN, YEAR_MAX = 1500, 2030
ROLES = ("primary", "comparison")


def parse_year(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return None
    if re.fullmatch(r"\d{3,4}", text):
        year = int(text)
        return year if YEAR_MIN <= year <= YEAR_MAX else None
    m = re.search(r"(1[5-9]\d{2}|20[0-3]\d)", text)
    if not m:
        return None
    year = int(m.group(1))
    return year if YEAR_MIN <= year <= YEAR_MAX else None


def decade(year: int | None) -> int | None:
    return None if year is None or pd.isna(year) else int(year) // 10 * 10


def period_5y(year: int | None) -> str:
    if year is None or pd.isna(year):
        return ""
    y = int(year)
    start = y - (y % 5)
    return f"{start}-{start + 4}"


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _filled_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna()
    return _nonempty(series)


def _first_map(df: pd.DataFrame, key: str, value_cols: list[str]) -> pd.DataFrame:
    """Keep the richest row per key; return key + value columns."""
    work = df[_nonempty(df[key])].copy()
    work[key] = work[key].astype(str).str.strip()
    for col in value_cols:
        if col not in work.columns:
            work[col] = pd.NA
        elif col.startswith("year_") or col.endswith("_year") or "year" in col:
            work[col] = work[col].map(parse_year)
        else:
            work[col] = work[col].fillna("").astype(str).str.strip().replace({"": pd.NA})
    score = sum(_filled_mask(work[c]).astype(int) for c in value_cols)
    work = work.assign(_score=score).sort_values("_score", ascending=False)
    return work.drop_duplicates(key, keep="first")[[key, *value_cols]]


def _title_year_map(titles: pd.Series, years: pd.Series) -> dict[str, int]:
    out: dict[str, int] = {}
    for title, year in zip(titles, years):
        key = normalize_title(title)
        y = parse_year(year)
        if key and y is not None and key not in out:
            out[key] = y
    return out


def _pick_year(row: pd.Series, fields: list[tuple[str, str]]) -> tuple[int | None, str]:
    for col, source in fields:
        y = parse_year(row.get(col))
        if y is not None:
            return y, source
    return None, "missing"


def load_books() -> pd.DataFrame:
    """One row per file in en/primary and en/comparison (via manifest raw_path)."""
    manifest = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    role_from_path = manifest["raw_path"].str.extract(r"/en/(primary|comparison)/", expand=False)
    books = manifest[role_from_path.isin(ROLES)].copy()
    books["path_role"] = role_from_path[role_from_path.isin(ROLES)]
    # Prefer the folder label when present; fall back to corpus_role.
    books["corpus_role"] = books["path_role"].fillna(books["corpus_role"])
    books["title_best"] = books["title"].where(_nonempty(books["title"]), books["file_title"])
    books["author_best"] = books["author_name"].where(
        _nonempty(books["author_name"]), books["file_author"]
    )
    books["title_norm"] = books["title_best"].map(normalize_title)
    return books.reset_index(drop=True)


def attach_union(books: pd.DataFrame) -> pd.DataFrame:
    union = pd.read_csv(UNION, dtype=str, low_memory=False)
    cols = [
        "candidate_id",
        "title",
        "author_name",
        "publication_year",
        "isbn_13",
        "isbn_10",
        "edition_language",
        "subtitle",
        "source_tags",
        "screen_status",
        "screen_rule",
        "match_confidence",
        "download_priority",
        "ol_work_id",
        "ol_edition_id",
        "dnb_id",
        "md5",
        "search_status",
        "found_format",
        "search_date",
    ]
    by_cid = _first_map(union, "candidate_id", [c for c in cols if c != "candidate_id"])
    out = books.merge(by_cid, on="candidate_id", how="left", suffixes=("", "_union"))

    # Fill identity fields from union when manifest left them blank.
    for col in [
        "title",
        "author_name",
        "publication_year",
        "isbn_13",
        "isbn_10",
        "edition_language",
        "subtitle",
        "ol_work_id",
        "ol_edition_id",
        "source_tags",
        "screen_status",
        "screen_rule",
        "match_confidence",
        "download_priority",
        "dnb_id",
        "search_status",
        "found_format",
        "search_date",
    ]:
        src = f"{col}_union"
        if src in out.columns:
            out[col] = out[col].where(_nonempty(out[col]), out[src])
            out.drop(columns=[src], inplace=True)
        elif col not in out.columns and col in by_cid.columns:
            out[col] = out.get(col, "")

    # md5-only matches for rows without candidate_id
    missing = ~_nonempty(out["candidate_id"])
    if missing.any():
        by_md5 = _first_map(union, "md5", [c for c in cols if c != "md5"])
        fill = out.loc[missing, ["md5"]].merge(by_md5, on="md5", how="left")
        for col in by_md5.columns:
            if col == "md5":
                continue
            out.loc[missing, col] = out.loc[missing, col].where(
                _nonempty(out.loc[missing, col]), fill[col].values
            )
    return out


def attach_seed(books: pd.DataFrame) -> pd.DataFrame:
    seed = pd.read_csv(SEED, dtype=str, low_memory=False).copy()
    seed["seed_candidate_id"] = "seed:" + seed["candidate_id"].astype(str)
    seed["year_seed_edition"] = seed["edition_year"].map(parse_year)
    seed["year_seed_first_work"] = seed["first_work_year"].map(parse_year)
    seed["year_seed_original"] = seed.get("original_year", pd.Series(dtype=str)).map(parse_year)
    seed["publisher_seed"] = seed.get("publisher", "")
    seed["subjects_seed"] = seed.get("subjects", "")
    seed["subtitle_seed"] = seed.get("subtitle", "")
    seed["isbn_13_seed"] = seed.get("isbn_13", "").map(
        lambda v: normalize_isbn(v) if isinstance(v, str) else None
    )
    seed["isbn_10_seed"] = seed.get("isbn_10", "")
    seed["ol_work_id_seed"] = seed.get("ol_work_id", "")
    seed["ol_edition_id_seed"] = seed.get("ol_edition_id", "")
    seed["period_scope_seed"] = seed.get("period_scope", "")

    seed_cols = [
        "seed_candidate_id",
        "year_seed_edition",
        "year_seed_first_work",
        "year_seed_original",
        "publisher_seed",
        "subjects_seed",
        "subtitle_seed",
        "isbn_13_seed",
        "isbn_10_seed",
        "ol_work_id_seed",
        "ol_edition_id_seed",
        "period_scope_seed",
    ]
    by_cid = seed[seed_cols].drop_duplicates("seed_candidate_id", keep="first")
    out = books.merge(
        by_cid,
        left_on="candidate_id",
        right_on="seed_candidate_id",
        how="left",
    )
    out.drop(columns=["seed_candidate_id"], inplace=True, errors="ignore")

    # Also match seed rows by ISBN-13 for non-seed candidate_ids.
    seed_isbn = (
        seed[seed_cols]
        .dropna(subset=["isbn_13_seed"])
        .drop_duplicates("isbn_13_seed", keep="first")
        .drop(columns=["seed_candidate_id"])
        .set_index("isbn_13_seed")
    )
    out["_isbn"] = out["isbn_13"].map(
        lambda v: normalize_isbn(v) if isinstance(v, str) else None
    )
    need_idx = out.index[out["year_seed_first_work"].isna() & out["_isbn"].notna()]
    for idx in need_idx:
        isbn = out.at[idx, "_isbn"]
        if isbn not in seed_isbn.index:
            continue
        row = seed_isbn.loc[isbn]
        for col in seed_isbn.columns:
            cur = out.at[idx, col]
            empty = pd.isna(cur) or (isinstance(cur, str) and not str(cur).strip())
            if empty:
                out.at[idx, col] = row[col]
    out.drop(columns=["_isbn"], inplace=True)
    return out


def attach_ol_cand(books: pd.DataFrame) -> pd.DataFrame:
    ol = pd.read_csv(OL_CAND, dtype=str, low_memory=False)
    ol["year_ol_cand_edition"] = ol["edition_year"].map(parse_year)
    keep = _first_map(
        ol,
        "ol_work_id",
        ["year_ol_cand_edition", "year_raw", "ol_edition_id", "isbn_13", "isbn_10", "subtitle"],
    ).rename(
        columns={
            "year_raw": "year_ol_cand_raw",
            "ol_edition_id": "ol_edition_id_olcand",
            "isbn_13": "isbn_13_olcand",
            "isbn_10": "isbn_10_olcand",
            "subtitle": "subtitle_olcand",
        }
    )
    return books.merge(keep, on="ol_work_id", how="left")


def _ensure_title_norm(books: pd.DataFrame) -> pd.DataFrame:
    out = books.copy()
    if "title_best" not in out.columns:
        title = out["title"] if "title" in out.columns else out.get("file_title", "")
        out["title_best"] = title
    out["title_norm"] = out["title_best"].map(normalize_title)
    return out


def attach_ol_search(books: pd.DataFrame) -> pd.DataFrame:
    ol = pd.read_csv(OL_SEARCH, dtype=str, low_memory=False)
    ol["year_ol_search_first_publish"] = ol["publication_year"].map(parse_year)
    keep = _first_map(
        ol,
        "ol_work_id",
        [
            "year_ol_search_first_publish",
            "publisher",
            "edition_count",
            "subjects",
            "isbn_13",
            "isbn_10",
            "subtitle",
        ],
    ).rename(
        columns={
            "publisher": "publisher_ol_search",
            "edition_count": "ol_edition_count",
            "subjects": "subjects_ol_search",
            "isbn_13": "isbn_13_ol_search",
            "isbn_10": "isbn_10_ol_search",
            "subtitle": "subtitle_ol_search",
        }
    )
    out = _ensure_title_norm(books).merge(keep, on="ol_work_id", how="left")
    out["year_ol_search_first_publish"] = pd.to_numeric(
        out["year_ol_search_first_publish"], errors="coerce"
    ).astype("Int64")
    out["year_ol_search_match"] = pd.Series(pd.NA, index=out.index, dtype="string")
    via_work = out["year_ol_search_first_publish"].notna()
    out.loc[via_work, "year_ol_search_match"] = "ol_work_id"

    # Title fallback for rows still missing first-publish year.
    title_map = _title_year_map(ol["title"], ol["publication_year"])
    miss = out["year_ol_search_first_publish"].isna()
    filled = out.loc[miss, "title_norm"].map(title_map)
    out.loc[miss, "year_ol_search_first_publish"] = pd.array(filled, dtype="Int64")
    out.loc[miss & filled.notna(), "year_ol_search_match"] = "title"
    return out


def attach_filtered_ol(books: pd.DataFrame) -> pd.DataFrame:
    filt = pd.read_csv(FILTERED, dtype=str, sep=";", engine="python")
    title_map = _title_year_map(filt["title"], filt["year"])
    books = _ensure_title_norm(books)
    books["year_filtered_ol"] = books["title_norm"].map(title_map)
    return books


def attach_opensyllabus(books: pd.DataFrame) -> pd.DataFrame:
    osp = pd.read_csv(OSP, dtype=str, low_memory=False)
    books_only = osp[osp["publication_type"].fillna("").str.lower().eq("book")].copy()
    title_map = _title_year_map(books_only["title"], books_only["year"])
    books = _ensure_title_norm(books)
    books["year_opensyllabus"] = books["title_norm"].map(title_map)

    # Citation / syllabus signal when title matches.
    osp_meta = books_only.copy()
    osp_meta["title_norm"] = osp_meta["title"].map(normalize_title)
    osp_meta = osp_meta[_nonempty(osp_meta["title_norm"])].drop_duplicates("title_norm", keep="first")
    meta = osp_meta.set_index("title_norm")[["citation_count", "publisher", "year"]].rename(
        columns={
            "citation_count": "opensyllabus_citation_count",
            "publisher": "publisher_opensyllabus",
            "year": "year_opensyllabus_raw",
        }
    )
    return books.merge(meta, left_on="title_norm", right_index=True, how="left")


def attach_stats(books: pd.DataFrame) -> pd.DataFrame:
    if not STATS.exists():
        return books
    stats = pd.read_csv(STATS, dtype=str, low_memory=False)
    keep = [
        "book_id",
        "edition_year",
        "year_source",
        "candidate_source",
        "identity_check",
        "in_analysis_set",
        "possible_duplicate",
        "work_group",
        "role_conflict",
        "surplus_duplicate",
        "needs_ocr",
        "short_text",
        "n_words",
        "body_words",
        "exclusion_reason",
        "corpus_type",
    ]
    keep = [c for c in keep if c in stats.columns]
    stats = stats[keep].rename(
        columns={
            "edition_year": "year_stats_cleaned",
            "year_source": "year_stats_source",
            "candidate_source": "candidate_source_stats",
            "n_words": "n_words_stats",
            "body_words": "body_words_stats",
            "exclusion_reason": "exclusion_reason_stats",
            "corpus_type": "corpus_type_stats",
        }
    )
    return books.merge(stats, on="book_id", how="left")


def coalesce_fields(books: pd.DataFrame) -> pd.DataFrame:
    out = books.copy()
    out["year_manifest"] = out["publication_year"].map(parse_year)

    # Suggested edition year: prefer catalogue match tied to the file, then other sources.
    edition_fields = [
        ("year_manifest", "union_or_manifest"),
        ("year_seed_edition", "seed"),
        ("year_ol_cand_edition", "openlibrary_candidates"),
        ("year_filtered_ol", "filtered_openlibrary_title"),
        ("year_opensyllabus", "opensyllabus_title"),
        ("year_stats_cleaned", "en_book_stats"),
    ]
    picked = out.apply(lambda r: _pick_year(r, edition_fields), axis=1, result_type="expand")
    out["edition_year"] = picked[0].astype("Int64")
    out["edition_year_source"] = picked[1]

    # If stats dropped the year for identity reasons, keep that warning visible.
    if "identity_check" in out.columns:
        bad = out["identity_check"].isin(["renamed", "unresolved"]) & out["year_manifest"].notna()
        # Still report the raw year, but mark that the cleaned analysis year may differ.
        out.loc[bad, "edition_year_identity_warning"] = out.loc[bad, "identity_check"]
    else:
        out["edition_year_identity_warning"] = ""

    # Suggested first-work year: seed first, then OL-search only when joined by
    # ol_work_id (title-only matches stay in year_ol_search_first_publish for audit
    # but are too collision-prone for common titles like "The creative writer").
    out["year_ol_search_first_publish_linked"] = out["year_ol_search_first_publish"].where(
        out.get("year_ol_search_match", pd.Series(index=out.index)).eq("ol_work_id")
    )
    first_fields = [
        ("year_seed_first_work", "seed"),
        ("year_ol_search_first_publish_linked", "ol_search"),
        ("year_seed_original", "seed_original"),
    ]
    picked_first = out.apply(lambda r: _pick_year(r, first_fields), axis=1, result_type="expand")
    out["first_work_year"] = picked_first[0].astype("Int64")
    out["first_work_year_source"] = picked_first[1]

    out["year_gap_edition_minus_first"] = (
        out["edition_year"].astype("Float64") - out["first_work_year"].astype("Float64")
    ).astype("Int64")
    out["decade_edition"] = out["edition_year"].map(decade).astype("Int64")
    out["decade_first_work"] = out["first_work_year"].map(decade).astype("Int64")
    out["period_5y_edition"] = out["edition_year"].map(period_5y)
    out["period_5y_first_work"] = out["first_work_year"].map(period_5y)

    # Publisher / ISBN / subtitle coalesce
    out["publisher"] = ""
    for col, source in [
        ("publisher_seed", "seed"),
        ("publisher_ol_search", "ol_search"),
        ("publisher_opensyllabus", "opensyllabus"),
    ]:
        if col not in out.columns:
            continue
        take = ~_nonempty(out["publisher"]) & _nonempty(out[col])
        out.loc[take, "publisher"] = out.loc[take, col]
        out.loc[take, "publisher_source"] = source
    if "publisher_source" not in out.columns:
        out["publisher_source"] = ""

    for dest, sources in {
        "isbn_13": ["isbn_13", "isbn_13_seed", "isbn_13_ol_search", "isbn_13_olcand"],
        "isbn_10": ["isbn_10", "isbn_10_seed", "isbn_10_ol_search", "isbn_10_olcand"],
        "subtitle": ["subtitle", "subtitle_seed", "subtitle_ol_search", "subtitle_olcand"],
        "subjects": ["subjects_seed", "subjects_ol_search"],
        "ol_work_id": ["ol_work_id", "ol_work_id_seed"],
        "ol_edition_id": ["ol_edition_id", "ol_edition_id_seed", "ol_edition_id_olcand"],
    }.items():
        if dest not in out.columns:
            out[dest] = ""
        for col in sources:
            if col == dest or col not in out.columns:
                continue
            take = ~_nonempty(out[dest]) & _nonempty(out[col])
            out.loc[take, dest] = out.loc[take, col]

    out["has_edition_year"] = out["edition_year"].notna()
    out["has_first_work_year"] = out["first_work_year"].notna()
    out["has_both_years"] = out["has_edition_year"] & out["has_first_work_year"]
    return out


def finalize(books: pd.DataFrame) -> pd.DataFrame:
    # Prefer catalogue title/author when present.
    title = books["title"].where(_nonempty(books["title"]), books["title_best"])
    author = books["author_name"].where(_nonempty(books["author_name"]), books["author_best"])

    out = pd.DataFrame({
        # identity
        "book_id": books["book_id"],
        "md5": books["md5"],
        "corpus_role": books["corpus_role"],
        "folder_lang": books["folder_lang"],
        "lang": books["lang"],
        "title": title,
        "author": author,
        "file_title": books["file_title"],
        "file_author": books["file_author"],
        "subtitle": books.get("subtitle", ""),
        "original_filename": books["original_filename"],
        "raw_path": books["raw_path"],
        "format": books["format"],
        "size_mb": books["size_mb"],
        "n_words": books.get("n_words", books.get("n_words_stats", "")),
        "n_chapters": books.get("n_chapters", ""),
        # catalogue identity
        "candidate_id": books["candidate_id"],
        "in_candidate_lists": books["in_candidate_lists"],
        "source_tags": books.get("source_tags", ""),
        "download_priority": books.get("download_priority", ""),
        "screen_status": books.get("screen_status", ""),
        "screen_rule": books.get("screen_rule", ""),
        "match_confidence": books.get("match_confidence", ""),
        "candidate_source_stats": books.get("candidate_source_stats", ""),
        # identifiers
        "isbn_13": books.get("isbn_13", ""),
        "isbn_10": books.get("isbn_10", ""),
        "ol_work_id": books.get("ol_work_id", ""),
        "ol_edition_id": books.get("ol_edition_id", ""),
        "dnb_id": books.get("dnb_id", ""),
        "ol_edition_count": books.get("ol_edition_count", ""),
        # suggested time fields
        "edition_year": books["edition_year"],
        "edition_year_source": books["edition_year_source"],
        "edition_year_identity_warning": books.get("edition_year_identity_warning", ""),
        "first_work_year": books["first_work_year"],
        "first_work_year_source": books["first_work_year_source"],
        "year_gap_edition_minus_first": books["year_gap_edition_minus_first"],
        "decade_edition": books["decade_edition"],
        "decade_first_work": books["decade_first_work"],
        "period_5y_edition": books["period_5y_edition"],
        "period_5y_first_work": books["period_5y_first_work"],
        "has_edition_year": books["has_edition_year"],
        "has_first_work_year": books["has_first_work_year"],
        "has_both_years": books["has_both_years"],
        # per-source years (for auditing / sensitivity)
        "year_manifest": books["year_manifest"],
        "year_seed_edition": books.get("year_seed_edition"),
        "year_seed_first_work": books.get("year_seed_first_work"),
        "year_seed_original": books.get("year_seed_original"),
        "year_ol_cand_edition": books.get("year_ol_cand_edition"),
        "year_ol_search_first_publish": books.get("year_ol_search_first_publish"),
        "year_ol_search_match": books.get("year_ol_search_match", ""),
        "year_filtered_ol": books.get("year_filtered_ol"),
        "year_opensyllabus": books.get("year_opensyllabus"),
        "year_stats_cleaned": books.get("year_stats_cleaned"),
        "year_stats_source": books.get("year_stats_source", ""),
        # other biblio
        "publisher": books.get("publisher", ""),
        "publisher_source": books.get("publisher_source", ""),
        "edition_language": books.get("edition_language", ""),
        "subjects": books.get("subjects", ""),
        "period_scope_seed": books.get("period_scope_seed", ""),
        "opensyllabus_citation_count": books.get("opensyllabus_citation_count", ""),
        "search_status": books.get("search_status", ""),
        "found_format": books.get("found_format", ""),
        "search_date": books.get("search_date", ""),
        # analysis flags (from en_book_stats when available)
        "identity_check": books.get("identity_check", ""),
        "in_analysis_set": books.get("in_analysis_set", ""),
        "possible_duplicate": books.get("possible_duplicate", ""),
        "work_group": books.get("work_group", ""),
        "role_conflict": books.get("role_conflict", ""),
        "surplus_duplicate": books.get("surplus_duplicate", ""),
        "needs_ocr": books.get("needs_ocr", ""),
        "short_text": books.get("short_text", ""),
        "exclusion_reason": books.get("exclusion_reason_stats", books.get("exclusion_reason", "")),
        "corpus_type": books.get("corpus_type_stats", books.get("corpus_type", "")),
        "title_match": books.get("title_match", ""),
        "translation_status": books.get("translation_status", ""),
        "original_language": books.get("original_language", ""),
        "role_source": books.get("role_source", ""),
        "prescreen_role": books.get("prescreen_role", ""),
        "prescreen_reason": books.get("prescreen_reason", ""),
    })

    # Stable sort for diffs
    return out.sort_values(["corpus_role", "edition_year", "title", "book_id"], na_position="last").reset_index(drop=True)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build() -> pd.DataFrame:
    log = setup_logger("build_en_biblio_timeline")
    books = load_books()
    log.info(f"Loaded {len(books)} books from manifest paths "
             f"{books['corpus_role'].value_counts().to_dict()}")

    books = attach_union(books)
    log.info(f"Union join: candidate_id filled "
             f"{_nonempty(books['candidate_id']).sum()} / {len(books)}; "
             f"ol_work_id {_nonempty(books['ol_work_id']).sum()}")

    books = attach_seed(books)
    log.info(f"Seed first_work_year: {books['year_seed_first_work'].notna().sum()}")

    books = attach_ol_cand(books)
    log.info(f"OL-cand edition year: {books['year_ol_cand_edition'].notna().sum()}")

    books = attach_ol_search(books)
    log.info(f"OL-search first_publish: {books['year_ol_search_first_publish'].notna().sum()}")

    books = attach_filtered_ol(books)
    log.info(f"Filtered-OL title year: {books['year_filtered_ol'].notna().sum()}")

    books = attach_opensyllabus(books)
    log.info(f"OpenSyllabus title year: {books['year_opensyllabus'].notna().sum()}")

    books = attach_stats(books)
    books = coalesce_fields(books)
    out = finalize(books)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "n_rows": int(len(out)),
        "n_primary": int((out["corpus_role"] == "primary").sum()),
        "n_comparison": int((out["corpus_role"] == "comparison").sum()),
        "n_edition_year": int(out["has_edition_year"].sum()),
        "n_first_work_year": int(out["has_first_work_year"].sum()),
        "n_both_years": int(out["has_both_years"].sum()),
        "edition_year_sources": out["edition_year_source"].value_counts(dropna=False).to_dict(),
        "first_work_year_sources": out["first_work_year_source"].value_counts(dropna=False).to_dict(),
        "edition_year_min": int(out["edition_year"].min()) if out["has_edition_year"].any() else None,
        "edition_year_max": int(out["edition_year"].max()) if out["has_edition_year"].any() else None,
        "sources": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "union_all": str(UNION.relative_to(ROOT)),
            "seed_candidates": str(SEED.relative_to(ROOT)),
            "openlibrary_candidates": str(OL_CAND.relative_to(ROOT)),
            "ol_search_en": str(OL_SEARCH.relative_to(ROOT)),
            "filtered_english_corp": str(FILTERED.relative_to(ROOT)),
            "opensyllabus": str(OSP.relative_to(ROOT)),
            "en_book_stats": str(STATS.relative_to(ROOT)) if STATS.exists() else None,
        },
        "output": str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info(
        f"Wrote {OUT_CSV.relative_to(ROOT)} "
        f"({meta['n_edition_year']} edition years, "
        f"{meta['n_first_work_year']} first-work years)"
    )
    log.close()
    return out


def main() -> None:
    build()


if __name__ == "__main__":
    main()
