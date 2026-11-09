"""Build data/interim/fulltext/manifest.csv: one row per book file.

For every file under data/interim/creative_writing/ (and data/raw/books/ once
moved): MD5 check, format, size, language detection on a body-text sample,
PDF text-layer triage, and a join to the candidate lists on md5.

Re-running keeps book_id and every downstream column (labels, extraction
status) from the existing manifest; only inventory columns are recomputed.

Usage:
  python -m src.fulltext.inventory              # build / refresh manifest
  python -m src.fulltext.inventory --move       # also move originals to data/raw/books/{lang}/{role}/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, ROOT
from src.fulltext.books import (
    CORPUS_LANGS,
    EBOOK_EXTS,
    FULLTEXT_DIR,
    MANIFEST_PATH,
    RAW_BOOKS,
    detect_format,
    detect_language,
    discover_books,
    ebook_sample,
    md5_of,
    normalize_text,
    parse_filename,
    pdf_page_texts,
    raw_book_path,
    word_count,
    write_sample,
)

UNION_PATH = DATA_INTERIM / "final_lists" / "union_all.csv"
UNION_COLS = ["candidate_id", "title", "author_name", "publication_year", "screen_status",
              "screen_rule", "download_priority", "edition_language"]

INVENTORY_COLS = [
    "book_id", "md5", "md5_verified", "original_filename", "original_path", "raw_path",
    "file_title", "file_author", "folder_lang", "lang_detected", "lang_confidence", "lang",
    "lang_source", "file_ext", "format", "size_mb", "n_pages", "text_layer", "words_per_page",
    "empty_page_share", "clean_token_share", "sample_words", "in_candidate_lists",
    *UNION_COLS,
]
DOWNSTREAM_COLS = [
    "title_match", "content_dup_of", "corpus_type", "corpus_role", "role_source", "prescreen_role", "prescreen_reason",
    "translation_status", "original_language", "exclusion_reason", "work_id",
    "extraction_round", "extraction_status", "n_words", "n_chapters",
]
MANIFEST_COLS = INVENTORY_COLS + DOWNSTREAM_COLS

CLEAN_TOKEN = re.compile(r"^[^\W\d_]{2,}$", re.UNICODE)
EMPTY_PAGE_WORDS = 30
LANG_MIN_CONFIDENCE = 0.5


def pdf_triage(pages: list[str]) -> dict:
    """Classify the text layer as none / poor / good from sampled pages."""
    if not pages:
        return {"text_layer": "none", "words_per_page": 0, "empty_page_share": 1.0, "clean_token_share": 0.0}
    counts = [word_count(p) for p in pages]
    tokens = [t for p in pages for t in p.split()]
    clean = sum(1 for t in tokens if CLEAN_TOKEN.match(t.strip(".,;:!?\"'()[]«»„“”‘’-–—")))
    wpp = sorted(counts)[len(counts) // 2]
    empty = sum(1 for c in counts if c < EMPTY_PAGE_WORDS) / len(counts)
    clean_share = clean / len(tokens) if tokens else 0.0
    if wpp < 20 or empty > 0.8:
        layer = "none"
    elif clean_share < 0.75 or empty > 0.3:
        layer = "poor"
    else:
        layer = "good"
    return {"text_layer": layer, "words_per_page": wpp, "empty_page_share": round(empty, 2),
            "clean_token_share": round(clean_share, 2)}


def load_union() -> dict[str, dict]:
    frame = pd.read_csv(UNION_PATH, dtype=str, encoding="utf-8").fillna("")
    frame["md5"] = frame["md5"].str.strip().str.lower()
    frame = frame[frame["md5"] != ""].sort_values("download_priority")
    frame = frame.drop_duplicates("md5", keep="first")
    return {row["md5"]: row for row in frame[["md5", *UNION_COLS]].to_dict(orient="records")}


def load_existing() -> dict[str, dict]:
    if not MANIFEST_PATH.is_file():
        return {}
    frame = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
    return {row["md5"]: row for row in frame.to_dict(orient="records")}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def inspect(path: Path, folder_lang: str, log) -> tuple[dict, str]:
    """Return inventory fields and the text sample for one file."""
    info = parse_filename(path.name)
    fmt = detect_format(path)
    row = {
        **info,
        "format": fmt,
        "original_filename": path.name,
        "folder_lang": folder_lang,
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }
    sample = ""
    try:
        if fmt == "pdf":
            n_pages, pages = pdf_page_texts(path)
            row["n_pages"] = n_pages
            row.update(pdf_triage(pages))
            sample = normalize_text(" ".join(pages))
        elif fmt in EBOOK_EXTS:
            sample = ebook_sample(path, fmt)
        else:
            log.info(f"format {fmt} not sampled: {path.name}")
    except Exception as exc:  # malformed files are recorded, not fatal
        log.error(f"read failed: {path.name}: {exc}")
        row["text_layer"] = "unreadable" if fmt == "pdf" else ""
    row["sample_words"] = word_count(sample)
    lang, conf = detect_language(sample)
    row["lang_detected"], row["lang_confidence"] = lang, conf
    return row, sample


def assign_lang(row: dict) -> None:
    detected, conf, folder = row["lang_detected"], float(row["lang_confidence"] or 0), row["folder_lang"]
    if detected and conf >= LANG_MIN_CONFIDENCE:
        row["lang"] = detected if detected in CORPUS_LANGS else "other"
        row["lang_source"] = "detected"
    else:
        row["lang"] = folder or "other"
        row["lang_source"] = "folder"


def move_to_raw(path: Path, row: dict, log) -> Path:
    title = row.get("title") or row.get("file_title") or ""
    author = row.get("author_name") or row.get("file_author") or ""
    dest = raw_book_path(
        row["lang"], row.get("corpus_role"), title, author, row["md5"], row["format"],
    )
    if path.resolve() == dest.resolve():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.error(f"destination exists, not moving: {path.name}")
        return path
    path.rename(dest)
    if md5_of(dest) != row["md5"]:
        dest.rename(path)
        raise RuntimeError(f"hash mismatch after move, restored: {path.name}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--move", action="store_true", help="move originals to data/raw/books/{lang}/")
    args = parser.parse_args()

    log = setup_logger(__file__)
    union = load_union()
    existing = load_existing()
    books = discover_books()
    log.info(f"{len(books)} book files found; {len(existing)} rows in existing manifest")

    rows = []
    for i, (path, folder_lang) in enumerate(books, 1):
        md5 = parse_filename(path.name)["md5"] or path.stem
        old = existing.get(md5, {})
        if old and old.get("raw_path") == rel(path):
            row = dict(old)  # already inventoried and moved; nothing to recompute
        else:
            row, sample = inspect(path, folder_lang or old.get("folder_lang", ""), log)
            row["md5"] = md5
            actual = md5_of(path)
            row["md5_verified"] = actual == md5
            if actual != md5:
                log.error(f"md5 mismatch: {path.name} (file hash {actual})")
            assign_lang(row)
            row["book_id"] = old.get("book_id") or f"{row['lang']}_{md5[:12]}"
            row["original_path"] = old.get("original_path") or rel(path)
            write_sample(row["book_id"], sample)
            for col in DOWNSTREAM_COLS:
                row[col] = old.get(col, "")
        hit = union.get(md5)
        row["in_candidate_lists"] = bool(hit)
        for col in UNION_COLS:
            row[col] = hit[col] if hit else ""
        if args.move:
            row["raw_path"] = rel(move_to_raw(path, row, log))
        else:
            row["raw_path"] = row.get("raw_path") or ""
        rows.append(row)
        if i % 50 == 0:
            log.info(f"  {i}/{len(books)}")

    frame = pd.DataFrame(rows).reindex(columns=MANIFEST_COLS).fillna("")
    frame = frame.sort_values(["lang", "format", "book_id"])
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    log.info(f"wrote {MANIFEST_PATH} ({len(frame)} rows)")
    log.info("by lang x format:\n" + frame.groupby(["lang", "format"]).size().to_string())
    mism = frame[(frame["folder_lang"] != "") & (frame["lang"] != frame["folder_lang"])]
    log.info(f"language differs from folder: {len(mism)}")
    log.info("PDF text layer:\n" + frame[frame["format"] == "pdf"]["text_layer"].value_counts().to_string())
    log.info(f"in candidate lists: {(frame['in_candidate_lists'].astype(str) == 'True').sum()}")
    log.close()


if __name__ == "__main__":
    main()
