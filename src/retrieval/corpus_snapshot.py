"""Freeze a versioned corpus snapshot + book/passage manifests (no source mutation).

Snapshot is labelled a *candidate* while cleaning remains ongoing.
Full texts stay private; manifests store pointers and cryptographic hashes only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.paragraphs import CorpusScanStats, iter_body_paragraphs, load_book_stats
from src.retrieval.run_meta import (
    DEFAULT_CORPUS_SNAPSHOT_ID,
    base_run_meta,
    file_sha256,
    peak_rss_mb,
    write_json,
)
from src.retrieval.textnorm import normalize_text, tokenize

JSON_ROOT = DATA_INTERIM / "fulltext" / "json"
MIN_PARA_CHARS = 40
EXTREME_CHARS = 100_000

BOOK_FIELDS = [
    "corpus_snapshot_id",
    "language",
    "book_id",
    "work_id",
    "title",
    "author",
    "publication_year",
    "edition_year",
    "corpus_role",
    "primary_group",
    "source_path",
    "source_file_hash",
    "number_of_words",
    "number_of_eligible_paragraphs",
    "processing_status",
    "in_analysis_set",
    "qc_notes",
]

PASSAGE_FIELDS = [
    "corpus_snapshot_id",
    "passage_id",
    "book_id",
    "work_id",
    "language",
    "chapter_idx",
    "para_idx",
    "source_path",
    "word_count",
    "character_count",
    "token_count",
    "raw_content_hash",
    "normalized_content_hash",
    "segmentation_status",
    "corpus_role",
    "primary_group",
    "qc_flags",
]

LINEAGE_FIELDS = [
    "corpus_snapshot_id",
    "previous_passage_id",
    "new_passage_id",
    "previous_content_hash",
    "new_content_hash",
    "change_type",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _qc_passage(text: str, lang: str) -> tuple[str, str]:
    flags = []
    status = "ok"
    n = len(text)
    if n == 0:
        flags.append("empty")
        status = "empty"
    elif n < MIN_PARA_CHARS:
        flags.append("below_min_chars")
        status = "short"
    if n >= EXTREME_CHARS:
        flags.append("extreme_length")
        status = "extreme_length"
    # Crude OCR / spacing heuristics
    if "  " in text or " - " in text:
        flags.append("possible_ocr_spacing")
    if text.count("\ufffd") > 0:
        flags.append("replacement_char")
    return status, ";".join(flags)


def build_book_manifest(lang: str, snapshot_id: str, log) -> tuple[Path, dict]:
    stats = load_book_stats(lang)
    root = JSON_ROOT / lang
    out = DATA_PROCESSED / f"corpus_manifest_{lang}_v1_candidate.csv"
    qc_issues = {
        "missing_files": [],
        "unreadable_files": [],
        "duplicate_book_ids": [],
        "missing_metadata": [],
        "possible_duplicate_works": [],
    }
    # Duplicate book_id check
    dup = stats["book_id"].value_counts()
    for bid, n in dup.items():
        if int(n) > 1:
            qc_issues["duplicate_book_ids"].append(str(bid))

    rows = []
    para_counts: dict[str, int] = {}
    # Count eligible paragraphs via a dedicated stream (analysis set only for counts
    # used in retrieval; still list ALL books in stats with status).
    scan = CorpusScanStats()
    for para in iter_body_paragraphs(lang, analysis_only=True, scan_stats=scan):
        para_counts[para.book_id] = para_counts.get(para.book_id, 0) + 1

    seen_ids: set[str] = set()
    for _, book in stats.iterrows():
        bid = str(book["book_id"])
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        source = root / f"{bid}.json"
        notes = []
        status = "ok"
        fhash = ""
        if not source.is_file():
            status = "missing_file"
            qc_issues["missing_files"].append(bid)
            notes.append("missing_json")
        else:
            try:
                fhash = file_sha256(source)
                # Quick JSON parse check
                json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                status = "unreadable"
                qc_issues["unreadable_files"].append(bid)
                notes.append(f"unreadable:{exc}")
        for field in ("title", "author"):
            if not str(book.get(field) or "").strip():
                notes.append(f"missing_{field}")
                qc_issues["missing_metadata"].append(f"{bid}:{field}")
        if str(book.get("possible_duplicate") or "").lower() in ("true", "1", "yes"):
            qc_issues["possible_duplicate_works"].append(bid)
            notes.append("possible_duplicate")
        if str(book.get("role_conflict") or "").lower() in ("true", "1", "yes"):
            notes.append("role_conflict")

        work_id = str(book.get("work_id") or book.get("work_group") or "")
        edition_year = str(book.get("edition_year") or "")
        pub_year = edition_year  # publication_year not always separate
        in_analysis = str(book.get("in_analysis_set")).lower() in ("true", "1")
        n_words = book.get("body_words") or book.get("n_words") or ""
        rows.append(
            {
                "corpus_snapshot_id": snapshot_id,
                "language": lang,
                "book_id": bid,
                "work_id": work_id,
                "title": book.get("title") or "",
                "author": book.get("author") or "",
                "publication_year": pub_year,
                "edition_year": edition_year,
                "corpus_role": book.get("corpus_role") or "",
                "primary_group": book.get("primary_group") or "",
                "source_path": f"data/interim/fulltext/json/{lang}/{bid}.json",
                "source_file_hash": fhash,
                "number_of_words": n_words,
                "number_of_eligible_paragraphs": para_counts.get(bid, 0) if in_analysis else "",
                "processing_status": status,
                "in_analysis_set": in_analysis,
                "qc_notes": ";".join(notes),
            }
        )

    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BOOK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = base_run_meta(
        kind="corpus_manifest",
        lang=lang,
        corpus_snapshot_id=snapshot_id,
        n_books=len(rows),
        n_analysis_books=int(sum(1 for r in rows if r["in_analysis_set"])),
        qc_issues={k: v for k, v in qc_issues.items() if v},
        expected_books=scan.expected_books,
        processed_books=scan.processed_books,
        skipped_missing=scan.skipped_missing,
        skipped_unreadable=scan.skipped_unreadable,
        candidate_label=True,
        note="Candidate snapshot while corpus cleaning ongoing; sources not modified.",
    )
    write_json(DATA_PROCESSED / f"corpus_manifest_{lang}_v1_candidate.meta.json", meta)
    log.info(f"wrote {out} ({len(rows)} books)")
    return out, meta


def build_passage_manifest(lang: str, snapshot_id: str, log) -> Path:
    stats = load_book_stats(lang)
    work_by_book = {}
    if "work_id" in stats.columns:
        for _, row in stats.iterrows():
            work_by_book[str(row["book_id"])] = str(row.get("work_id") or "")
    elif "work_group" in stats.columns:
        for _, row in stats.iterrows():
            work_by_book[str(row["book_id"])] = str(row.get("work_group") or "")

    scan = CorpusScanStats()
    out = DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv"
    seen_ids: set[str] = set()
    dup_ids = []
    extreme = []
    n = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PASSAGE_FIELDS)
        w.writeheader()
        for i, para in enumerate(iter_body_paragraphs(lang, scan_stats=scan)):
            if para.passage_id in seen_ids:
                dup_ids.append(para.passage_id)
            seen_ids.add(para.passage_id)
            seg, flags = _qc_passage(para.text, lang)
            if "extreme_length" in flags:
                extreme.append(para.passage_id)
            raw_h = _sha256_text(para.text)
            norm_h = _sha256_text(normalize_text(para.text))
            toks = tokenize(para.text)
            words = para.text.split()
            source = f"data/interim/fulltext/json/{lang}/{para.book_id}.json"
            w.writerow(
                {
                    "corpus_snapshot_id": snapshot_id,
                    "passage_id": para.passage_id,
                    "book_id": para.book_id,
                    "work_id": work_by_book.get(para.book_id, ""),
                    "language": lang,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "source_path": source,
                    "word_count": len(words),
                    "character_count": len(para.text),
                    "token_count": len(toks),
                    "raw_content_hash": raw_h,
                    "normalized_content_hash": norm_h,
                    "segmentation_status": seg,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "qc_flags": flags,
                }
            )
            n += 1
            if i and i % 50000 == 0:
                log.info(f"  hashed manifest {lang}: {i}")

    meta = base_run_meta(
        kind="passage_manifest_v1_candidate",
        lang=lang,
        corpus_snapshot_id=snapshot_id,
        n_passages=n,
        expected_books=scan.expected_books,
        processed_books=scan.processed_books,
        skipped_missing=scan.skipped_missing,
        skipped_unreadable=scan.skipped_unreadable,
        n_duplicate_passage_ids=len(dup_ids),
        duplicate_passage_ids_sample=dup_ids[:20],
        extreme_length_passages=extreme[:50],
        n_extreme_length=len(extreme),
        duration_sec=round(time.time() - t0, 1),
        peak_rss_mb=round(peak_rss_mb(), 1),
        note=(
            "Content hashes enable incremental embedding updates. "
            "Do not truncate extreme paragraphs without recording impact on IDs."
        ),
    )
    write_json(DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.meta.json", meta)
    log.info(f"wrote {out} ({n} passages); extreme={len(extreme)}")
    return out


def write_lineage_stub(snapshot_id: str, log) -> Path:
    """First snapshot: empty prev links; schema ready for future cleaning diffs."""
    out = DATA_PROCESSED / "passage_lineage_v1_candidate.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LINEAGE_FIELDS)
        w.writeheader()
    write_json(
        DATA_PROCESSED / "passage_lineage_v1_candidate.meta.json",
        base_run_meta(
            kind="passage_lineage",
            corpus_snapshot_id=snapshot_id,
            n_rows=0,
            note=(
                "Initial candidate snapshot has no previous IDs. "
                "Identifier stability ≠ content stability across future cleaning."
            ),
        ),
    )
    log.info(f"wrote empty lineage stub {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--snapshot-id", default=DEFAULT_CORPUS_SNAPSHOT_ID)
    parser.add_argument("--skip-passages", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)
    langs = ["en", "de"] if args.lang == "all" else [args.lang]
    for lang in langs:
        build_book_manifest(lang, args.snapshot_id, log)
        if not args.skip_passages:
            build_passage_manifest(lang, args.snapshot_id, log)
    write_lineage_stub(args.snapshot_id, log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
