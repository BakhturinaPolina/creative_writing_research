"""Stable corpus passage manifest for future semantic retrieval.

Streams every eligible analysis-set body paragraph — including those with
no lexical hits — and writes pointer metadata only (no fulltext).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.paragraphs import CorpusScanStats, iter_body_paragraphs, load_book_stats
from src.retrieval.run_meta import base_run_meta, file_sha256, peak_rss_mb, write_json
from src.retrieval.textnorm import tokenize

OUT = DATA_INTERIM / "retrieval"
MANIFEST_FIELDS = [
    "passage_id",
    "book_id",
    "work_id",
    "language",
    "corpus_role",
    "primary_group",
    "chapter_idx",
    "para_idx",
    "source_json",
    "corpus_version",
    "word_count",
    "token_count",
]


def build_manifest(lang: str, log, *, corpus_version: str = "analysis_set_v1") -> Path:
    stats = load_book_stats(lang)
    work_by_book = {}
    if "work_id" in stats.columns:
        for _, row in stats.iterrows():
            work_by_book[str(row["book_id"])] = str(row.get("work_id") or "")

    scan = CorpusScanStats()
    out = DATA_PROCESSED / f"passage_manifest_{lang}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for i, para in enumerate(iter_body_paragraphs(lang, scan_stats=scan)):
            toks = tokenize(para.text)
            words = para.text.split()
            # source path relative pointer (filename only if absolute unavailable)
            source = f"data/interim/fulltext/json/{lang}/{para.book_id}.json"
            w.writerow(
                {
                    "passage_id": para.passage_id,
                    "book_id": para.book_id,
                    "work_id": work_by_book.get(para.book_id, ""),
                    "language": lang,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "source_json": source,
                    "corpus_version": corpus_version,
                    "word_count": len(words),
                    "token_count": len(toks),
                }
            )
            n += 1
            if i and i % 50000 == 0:
                log.info(f"  manifest {lang}: {i} paragraphs")

    meta = base_run_meta(
        kind="passage_manifest",
        lang=lang,
        n_passages=n,
        corpus_version=corpus_version,
        expected_books=scan.expected_books,
        processed_books=scan.processed_books,
        skipped_missing=scan.skipped_missing,
        skipped_unreadable=scan.skipped_unreadable,
        book_stats_sha256=file_sha256(
            DATA_PROCESSED / f"{lang}_book_stats.csv"
        ),
        peak_rss_mb=round(peak_rss_mb(), 1),
        note="All eligible paragraphs; lexical relevance is not a prerequisite.",
    )
    write_json(DATA_PROCESSED / f"passage_manifest_{lang}.meta.json", meta)
    log.info(
        f"wrote {out} ({n} passages); books {scan.processed_books}/{scan.expected_books}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--corpus-version", default="analysis_set_v1")
    args = parser.parse_args()
    log = setup_logger(__file__)
    langs = ["en", "de"] if args.lang == "all" else [args.lang]
    for lang in langs:
        build_manifest(lang, log, corpus_version=args.corpus_version)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
