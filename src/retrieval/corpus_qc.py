"""Corpus quality-control table for the frozen v1_candidate snapshot.

Does not modify or drop problematic records — only documents them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_PROCESSED, ROOT
from src.retrieval.run_meta import base_run_meta, write_json

SNAPSHOT_ID = "corpus_snapshot_analysis_v1_candidate_20261122"


def build_corpus_quality(log) -> pd.DataFrame:
    rows = []
    for lang in ("en", "de"):
        cm_path = DATA_PROCESSED / f"corpus_manifest_{lang}_v1_candidate.csv"
        pm_path = DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv"
        cm = pd.read_csv(cm_path, dtype=str).fillna("")
        pm = pd.read_csv(pm_path, dtype=str).fillna("")
        analysis = cm[cm["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))]

        def _missing_year(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip().isin(("", "nan", "None", "NaN"))

        def _missing_work(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip().isin(("", "nan", "None", "NaN"))

        extreme = pm[
            pm["segmentation_status"].astype(str).str.contains("extreme", case=False)
            | pm["qc_flags"].astype(str).str.contains("extreme_length", case=False)
        ]
        dup_passage = int(pm["passage_id"].duplicated().sum())
        missing_files = analysis[
            analysis["processing_status"].astype(str).str.contains("missing", case=False)
        ]
        possible_dup_books = analysis[
            analysis["qc_notes"].astype(str).str.contains("possible_duplicate", case=False)
        ]

        # Word / paragraph consistency: sum of passage word_count vs book number_of_words.
        pm["word_count_num"] = pd.to_numeric(pm["word_count"], errors="coerce").fillna(0)
        by_book_words = pm.groupby("book_id")["word_count_num"].sum()
        analysis = analysis.copy()
        analysis["number_of_words_num"] = pd.to_numeric(
            analysis["number_of_words"], errors="coerce"
        )
        analysis["passage_words_sum"] = analysis["book_id"].map(by_book_words)
        word_mismatch = analysis[
            analysis["passage_words_sum"].notna()
            & analysis["number_of_words_num"].notna()
            & (
                (analysis["passage_words_sum"] - analysis["number_of_words_num"]).abs()
                > 1
            )
        ]

        # Referential integrity: retrieval keys ⊂ passage manifest.
        lex = DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv"
        missing_ref = 0
        if lex.is_file():
            L = pd.read_csv(lex, dtype=str, usecols=["passage_id"]).fillna("")
            missing_ref = int((~L["passage_id"].isin(set(pm["passage_id"]))).sum())

        invalid_group = analysis[
            ~analysis["primary_group"].isin(
                ["fiction_manual", "pedagogy_genre_craft", "contrast", ""]
            )
            & analysis["primary_group"].astype(str).str.len().gt(0)
        ]

        rows.append(
            {
                "corpus_snapshot_id": SNAPSHOT_ID,
                "language": lang,
                "n_listed_books": len(cm),
                "n_analysis_set_books": len(analysis),
                "n_excluded_books": len(cm) - len(analysis),
                "n_missing_or_unreadable_fulltext": len(missing_files),
                "n_eligible_passages": len(pm),
                "n_extreme_length_passages": len(extreme),
                "n_duplicate_passage_ids": dup_passage,
                "pct_analysis_books_missing_publication_year": round(
                    100.0 * _missing_year(analysis["publication_year"]).mean(), 2
                ),
                "pct_analysis_books_missing_work_id": round(
                    100.0 * _missing_work(analysis["work_id"]).mean(), 2
                ),
                "n_analysis_books_possible_duplicate": len(possible_dup_books),
                "n_analysis_books_wordcount_mismatch": len(word_mismatch),
                "n_analysis_books_invalid_primary_group": len(invalid_group),
                "n_lexical_assignments_missing_from_passage_manifest": missing_ref,
                "total_analysis_words": int(
                    pd.to_numeric(analysis["number_of_words"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "total_analysis_eligible_paragraphs": int(
                    pd.to_numeric(
                        analysis["number_of_eligible_paragraphs"], errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                ),
                "notes": (
                    "No records removed. Snapshot remains candidate. "
                    "work_id missingness prevents automatic edition-safe work grouping."
                ),
            }
        )

        # Detail rows for extreme passages (for notebook).
        detail_path = DATA_PROCESSED / f"corpus_quality_extreme_passages_{lang}_v1_candidate.csv"
        extreme[
            [
                c
                for c in (
                    "passage_id",
                    "book_id",
                    "language",
                    "character_count",
                    "word_count",
                    "segmentation_status",
                    "qc_flags",
                    "primary_group",
                )
                if c in extreme.columns
            ]
        ].to_csv(detail_path, index=False)
        log.info(f"wrote {detail_path} ({len(extreme)} rows)")

    out_df = pd.DataFrame(rows)
    out = DATA_PROCESSED / "corpus_quality_v1_candidate.csv"
    out_df.to_csv(out, index=False)
    write_json(
        DATA_PROCESSED / "corpus_quality_v1_candidate.meta.json",
        base_run_meta(
            kind="corpus_quality_v1_candidate",
            corpus_snapshot_id=SNAPSHOT_ID,
            n_languages=len(out_df),
            note="Diagnostic only; does not alter the frozen snapshot.",
        ),
    )
    log.info(f"wrote {out}")
    return out_df


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    log = setup_logger(__file__)
    build_corpus_quality(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
