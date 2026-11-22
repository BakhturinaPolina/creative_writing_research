# Corpus coverage audit (analysis v2)

Generated: `2026-11-22T22:52:28.992196+00:00`
Source commit: `124f0683c6ad7b26a81cd20ca24ccf2c5be9cdbd`
Corpus snapshot: `corpus_snapshot_analysis_v1_candidate_20261122`
Lexicon version: `0.3-draft`
Retrieval execution: `v03_lexical_bm25_20261122`
Evaluation infrastructure: `v0.5`

## Snapshot vs retrieval agreement

Lexical and BM25 v0.3 hit tables were executed against the same analysis-set book-stats
hashes that the v1 candidate snapshot freezes. v0.4 and v0.5 did not re-run retrieval.

| Language | book_stats sha256 matches lexical v0.3 | matches snapshot meta |
|---|---|---|
| en | True | True |
| de | True | True |

## Pipeline funnel

| language | language_label | listed_books | books_with_available_text | successfully_preprocessed_or_ruled | books_included_in_retrieval | books_with_at_least_one_match | books_without_matches | books_excluded | analysis_books_flagged_potential_duplicate | excluded_needs_ocr | excluded_short_text | excluded_surplus_duplicate | excluded_role_conflict | total_eligible_passage_words | total_cleaned_body_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | English | 503 | 486 | 486 | 442 | 407 | 35 | 61 | 27 | 17 | 17 | 23 | 4 | 34218544.0 | 34581264.0 |
| de | German | 78 | 77 | 77 | 67 | 67 | 0 | 11 | 4 | 1 | 5 | 5 | 0 | 4619105.0 | 4665385.0 |

## Exclusion rules

Exclusions come from `exclusion_step` in `{en,de}_book_stats.csv` (first match wins):
`needs_ocr`, `short_text` (<10,000 words), `role_conflict`, `surplus_duplicate`.
Books with zero lexical matches remain in the analysis population.
Fulltext catalogue `negative` roles were never part of this analysis corpus.

Audit rows: 581. Unexpected empty-exclusion available-text books: 0.

Machine-readable inventory: `data/processed/input_inventory_v2.json`.
Book table: `reports/tables/02_book_coverage_audit.csv`.
Funnel table: `reports/tables/02_pipeline_funnel.csv`.
