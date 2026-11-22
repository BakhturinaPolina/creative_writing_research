# Analysis v2 implementation summary

## 1. Issues identified

- Previous notebook/analysis used frozen lexical/BM25 **v0.3** hits; v0.4/v0.5 did not re-run retrieval (filename versions ≠ execution).
- Hardcoded / disabled integrity check (`or True` on English passage total).
- Standardization weights calculated but never applied.
- Book-level tables covered only three concepts.
- Body-word vs eligible-passage word “mismatches” were definition gaps, not silent loss.
- 56 extreme passages (≥100k chars) are collapsed chapters without recoverable blank-line breaks.
- Historical BM25 raw heap (`Braw`) is not on disk.
- Evaluation sample v0.5 remains unlabelled (no precision/recall).
- No `tests/analysis/` coverage.

## 2. Corrections implemented

- Machine-readable input inventory with checksums and hash agreement checks.
- Full book-coverage audit with exclusion reasons and recommended actions.
- Word-count fields split (`raw`, `cleaned_body`, `eligible_passage`, `excluded`).
- Extreme-passage diagnosis + resegment gate (did not fire).
- Analysis library v2: eligible-word denominators, all-concept dispersion, standardization, expanded sensitivity S1–S11, co-occurrence with evidence-overlap flags, book bootstrap, long assignment schema.
- Meaningful integrity assertions; `tests/analysis/` (11 tests).
- Notebook 06 executed end-to-end; interactive offline HTML with lexicon explorer and local suggestions; aggregator script.

## 3. Final corpus size and coverage

| Metric | English | German |
|---|---:|---:|
| Listed books | 503 | 78 |
| Analysis set (in retrieval) | 442 | 67 |
| Excluded | 61 | 11 |
| Eligible words | ~34.3M (passage sum) | ~4.6M |
| Eligible paragraphs | 315,302 | 42,623 |
| Lexical assignments / passages | 10,643 / 7,380 | 1,380 / 941 |

Exact eligible-word totals are in `reports/tables/02_corpus_overview.csv` and `02_pipeline_funnel.csv`.

## 4. Books still excluded

Intentional research rules from `exclusion_step`:

- EN: `needs_ocr` 17, `short_text` 17, `surplus_duplicate` 23, `role_conflict` 4.
- DE: `needs_ocr` 1, `short_text` 5, `surplus_duplicate` 5.

No book with available text and empty `exclusion_step` was found outside the analysis set. Zero-match analysis books remain included.

## 5. Retrieval versions used

- Lexicon: `0.3-draft`
- Retrieval execution: `v03_lexical_bm25_20261122` (frozen `*_v03_*` CSVs)
- Corpus snapshot: `corpus_snapshot_analysis_v1_candidate_20261122`
- Analysis: `v2`
- Evaluation infrastructure: `v0.5` (unlabelled)
- `Braw_available`: false

## 6. Changes from previous results

- Primary rates now use **eligible searchable words** (body-word rates kept as supplementary).
- Standardization fully applied for all concepts.
- Dispersion/sensitivity extended to all 36 concepts; scenarios S6–S11 added.
- Book bootstrap CIs for focus concepts (1000 iterations, seed 20261122).
- Integrity checks dynamic; no hardcoded 7380 assert.
- Interactive report + terminology review workflow added.

## 7. Important scientific findings (retrieval patterns)

- English corpus is much larger; German contrast materials hold a larger word share — composition matters.
- German stance and zero-focalization rates are high and **sensitive to lexical rules** (S4 allwissen*/omniscient_de; S7 Stanzel stance vocabulary).
- Several German concepts have zero lexical matches (`voice_homodiegetic`, `voice_eyewitness`, `voice_participant`, `focalization_fixed` on current run).
- BM25-new expansion is large in German relative to lexical hits; this is complementarity, not validated recall.
- Missing publication years are common; no historical concept trend is claimed from sparse dated subsets.

## 8. Remaining limitations

- Dictionary bias; asymmetric EN/DE vocabularies.
- No human gold labels → no precision/recall/F1.
- Extreme passages not resegmented (no recoverable breaks).
- `work_id` ~94% missing → limited work-level dedup.
- Bootstrap does not capture dictionary bias.

## 9. Tests executed

```text
pytest tests/retrieval tests/analysis  →  166 passed
notebook 06 execute                     →  14/14 code cells, 0 errors
HTML offline check                      →  no external http script/link
Browser                                 →  metric switch, 36 concept cards,
                                            localStorage suggestion persist/reload,
                                            mobile viewport header/nav usable
```

## 10. Regenerate notebook and HTML

```bash
.venv/bin/python -m src.analysis.inventory
.venv/bin/python -m src.analysis.extreme_passages
.venv/bin/python -m src.analysis.run_np_exploration --edition v2
.venv/bin/python -m src.analysis.build_interactive_report
PYTHONPATH=. .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/06_narrative_perspective_full_analysis.ipynb
```

## 11. Colleague terminology suggestions

Open `reports/02_narrative_perspective_interactive.html` offline. See `docs/terminology_review_guide.md`. Export JSON and aggregate with:

```bash
.venv/bin/python -m src.lexicon_review.aggregate_suggestions reviewer_*.json
```

## 12. Recommended next steps before semantic retrieval

1. Human review of German priority / ambiguous terms; decide lexicon edits.
2. Annotate `manual_eval_sample_prob_v05` (independent of dictionary development).
3. Improve extraction/segmentation for extreme chapters before embeddings.
4. Export full BM25 `Braw` on any future BM25 run.
5. Improve `work_id` / duplicate resolution.
6. Then add dense multilingual retrieval rows into `02_assignments_long.csv`.
