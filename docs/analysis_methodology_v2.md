# Analysis methodology v2

## Purpose

Descriptive bilingual analysis of narrative-perspective retrieval in English and German creative-writing manuals. Results are **retrieval patterns**, not validated PsyPer annotations.

## Version layers

| Layer | Value used by analysis v2 |
|---|---|
| Corpus snapshot | `corpus_snapshot_analysis_v1_candidate_20261122` |
| Lexicon | `0.3-draft` (`config/pov_lexicon_psyper_en_de_v0.3.yml`) |
| Retrieval algorithm | v0.3 lexical matcher + streaming BM25 |
| Retrieval execution | Frozen `*_v03_*` hit tables (execution id `v03_lexical_bm25_20261122`) |
| Evaluation infrastructure | v0.5 sampling frame (unlabelled) |
| Analysis | v2 |

v0.4/v0.5 did not regenerate hit tables. Book-stats hashes agree across snapshot and lexical v0.3 metas.

## Denominators

- **Primary:** eligible searchable words = sum of passage `word_count` for paragraphs with ≥40 characters.
- **Supplementary:** cleaned body words (`body_words` / manifest `number_of_words`).
- Short paragraphs and front/back filtering explain body-vs-eligible gaps; they are not silent data loss.

Numerators and denominators are paired in every rate table.

## Extreme passages

Passages with ≥100,000 characters were diagnosed. The resegment gate fires only for collapsed chapters with recoverable blank-line breaks. In the current corpus the gate did **not** fire (0 recoverable). Scenario S3 drops extremes from numerator and denominator.

## Integrity

Automated checks cover unique IDs, referential integrity, duplicate assignments, language/concept coverage including zeros, summary agreement, zero-hit books in denominators, and family deduplication notes. Hardcoded passage totals are not used.

## Standardization

Category-specific rates use category eligible-word denominators. Shared weights \(w_g\) are the mean of English and German category word shares. Standardized rates do not remove dictionary bias.

## Sensitivity

S1–S5 as before, plus high-confidence evidence (S6), German stance-trigger exclusion (S7), drop top-5 books (S8), fiction-only (S9), fiction+pedagogy (S10), body-word denominator (S11). S4 isolates `allwissen*` / `omniscient_de`; S7 isolates Stanzel stance vocabulary.

## Uncertainty

Book-level stratified bootstrap (default 1000 iterations, seed `20261122`) for focus concepts. Intervals do not capture dictionary bias. Precision/recall remain pending.

## Future retrieval methods

`reports/tables/02_assignments_long.csv` uses:

`language, book_id, passage_id, concept_id, retrieval_method, retrieval_version, score, rank, query_id, corpus_snapshot_id`

Dense, hybrid, RRF and reranked methods can append rows without rewriting the analysis framework.

## Regenerate

```bash
.venv/bin/python -m src.analysis.inventory
.venv/bin/python -m src.analysis.extreme_passages
.venv/bin/python -m src.analysis.run_np_exploration --edition v2
.venv/bin/python -m src.analysis.build_interactive_report
PYTHONPATH=. .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/06_narrative_perspective_full_analysis.ipynb
```

Retrieval rerun commands are required only if `data/processed/extreme_passage_gate_v2.json` reports `resegment_gate_fires: true`.

## Outputs

- Tables: `reports/tables/02_*.csv`
- Figures: `reports/figures/02_narrative_perspective/`
- Meta: `reports/02_narrative_perspective_run_meta.json`
- HTML: `reports/02_narrative_perspective_interactive.html`
- Notebook: `notebooks/06_narrative_perspective_full_analysis.ipynb`

Historical `reports/01_*` and notebook 05 are preserved.
