# Lexical retrieval finalization report (v0.4)

**Date:** 2026-11-22  
**Frozen baseline:** v0.3 @ `89b3c88`  
**Scope:** Finalize lexical IR infrastructure before semantic retrieval. No embeddings / fusion / RAG.

## 1. Implemented changes

| Area | Change |
|---|---|
| Sampling | BLAKE2b stratum seeds; Design A probability sample + Design B diversity sample; reconciliation vs frozen v0.3 |
| BM25 reporting | Named set algebra `L, Bq, B, B\L, L\B, L∩B`; passage-level new vs new-on-known |
| BM25 code | Future runs subtract `L` (not only `Bq`) for genuinely-new export |
| Metadata | Dirty-tree fingerprint, snapshot ID, denominators in `run_meta` |
| Tests | 143 passed (sampling cross-process, schema, matcher edges) |
| Matcher audit | 200 EN + 200 DE; 0 prefilter≠exhaustive diffs (finite sample) |
| German audit | Gain attribution, 78 review rows, provenance 2042 terms |
| Corpus snapshot | `corpus_snapshot_analysis_v1_candidate_20261122` + hashed manifests |

## 2. Unresolved methodological issues (human judgment)

1. Overlap of Stanzel *auktorial* cues with Genette zero focalization in the v0.3 dictionary.
2. False-positive risk of broad `allwissen*` regex outside craft instruction.
3. DE zero lexical hits on `focalization_fixed`, `voice_eyewitness`, `voice_participant`, `voice_homodiegetic` despite BM25-only candidates — absence vs retrieval failure unresolved.
4. Extreme “paragraphs” (≥100k chars; EN 49 / DE 7 including `de_26856f179d5e:c7:p0`) — segmentation policy for embeddings.
5. Manual relevance coding of eval samples still outstanding.

## 3. Before/after retrieval statistics

Lexical/BM25 **assignment CSVs unchanged** (no vocabulary/matcher regen). Reporting clarified:

| lang | \|L\| | \|Bq\| | \|B\| | \|B\L\| | \|L\B\| | \|L∩B\| | new passages | new concepts on known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 10643 | 5756 | 8568 | 2812 | 4887 | 5756 | 820 | 958 |
| de | 1380 | 1339 | 4534 | 3195 | 41 | 1339 | 1263 | 330 |

`Bonly ∩ L = 0` on frozen v0.3 exports.

## 4. German vocabulary audit

- Gains almost entirely vocabulary/regex (`focalization_zero` +125 vocab; `stance_narratorial` +98; `stance_figural` +58; `grammatical_third_person` +42).
- Algo-fix contribution small (+1–7).
- Annotation-ready: `data/interim/retrieval/german_audit_review_v04.csv` (78 rows; text in gitignored sidecar).
- AI categories are **provisional** (`assessment_source=provisional_ai_assessment`).
- No lexicon edits applied pending human review.

## 5. Corpus snapshot

- ID: `corpus_snapshot_analysis_v1_candidate_20261122` (**candidate**, cleaning ongoing).
- Books: EN 503 / DE 78 listed; analysis-set EN 442 / DE 67 with fulltext present.
- Passages: EN 315,302 / DE 42,623 with SHA-256 raw + normalized hashes.
- Extreme-length flags: EN 49 / DE 7.
- Lineage stub empty (first snapshot).
- Sources not modified; full texts remain gitignored.

## 6. Reproduction

```bash
.venv/bin/python -m pytest tests/retrieval -q
.venv/bin/python -m src.retrieval.report --version set_algebra
.venv/bin/python -m src.retrieval.eval_sample --version v04
.venv/bin/python -m src.retrieval.matcher_audit --n-per-lang 200
.venv/bin/python -m src.retrieval.de_vocab_audit --per-trigger 25
.venv/bin/python -m src.retrieval.corpus_snapshot --lang all
```

Do **not** overwrite v0.3 stems. Full lexical/BM25 regen only after approved lexicon/regex edits (then use new `*_v04_*` stems).

## 7. Generated artifacts

**Code/tests/docs:** `sampling.py`, `eval_sample.py`, `run_meta.py`, `bm25_passages.py`, `report.py`, `matcher_audit.py`, `de_vocab_audit.py`, `corpus_snapshot.py`, new tests, this report, plan, v0.4 changelog.

**Data (pointers / aggregates):**
- `manual_eval_sample_prob_v04.*`, `manual_eval_sample_diversity_v04.*`, reconciliation CSV
- `retrieval_set_algebra_v04.*`
- `matcher_prefilter_audit_v04.*`
- `german_*_v04.*`, `lexicon_term_provenance_v04.csv`, `low_coverage_concepts_v04.csv`
- `corpus_manifest_{en,de}_v1_candidate.csv`
- `passage_manifest_{en,de}_v1_candidate.csv`
- `passage_lineage_v1_candidate.csv`

**Private (gitignored):** `*_passages.csv` sidecars including German audit passages.

**Preserved:** all `pov_baseline_*_v0{1,2,3}_*`, `manual_eval_sample_v03.*`.

## 8. Readiness for semantic retrieval

**Conditional go:** infrastructure is ready for embedding work **after** (a) human review of German audit decisions that might change the lexicon, and (b) a policy for extreme-length passages. Hashed passage manifests support incremental embedding updates. Do **not** treat lexical hits as gold PsyPer labels. Sampling Design A weights apply only within recorded stratum populations of the independent evaluation set — not corpus-wide prevalence.
