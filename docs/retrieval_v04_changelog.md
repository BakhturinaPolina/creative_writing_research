# PsyPer retrieval v0.4 changelog (infrastructure finalization)

Lexical baseline only — **not** manually validated. No semantic retrieval / embeddings / RAG in this release.

**Frozen:** all v0.1–v0.3 retrieval CSVs and `manual_eval_sample_v03.*` (247 passages).

## What changed (algorithm / tooling)

- Deterministic BLAKE2b stratum seeds (`src/retrieval/sampling.py`) — independent of `PYTHONHASHSEED`.
- Two evaluation sampling designs:
  - **A** probability-based (`manual_eval_sample_prob_v04.*`) with recorded `N_h`, `n_h`, `π_h`, weights.
  - **B** book-diversity diagnostic (`manual_eval_sample_diversity_v04.*`) — no weights.
- BM25 genuinely-new definition documented as `B \ L` at `(passage_id, retrieval_concept_id)`; set algebra report added.
- Reproducibility metadata: dirty-tree fingerprint, `corpus_snapshot_id`, denominators.
- Matcher edge-case tests + finite bilingual prefilter audit (200/lang; 0 diffs — **not** a proof of equivalence).
- German vocabulary audit + term provenance table (AI assessments provisional).
- Corpus snapshot candidate + hashed passage manifests for future incremental embeddings.

## What did **not** change

- Lexicon YAML and regex maps remain at v0.3 (no automatic Genette/Stanzel remaps).
- No regeneration of `pov_baseline_*_v03_*` assignment CSVs (sampling/report/meta only).
- Independent evaluation labels untouched.

## Human decisions still open

- Whether `auktoriale Erzählsituation` should remain under `focalization_zero`.
- Whether to demote bare `allwissen*` outside narratological collocates.
- Review of `german_audit_review_v04.csv` (78 rows, provisional AI categories).
