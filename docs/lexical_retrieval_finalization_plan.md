# Lexical retrieval finalization plan (v0.4 infrastructure)

Lexical baseline only — **not** manually validated. Does not start semantic retrieval, embeddings, fusion, or RAG.

Frozen baseline: PsyPer retrieval **v0.3** at commit `89b3c88`. New methodological artifacts use **v0.4** stems. Do not overwrite v0.1–v0.3 outputs.

## Phase 0 findings

| Item | Finding |
|---|---|
| HEAD | `89b3c88` (reviewed v0.3); clean tree at plan time |
| Corpus | EN 442 / DE 67 analysis-set books; JSON present for all |
| Denominators | EN 34.58M / DE 4.67M body words |
| Eval sample | 247 passages in `manual_eval_sample_v03.*` — preserve |
| Open | Deterministic hashing, probability vs diversity sampling, BM25 set algebra reporting, matcher audit expansion, German vocab audit, corpus snapshot hashes |

### Issue triage

| ID | Status |
|---|---|
| 1.1 Deterministic hashing | Open — `hash((seed, stratum))` in `sampling.py` |
| 1.2 Sampling protocol | Open — reservoir π then book-diversity downselect |
| 1.3 BM25-only definition | Partial — assignment keys used; named set algebra missing |
| 1.4 Matcher tests | Partial — fixtures exist; edge/corpus audit incomplete |
| 1.5 Schema | Mostly done on v0.3 writers |
| 1.6 Repro metadata | Partial — dirty-tree fingerprint / snapshot ID missing |
| 2.x German audit | Open |
| 3.x Corpus snapshot | Partial — pointer manifest exists; hashes/QC missing |

## Files to modify

- `src/retrieval/sampling.py` — blake2b stratum seeds
- `src/retrieval/eval_sample.py` — probability + diversity v0.4 designs
- `src/retrieval/run_all.py` — held-out wiring; v0.4 sampling
- `src/retrieval/bm25_passages.py` — subtract `L` for genuinely-new BM25
- `src/retrieval/report.py` — set algebra + dual unit stats
- `src/retrieval/run_meta.py` — snapshot ID, dirty fingerprint, duration
- `src/retrieval/passage_manifest.py` — hashed versioned manifests
- `src/retrieval/README.md`, `docs/research_log.md`

## New files

- `docs/lexical_retrieval_finalization_plan.md` (this file)
- `docs/lexical_retrieval_finalization_report.md`
- `docs/retrieval_v04_changelog.md` (if configs change)
- `tests/retrieval/test_sampling_reproducibility.py`
- `tests/retrieval/test_matcher_edge_cases.py`
- `tests/retrieval/test_schema_fields.py`
- `src/retrieval/matcher_audit.py`
- `src/retrieval/de_vocab_audit.py`
- `src/retrieval/corpus_snapshot.py`
- `data/interim/retrieval/manual_eval_sample_prob_v04.*`
- `data/interim/retrieval/manual_eval_sample_diversity_v04.*`
- `data/interim/retrieval/manual_eval_sample_v03_v04_reconciliation.csv`
- `data/interim/retrieval/retrieval_set_algebra_v04.csv`
- `data/interim/retrieval/lexicon_term_provenance_v04.csv`
- `data/interim/retrieval/german_lexical_audit_v04.md`
- `data/interim/retrieval/german_audit_review_v04.csv`
- `data/processed/corpus_manifest_{en,de}_v1_candidate.csv`
- `data/processed/passage_manifest_{lang}_v1_candidate.csv`
- `data/processed/passage_lineage_v1_candidate.csv`

## Dependencies

1. Sampling seed fix → probability/diversity samples → reconciliation  
2. `run_meta` enrichment → all new writers  
3. BM25 `L`-subtraction + report set algebra (report-only if rows unchanged)  
4. Matcher tests / audit (independent of sampling)  
5. German audit (reads frozen v0.3 outputs; may propose v0.4 config demotions)  
6. Corpus snapshot + hashed manifests  
7. Selective regen → final report + commits  

## Regression tests required

- Subprocess sampling identity under different `PYTHONHASHSEED`
- Probability meta records `N_h`, `n_h`, `π_h`, weights
- Schema field separation + hierarchical PsyPer string codes
- Matcher edge cases (DE morphology, OCR spacing, plurals, contrasts, …)
- Exhaustive-vs-prefilter sample audit (finite; not claimed perfect)

## Runtime / compute

| Step | Estimate |
|---|---|
| Unit + regression tests | &lt; 5 min |
| Sampling + reports | 10–30 min |
| Corpus snapshot + hashed manifests | 20–40 min |
| German audit extraction | 15–45 min |
| Full EN+DE lexical+BM25 (only if rules change) | 1.5–3 h; peak RSS ~150–200 MB |

Streaming / small batches / CPU only. Target peak RSS &lt; 2 GB.

## Methodological risks

- Finite prefilter audits cannot prove matcher equivalence.
- Provisional AI German audit labels ≠ human gold.
- Stanzel ↔ Genette dictionary overlap remains ambiguous.
- Extreme “paragraphs” (e.g. 100k+ chars) break passage assumptions.
- Prior v0.3 metas may cite a dirty-tree commit; v0.4 must fingerprint correctly.

## Rerun conditions

| Change | Action |
|---|---|
| Sampling / meta / report only | Do **not** rewrite v0.3 retrieval CSVs |
| Lexicon / regex / evidence / matcher | New `*_v04_*` lexical (+ BM25 if gates change) |
| Corpus text / segmentation | New snapshot ID + remanifest; versioned retrieval regen |

## Acceptance criteria (from task brief)

- Sampling reproducible across processes  
- Sampling probabilities match the implemented design  
- BM25-only distinguished from lexical at assignment level  
- Eval and development data remain separate  
- German matching rules inspected and documented  
- Vocabulary provenance traceable  
- Corpus snapshot and passage manifests versioned  
- Existing experimental outputs preserved  
- Outstanding human-judgment issues explicit  
- No semantic retrieval phase yet  
