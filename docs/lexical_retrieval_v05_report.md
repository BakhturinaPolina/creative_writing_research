# Lexical retrieval v0.5 + bilingual analysis — final summary

**Date:** 2026-11-22

## 1. Implemented changes

1. **Exclusive probability sampling (v0.5)** — passage membership table with mutually exclusive primary strata fixed before SRS; π = n/N, w = N/n; v04 preserved; reconciliation with content-hash transfer rule.
2. **Lexicon provenance v0.5** — controlled categories; inherited/unknown recorded explicitly; PsyPer vs 36 retrieval concepts documented.
3. **German audit clarification** — heuristic assessment rename; empty human fields; Wissensvorsprung dual-relevance notes; not an evaluation set.
4. **BM25 terminology** — L, Bq, Braw, Bnew, U / B_combined clarified without full BM25 regen.
5. **Corpus QC table** — missing years/work_id, extreme paragraphs, integrity checks; snapshot frozen.
6. **Analysis package** — concept families, helpers, report runner, executed notebook + HTML/tables/figures.

## 2. Verification results

- `tests/retrieval/test_eval_sample_v05.py`: 12 passed (exclusivity, multi-rank, census π=1, PYTHONHASHSEED).
- Membership: 298,787 eligible passages; pairwise exclusive strata; sample n=250 (125 EN / 125 DE).
- Notebook executed end-to-end; HTML exported without copyrighted full texts / audit passage sidecars.
- Corpus overview recomputed: EN 503/442 / 34,581,264 / 315,302; DE 78/67 / 4,665,385 / 42,623.

## 3. Generated files (selected)

- `docs/lexical_retrieval_v05_analysis_plan.md`
- `data/interim/retrieval/eval_passage_membership_v05.*`
- `data/interim/retrieval/manual_eval_sample_prob_v05.*`
- `data/interim/retrieval/manual_eval_sample_v04_v05_reconciliation.*`
- `data/interim/retrieval/lexicon_term_provenance_v05.csv`
- `docs/psyper_vs_retrieval_concepts_v05.md`
- `data/interim/retrieval/german_audit_review_v05.csv`
- `data/interim/retrieval/retrieval_set_algebra_v05.*`
- `data/processed/corpus_quality_v1_candidate.csv`
- `config/concept_families_v05.yml`
- `src/analysis/narrative_perspective.py`, `run_np_exploration.py`
- `notebooks/05_narrative_perspective_exploration.ipynb`
- `reports/01_narrative_perspective_exploration.html`
- `reports/tables/*`, `reports/figures/01_narrative_perspective/*`

## 4. Main preliminary research findings (retrieval coverage)

- EN and DE lexical passage rates per 10,000 words are similar (~2), while absolute counts differ mainly by corpus size.
- German contrast materials contribute a much larger word share (~29% vs ~5% EN); composition must be controlled when interpreting language differences.
- v0.3 German gains concentrate in focalization_zero / stance / third-person vocabulary expansions — not proof of improved recall.
- BM25 adds many genuinely new passages, with larger relative expansion in German; causes are plural (size, K, morphology, lexicon), not inherent BM25 superiority.
- Publication-year and work_id missingness (esp. DE) make historical and edition-safe claims premature.

## 5. Unresolved methodological limitations

- Manual relevance coding of evaluation samples still outstanding.
- Stanzel↔Genette dictionary overlap unresolved pending human audit.
- Historical `Braw` not fully reconstructable; use `B_combined` terminology carefully.
- Extreme-length paragraphs remain in the frozen snapshot.
- Sampling weights apply only within recorded exclusive stratum populations of the independent eval set (held-out books excluded) — not corpus-wide prevalence.

## 6. Recommended next steps

1. Human review of German audit + priority provenance terms; only then consider lexicon edits as a new versioned stem.
2. Annotate `manual_eval_sample_prob_v05` (transfer from v04 only where content hashes agree).
3. Policy for extreme-length passages before embedding / semantic retrieval.
4. Future BM25 runs: export complete Braw before subtracting L.
5. Extend notebook tables with dense/hybrid method columns when semantic retrieval begins.
