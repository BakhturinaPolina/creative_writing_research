# Retrieval changelog v0.5

**Date:** 2026-11-22  
**Frozen retrieval CSVs:** v0.3 lexical / BM25 (unchanged)  
**Infrastructure baseline:** v0.4 @ `e86b2da` plus v0.5 sampling, provenance, audit clarification, BM25 terminology, corpus QC, and bilingual analysis notebook.

## What changed

| Area | Change |
|---|---|
| Evaluation sampling | Mutually exclusive primary strata **before** SRS (`eval_passage_membership_v05`, `manual_eval_sample_prob_v05`) |
| Reconciliation | `manual_eval_sample_v04_v05_reconciliation.csv` with content-hash transfer safety |
| Provenance | Controlled vocabulary in `lexicon_term_provenance_v05.csv`; PsyPer vs retrieval concepts documented |
| German audit | `heuristic_provisional_assessment`; empty human fields; development-only labelling |
| BM25 reporting | `L / Bq / Braw / Bnew / U` terminology; `B_combined` = reconstructed pool, not independent top-K |
| Corpus QC | `corpus_quality_v1_candidate.csv` on frozen snapshot |
| Analysis | Concept families + helpers; notebook `05` + `reports/` package |

## What did not change

- Lexical/BM25 assignment CSVs (`*_v03_*`) were **not** regenerated.
- No semantic retrieval / embeddings / RAG.
- No Genette↔Stanzel conceptual remaps pending human audit.
- v0.3 / v0.4 evaluation samples preserved.

## Reproduction

```bash
.venv/bin/python -m pytest tests/retrieval/test_eval_sample_v05.py -q
.venv/bin/python -m src.retrieval.eval_sample --version v05 --skip-passages
.venv/bin/python -m src.retrieval.lexicon_provenance
.venv/bin/python -m src.retrieval.german_audit_v05
.venv/bin/python -m src.retrieval.report --version v05
.venv/bin/python -m src.retrieval.corpus_qc
.venv/bin/python -m src.analysis.run_np_exploration
PYTHONPATH=. .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/05_narrative_perspective_exploration.ipynb
PYTHONPATH=. .venv/bin/jupyter nbconvert --to html --no-input \
  --output-dir reports --output 01_narrative_perspective_exploration \
  notebooks/05_narrative_perspective_exploration.ipynb
```
