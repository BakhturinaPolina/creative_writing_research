# Lexical retrieval v0.5 + bilingual analysis notebook

**Status:** implementation plan  
**Baseline frozen:** v0.3 retrieval CSVs @ prior commits; v0.4 infrastructure @ `e86b2da`  
**Out of scope:** semantic retrieval, embeddings, vector DBs, cross-encoder reranking, RAG, lexicon remaps pending human audit.

## Objectives

1. Correct probability-based evaluation sampling so stratum **populations** are mutually exclusive before sampling.
2. Improve lexicon provenance transparency and PsyPer vs project-concept documentation.
3. Clarify German heuristic audit (not human validation / not precision).
4. Clarify BM25 candidate-pool terminology (`L`, `Bq`, `Braw`, `Bnew`, `U`) without full historical regen.
5. Produce a notebook-ready corpus QC table on the frozen candidate snapshot.
6. Ship a fully executed bilingual exploratory research notebook + HTML/tables/figures.

## Path conventions

| Artifact | Path |
|---|---|
| This plan | `docs/lexical_retrieval_v05_analysis_plan.md` |
| Notebook | `notebooks/05_narrative_perspective_exploration.ipynb` (01–04 already used) |
| HTML report | `reports/01_narrative_perspective_exploration.html` |
| Figures | `reports/figures/01_narrative_perspective/` |
| Tables | `reports/tables/` |

Preserve all v0.3 / v0.4 sample and retrieval stems. New artifacts use `*_v05_*` stems.

## Issue triage (from inspection)

| ID | Issue | Action |
|---|---|---|
| A1 | Stratum populations overlap (multi-rank BM25; passage in C and only) | Exclusive priority partition before sampling |
| A2 | Provenance mostly `prior_lexicon_v02` | Controlled vocabulary + completeness summary |
| A3 | Audit `provisional_ai_assessment`; risk of 71/78 as precision | Rename; empty human fields; development-only label |
| A4 | BM25 `B` = reconstructed union; missing Braw | Terminology + set-algebra v05 report-only |
| A5 | No notebook QC table; work_id ~94% missing | `corpus_quality_v1_candidate.csv` |
| B | No bilingual concept notebook | Notebook 05 + helpers + concept families |

Verified corpus denominators (analysis-set): EN 442 books / 34,581,264 words / 315,302 paragraphs; DE 67 / 4,665,385 / 42,623.

---

## Part A — Deliverables

### A1. Exclusive sampling

**Code:** `src/retrieval/eval_sample.py` (+ helpers as needed)

**Membership table** `eval_passage_membership_v05.csv` columns:

`passage_id, book_id, work_id, language, primary_group, in_lexical, in_qualified_bm25, in_bm25_expansion, has_new_concept_assignment, best_bm25_rank, n_lexically_retrieved_concepts, n_bm25_retrieved_concepts, primary_stratum`

**Exclusive priority strata:**

1. `high_ranked_qualified_bm25` — qualified BM25 with best rank 1–50  
2. `lower_ranked_qualified_bm25` — qualified BM25, no rank ≤50  
3. `lexical_only` — lexical only (not Bq, not exploratory BM25)  
4. `bm25_expansion_known` — lexical + additional BM25 concepts; not 1–3  
5. `new_bm25_passages` — BM25-only passages absent from lexical  
6. `non_retrieved` — split by `language × primary_group` (unknown group retained)

**Formulas:** \(\pi_h = n_h / N_h\), \(w_h = N_h / n_h\); if census of stratum, \(\pi_h = 1\).

**Quota redistribution:** unused targets → depleted retrieved strata → non-retrieved by language → `fiction_manual` leftover. Target ~250–300 passages.

**Outputs:** `manual_eval_sample_prob_v05.*`, `manual_eval_sample_v04_v05_reconciliation.csv`  
**Tests:** `tests/retrieval/test_eval_sample_v05.py`  
**Runtime:** ~5–15 min

### A2. Provenance

**Code:** `src/retrieval/lexicon_provenance.py`  
**Outputs:** `lexicon_term_provenance_v05.csv`, `lexicon_provenance_completeness_v05.md`, `docs/psyper_vs_retrieval_concepts_v05.md`  
**Runtime:** ~10–30 min

### A3. German audit v05

**Outputs:** `german_audit_review_v05.csv`, `german_lexical_audit_v05.md`  
Keep v04 frozen. Assessment source: `heuristic_provisional_assessment`.

### A4. BM25 terminology

**Code:** `src/retrieval/report.py`, `bm25_passages.py` meta comments  
**Outputs:** `retrieval_set_algebra_v05.md/.csv` (report-only from v0.3 files)

### A5. Corpus QC

**Code:** `src/retrieval/corpus_qc.py`  
**Outputs:** `data/processed/corpus_quality_v1_candidate.csv`

---

## Part B — Notebook

**Helpers:** `src/analysis/narrative_perspective.py`  
**Families:** `config/concept_families_v05.yml`  
**Notebook sections 1–18** as specified in the task brief.  
**Sensitivity:** S1 full analysis-set; S2 exclude contrast; S3 exclude extreme-length; S4 conservative DE ambiguous rules; S5 duplicate-edition sensitivity if feasible.

**Runtime:** notebook execute ~5–20 min (no fulltext load).

---

## Dependencies

- Frozen v0.3 lexical / BM25 pointer CSVs under `data/processed/`
- `corpus_manifest_*_v1_candidate.csv`, `passage_manifest_*_v1_candidate.csv`
- `lexicon_dev_held_out_books_v03.txt`
- Existing v04 eval samples for reconciliation
- Project `.venv` with pandas, numpy, matplotlib, seaborn, jupyter, nbconvert

## Tests

```bash
.venv/bin/python -m pytest tests/retrieval/test_eval_sample_v05.py tests/retrieval -q
```

## Commits (logical separation)

1. docs: v0.5 analysis plan  
2. fix(retrieval): exclusive eval strata + v05 sample + tests  
3. docs/data: provenance v05 + PsyPer distinction + German audit v05  
4. docs(retrieval): BM25 terminology + set algebra v05  
5. feat: corpus QC table  
6. feat(analysis): helpers + concept families  
7. feat(notebook): executed notebook + reports  
8. docs: research log + v05 report/changelog  

## Acceptance criteria

- Evaluation strata mutually exclusive at population level  
- Sampling probabilities scientifically recorded  
- Prior annotations/samples preserved  
- Provenance limitations transparent  
- German heuristic audit not misrepresented  
- BM25 pool naming accurate  
- Corpus QC documented; snapshot frozen  
- Notebook fully executed, English prose, no copyrighted full texts  
- Research log updated  
