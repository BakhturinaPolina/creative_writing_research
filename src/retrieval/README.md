# POV passage retrieval (research plan weeks 15–16)

Find candidate POV passages with the bilingual PsyPer lexicon, then (later) sentence embeddings / hybrid retrieval.

High recall matters more than precision at this stage. Book-level `corpus_stats` still uses lexicon **v0.1** (`config/pov_lexicon.yml`); this package uses the additive **v0.2-draft** search lexicon plus an explicit regex→concept map.

**Lexical baselines are not manually validated.** Do not treat candidate counts as concept prevalence.

## Lexicons and maps

| File | Role |
|---|---|
| [`config/pov_lexicon.yml`](../../config/pov_lexicon.yml) | v0.1 — Baseline A + book-level POV density |
| [`config/pov_lexicon_psyper_en_de_v0.2.yml`](../../config/pov_lexicon_psyper_en_de_v0.2.yml) | v0.2-draft — PsyPer retrieval cues |
| [`config/regex_concept_map_v01.yml`](../../config/regex_concept_map_v01.yml) | Explicit regex → permitted concept IDs + roles |

## Three retrieval baselines

| Baseline | Method | Output stem |
|---|---|---|
| **A** | v0.1 keyword (dimensions) | `pov_baseline_a_v01_{lang}.csv` |
| **B** | v0.2 lexical + concept-specific evidence (uncapped) | `pov_baseline_b_v02_lexical_{lang}.csv` |
| **C** | streaming BM25 top-k + literal gate | `pov_baseline_c_bm25_{lang}.csv` |
| C′ | BM25-only exploratory (`bm25_only_unvalidated`) | `pov_baseline_c_bm25_only_unvalidated_{lang}.csv` |

Also: compact `pov_baseline_c_bm25_top50_{lang}.csv` and legacy alias `pov_passage_candidates_{lang}.csv`.

## Evidence tiers (shared by audit, B/C, tests)

| Tier | Qualifies alone? |
|---|---|
| Academic / high | Yes (unless weak POV surface) |
| Supporting craft | Only with another independent cue (or weak POV + one) |
| Ambiguous / demoted | Never |
| Context-only | Never |
| Paraphrase | Counts as supporting if matched |
| Regex | Only for mapped concepts; `context_only` / `supporting_insufficient` roles respected |

Grammatical person never validates diegesis, focalization, or filter words by itself.

## Memory model

Streaming only: one book JSON resident; BM25 uses two corpus passes and top-k heaps. Target peak RSS **< 2 GB**.

## BM25 conventions

- Tokenization: `textnorm.tokenize` (NFKC, casefold, Unicode word tokens)
- IDF: `log((N − df + 0.5) / (df + 0.5))`; negative IDF → `ε × mean_idf` (ε=0.25)
- `k1=1.5`, `b=0.75`
- Query construction: dedupe expressions and tokens; drop function words; optional weights via `QueryConfig`
- Default score floor: 2.0; default top-k: 500 (keep fewer if fewer qualify)
- **Do not** compare raw BM25 scores across languages or concepts

## Commands

```bash
# Regression tests (required before full runs)
.venv/bin/python -m pytest tests/retrieval -q

# Keyword research
.venv/bin/python -m src.retrieval.term_audit --lang all

# Baselines
.venv/bin/python -m src.retrieval.keyword_baselines --baseline all --lang all
.venv/bin/python -m src.retrieval.bm25_passages --lang all --top-k 500

# Report + manual eval sample
.venv/bin/python -m src.retrieval.report
.venv/bin/python -m src.retrieval.eval_sample

# Or orchestrate (tests → A/B/C → report → sample)
.venv/bin/python -m src.retrieval.run_all --lang all --with-passages
```

Optional `--with-passages` writes gitignored sidecars under `data/interim/retrieval/*_passages.csv`.

## Hugging Face Jobs (CPU) fallback

If local RAM is insufficient, run the same CLIs on a CPU Job with private storage for restricted fulltexts. Mount or sync `data/interim/fulltext/json/` and `data/processed/{en,de}_book_stats.csv`, then execute `python -m src.retrieval.run_all --lang all`. Do not publish passage sidecars.

## Outputs

| Path | Content |
|---|---|
| `data/processed/pov_baseline_*_{lang}.csv` | pointers + scores + evidence — **no book text** |
| `data/interim/retrieval/*_passages.csv` | full passages (gitignored) |
| `data/interim/retrieval/retrieval_comparative_report.md` | comparative stats |
| `data/interim/retrieval/manual_eval_sample.csv` | stratified review sheet (empty gold fields) |
| `tests/retrieval/test_evidence_regression.py` | ≥60 bilingual concept-specific cases |

## Compatibility

Passage IDs (`book_id:c{chapter}:p{para}`) are stable across baselines for later RRF / embedding fusion. Keep lexical and semantic evidence columns separate when hybrid retrieval is added.
