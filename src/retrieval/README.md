# POV passage retrieval (research plan weeks 15–16)

Find candidate POV passages with the bilingual PsyPer lexicon, then (later) sentence embeddings / hybrid retrieval.

High recall matters more than precision at this stage. Book-level `corpus_stats` still uses lexicon **v0.1** (`config/pov_lexicon.yml`); passage retrieval uses additive **v0.2** / **v0.3** search lexicons plus explicit regex→concept maps.

**Lexical baselines are not manually validated.** Do not treat candidate counts as concept prevalence.

## Lexicons and maps

| File | Role |
|---|---|
| [`config/pov_lexicon.yml`](../../config/pov_lexicon.yml) | v0.1 — Baseline A + book-level POV density |
| [`config/pov_lexicon_psyper_en_de_v0.2.yml`](../../config/pov_lexicon_psyper_en_de_v0.2.yml) | v0.2-draft — frozen PsyPer retrieval cues |
| [`config/pov_lexicon_psyper_en_de_v0.3.yml`](../../config/pov_lexicon_psyper_en_de_v0.3.yml) | v0.3-draft — expanded cues + `evidence_overrides` |
| [`config/regex_concept_map_v01.yml`](../../config/regex_concept_map_v01.yml) | Regex map for v0.2 |
| [`config/regex_concept_map_v03.yml`](../../config/regex_concept_map_v03.yml) | Regex map for v0.3 (+ DE person/omniscience) |
| [`docs/retrieval_v03_changelog.md`](../../docs/retrieval_v03_changelog.md) | Algorithm vs vocabulary change log |

## Retrieval baselines

| Baseline | Method | Output stem |
|---|---|---|
| **A** | v0.1 keyword (dimensions) | `pov_baseline_a_v01_{lang}.csv` |
| **B v0.2** | v0.2 lexical (frozen prior run) | `pov_baseline_b_v02_lexical_{lang}.csv` |
| **B algo-fix** | v0.2 vocab + shared variant prefilter | `pov_baseline_b_v02_algo_fix_{lang}.csv` |
| **B v0.3** | v0.3 lexical + evidence tiers (uncapped) | `pov_baseline_b_v03_lexical_{lang}.csv` |
| **C v0.3** | streaming BM25 top-k + literal gate | `pov_baseline_c_v03_bm25_{lang}.csv` |
| C′ v0.3 | BM25-only exploratory | `pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv` |

Also: compact `pov_baseline_c_v03_bm25_top50_{lang}.csv`. Legacy v0.2 C stems remain under `pov_baseline_c_bm25_*`.

## Evidence tiers (v0.3)

Explicit strength is independent of vocabulary origin (`academic` / `craft` / `paraphrase` / `regex` / `context`). Origin is retained as `vocab_origin` on matched spans.

| Evidence tier | Qualifies alone? |
|---|---|
| `strong_academic` / `strong_craft` / `strong_paraphrase` | Yes (unless weak POV surface) |
| `supporting` | Needs another independent cue (or weak POV + one) |
| `context_only` | Never |
| `ambiguous_demoted` | Never |

Overrides live under each language block as `evidence_overrides`. Grammatical person never validates diegesis, focalization, or filter words by itself.

Output metadata (v0.3 writers):

- `retrieval_concept_id` — research concept ID
- `psyper_related_tags` — string PsyPer codes from the lexicon (e.g. `2.2.1`)
- `human_validated_psyper_tags` — empty until manual review

## Shared matching

Prefilter, detailed matching, and needle construction share `expand_phrase_variants` / `build_concept_prefilter_needles` so morphological and orthographic variants are not dropped before evidence collection.

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

# v0.3 pipeline (algo-fix → B v0.3 → C v0.3 → report → v0.4 eval → manifest/snapshot)
.venv/bin/python -m src.retrieval.run_all --lang all --pipeline v03

# Individual steps
.venv/bin/python -m src.retrieval.keyword_baselines --baseline b --lang all --lexicon-version v02_algo_fix
.venv/bin/python -m src.retrieval.keyword_baselines --baseline b --lang all --lexicon-version v03
.venv/bin/python -m src.retrieval.bm25_passages --lang all --lexicon-version v03 --top-k 500
.venv/bin/python -m src.retrieval.report --version v03
.venv/bin/python -m src.retrieval.report --version set_algebra
.venv/bin/python -m src.retrieval.eval_sample --version v04
.venv/bin/python -m src.retrieval.passage_manifest --lang all
.venv/bin/python -m src.retrieval.corpus_snapshot --lang all
.venv/bin/python -m src.retrieval.dev_sample   # BM25-only lexicon-expansion holdout
.venv/bin/python -m src.retrieval.matcher_audit --n-per-lang 200
.venv/bin/python -m src.retrieval.de_vocab_audit
```


Optional `--with-passages` writes gitignored sidecars under `data/interim/retrieval/*_passages.csv`.

## Evaluation samples

| Sample | Role |
|---|---|
| `manual_eval_sample_v03.*` | **Frozen** 247-passage sample (do not overwrite) |
| `manual_eval_sample_prob_v04.*` | Design A — probability-based; records `N_h`, `n_h`, `π_h`, weights |
| `manual_eval_sample_diversity_v04.*` | Design B — book-diversity diagnostic; **no** probability weights |
| `manual_eval_sample_v03_v04_reconciliation.csv` | Passage overlap across samples |

Design A estimates refer to stratum populations inside the **independent evaluation population** (analysis set minus lexicon-dev held-out books). Design B is qualitative only. Annotator CSVs hide method/rank/predicted concept. Review fields stay empty until human coding.

```bash
.venv/bin/python -m src.retrieval.eval_sample --version v04
```

## BM25 set algebra (v0.4)

Canonical key: `(passage_id, retrieval_concept_id)`. Report: `retrieval_set_algebra_v04.md`.
Genuinely new BM25 assignments are **`B \ L`**, not merely “absent from qualified BM25”.

## Corpus snapshot (candidate)

```bash
.venv/bin/python -m src.retrieval.corpus_snapshot --lang all
```

Writes `corpus_manifest_{lang}_v1_candidate.csv` and hashed `passage_manifest_{lang}_v1_candidate.csv` under snapshot ID `corpus_snapshot_analysis_v1_candidate_20261122`. Full texts stay private.

## German lexical audit

```bash
.venv/bin/python -m src.retrieval.de_vocab_audit
```

Provisional AI assessments only — see `german_lexical_audit_v04.md`.

## Hugging Face Jobs (CPU) fallback

If local RAM is insufficient, run the same CLIs on a CPU Job with private storage for restricted fulltexts. Do not launch paid jobs without approval. Do not publish passage sidecars.

## Outputs

| Path | Content |
|---|---|
| `data/processed/pov_baseline_*_{lang}.csv` | pointers + scores + evidence — **no book text** |
| `data/processed/passage_manifest_{lang}.csv` | all eligible passages for future embeddings |
| `data/interim/retrieval/*_passages.csv` | full passages (gitignored) |
| `data/interim/retrieval/retrieval_comparative_report_v03.md` | v0.2 vs v0.3 comparative stats |
| `data/interim/retrieval/retrieval_coverage_audit_v03.csv` | EN/DE lexicon coverage audit |
| `data/interim/retrieval/manual_eval_sample_v03.csv` | stratified review sheet (empty gold fields) |
| `tests/retrieval/` | bilingual regressions (v0.2 + v0.3) |

## Future semantic retrieval

Stable `passage_id` (`book_id:c{chapter}:p{para}`) joins lexical, BM25, and future embedding results. The passage manifest includes every eligible paragraph (including zero lexical hits). Planned comparison protocol: A · B v0.2 · B v0.3 · BM25 · dense · RRF · cross-encoder rerank — after manual judgments.

## Compatibility

Do not overwrite frozen v0.2 outputs when iterating on v0.3. Keep lexical and semantic evidence columns separate when hybrid retrieval is added.
