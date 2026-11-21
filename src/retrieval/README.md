# POV passage retrieval (research plan weeks 15–16)

Find candidate POV passages with the bilingual PsyPer lexicon, then (later) sentence embeddings.

High recall matters more than precision at this stage. Book-level `corpus_stats` still uses lexicon **v0.1** (`config/pov_lexicon.yml`); this package uses the additive **v0.2-draft** search lexicon only.

## Lexicons

| File | Role |
|---|---|
| [`config/pov_lexicon.yml`](../../config/pov_lexicon.yml) | v0.1 — book-level POV density in `src.analysis.corpus_stats` |
| [`config/pov_lexicon_psyper_en_de_v0.2.yml`](../../config/pov_lexicon_psyper_en_de_v0.2.yml) | v0.2-draft — PsyPer-aligned retrieval cues (academic / craft / paraphrases / context) |

## Memory model

On machines with ~5 GB RAM, never load all analysis-set paragraphs into a list. Both CLIs **stream one paragraph at a time** (one book JSON resident). BM25 uses two corpus passes and keeps only a top-k heap per concept (~150 MB peak RSS for English ~315k paragraphs).

## Commands

```bash
# Keyword research: DF/TF of academic+craft terms; write tiers + pilot smoke
python -m src.retrieval.term_audit --lang all   # or en|de

# BM25 passage candidates (streaming Okapi BM25)
python -m src.retrieval.bm25_passages --lang all
python -m src.retrieval.bm25_passages --lang en --top-k 50 --score-floor 2.0
```

Optional dependency: `rank_bm25` (listed in `requirements.txt`). The streaming path uses a small local Okapi implementation and does not require it.

## Outputs

| Path | Content |
|---|---|
| `data/interim/retrieval/lexicon_term_audit_{en,de}.csv` | per-term DF/TF by role |
| `data/interim/retrieval/lexicon_term_tiers.yml` | high / supporting / demoted / unvalidated + catalogue candidates |
| `data/interim/retrieval/lexicon_pilot_smoke.md` | pilot + everyday false-positive checks |
| `data/processed/pov_passage_candidates_{en,de}.csv` | ranked hits: book/paragraph pointers, scores, cue evidence — **no book text** |
| `data/interim/retrieval/pov_passage_candidates_{en,de}_passages.csv` | optional sidecar with passage text (`--with-passages`); **gitignored** |

## Candidate rule

A paragraph is kept if BM25 score ≥ floor **and** literal evidence passes:

- a *strong* high-tier / regex hit, or
- a weak POV surface (`point of view`, `regex:general_pov`, …) **plus** an independent supporting cue, or
- ≥2 independent supporting craft hits

Context-only terms never retrieve alone. Paraphrases are BM25 query text, not literal DF features.
