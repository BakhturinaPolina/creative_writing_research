# PsyPer retrieval v0.3 changelog

Lexical baseline only — **not** manually validated. Provisional vocabulary additions require human confirmation.

## Algorithm vs vocabulary

### Algorithm / implementation (apply even with v0.2 vocabulary)

- Shared variant expansion for prefilter, detailed matching, term audit, and BM25 needle gates (`expand_phrase_variants` → `build_concept_prefilter_needles`).
- Streaming reservoir sampling for non-retrieved evaluation passages (no early stop at 5,000).
- PsyPer tag metadata: load YAML `psyper` codes as strings; emit `retrieval_concept_id`, `psyper_related_tags`, `human_validated_psyper_tags` (empty until manual review).
- Explicit evidence tiers independent of academic/craft/paraphrase origin (`evidence_overrides` + `resolve_evidence_tier`).
- Versioned output stems so v0.2 results are preserved.

### Vocabulary (v0.3 lexicon only)

- Bilingual paraphrases and craft expansions for low-coverage concepts: `focalization_fixed`, `voice_homodiegetic`, `voice_participant`, `voice_eyewitness`, `filter_words`, `focalization_zero`, `voice_heterodiegetic`.
- German coverage: Ich-/Du-/Er-/Sie-Perspektive and *-Form*, personale/auktoriale/allwissende terminology, `Nullfokalisierung`.
- Evidence overrides promoting highly specific craft/paraphrase cues (`filter words`, `head hopping`, `Filterwörter`, selected paraphrases).
- Additional DE regex patterns mapped in `regex_concept_map_v03.yml`.

## Provisional additions (human confirmation still required)

All newly added paraphrases and craft phrases in `config/pov_lexicon_psyper_en_de_v0.3.yml` are provisional. Seeded from the task brief and BM25-only discovery priorities; development passages are listed in `data/interim/retrieval/lexicon_dev_sample_v03.csv` (60 held-out books).

Priority human review after the v0.3 run:

- DE `focalization_zero` (5 → 133) and `stance_narratorial` / `stance_figural` jumps driven largely by new regex/craft (`allwissen*`, `auktorial*`, `personal*`) — check false positives outside narratological instruction.
- Remaining DE zero-hit concepts: `focalization_fixed`, `voice_eyewitness`, `voice_homodiegetic`, `voice_participant` (paraphrases too rare or still missing in corpus wording).
- EN `filter_words` (1 → 10): confirm craft/paraphrase hits discuss filters rather than fictional perception.

## Intermediate experiment

`pov_baseline_b_v02_algo_fix_{en,de}.csv` — unchanged v0.2 vocabulary with corrected shared prefilter/evidence pipeline — isolates algorithmic gains from vocabulary expansion.

## Files

| Path | Role |
|---|---|
| `config/pov_lexicon_psyper_en_de_v0.2.yml` | Frozen v0.2 lexicon |
| `config/pov_lexicon_psyper_en_de_v0.3.yml` | Expanded v0.3 lexicon |
| `config/regex_concept_map_v01.yml` | Frozen regex map |
| `config/regex_concept_map_v03.yml` | Extended regex map |
| `docs/retrieval_v03_changelog.md` | This file |
