# Comparative PsyPer retrieval report (v0.2 vs v0.3)

Lexical baseline only — **not** manually validated. Increased candidate counts are not evidence of improved recall without relevance judgments.

## Corpus denominators

| language | primary_group | n_books | body_words |
|---|---|---:|---:|
| en | contrast | 21 | 1,744,762 |
| en | fiction_manual | 290 | 22,615,325 |
| en | pedagogy_genre_craft | 131 | 10,221,177 |
| en | ALL | 442 | 34,581,264 |
| de | contrast | 11 | 1,331,992 |
| de | fiction_manual | 38 | 2,155,622 |
| de | pedagogy_genre_craft | 18 | 1,177,771 |
| de | ALL | 67 | 4,665,385 |

## Baseline summary

| baseline | language | rows | unique passages | concepts |
|---|---|---:|---:|---:|
| B_v02_lexical | en | 10611 | 7361 | 36 |
| B_v02_lexical | de | 1034 | 771 | 31 |
| B_v02_algo_fix | en | 10630 | 7374 | 36 |
| B_v02_algo_fix | de | 1050 | 777 | 31 |
| B_v03_lexical | en | 10643 | 7380 | 36 |
| B_v03_lexical | de | 1380 | 941 | 32 |
| C_v03_bm25 | en | 5756 | 4204 | 36 |
| C_v03_bm25 | de | 1339 | 906 | 32 |
| C_v03_bm25_only | en | 2812 | 1778 | 36 |
| C_v03_bm25_only | de | 3195 | 1593 | 36 |

## Algorithm vs vocabulary contributions

`B_v02_algo_fix` uses the v0.2 vocabulary with the corrected shared prefilter/evidence pipeline. Gains over `B_v02_lexical` are algorithmic; further gains in `B_v03_lexical` reflect vocabulary/tier changes.

### en
- `B_v02_lexical`: 10611 assignments, 7361 passages, 36 concepts
- `B_v02_algo_fix`: 10630 assignments, 7374 passages, 36 concepts
- `B_v03_lexical`: 10643 assignments, 7380 passages, 36 concepts

### de
- `B_v02_lexical`: 1034 assignments, 771 passages, 31 concepts
- `B_v02_algo_fix`: 1050 assignments, 777 passages, 31 concepts
- `B_v03_lexical`: 1380 assignments, 941 passages, 32 concepts

## Concepts with largest absolute gains (v0.2 → v0.3)

### en
- `grammatical_first_person`: 2497 → 2510 (algo +13/−0, vocab +0/−0)
- `filter_words`: 1 → 10 (algo +0/−0, vocab +9/−0)
- `voice_heterodiegetic`: 6 → 10 (algo +0/−0, vocab +4/−0)
- `grammatical_third_person`: 2028 → 2031 (algo +3/−0, vocab +0/−0)
- `grammatical_second_person`: 550 → 552 (algo +2/−0, vocab +0/−0)
- `thought_representation`: 376 → 377 (algo +1/−0, vocab +0/−0)
- `choice_tradeoffs`: 100 → 100 (algo +0/−0, vocab +0/−0)
- `consciousness_access`: 212 → 212 (algo +0/−0, vocab +0/−0)
- `distance_depth`: 153 → 153 (algo +0/−0, vocab +0/−0)
- `focalization_general`: 69 → 69 (algo +0/−0, vocab +0/−0)

### de
- `focalization_zero`: 5 → 133 (algo +3/−0, vocab +125/−0)
- `stance_narratorial`: 2 → 103 (algo +3/−0, vocab +98/−0)
- `stance_figural`: 10 → 75 (algo +7/−0, vocab +58/−0)
- `grammatical_third_person`: 46 → 89 (algo +1/−0, vocab +42/−0)
- `grammatical_second_person`: 8 → 12 (algo +0/−0, vocab +4/−0)
- `reliability`: 1 → 3 (algo +2/−0, vocab +0/−0)
- `filter_words`: 0 → 2 (algo +0/−0, vocab +2/−0)
- `grammatical_first_person`: 246 → 247 (algo +0/−0, vocab +1/−0)
- `distance_depth`: 11 → 11 (algo +0/−0, vocab +0/−0)
- `consciousness_access`: 32 → 32 (algo +0/−0, vocab +0/−0)

## Zero-hit / low-hit audit (v0.3 lexical)

### en: 0 concepts with zero lexical hits
Lowest non-zero (en):
- `pov_object`: 3 assignments (0.0009/10k)
- `voice_participant`: 3 assignments (0.0009/10k)
- `multiple_focalization_same_event`: 4 assignments (0.0012/10k)
- `pov_subject`: 8 assignments (0.0023/10k)
- `voice_heterodiegetic`: 10 assignments (0.0029/10k)
- `filter_words`: 10 assignments (0.0029/10k)
- `focalization_fixed`: 14 assignments (0.004/10k)
- `stance_narratorial`: 16 assignments (0.0046/10k)

### de: 4 concepts with zero lexical hits
- `focalization_fixed`
- `voice_eyewitness`
- `voice_participant`
- `voice_homodiegetic`
Lowest non-zero (de):
- `pov_object`: 1 assignments (0.0021/10k)
- `voice_heterodiegetic`: 1 assignments (0.0021/10k)
- `pov_subject`: 1 assignments (0.0021/10k)
- `focalization_external`: 1 assignments (0.0021/10k)
- `filter_words`: 2 assignments (0.0043/10k)
- `reader_effect`: 3 assignments (0.0064/10k)
- `multiple_focalization_same_event`: 3 assignments (0.0064/10k)
- `focalization_general`: 3 assignments (0.0064/10k)

## Overlaps

- **en** B_v02_lexical ∩ B_v02_algo_fix: Jaccard=0.9982 (∩=10611, a_only=0, b_only=19)
- **en** B_v02_algo_fix ∩ B_v03_lexical: Jaccard=0.9988 (∩=10630, a_only=0, b_only=13)
- **en** B_v02_lexical ∩ B_v03_lexical: Jaccard=0.997 (∩=10611, a_only=0, b_only=32)
- **en** B_v03_lexical ∩ C_v03_bm25: Jaccard=0.5408 (∩=5756, a_only=4887, b_only=0)
- **de** B_v02_lexical ∩ B_v02_algo_fix: Jaccard=0.9848 (∩=1034, a_only=0, b_only=16)
- **de** B_v02_algo_fix ∩ B_v03_lexical: Jaccard=0.7609 (∩=1050, a_only=0, b_only=330)
- **de** B_v02_lexical ∩ B_v03_lexical: Jaccard=0.7493 (∩=1034, a_only=0, b_only=346)
- **de** B_v03_lexical ∩ C_v03_bm25: Jaccard=0.9703 (∩=1339, a_only=41, b_only=0)

## Multi-concept passages (v0.3 lexical)

- **en**: max=21, mean=1.442, ≥10 concepts: 8
- **de**: max=15, mean=1.467, ≥10 concepts: 3

## Cross-linguistic coverage caveat

Before interpreting EN–DE frequency differences as literary or cultural, consult `retrieval_coverage_audit_v03.csv`. Unequal lexical coverage (especially paraphrases and DE morphology) may explain residual gaps.

## Evaluation protocol (future)

Compare on the held-out manual sample (plus later semantic-only additions): A v0.1 · B v0.2 · B v0.3 · BM25 · dense · RRF · cross-encoder rerank. Do not claim scientific validation until judgments are collected.

## Methodological limitations

- Lexical hits ≠ validated concept discussion.
- Stratified candidate samples do not estimate absolute corpus-wide recall.
- Provisional lexicon additions require human confirmation.
- Full passages remain local/private; public CSVs are pointers only.

