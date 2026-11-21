# Comparative PsyPer lexical retrieval report

Lexical baseline only — **not** manually validated. Do not treat candidate counts as concept prevalence.

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
| A_v01 | en | 16649 | 10964 | 12 |
| A_v01 | de | 2138 | 1531 | 12 |
| B_v02_lexical | en | 10611 | 7361 | 36 |
| B_v02_lexical | de | 1034 | 771 | 31 |
| C_bm25 | en | 5741 | 4202 | 36 |
| C_bm25 | de | 1021 | 762 | 31 |
| C_bm25_only_unvalidated | en | 2813 | 1804 | 36 |
| C_bm25_only_unvalidated | de | 3320 | 1647 | 36 |

## Overlap Baseline B ∩ C

- **en**: Jaccard=0.541 (B=10611, C=5741, ∩=5741)
- **de**: Jaccard=0.9874 (B=1034, C=1021, ∩=1021)

## Concepts per passage (Baseline B)

- **en**: max=21, mean=1.442, ≥10 concepts: 8
- **de**: max=15, mean=1.341, ≥10 concepts: 2

## Likely lexical-insufficient concepts

Concepts with zero or very few Baseline-B hits relative to corpus size are priority targets for later semantic retrieval.
### en (lowest assignment counts)
- `filter_words`: 1 assignments, 1 passages, 0.0003/10k words
- `pov_object`: 3 assignments, 3 passages, 0.0009/10k words
- `voice_participant`: 3 assignments, 3 passages, 0.0009/10k words
- `multiple_focalization_same_event`: 4 assignments, 4 passages, 0.0012/10k words
- `voice_heterodiegetic`: 6 assignments, 6 passages, 0.0017/10k words
- `pov_subject`: 8 assignments, 8 passages, 0.0023/10k words
- `focalization_fixed`: 14 assignments, 14 passages, 0.004/10k words
- `stance_narratorial`: 16 assignments, 16 passages, 0.0046/10k words

### de (lowest assignment counts)
- `focalization_external`: 1 assignments, 1 passages, 0.0021/10k words
- `reliability`: 1 assignments, 1 passages, 0.0021/10k words
- `pov_subject`: 1 assignments, 1 passages, 0.0021/10k words
- `pov_object`: 1 assignments, 1 passages, 0.0021/10k words
- `voice_heterodiegetic`: 1 assignments, 1 passages, 0.0021/10k words
- `stance_narratorial`: 2 assignments, 2 passages, 0.0043/10k words
- `multiple_focalization_same_event`: 3 assignments, 3 passages, 0.0064/10k words
- `focalization_general`: 3 assignments, 3 passages, 0.0064/10k words

## Before vs after (reference commit 83706dd)

Prior run (commit family around streaming BM25 fix): EN/DE each had exactly 1,800 candidate rows (50×36) with heavy regex-only high evidence leakage. This report replaces that with concept-specific evidence and uncapped Baseline B.

## Methodological limitations

- Lexical hits ≠ validated concept discussion; manual review is required.
- BM25 scores are not comparable across languages or concepts.
- Demoted/zero-hit terms remain in the master lexicon for semantic retrieval.
- Full passages are local/private; public CSVs are pointers only.

