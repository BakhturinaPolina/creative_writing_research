# German lexical audit (v0.4)

**All AI assessments are provisional** and are not human ground truth.
Independent evaluation labels were not modified.

## Gain attribution (DE focus concepts)

| concept | v0.2 | algo_fix | v0.3 | +algo | +vocab |
|---|---:|---:|---:|---:|---:|
| `focalization_zero` | 5 | 8 | 133 | 3 | 125 |
| `stance_narratorial` | 2 | 5 | 103 | 3 | 98 |
| `stance_figural` | 10 | 17 | 75 | 7 | 58 |
| `grammatical_third_person` | 46 | 47 | 89 | 1 | 42 |

Cause breakdowns (vocab-new) are in `german_gain_attribution_v04.csv`.

## Conceptual distinctions (human decisions required)

- Do **not** equate auktoriales Erzählen with zero focalization (regex map already routes `auktorial` → `stance_narratorial`).
- Do **not** equate personales Erzählen with internal focalization automatically.
- Lexicon still lists some overlapping academic cues (e.g. `auktoriale Erzählsituation` under `focalization_zero`) — **human decision** whether to demote/remove.
- Prefer additional evidence requirements for ambiguous terms over silent deletion.

## Trigger sample

- Annotation-ready rows: 78 (`german_audit_review_v04.csv`)
- Provisional category counts: `{'genuine_narratological_discussion': 71, 'unrelated_or_unclear': 6, 'general_literary_discussion': 1}`

## Low-coverage concepts

- `de` `focalization_fixed`: lexical=0, bm25_only=100 → possible_retrieval_failure_or_rare_wording
- `de` `voice_eyewitness`: lexical=0, bm25_only=100 → possible_retrieval_failure_or_rare_wording
- `de` `voice_participant`: lexical=0, bm25_only=100 → possible_retrieval_failure_or_rare_wording
- `de` `voice_homodiegetic`: lexical=0, bm25_only=100 → possible_retrieval_failure_or_rare_wording
- `en` `pov_object`: lexical=3, bm25_only=100 → low_lexical_hit_with_exploratory_bm25_candidates
- `en` `pov_subject`: lexical=8, bm25_only=99 → low_lexical_hit_with_exploratory_bm25_candidates
- `en` `voice_participant`: lexical=3, bm25_only=100 → low_lexical_hit_with_exploratory_bm25_candidates
- `en` `multiple_focalization_same_event`: lexical=4, bm25_only=97 → low_lexical_hit_with_exploratory_bm25_candidates

Preserve zero/low-coverage concepts as diagnostic cases for semantic retrieval.

## Proposed rule changes

1. **Human decision:** whether `auktoriale Erzählsituation` should remain under `focalization_zero` academic terms or move to stance-only.
2. **Mechanical (optional):** demote bare `allwissen*` hits outside narratological collocates — needs human confirmation before v0.4 lexicon edit.
3. No automatic conceptual remaps applied in this finalization pass.

## Known segmentation anomaly

- `de_26856f179d5e:c7:p0` ≈ 102k characters (single paragraph chapter) — flag in corpus QC; do not silently truncate.
