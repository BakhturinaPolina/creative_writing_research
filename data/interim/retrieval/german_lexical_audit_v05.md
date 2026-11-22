# German lexical audit clarification (v0.5)

**Rows:** 78 (frozen from v0.4 sample; development held-out books only).

## Critical interpretation warning

The heuristic labelled approximately **71/78** rows as `genuine_narratological_discussion`.

**This proportion must NOT be interpreted as retrieval precision.**

Reasons:

1. Categories are `heuristic_provisional_assessment` (surface heuristics).
2. The heuristic partly uses the triggering lexical expressions themselves as evidence of relevance.
3. This table is a **lexicon development** dataset, not the independent probability-based evaluation sample.
4. Human-review fields are present but **empty** — no human gold labels here.

## Wissensvorsprung

Several passages discuss dramatic irony, suspense, audience knowledge and information asymmetry. Although relevance to focalization may be uncertain, they may be highly relevant to `information_control` or `reader_effect`.
They are **not** automatically classified as unrelated.

## Heuristic category counts (not precision)

- `genuine_narratological_discussion`: 71
- `unrelated_or_unclear`: 6
- `general_literary_discussion`: 1

## Human review fields (empty)

`human_category`, `human_error_type`, `human_ambiguity`, `human_notes`, `reviewer_id`, `review_date`

Preserve v0.4 artifacts unchanged. Keep this audit separate from `manual_eval_sample_prob_v05`.

