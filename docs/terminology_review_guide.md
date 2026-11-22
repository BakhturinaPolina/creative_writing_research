# Terminology review guide (colleagues)

This guide explains how literary-studies and creative-writing colleagues can propose lexicon changes without editing the active retrieval dictionary.

## What you will use

Open the standalone report:

`reports/02_narrative_perspective_interactive.html`

It works offline. Suggestions stay in your browser until you export a file.

## Reviewer identity

On first open, the report creates a stable `reviewer_id` in local storage. Keep using the same browser profile if you want one continuous suggestion list.

## How to suggest a change

1. Go to section **13. Interactive bilingual lexicon**.
2. Search or filter vocabulary, or open a **concept card**.
3. Answer the short questions:
   - What term or expression is missing (or should change)?
   - Which language?
   - How should it be interpreted?
   - Is it a theoretical term, a practical writing expression, or a paraphrase?
   - Could it have another meaning that needs context?
4. Click **Save suggestion**.

Optional fields: source/reference, constructed example sentence, confidence.

Allowed actions include adding terms, paraphrases or variants; removing misleading terms; and proposing definition, PsyPer-mapping or matching-condition changes.

## Export and share

Use **Export JSON** (preferred for aggregation) or **Export CSV**.

Send the file to the project maintainers. Nothing is uploaded automatically.

## Import

**Import JSON** merges a previously exported file into your local list. Exact duplicates are skipped.

## Aggregation (maintainers)

```bash
.venv/bin/python -m src.lexicon_review.aggregate_suggestions \
  path/to/reviewer_a.json path/to/reviewer_b.json \
  --out data/interim/retrieval/terminology_review_consolidated.csv
```

Conflicting suggestions are kept as separate rows and labelled with a `conflict_group`. The active lexicon is never modified by this script.

## Important boundaries

- Suggestions remain `pending_review` until a human curator decides.
- Constructed examples must be labelled as constructed; do not paste copyrighted book passages into the shared report.
- Retrieval patterns in the report are dictionary matches, not validated narratological coding.
