# Retrieval set algebra (v0.4)

Canonical assignment key: `(passage_id, retrieval_concept_id)`.

- **L** — lexical candidates (B v0.3)
- **Bq** — BM25 with lexical qualification (C v0.3)
- **B** — exploratory BM25 pool (reconstructed as Bq ∪ BM25-only file)
- **B\\L** — genuinely new BM25 assignments (not in lexical baseline)
- **L\\B** — lexical assignments absent from exploratory BM25
- **L∩B** — shared assignments

A candidate is **not** exclusively BM25-retrieved if the same passage–concept pair is already in L.

Estimates refer to the analysis-set retrieval candidate pools, not absolute corpus-wide prevalence.

## en

- Assignments: |L|=10643, |Bq|=5756, |B|=8568, |B\\L|=2812, |L\\B|=4887, |L∩B|=5756
- BM25-only file ∩ L (should be 0 if file stores B\\L or B\\Bq with Bq⊆L): 0
- Passages: L=7380, B=5213; genuinely new passages=820; new concept-assignments on already-lexical passages=958

## de

- Assignments: |L|=1380, |Bq|=1339, |B|=4534, |B\\L|=3195, |L\\B|=41, |L∩B|=1339
- BM25-only file ∩ L (should be 0 if file stores B\\L or B\\Bq with Bq⊆L): 0
- Passages: L=941, B=2170; genuinely new passages=1263; new concept-assignments on already-lexical passages=330

