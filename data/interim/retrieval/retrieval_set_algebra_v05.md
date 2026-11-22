# Retrieval set algebra (v0.5 terminology)

Canonical assignment key: `(passage_id, retrieval_concept_id)`.

## Symbols

| Symbol | Meaning |
|---|---|
| **L** | Lexical retrieval assignments (B v0.3) |
| **Bq** | Lexical-qualified BM25 assignments (C v0.3; top-K = 500) |
| **Braw** | Independent exploratory BM25 heap (top-K = 100) *before* subtracting L |
| **Bnew** | Genuinely new exploratory assignments: ideally Braw \\ L |
| **U** | Union of lexical and exploratory pools: ideally L ∪ Braw |
| **B_combined** | Reconstructed pool Bq ∪ bm25_only_file (historical stand-in) |

**Important:** The complete historical `Braw` output is not available on disk.
Do **not** interpret `B_combined` as an independent top-K BM25 ranking.
Qualified and exploratory heaps used different K limits (500 vs 100).

Passage-level and assignment-level comparisons are reported separately.

## en

- Assignments: |L|=10643, |Bq|=5756, |B_combined|=8568, |Bnew_file|=2812, |Bnew_from_combined|=2812, |U_combined|=13455
- Bnew_file ∩ L (expect 0): 0
- Passages: L=7380, B_combined=5213, U_combined=8200; genuinely new passages=820; new concepts on known passages=958
- Observed max ranks: Bq=500, Bonly_file=100 (configs 500 / 100)
- `Braw` not reconstructed from historical files.

## de

- Assignments: |L|=1380, |Bq|=1339, |B_combined|=4534, |Bnew_file|=3195, |Bnew_from_combined|=3195, |U_combined|=4575
- Bnew_file ∩ L (expect 0): 0
- Passages: L=941, B_combined=2170, U_combined=2204; genuinely new passages=1263; new concepts on known passages=330
- Observed max ranks: Bq=240, Bonly_file=100 (configs 500 / 100)
- `Braw` not reconstructed from historical files.

