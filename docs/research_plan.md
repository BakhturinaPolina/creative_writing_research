# Research plan (24 weeks)

Assumption: 10-15 h per week, so about 320-470 h in total. Phases overlap on purpose. Library negotiations run in the background while open material is processed.

## What changed relative to the original notes

1. **Existing candidates come before new harvesting.** The colleague's OpenLibrary data and the seed bibliography are migrated and benchmarked first. New DNB/HeBIS retrieval then becomes targeted gap-filling, not a restart.
2. **Library/DNB contact moves to week 2.** Negotiating TDM access is slow and does not depend on the bibliography being finished.
3. **The period decision comes after the catalogue counts.** 1970/1980 stays provisional (DEC-002) until the Ngram and publication-frequency evidence is in.
4. **The POV lexicon is drafted early (v0) but finalized only after the pilot texts exist (v2).**

## Milestones

| Milestone | Weeks | Question | Output |
|---|---|---|---|
| M1 | 1-5 | What counts, and what do we already have? | Corpus protocol v1, gold benchmark, Ngram figures |
| M2 | 6-11 | What exists, and can we justify the period? | Master bibliography, publication history, period decision |
| M3 | 11-18 | What can we obtain, and can POV be found reliably? | Pilot corpus, retrieval prototype, gold annotations |
| M4 | 18-24 | What does the corpus tell us? | Validated classifier, book-level dataset, first results |

## Week by week

| Week | Work | Deliverable | Hours | Depends on |
|---|---|---|---:|---|
| 1 | Freeze the operational definition, corpus types A-F, exclusion reasons (`config/inclusion_rules.yml`) | Corpus protocol v0.9 | 10-12 | none |
| 2 | Add the missing historical seeds (Besant, James, Lubbock, Wharton editions, Forster, Brande). Map the seed `Book_type` to A-F. Draft the TDM letters | Seed bibliography v1 (~110 titles). Letters to Goethe UB DH and DNB sent | 10-15 | W1 |
| 3 | Migrate the colleague OpenLibrary CSVs into the master schema (header repair, W/M IDs, provenance) | `data/interim/catalogue/openlibrary_candidates.csv` | 10-12 | W1 |
| 4 | Rule-based pre-screen (`likely_primary/contextual/exclude/uncertain`). Gold benchmark recall per source | Pre-screened candidates, recall table | 10-12 | W2, W3 |
| 5 | Ngram figures: corpus terms, POV terms, current corpus vs 2019. DNB SRU query testing by hand, subject-heading discovery | Ngram figures and milestones, revised `catalogue_queries.yml` | 10-15 | W1 |
| 6 | DNB harvest (title + subject queries, raw XML, query-hit table). Switch to MARC21-XML | `dnb_hits_*.csv` | 12-15 | W5 |
| 7 | HeBIS client and harvest. KVK manual gap checks for benchmark titles that are still missing | `hebis_hits_*.csv`, KVK notes in `data/raw/manual_kvk/` | 12-15 | W6 |
| 8 | Merge all sources into the master query-hit table (provenance kept) | `data/interim/catalogue_merged.parquet` | 10-12 | W4, W6, W7 |
| 9 | Enrichment of the screened subset: ISBN, publisher, subjects, OCLC, DNB_ID, translation fields | Enriched candidates | 12-15 | W8 |
| 10 | Deduplication: identifiers, then normalized title/author, then fuzzy matching. Assign `work_id` / `edition_id` | `catalogue_deduplicated.csv` | 12-15 | W9 |
| 11 | Manual eligibility coding (at least 500 records). Publication frequency (new works vs editions; EN / DE-original / DE-translated; 5-year bins). **Period decision** | `publication_frequency.csv`, figures, DEC entry | 12-15 | W5, W10 |
| 12 | Full-text access screen: OpenLibrary/IA `ebook_access`, DNB free digital objects, Gutenberg/HathiTrust, Frankfurt holdings. Follow up the TDM requests | `fulltext_status`, `rights_TDM_status` filled | 10-12 | W2, W11 |
| 13 | Acquisition of the pilot (100 EN + 100 DE, all eligible types, not only POV-positive books) | Pilot files | 12-15 | W12 |
| 14 | Text extraction (PDF/EPUB), OCR, cleaning, segmentation into chapters/sections/paragraphs | Cleaned pilot corpus | 12-15 | W13 |
| 15 | POV lexicon v2 (incorporate the cue xlsx and video terminology). Lexical retrieval | Lexical candidates | 10-12 | W14 |
| 16 | Embedding retrieval against semantic prototypes; union of candidates | Retrieval pipeline v1 | 12-15 | W15 |
| 17 | Annotation guidelines, then annotate 400-700 passages, with a double-annotated subset | Gold set part 1, agreement scores | 12-15 | W16 |
| 18 | Annotate up to 800-1,500 passages. Revise the taxonomy | Gold set v1 | 12-15 | W17 |
| 19 | Train the classifier (TF-IDF + LogReg baseline, then embedding classifier). Measure precision/recall/F1 | Model report | 12-15 | W18 |
| 20 | Error analysis. Recall check on historical vocabulary (target recall at least .90) | Validated POV detector | 10-12 | W19 |
| 21 | Scale acquisition to all accessible eligible books. Full-corpus inference | Passage predictions | 12-15 | W12, W20 |
| 22 | Book-level measures and decade aggregation | Analysis dataset | 10-12 | W21 |
| 23 | Diachronic and EN/DE models. Translation-mediation analysis (RQ6) | Results tables and figures | 12-15 | W22 |
| 24 | Close reading of representative passages. Write-up of methods and results | Draft paper / report | 12-15 | W23 |

## Running in parallel

- **Library/TDM access (from W2):** emails, meetings, agreement on formats and computing environment.
- **Research log:** at least one entry per session ([research_log.md](research_log.md)).
- **Literature:** history of literary-advice books, *Kreatives Schreiben* in Germany, narratology of perspective.

## Exit criteria per milestone

- **M1:**
  - The protocol is frozen.
  - The benchmark has recall per source.
  - Ngram figures exist for EN and DE.
- **M2:**
  - At least 1,500 candidate records are deduplicated into works/editions.
  - At least 500 are manually coded.
  - The period decision is recorded as a DEC entry.
- **M3:**
  - The 200-book pilot is cleaned.
  - The retrieval prototype runs.
  - At least 800 passages are annotated.
- **M4:**
  - Classifier recall is at least .90 on substantive POV passages.
  - Book-level dataset and first comparative results exist.
