# Research log

Newest entry first. One entry per working session (or per day). Always link outputs to a path and to a git commit, so that results can be traced back.

## How to write an entry

```markdown
## YYYY-MM-DD | Week N | Phase: <Protocol|Candidates|Ngram|Catalogue|Dedup|Access|Fulltext|Retrieval|Annotation|Model|Analysis> | X.X h
**Goal:** what this session was meant to achieve
**Done:** what actually happened (facts, numbers)
**Data/outputs:** paths + row counts + commit hash
**Decisions:** DEC-XXX (one line; full entry in the register)
**Problems / open questions:** ...
**Next:** concrete next action
```

Rules:

- Record negative results too. A query that produced only noise is evidence.
- Any change to inclusion rules, cutoffs, lexicon or taxonomy becomes a DEC entry.
- Counts always state their unit: records, works, editions or passages.

---

## Decision register

| ID | Date | Decision | Evidence / reason | Status |
|---|---|---|---|---|
| DEC-001 | 2026-10-08 | Population corpus (dataset A) has **no POV requirement** | Avoids circular selection that would inflate prevalence (RQ1) | frozen |
| DEC-002 | 2026-10-08 | Main period 1970-2025, historical extension 1880-1969 | Literature on EN/DE history. To be confirmed by Ngram and catalogue counts | provisional |
| DEC-003 | 2026-10-08 | Colleague OpenLibrary data is kept as a frozen raw discovery layer, not merged into the corpus directly | Noise (DE ~72% non-primary), seed recall DE ~3/50 | frozen |
| DEC-004 | 2026-10-08 | DNB + HeBIS are the primary German sources; WorldCat optional; KVK manual only | Coverage, open access to APIs, KVK has no database of its own | provisional |
| DEC-005 | 2026-10-08 | Works and editions are modelled separately (`work_id`, `edition_id`) | New-works vs editions-per-year are different measures | frozen |

Status values: `provisional` (open for revision), `frozen` (changes need a new DEC that supersedes it), `superseded`.

---

## Hours tracker

| Phase | Budget (h) | Spent (h) | Notes |
|---|---:|---:|---|
| Protocol & vocabulary | 20-30 | 0 | |
| Existing candidates & gold benchmark | 15-20 | 0 | |
| Ngram analysis | 10-15 | 0 | |
| Catalogue harvesting (DNB/HeBIS/KVK) | 25-35 | 0 | |
| Enrichment & deduplication | 30-45 | 0 | |
| Eligibility coding & publication history | 20-30 | 0 | |
| Library / TDM access | 15-25 | 0 | runs in parallel |
| Full-text acquisition, OCR, cleaning | 50-80 | 0 | |
| Retrieval prototype | 30-40 | 0 | |
| Annotation | 30-45 | 0 | |
| Classifier & evaluation | 25-35 | 0 | |
| Full inference & analysis | 50-70 | 0 | |
| **Total** | **320-470** | **0** | |

---

## Entries

## 2026-10-10 | Week 5–7 | Phase: Catalogue | — h
**Goal:** Reorganize `src/` by research-plan phase; harden DNB; implement HeBIS + improved Open Library; shared logging.
**Done:**
- `src/` now has phase folders (`common`, `candidates`, `ngrams`, `catalogue`, `merge_enrich`, `publication`, plus README-only later phases). Each folder has a plain-English README. Run logs go to `logs/`.
- DNB practice (1 page each): `erzählperspektive` 61, `schriftstellerei` 78, `kreatives schreiben` 100 of 585. Filled starter `subject_headings` in `config/catalogue_queries.yml` (DDC 800/808/830; GND label starters).
- `dnb_seed_gaps --lang de --limit 8`: **8/8** seeds hit (23 hit rows) → `data/interim/dnb/dnb_seed_gaps_de.csv`.
- HeBIS client implemented (`src/catalogue/hebis.py`, `hebis_seed_holdings.py`). From this host `sru.hebis.de` **times out** (connect); empty CSVs written cleanly.
- Improved OL smoke: DE 206 rows / **97% ISBN**; EN 193 / **95% ISBN**. Seed recall on smoke slices only: EN 13/50, DE 2/50 (full harvest not run yet). DE OLID overlap with filtered: 79/80.
**Data/outputs:**
- `data/interim/dnb/dnb_hits_tit_{erzählperspektive,schriftstellerei,kreatives_schreiben}.csv`
- `data/interim/dnb/dnb_seed_gaps_de.csv`
- `data/interim/openlibrary/ol_search_{en,de}.csv`
- `data/interim/hebis/hebis_seed_holdings.csv` (0 rows; host unreachable)
- `logs/*_20261010_*.log`
**Decisions:** none
**Problems / open questions:** HeBIS SRU unreachable from current network; retry from campus or another path. oai_dc subjects are DDC-style — switch to MARC21 for real GND headings.
**Next:** Full DE DNB seed-gap pass; fuller OL harvest; access MD5 search on `final_list_*.csv`.

## 2026-10-08 | Week 3 | Phase: Candidates | — h
**Goal:** File layout for filtered OL + seed → final candidate lists; document MD5 handoff.
**Done:**
- Moved MiniMax scrapers to `data/raw/colleague_openlibrary/scripts/openlibrary_scrape_{en,de}.py`.
- Moved cleaned filtered CSVs to `data/interim/filtered_openlibrary/` (UTF-8; 1,086 EN + 80 DE; all OLIDs ⊆ raw).
- Reorganized interim: `final_lists/`, `catalogue/`, `filtered_openlibrary/`, `dnb/`; wrote `data/interim/README.md`.
- `src/candidates/build_final_lists.py` merges filtered + seed; dedupe on base title + surname (seed wins). Lists: EN 1,120 (50 seed + 1,070 filtered), DE 128 (50 seed + 78 filtered); 18 drops in `dedupe_log.csv` (6 seed∩filtered, 12 filtered∩filtered).
- Broader lists under `catalogue/prescreen_*.csv` (still no seed). `md5` empty until access search; download needs 32-char lowercase `md5` from the access pipeline.
**Data/outputs:**
- `data/interim/final_lists/final_list_en.csv` (1,120), `final_list_de.csv` (128), `dedupe_log.csv` (18)
- `data/interim/filtered_openlibrary/filtered_*_corp.csv`
- `data/interim/catalogue/*` (moved from interim root)
- not committed
**Decisions:** none
**Problems / open questions:** none new
**Next:** Run access checks on `final_list_*.csv` one language at a time (`--formats epub,pdf`).

## 2026-10-08 | Week 3 | Phase: Candidates | — h
**Goal:** Migrate the colleague OpenLibrary lists and the seed corpus into the master schema, pre-screen them, and measure seed recall.
**Done:**
- `src/migrate_candidates.py` reads both OpenLibrary CSVs (header repair, W/M ids, `/works/OL...M` → `/books/OL...M`) and the seed sheet. Nothing is deleted.
- OpenLibrary pre-screen, records: EN exclude 68, contextual 175, primary 379, uncertain 5,820 (6,442). DE exclude 505, contextual 24, primary 3, uncertain 292 (824).
- Seed recall on normalized base title (before `:`) and author surname: EN 27/50 (0.540), DE 2/50 (0.040). German hits are DE-035 Egri and DE-046 Rico. DE-001 Gesing is in the German list as `Kreativ schreiben. Handwerk...` (period, not colon), so the colon rule misses it.
- The same rules flag two gold titles: EN-026 (`abc` → `likely_exclude`) and DE-026 (`journal|tagebuch` → `likely_contextual`).
- Prescreen lists, `likely_primary` + `uncertain` only: 5,381 EN and 378 DE. Not searched yet.
**Data/outputs:** (paths later moved under `data/interim/catalogue/`; see next entry)
- `openlibrary_candidates.csv` (7,266 records)
- `seed_candidates.csv` (100 records)
- `seed_recall.csv` (100 records)
- `prescreen_en.csv` / `prescreen_de.csv` (5,381 / 378)
- not committed
**Decisions:** none
**Problems / open questions:**
- `corpus_type` is still empty. Seed `Book_type` → A-F is separate work.
- A subtitle set off by a period rather than a colon does not count as the base title.
**Next:** Sample the pre-screen classes (step 9). Map seed `Book_type` to A-F.

## 2026-10-08 | Week 1 | Phase: Protocol | — h
**Goal:** Set up the repository, a first Ngram and DNB smoke test, and the documentation.
**Done:**
- Created the scaffold (venv on Python 3.14, configs, `src/` modules).
- Ngram EN/DE corpus terms retrieved (smoothing 0, cached).
- DNB `tit="kreatives schreiben"`: 585 hits, 200 parsed.
- Colleague OpenLibrary CSVs and the seed corpus moved to `data/raw/`; seed xlsx exported to CSV.
- Added `docs/` (protocol, plan, log, candidate-list description) and the POV lexicon v0.
**Data/outputs:**
- `data/processed/ngram_*`
- `data/interim/dnb/dnb_hits_tit_kreatives_schreiben.csv` (path updated later)
- `data/raw/colleague_openlibrary/`
- `data/raw/seed_corpus/`
**Decisions:** DEC-001 to DEC-005
**Problems / open questions:**
- Missing inputs (at the time): MiniMax scripts (now under `colleague_openlibrary/scripts/`), the cue-words xlsx, the 27-item curated German list, and the ~30-book EN POV set.
- The English CSV's processing history is unclear (it looks like a spreadsheet re-export).
**Next:** Freeze corpus protocol v1. Add the historical seeds. Map seed `Book_type` to A-F.
