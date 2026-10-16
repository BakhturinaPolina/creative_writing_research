# `data/interim/` — working tables

Raw inputs stay under `data/raw/` and are never edited. Everything here is derived.

## Subfolders

| Folder | What belongs here |
|---|---|
| [`final_lists/`](final_lists/) | **Colleague review**: true-union audit table, tiered feeds, review CSVs, dedupe log |
| [`catalogue/`](catalogue/) | Migrated master-schema tables and broader auto-prescreen lists (provenance; do not review here first) |
| [`filtered_openlibrary/`](filtered_openlibrary/) | Manually cleaned subsets of the colleague Open Library scrapes (UTF-8) |
| [`dnb/`](dnb/) | DNB SRU hit tables (title harvests + seed-gap lookups) |
| [`hebis/`](hebis/) | HeBIS SRU hit tables (seed holdings checks) |
| [`openlibrary/`](openlibrary/) | Improved Open Library harvest CSVs (`ol_search_en.csv`, `ol_search_de.csv`) |
| [`opensyllabus/`](opensyllabus/) | Open Syllabus Project creative-writing query results |

## Final lists (`final_lists/`) — start here for review

**Colleagues:** open **`review_en.csv`** / **`review_de.csv`** only. Other interim folders are discovery provenance for coders.

| File | Role |
|---|---|
| `review_{en,de}.csv` | **Colleague-facing** review table (P0+P1 fully + sample of P2/P3) |
| `final_list_{en,de}.csv` | Tiered feed (P0–P2, excludes `likely_exclude`) |
| `union_all.csv` | Full deduped union across all discovery sources (audit) |
| `dedupe_log.csv` | Kept/dropped collisions (ISBN → OL work → DNB → title+surname) |

### Columns

Tiered feeds (`final_list_*`):

`candidate_id, title, author_name, publication_year, isbn_13, isbn_10, edition_language, md5, search_status, found_format, search_date`

Review / union extras:

`source_tags, screen_status, screen_rule, match_confidence, download_priority, ol_work_id, ol_edition_id, dnb_id, subtitle`

### Priority tiers

| Tier | Who | Review |
|---|---|---|
| **P0** | Seed bibliography | Always first |
| **P1** | Filtered OL + OpenSyllabus **books** | High-confidence curated |
| **P2** | `likely_primary` from harvests / prescreen | Included in `final_list_*` |
| **P3** | `uncertain` (+ contextual) | Union/audit only; sample in review |

### Rebuild

```bash
python -m src.candidates.build_final_lists   # union + tiers + ISBN backfill from ol_search
python -m src.candidates.backfill_isbn       # optional re-enrich of existing CSVs
```

Sources folded into the union: seed, filtered Open Library, catalogue `prescreen_*`, `ol_search_*` (`likely_primary` + `uncertain`), DNB title hits, OpenSyllabus **books** only (articles/chapters dropped).

## Catalogue (`catalogue/`)

| File | Role |
|---|---|
| `openlibrary_candidates.csv` | Full colleague OL lists in master schema + pre-screen |
| `seed_candidates.csv` | Seed bibliography in master schema |
| `seed_recall.csv` | Seed hit/miss against full OL candidates |
| `prescreen_en.csv` / `prescreen_de.csv` | Broader lists (`likely_primary` + `uncertain` from full OL, **no seed**). Fed into the union builder |

Rebuild catalogue tables:

```bash
python -m src.candidates.migrate_candidates
```

## Filtered Open Library (`filtered_openlibrary/`)

Manual clean of the raw scrapes in `data/raw/colleague_openlibrary/` (1,086 EN + 80 DE). Same columns as the root drop (`title;author;…;link`), re-encoded UTF-8. Source for `build_final_lists`, not for re-scraping.

## DNB (`dnb/`)

Hit tables from `python -m src.catalogue.dnb`, e.g. `dnb_hits_tit_kreatives_schreiben.csv`.
Seed-gap fills: `python -m src.catalogue.dnb_seed_gaps --lang de` → `dnb_seed_gaps_de.csv`.

## HeBIS (`hebis/`)

Seed holdings checks: `python -m src.catalogue.hebis_seed_holdings` → `hebis_seed_holdings.csv`.
Single lookups: `python -m src.catalogue.hebis --isbn …`.

## Open Library (`openlibrary/`)

Improved harvest (not the frozen colleague MiniMax CSVs):

```bash
python -m src.catalogue.openlibrary --lang de --max-pages 1
python -m src.catalogue.compare_ol_to_colleague
```

## Open Syllabus (`opensyllabus/`)

Creative-writing query exports; only `publication_type=book` enters the union.

## Logs

Every CLI also writes a timestamped file under repo-root `logs/` (gitignored).
