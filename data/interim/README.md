# `data/interim/` — working tables

Raw inputs stay under `data/raw/` and are never edited. Everything here is derived.

## Subfolders

| Folder | What belongs here |
|---|---|
| [`final_lists/`](final_lists/) | Deduped filtered-OL + seed candidate lists for access checking, plus the merge dedupe log |
| [`catalogue/`](catalogue/) | Migrated master-schema tables and broader auto-prescreen lists |
| [`filtered_openlibrary/`](filtered_openlibrary/) | Manually cleaned subsets of the colleague Open Library scrapes (UTF-8) |
| [`dnb/`](dnb/) | DNB SRU hit tables (title harvests + seed-gap lookups) |
| [`hebis/`](hebis/) | HeBIS SRU hit tables (seed holdings checks) |
| [`openlibrary/`](openlibrary/) | Improved Open Library harvest CSVs (`ol_search_en.csv`, `ol_search_de.csv`) |

## Final lists (`final_lists/`)

| File | Role |
|---|---|
| `final_list_en.csv` | English candidate list (filtered OL + seed, deduped) |
| `final_list_de.csv` | German candidate list (same) |
| `dedupe_log.csv` | Rows dropped when title+surname (or OLID) collided; seed wins over filtered |

Columns (order fixed for downstream access tooling):

`candidate_id, title, author_name, publication_year, isbn_13, isbn_10, edition_language, md5, search_status, found_format, search_date`

Rebuild:

```bash
python -m src.candidates.build_final_lists
```

### IDs and handoff

- Access search needs `title` + `author_name`. `md5` stays empty until a file hash is filled by the access pipeline.
- Download tooling may require a 32-char lowercase `md5`.
- `candidate_id` is our join key (`filtered_ol:OL…W` or `seed:EN-001`). It is **not** a Goodreads `work_id`.
- ISBN is present on seed rows; filtered colleague rows have none.
- Open Library IDs on filtered `link`s are catalogue identity only.

## Catalogue (`catalogue/`)

| File | Role |
|---|---|
| `openlibrary_candidates.csv` | Full colleague OL lists in master schema + pre-screen |
| `seed_candidates.csv` | Seed bibliography in master schema |
| `seed_recall.csv` | Seed hit/miss against full OL candidates |
| `prescreen_en.csv` / `prescreen_de.csv` | Broader lists (`likely_primary` + `uncertain` from full OL, **no seed**). Kept for later scale-up; not the first access-check input |

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

## Logs

Every CLI also writes a timestamped file under repo-root `logs/` (gitignored).
