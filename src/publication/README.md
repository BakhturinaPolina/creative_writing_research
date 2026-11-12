# Publication history (research plan week 11)

Count new works and editions over time (by language and translation status) so we can freeze the corpus period.

| Script | Status |
|---|---|
| `build_en_biblio_timeline.py` | Builds a detailed biblio CSV for English `primary` + `comparison` books |
| `publication_stats.py` | Stub (new works / editions per period) |

## English biblio timeline table

```bash
python -m src.publication.build_en_biblio_timeline
```

Reads interim catalogue sources plus `data/interim/fulltext/manifest.csv` and writes:

| File | Content |
|---|---|
| `data/processed/en_primary_comparison_biblio.csv` | one row per EN primary/comparison file: identifiers, suggested edition/first-work years, per-source year columns, publisher, analysis flags |
| `data/processed/en_primary_comparison_biblio.meta.json` | build time, coverage counts, source paths |

**Suggested years**

- `edition_year`: best available edition/publication year for the matched catalogue record (union/manifest → seed → OL candidates → filtered OL title → OpenSyllabus title → cleaned stats year).
- `first_work_year`: earliest work-level year (seed `first_work_year` → Open Library search `first_publish` → seed original year).
- Raw per-source columns stay in the table for sensitivity checks (`year_manifest`, `year_seed_*`, `year_ol_*`, `year_filtered_ol`, `year_opensyllabus`, `year_stats_cleaned`).
- `edition_year_identity_warning` flags `renamed` / `unresolved` cases where the catalogue year may not belong to the file on disk.

`publication_stats.py` can operate on this table once work-level dedupe is settled.
