# Full-text acquisition and cleaning (research plan weeks 13–14)

Turn the downloaded books into one uniform, machine-readable format.

```bash
python -m src.fulltext.inventory --move     # manifest + move originals into raw books tree
python -m src.fulltext.prescreen_books      # proposed corpus_role for every book
python -m src.fulltext.fix_titles suggest   # proposed title/author fixes (en) → title_author_review.csv
python -m src.fulltext.fix_titles apply     # accepted fixes → config/book_metadata_overrides.csv
python -m src.fulltext.reorganize_books     # readable names + role subfolders (after labelling)
python -m src.fulltext.stage_title_review  # move pending title fixes → en/_title_review/{confidence}/
python -m src.fulltext.sync_raw_layout     # after hand moves/renames: sync manifest + overrides
python -m src.fulltext.extract_ebook        # round 1: EPUB/MOBI/AZW3/DOCX/page-text zips → per-book JSON
```

- `books.py`: shared helpers (format sniffing, EPUB/Kindle/DOCX readers, PDF sampling, language detection, DRM check).
- Book ID: `{lang}_{md5[:12]}`; `lang` is detected from the text (`en`, `de`, `other`), falling back to the download folder.
- `format` is the real format from the file's magic bytes; `file_ext` is the downloaded extension (many `.epub` files are PDFs).
- Raw originals (gitignored): `data/raw/books/{lang}/{primary|review|comparison|negative}/{Title} - {Author} [{md5}].{ext}`.
  `unreviewed` maps to `review/`. Re-run `reorganize_books` after changing labels.
  Files moved or renamed by hand, and anything under `_title_review/`, are left in place.
  `inventory --move` only relocates files still under `data/interim/creative_writing/`.

## Outputs (`data/interim/fulltext/`)

| File | Tracked | Content |
|---|---|---|
| `manifest.csv` | yes | one row per file: identity, language, format, PDF triage, list join, labels, extraction status |
| `review_unlisted.csv` | yes | books not in the candidate lists, with prescreen signals per 1,000 words |
| `extraction_stats_round1.csv`, `spotcheck_round1.md` | yes | extraction statistics and spot-check report (no book text) |
| `title_author_review.csv` | yes | proposed title/author fixes with every source, confidence and an `accept` column |
| `lookup_cache.json` | no | cached Open Library responses |
| `json/{lang}/{book_id}.json` | no | chapters (`front` / `body` / `back`) and paragraphs |
| `excerpts/` | no | text samples used for language detection, prescreen and spot-checks |

## Labels

`corpus_role` = `primary` (types A–D, incl. German translations), `comparison` (E–F, school, therapy,
academic, screenwriting, poetry), `negative` (fiction, unrelated, language courses, other languages),
`unreviewed`. The prescreen proposes; [`config/book_role_overrides.csv`](../../config/book_role_overrides.csv)
decides (`reviewer` column; `agent-firstpass` rows are proposals awaiting confirmation).

## Titles and authors

Filenames use [`config/book_metadata_overrides.csv`](../../config/book_metadata_overrides.csv) first, then the
candidate lists, then the download filename. `fix_titles suggest` compares the filename, candidate list, metadata
embedded in the file and Open Library (ISBN from the copyright page, else a search) against the book's first pages
and proposes a fix with a `confidence`:

- `high`: title and author are on the title page and every changed field is backed by two sources. These are applied unless `accept=no`.
- `medium` / `conflict`: need `accept=yes`.
- `wrong_book`: the file's text is a different book than its name says. These need `accept=yes`, and once applied they are marked `identity_changed=True` (re-check `corpus_role`).

You can edit `proposed_title` / `proposed_author` before running `apply`. Re-running `suggest` keeps `accept`
values for unchanged proposals.

For manual review, `stage_title_review` gathers remaining suggestions into
`data/raw/books/en/_title_review/{wrong_book|conflict|medium}/` (books already in
`comparison/` or `negative/` are left there). Rename the file, move it into the right
role folder, then run `sync_raw_layout` so the manifest and
[`config/book_role_overrides.csv`](../../config/book_role_overrides.csv) /
[`config/book_metadata_overrides.csv`](../../config/book_metadata_overrides.csv) match the disk.

Check columns before analysis: `title_match=False` means the file's text does not contain its title words
(often a different book than the filename says); `content_dup_of` points to another file with the same text.

Not yet: PDF text-layer extraction (round 1b), OCR of scans (round 2), Parquet export.
