# Full-text acquisition and cleaning (research plan weeks 13–14)

Turn the downloaded books into one uniform, machine-readable format.

```bash
python -m src.fulltext.inventory --move     # manifest + move originals into raw books tree
python -m src.fulltext.prescreen_books      # proposed corpus_role for every book
python -m src.fulltext.reorganize_books     # readable names + role subfolders (after labelling)
python -m src.fulltext.extract_ebook        # round 1: EPUB/MOBI/AZW3/DOCX/page-text zips → per-book JSON
```

- `books.py`: shared helpers (format sniffing, EPUB/Kindle/DOCX readers, PDF sampling, language detection, DRM check).
- Book ID: `{lang}_{md5[:12]}`; `lang` is detected from the text (`en`, `de`, `other`), falling back to the download folder.
- `format` is the real format from the file's magic bytes; `file_ext` is the downloaded extension (many `.epub` files are PDFs).
- Raw originals (gitignored): `data/raw/books/{lang}/{primary|review|comparison|negative}/{Title} - {Author} [{md5}].{ext}`.
  `unreviewed` maps to `review/`. Re-run `reorganize_books` after changing labels.

## Outputs (`data/interim/fulltext/`)

| File | Tracked | Content |
|---|---|---|
| `manifest.csv` | yes | one row per file: identity, language, format, PDF triage, list join, labels, extraction status |
| `review_unlisted.csv` | yes | books not in the candidate lists, with prescreen signals per 1,000 words |
| `extraction_stats_round1.csv`, `spotcheck_round1.md` | yes | extraction statistics and spot-check report (no book text) |
| `json/{lang}/{book_id}.json` | no | chapters (`front` / `body` / `back`) and paragraphs |
| `excerpts/` | no | text samples used for language detection, prescreen and spot-checks |

## Labels

`corpus_role` = `primary` (types A–D, incl. German translations), `comparison` (E–F, school, therapy,
academic, screenwriting, poetry), `negative` (fiction, unrelated, language courses, other languages),
`unreviewed`. The prescreen proposes; [`config/book_role_overrides.csv`](../../config/book_role_overrides.csv)
decides (`reviewer` column; `agent-firstpass` rows are proposals awaiting confirmation).

Check columns before analysis: `title_match=False` means the file's text does not contain its title words
(often a different book than the filename says); `content_dup_of` points to another file with the same text.

Not yet: PDF text-layer extraction (round 1b), OCR of scans (round 2), Parquet export.
