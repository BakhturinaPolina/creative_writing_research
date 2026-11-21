# Full-text acquisition and cleaning (research plan weeks 13–14)

Turn the downloaded books into one uniform, machine-readable format. **English and German
use the same pipeline and the same cleaning rules.**

```bash
python -m src.fulltext.inventory --move              # manifest + move originals into raw books tree
python -m src.fulltext.prescreen_books               # proposed corpus_role for every book
python -m src.fulltext.fix_titles suggest --lang en # proposed title/author fixes → title_author_review.csv
python -m src.fulltext.fix_titles suggest --lang de # same for German (keeps other langs in the CSV)
python -m src.fulltext.fix_titles apply              # accepted fixes → config/book_metadata_overrides.csv
python -m src.fulltext.reorganize_books              # readable names + role subfolders (after labelling)
python -m src.fulltext.stage_title_review --lang en  # pending title fixes → {lang}/_title_review/
python -m src.fulltext.stage_title_review --lang de
python -m src.fulltext.sync_raw_layout               # after hand moves/renames: sync manifest + overrides
python -m src.fulltext.extract_ebook                 # round 1: EPUB/MOBI/AZW3/DOCX/page-text zips → JSON
python -m src.fulltext.extract_pdf                   # round 1b: PDFs with text_layer=good → JSON
python -m src.fulltext.extract_pdf --lang de         # required for DE parity (EN already done)
python -m src.fulltext.clean_fulltext --lang all     # residual hyphens + running headers (both langs)
python -m src.fulltext.preprocess_qc                 # quality metrics → preprocess_qc.csv / .md
```

- `books.py`: shared helpers (format sniffing, EPUB/Kindle/DOCX readers, PDF sampling, language detection, DRM check, `normalize_block`).
- `cleaning.py`: shared residual-hyphen and running-header rules (EN + DE).
- Book ID: `{lang}_{md5[:12]}`; `lang` is detected from the text (`en`, `de`, `other`), falling back to the download folder.
- `format` is the real format from the file's magic bytes; `file_ext` is the downloaded extension (many `.epub` files are PDFs).
- Raw originals (gitignored): `data/raw/books/{lang}/{primary|review|comparison|negative}/{Title} - {Author} [{md5}].{ext}`.
  `unreviewed` maps to `review/`. Re-run `reorganize_books` after changing labels.
  Files moved or renamed by hand, and anything under `_title_review/`, are left in place.
  `inventory --move` only relocates files still under `data/interim/creative_writing/`.

## Bilingual preprocessing standard

Same steps and thresholds for `en` and `de`:

1. **Ebook extract (round 1)** then **PDF text-layer extract (round 1b)** for `text_layer=good`.
2. **`text_layer=none|poor` → `extraction_status=needs_ocr`** (OCR round 2 not implemented yet).
3. **Post-extract cleaner** (`clean_fulltext`, rules_version in JSON under `extraction.cleaning`):
   - NFC / soft-hyphen / whitespace (via extractors).
   - Residual spaced hyphens (`inci- dents`, `gezwun- gen`) rejoined; coordinated compounds kept
     (`Menschheits- und`, `pre- and`).
   - Hyphenation split across adjacent HTML paragraphs rejoined inside a chapter.
   - Short paragraphs (≤12 words) repeating ≥10 times dropped as running headers/footers.
4. **QC** (`preprocess_qc`): residual hyphen rate, header share, encoding flags — report must
   cover both languages.
5. **Spot-check**: every extraction round samples up to 10 EN + 10 DE (stratified by role).
6. **Analysis gates** (in `src.analysis.corpus_stats`): `needs_ocr`, short text (&lt;10k words),
   role conflict, surplus duplicate — apply the same order when DE analysis is added.

Minimum body size for a successful extract: 1,000 words (`MIN_BOOK_WORDS` in the extractors).

## Outputs (`data/interim/fulltext/`)

| File | Tracked | Content |
|---|---|---|
| `manifest.csv` | yes | one row per file: identity, language, format, PDF triage, list join, labels, extraction status |
| `review_unlisted.csv` | yes | books not in the candidate lists, with prescreen signals per 1,000 words |
| `primary_check.csv` | yes | `primary` books whose label conflicts with the prescreen rules, with the reason (to check by hand) |
| `extraction_stats_round1.csv`, `spotcheck_round1.md` | yes | ebook extraction statistics and spot-check report (no book text) |
| `extraction_stats_round1b.csv`, `spotcheck_round1b.md` | yes | PDF text-layer extraction statistics and spot-check report |
| `cleaning_stats.csv` | yes | per-book cleaner counts (hyphen joins, headers removed) |
| `preprocess_qc.csv`, `preprocess_qc.md` | yes | residual-hyphen / header / encoding QC by language |
| `title_author_review.csv` | yes | proposed title/author fixes with every source, confidence and an `accept` column |
| `lookup_cache.json` | no | cached Open Library responses |
| `json/{lang}/{book_id}.json` | no | chapters (`front` / `body` / `back`) and paragraphs |
| `excerpts/` | no | text samples used for language detection, prescreen and spot-checks |

## Labels

`corpus_role` = `primary` (types A–D, incl. German translations), `comparison` (E–F, school, therapy,
academic, screenwriting, poetry), `negative` (fiction, unrelated, language courses, other languages),
`unreviewed`. The prescreen proposes; [`config/book_role_overrides.csv`](../../config/book_role_overrides.csv)
decides (`reviewer` column; `agent-*` rows are proposals awaiting confirmation).

**Primary (analysis definition, 2026-11-12):** books whose subject is narrative / creative
writing — fiction craft, school creative-writing pedagogy and exercises, screen / poetry /
drama craft, creative nonfiction, therapy writing that teaches narrative writing. The
prescreen does not count POV words (DEC-001). Seed titles go to `primary` unless the file is
another book (`wrong_book`).

**Comparison (small contrast group):** narratology / narrative theory, and academic writing
about writing (composition handbooks, research on creative writing, literary scholarship on
writing). Journalism and songwriting are `negative`.

Aligned with `docs/scientific_readme.md` §2 (updated 2026-11-21).

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

For manual review, `stage_title_review --lang {en|de}` gathers remaining suggestions into
`data/raw/books/{lang}/_title_review/{wrong_book|conflict|medium}/` (books already in
`comparison/` or `negative/` are left there). Rename the file, move it into the right
role folder, then run `sync_raw_layout` so the manifest and
[`config/book_role_overrides.csv`](../../config/book_role_overrides.csv) /
[`config/book_metadata_overrides.csv`](../../config/book_metadata_overrides.csv) match the disk.

Check columns before analysis: `title_match=False` means the file's text does not contain its title words
(often a different book than the filename says); `content_dup_of` points to another file with the same text.

Round 1b (`extract_pdf`) covers PDFs with `text_layer=good` (and optionally `--include-poor`).
`text_layer=none|poor` are recorded as `needs_ocr` for both languages. Not yet: OCR of scans,
Parquet export.

## Role taxonomy (EN and DE, 2026-11-12)

Same rules for `data/raw/books/{en,de}/`:

| Goes to | Examples |
|---|---|
| `primary` | Fiction / narrative craft manuals; novel-writing; specialized craft (POV, scene, dialogue); writer memoirs with substantial craft; genre fiction craft; **children's literature** craft when it teaches story/narrative writing; school CW pedagogy and exercises; screen / poetry / drama craft; creative nonfiction; therapy writing that teaches narrative writing |
| `comparison` | Narratology / narrative theory (F); academic writing *about* writing (research on creative writing, composition studies as research, literary scholarship on writing) |
| `negative` | Journalism / copywriting / blogging / songwriting; academic / essay / scientific writing handbooks; business writing; dictionaries, language courses, and general grammar/style guides; literary scholarship that is **not** narratology; fiction itself; tech / programming / unrelated subjects; administering a CW programme; empty or wrong-book files |

Inside primary, analysis tables split `fiction_manual` vs `pedagogy_genre_craft` for robustness; comparison books are labelled `contrast` in that split.
