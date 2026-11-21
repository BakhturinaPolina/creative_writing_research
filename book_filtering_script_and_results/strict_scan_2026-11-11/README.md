# Strict bilingual book scan — 2026-11-11

## Provenance

Original scanner supplied by Danil Fokin on 11 November 2026 in the correspondence “Creative writing corpora”. The available attachment was named `read.me.txt` and contained Python source. It is archived here as `analyze_books_bilingual_strict.py`, with the source code unchanged.

The email reports 207 selected English books and 76 selected German books. These are reported counts, not independently reproduced results. The forwarded message contains one script attachment; no per-book CSV results or second script were attached to the available copy.

## Usage

The script supports Python 3.6 and requires `PyPDF2==1.26.0` for PDF extraction, as specified in the original source.

```bash
python analyze_books_bilingual_strict.py /path/to/raw_eng_cw_corp /path/to/raw_de_cw_corp --output-dir book_scan_results_strict
```

Default selection requires matches in at least two of three cue groups: point of view, omniscience and first-person narration. The scanner also supports `--min-categories`, `--require-pair` and `--rescan`. Outputs include `all_books.csv`, `relevant_books.csv`, `review_books.csv` and `relevant_books.txt`.

This version is preserved for provenance and comparison with the earlier scanners. It does not redefine the independently constructed population corpus or replace the current POV lexicon and corpus-statistics pipeline.
