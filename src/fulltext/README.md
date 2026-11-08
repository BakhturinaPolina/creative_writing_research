# Full-text acquisition and cleaning (research plan weeks 13–14)

Turn the downloaded books into one uniform, machine-readable format.

```bash
python -m src.fulltext.inventory --move   # manifest + move originals to data/raw/books/{lang}/{md5}.{format}
python -m src.fulltext.prescreen_books    # proposed corpus_role for every book
python -m src.fulltext.extract_ebook      # round 1: EPUB/MOBI/AZW3/DOCX/page-text zips → per-book JSON
```

- `books.py`: shared helpers (format sniffing, EPUB/Kindle/DOCX readers, PDF sampling, language detection).
- Book ID: `{lang}_{md5[:12]}`; `lang` is detected from the text (`en`, `de`, `other`), falling back to the download folder.
- `format` is the real format from the file's magic bytes; `file_ext` is the downloaded extension (many `.epub` files are PDFs).
- Outputs: `data/interim/fulltext/manifest.csv` (tracked), `json/` and `excerpts/` (gitignored, copyrighted text).

Not yet: PDF text-layer extraction (round 1b), OCR of scans (round 2), Parquet export.
