# Book filtering scripts: short analysis

These keyword scanners were written by a colleague. They search PDF and EPUB books
for words about narrative perspective ("point of view", "omniscient",
"first-person", and German equivalents) and mark each book as relevant or not.

In the scripts, plain `#` comments explain the code and `# ISSUE:` comments mark
problems. Only comments were added; the code is unchanged.

## What is here

| File | What it is |
|---|---|
| `analyze_books_initial_script.py` | First version. Python 3.7+, uses `pdfplumber`, `ebooklib`, `beautifulsoup4`. |
| `analyze_books_english_script.py` | Rewrite for an old Python 3.6 Jupyter kernel, uses `PyPDF2==1.26.0`. Produced `book_scan_results/`. |
| `analyze_german_books_german_script.py` | Same as the English script, with German terms, resume after a crash, and saving after every book. |
| `book_scan_results/` | Real run on 588 English books (`raw_eng_CW_corp`). |
| `example_results/` | Test run on 2 books. |

## How the scripts work

1. Find all `.pdf` and `.epub` files in the given folders.
2. Extract the text: one page at a time for PDFs, one HTML file at a time for EPUBs.
3. Count regex matches for each term and keep up to 3 text snippets per term.
4. Mark the book `relevant = yes` if it has at least 1 match.
5. Write `all_books.csv`, `relevant_books.csv` and `relevant_books.txt`.

## Results in `book_scan_results/`

- 588 books scanned: 351 marked relevant, 237 not relevant, 10 read errors.
- **PDFs often come back empty:** 54% of readable PDFs have 0 matches, against
  20% of EPUBs. The cause is weak text extraction, not the content.
- **Clearly relevant books got 0 matches**, for example *Narrative in Fiction and
  Film* (Lothe), *Making Shapely Fiction*, *The Art of Writing Fiction* and
  *Metafiction*.
- **The same book gives different results in different files:** about 46 books
  exist twice. For example, *Building Fiction* got 205 matches as PDF and 4 as
  EPUB.
- *The Power of Point of View* could not be read (encrypted PDF), so it was
  marked "no".

## Good points

- Simple and easy to check. The saved snippets let a person verify each hit.
- Handles PDF and EPUB, and whole folder trees.
- One broken book does not stop the run; its error is written to the CSV.
- The German script is the strongest version: it can resume after a crash,
  writes files safely, retries failed books, and has a well-chosen technical
  vocabulary (`Fokalisierung`, `auktorial`, `homodiegetisch`).

## Main issues (most important first)

1. **Bad PDF text extraction.** `PyPDF2 1.26.0` is old and weak, and there is no
   OCR for scanned books. Books with no readable text quietly get 0 matches, and
   the report does not show how much text was read, so you cannot spot them.
2. **Many real mentions are missed.**
   - "first person" written with a space is not found; only "first-person"
     with a hyphen is.
   - "omniscience" is not found, only "omniscient".
   - Common synonyms are missing: POV, viewpoint, third person, narrator,
     perspective.
   - The text is not cleaned first, so PDF ligatures ("ﬁrst"), words split by a
     hyphen at the end of a line, and some umlaut encodings break matches.
3. **Many false hits.** "point of view" is an everyday phrase ("from a business
   point of view"), and so are `Blickwinkel` and `Blickpunkt` in German.
   62 books are marked relevant only because of 1–2 such hits.
4. **The yes/no rule is too crude.** A single match makes a book relevant, and
   long books are not compared fairly with short ones.
5. **Errors are reported misleadingly.** A book that could not be read is
   marked "no" instead of "unknown". A book that failed halfway keeps its
   partial counts.
6. **No duplicate handling.** The same book is reported twice, often with
   different results.
7. **Hard to reuse.**
   - The reports store absolute Windows paths and have no book ID column.
   - The English scripts lose all results if they crash.
   - The three scripts are near copies, so every fix has to be made three times.

## How to use these results

- Treat `relevant = yes` as a rough first list, mostly reliable for EPUBs.
- **Do not use `relevant = no` to drop PDFs.** Many of them simply had no
  readable text.

## Suggested fixes

1. Merge the three scripts into one, keeping the German script's resume and
   safe saving. Switch PDF extraction back to `pdfplumber` or PyMuPDF.
2. Clean the text before searching: Unicode NFKC normalization, joining
   hyphenated words, collapsing whitespace. Then widen the term lists.
3. Add `text_chars`, `pages` and `status` (`ok` / `no_text` / `error`) columns.
   Send `no_text` PDFs to OCR.
4. Replace yes/no with a score (hits per 100,000 words, with technical terms
   weighted higher) and choose the cutoff by checking a small sample by hand.
5. Remove duplicates using title and author, and use the MD5 hash in the file
   name as the book ID.
