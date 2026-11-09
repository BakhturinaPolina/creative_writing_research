# -*- coding: utf-8 -*-
"""Python 3.6 compatible PDF/EPUB keyword scanner.
PDF extraction requires: PyPDF2==1.26.0
"""
# =============================================================================
# WHAT THIS SCRIPT DOES (second version, used for book_scan_results/)
#   Same job as analyze_books_initial_script.py, rewritten to run on an old
#   Python 3.6 Jupyter kernel with fewer libraries:
#   1. Finds every .pdf / .epub file in the folders you give it.
#   2. Pulls plain text out of each book (PDF: page by page, EPUB: per HTML file).
#   3. Counts 3 English terms ("point of view", "omniscient", "first-person")
#      and keeps up to 3 short text snippets per term.
#   4. Writes all_books.csv, relevant_books.csv, relevant_books.txt.
#      A book is "relevant" if it has at least ONE match of ANY term.
#
# HOW TO READ THE COMMENTS
#   "# ..."        -> plain explanation of what the code does
#   "# ISSUE: ..." -> a known problem / limitation (see README.md)
#
# Messages printed to the screen are in Russian.
# =============================================================================
from __future__ import print_function
import argparse
import csv
import os
import re
import sys
import zipfile
from html.parser import HTMLParser

# PyPDF2 is optional: without it only EPUB files can be scanned.
# ISSUE: PyPDF2 1.26.0 (from 2016) is a weak PDF text extractor: it often loses
#        spaces, mixes up characters or returns nothing. This is the main reason
#        why 54% of readable PDFs got 0 matches (vs 20% of EPUBs) in the results.
#        The first version used pdfplumber, which extracts text better.
try:
    from PyPDF2 import PdfFileReader
except ImportError:
    PdfFileReader = None

# -----------------------------------------------------------------------------
# SEARCH TERMS: name shown in the report -> regular expression to search for.
# re.I = ignore upper/lower case. \b = word boundary (whole words only).
# -----------------------------------------------------------------------------
# ISSUE: "point of view" is a very common everyday phrase ("from a business
#        point of view"). 62 books in the results are "relevant" only because of
#        1-2 such hits.
# ISSUE: "omnisci(ent)" finds only "omniscient", NOT "omniscience".
# ISSUE: "first-person" only matches with a hyphen/dash. "first person" with a
#        normal space is NOT found (very common spelling!).
# ISSUE: Missing obvious synonyms: POV, viewpoint, third person, second person,
#        narrator, perspective, focalization ...
# ISSUE: The text is not cleaned before searching. PDFs often contain the
#        ligature "ﬁ" (one symbol) instead of "fi", so "ﬁrst person" is missed.
#        Words broken by a line-end hyphen ("first-\nperson") are missed too.
TERMS = {
    "point of view": re.compile(r"\bpoint\s+of\s+view\b", re.I),
    "omniscient": re.compile(r"\bomnisci(ent)\b", re.I),
    "first-person": re.compile(r"\bfirst[\u2010-\u2015-]person\b", re.I),
}

# Small HTML reader: throws away the tags and keeps only the visible text pieces.
# ISSUE: Text inside <script> and <style> tags is kept as well (the German
#        script fixes this).
class TextParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)

# -----------------------------------------------------------------------------
# EPUB -> text. An EPUB is just a zip file with HTML pages inside.
# Opens the zip and gives back the text of one HTML file at a time.
# -----------------------------------------------------------------------------
# ISSUE: Files are read in zip storage order, not in the book's reading order.
#        Counts are not affected, but the "first 3 examples" may come from a
#        random place in the book.
# ISSUE: Every file is decoded as UTF-8; books in another encoding get garbled.
# ISSUE: Table of contents / index pages are counted as normal text.
# ISSUE: A file named .epub that is not really a zip fails with "BadZipFile".
def epub_chunks(path):
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            low = name.lower()
            if low.endswith((".html", ".htm", ".xhtml")):
                raw = archive.read(name)
                html = raw.decode("utf-8", "replace")
                parser = TextParser()
                parser.feed(html)
                yield " ".join(parser.parts)
                parser.close()

# -----------------------------------------------------------------------------
# PDF -> text. Gives back the text of one page at a time.
# -----------------------------------------------------------------------------
# ISSUE: No OCR. Scanned PDFs (pictures of pages) give empty text, so the book
#        silently gets 0 matches and looks "not relevant". The amount of text
#        read is not saved, so you cannot spot these books in the report.
# ISSUE: Encrypted PDFs fail ("File has not been decrypted") - this happened to
#        "The Power of Point of View", the most relevant book in the corpus.
def pdf_chunks(path):
    if PdfFileReader is None:
        raise RuntimeError("Для PDF установите PyPDF2==1.26.0 в это ядро Jupyter")
    stream = open(path, "rb")
    try:
        reader = PdfFileReader(stream)
        for index in range(reader.getNumPages()):
            yield reader.getPage(index).extractText() or ""
    finally:
        stream.close()

# -----------------------------------------------------------------------------
# Cut short snippets around each match (100 chars before, 140 after) so a human
# can check in the CSV whether the match is really about narrative perspective.
# -----------------------------------------------------------------------------
def excerpts(text, pattern, limit=3):
    result = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 140)
        result.append(" ".join(text[start:end].split()))
        if len(result) >= limit:
            break
    return result

# -----------------------------------------------------------------------------
# Collect all .pdf/.epub files from the given files and folders (recursively).
# -----------------------------------------------------------------------------
# ISSUE: No duplicate detection. ~46 books exist twice (PDF+EPUB or two copies)
#        and are scanned/reported separately, often with very different results
#        (e.g. "Building Fiction": PDF 205 matches, EPUB 4).
def collect_files(inputs):
    found = set()
    for item in inputs:
        if os.path.isfile(item):
            if os.path.splitext(item)[1].lower() in (".pdf", ".epub"):
                found.add(os.path.abspath(item))
        elif os.path.isdir(item):
            for root, dirs, names in os.walk(item):
                for name in names:
                    if os.path.splitext(name)[1].lower() in (".pdf", ".epub"):
                        found.add(os.path.abspath(os.path.join(root, name)))
        else:
            raise ValueError("Не найден путь: " + item)
    return sorted(found)

def main():
    # --- Read command line options: input files/folders and output folder ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Файлы PDF/EPUB или папки")
    parser.add_argument("--output-dir", default="book_scan_results")
    args = parser.parse_args()
    # Stop early if there are PDFs but PyPDF2 is not installed.
    # ISSUE: collect_files() walks all folders here and again below (done twice).
    if PdfFileReader is None and any(os.path.splitext(p)[1].lower() == ".pdf"
                                     for p in collect_files(args.inputs)):
        parser.error("Установите PyPDF2==1.26.0: %pip install PyPDF2==1.26.0")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    files = collect_files(args.inputs)
    rows = []

    # --- Scan books one by one ---
    # ISSUE: Results are kept only in memory and written at the very end.
    #        If the script crashes on book 500, books 1-499 are lost
    #        (the German script fixes this with saving after every book).
    for number, path in enumerate(files, 1):
        print("[%d/%d] %s" % (number, len(files), os.path.basename(path)), file=sys.stderr)
        counts = dict((term, 0) for term in TERMS)
        samples = dict((term, []) for term in TERMS)
        error = ""
        try:
            chunks = pdf_chunks(path) if path.lower().endswith(".pdf") else epub_chunks(path)
            # Search each page/chapter separately and add up the counts.
            # ISSUE: A phrase split between two pages/chapters is not found.
            for chunk in chunks:
                for term, pattern in TERMS.items():
                    counts[term] += sum(1 for unused in pattern.finditer(chunk))
                    if len(samples[term]) < 3:
                        samples[term].extend(excerpts(chunk, pattern, 3 - len(samples[term])))
        # If a book cannot be read, remember the error and continue with the next.
        # ISSUE: A book that failed is still marked relevant="no" (it was never
        #        really checked; 8 of 10 errors in the results). If it failed
        #        half-way, the partial counts are kept and it can be marked "yes".
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        total = sum(counts.values())
        # ISSUE: "Relevant" = at least 1 match. One random mention counts the same
        #        as 200 mentions, and long books are not compared fairly with short ones.
        # ISSUE: Term names are typed again here by hand; if TERMS is changed,
        #        this block must be changed too (the German script avoids this).
        row = {
            "path": path, "format": os.path.splitext(path)[1][1:].lower(),
            "relevant": "yes" if total else "no", "total_matches": total,
            "point of view": counts["point of view"],
            "omniscient": counts["omniscient"],
            "first-person": counts["first-person"], "error": error,
        }
        for term in TERMS:
            row["examples: " + term] = " || ".join(samples[term])
        rows.append(row)

    # --- Write reports ---
    # all_books.csv      = every book
    # relevant_books.csv = only books with relevant == "yes"
    # relevant_books.txt = only the file paths of relevant books
    # ISSUE: Reports store full absolute paths (e.g. C:\Users\<name>\...), which
    #        only make sense on the computer where the scan was run. There is no
    #        stable book ID column (the MD5 hash in the file name could be used).
    columns = ["path", "format", "relevant", "total_matches"] + list(TERMS.keys()) + ["error"]
    columns += ["examples: " + term for term in TERMS]
    relevant = [row for row in rows if row["relevant"] == "yes"]
    for filename, selected in (("all_books.csv", rows), ("relevant_books.csv", relevant)):
        outpath = os.path.join(args.output_dir, filename)
        with open(outpath, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)
    listing = os.path.join(args.output_dir, "relevant_books.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for row in relevant:
            handle.write(row["path"] + "\n")
    # Final summary: books processed, books with matches, read errors.
    errors = sum(1 for row in rows if row["error"])
    print("Готово: обработано книг — %d; с совпадениями — %d; ошибок — %d." %
          (len(rows), len(relevant), errors))
    print("Отчёты сохранены: " + os.path.abspath(args.output_dir))

if __name__ == "__main__":
    main()
