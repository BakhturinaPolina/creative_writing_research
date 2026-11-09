# -*- coding: utf-8 -*-
"""Resumable PDF/EPUB keyword scanner for German narratology terms.

Compatible with Python 3.6.
PDF support requires PyPDF2==1.26.0.
Use a new output folder for this German-language scan.
"""
# =============================================================================
# WHAT THIS SCRIPT DOES (third version, for German books)
#   Same idea as analyze_books_english_script.py, but:
#   - searches German narratology terms (plus their English equivalents),
#   - can be stopped and restarted: results are saved after EVERY book and the
#     next run skips books that were already scanned successfully,
#   - books that failed last time are tried again automatically,
#   - ignores text inside <script>/<style> tags in EPUBs.
#   Output: all_books.csv, relevant_books.csv, relevant_books.txt.
#   A book is "relevant" if it has at least ONE match of ANY term.
#
# HOW TO READ THE COMMENTS
#   "# ..."        -> plain explanation of what the code does
#   "# ISSUE: ..." -> a known problem / limitation (see README.md)
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
#        spaces, mixes up characters or returns nothing. In the English scan
#        54% of readable PDFs got 0 matches (vs 20% of EPUBs) - expect the same here.
try:
    from PyPDF2 import PdfFileReader
except ImportError:
    PdfFileReader = None

# Search German terminology and common English equivalents.
# -----------------------------------------------------------------------------
# SEARCH TERMS: 3 groups. Each group = several alternative words joined by "|".
# \w* at the end = "any word ending", so inflected forms are found too
# (Erzählperspektive, Erzählperspektiven, ...). re.I = ignore case.
# -----------------------------------------------------------------------------
# ISSUE: "Blickwinkel" and "Blickpunkt" are everyday words ("angle", "focus,
#        point of interest") and "point of view" is a common English phrase ->
#        many false "relevant" books.
# ISSUE: Missing terms, e.g. Erzählsituation, personale Erzählsituation,
#        Er-Erzähler / Er-Form, Erzählinstanz, Erzähler, heterodiegetisch,
#        Innen-/Außenperspektive, POV, third person.
# ISSUE: The text is not cleaned before searching:
#        - "ä" can be stored as "a" + a separate dots symbol (Unicode NFD) in
#          some PDFs/EPUBs; then "Erzähl..." is not found;
#        - words split by a line-end hyphen ("Erzähl-\nperspektive") are very
#          common in German PDFs and are missed;
#        - PDF ligatures ("ﬁ" as one symbol) break words like "ﬁrst person".
TERMS = {
    "Erzählperspektive / point of view": re.compile(
        r"\b(?:Erzählperspektiv\w*|Erzählstandpunkt\w*|Erzählerstandort\w*|"
        r"Blickpunkt\w*|Blickwinkel\w*|Fokalisierung\w*|point\s+of\s+view)\b",
        re.I
    ),
    "Allwissenheit / auktorial / omniscient": re.compile(
        r"\b(?:allwissend\w*|auktorial\w*|Nullfokalisierung\w*|omniscien\w*)\b",
        re.I
    ),
    "Ich-Erzählung / first-person": re.compile(
        r"\b(?:Ich[-‐‑– ]?(?:Erzähler\w*|Perspektive\w*|Form\w*|Erzählung\w*)|"
        r"homodiegetisch\w*|first[-‐‑– ]+person)\b",
        re.I
    ),
}
# Column order of the CSV reports.
COLUMNS = ["path", "format", "relevant", "total_matches"]
COLUMNS += list(TERMS.keys()) + ["error"]
COLUMNS += ["examples: " + term for term in TERMS]

# Small HTML reader: throws away the tags and keeps only the visible text.
# Text inside <script> and <style> (code, not book text) is skipped.
class TextParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style"):
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style") and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

# -----------------------------------------------------------------------------
# EPUB -> text. An EPUB is just a zip file with HTML pages inside.
# Opens the zip and gives back the text of one HTML file at a time.
# -----------------------------------------------------------------------------
# ISSUE: Files are read in zip storage order, not in the book's reading order
#        (counts are fine, but "first 3 examples" may come from anywhere).
# ISSUE: Every file is decoded as UTF-8; books in another encoding get garbled.
# ISSUE: Table of contents / index pages are counted as normal text.
def epub_chunks(path):
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if name.lower().endswith((".html", ".htm", ".xhtml")):
                raw = archive.read(name)
                html = raw.decode("utf-8-sig", "replace")
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
# ISSUE: Encrypted or damaged PDFs fail with an error.
def pdf_chunks(path):
    if PdfFileReader is None:
        raise RuntimeError("Install PyPDF2==1.26.0 in this Jupyter kernel")
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
# ISSUE: No duplicate detection: the same book as PDF and EPUB (or two copies)
#        is scanned and reported twice.
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
            raise ValueError("Input path does not exist: " + item)
    return sorted(found)

# Safe CSV writing: write to a temporary file first, then swap it in.
# If the script is killed mid-write, the old report stays intact.
def write_csv_atomic(path, rows):
    temp = path + ".tmp"
    with open(temp, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

# Write all 3 reports (all books / relevant books / list of relevant paths).
# ISSUE: All reports are rewritten completely after every single book. Fine for
#        a few hundred books, slow for many thousands.
def save_reports(outdir, rows):
    relevant = [row for row in rows if row.get("relevant") == "yes"]
    write_csv_atomic(os.path.join(outdir, "all_books.csv"), rows)
    write_csv_atomic(os.path.join(outdir, "relevant_books.csv"), relevant)
    temp = os.path.join(outdir, "relevant_books.txt.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        for row in relevant:
            handle.write(row["path"] + "\n")
    os.replace(temp, os.path.join(outdir, "relevant_books.txt"))

# Load results of an earlier (interrupted) run from the output folder, so we
# can continue where we stopped. Refuses to mix with a report from a different
# scan (e.g. the English one), because the columns would not match.
def load_previous(outdir):
    path = os.path.join(outdir, "all_books.csv")
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if set(COLUMNS) - fields:
            raise ValueError(
                "The output folder contains a report from another scan. "
                "Choose a new output folder for German books."
            )
        return list(reader)

def main():
    # --- Read command line options ---
    # --rescan = ignore earlier results and scan everything again.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="PDF/EPUB files or folders")
    parser.add_argument("--output-dir", default="german_book_scan_results")
    parser.add_argument("--rescan", action="store_true",
                        help="Scan all input books again instead of resuming")
    args = parser.parse_args()

    files = collect_files(args.inputs)
    if PdfFileReader is None and any(path.lower().endswith(".pdf") for path in files):
        parser.error("Install PyPDF2==1.26.0 in this Jupyter kernel before scanning PDFs")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Resume logic: skip books that were already scanned without error ---
    # ISSUE: Books are recognised by their full path. If the folder is moved or
    #        renamed, all books are scanned again AND the old rows stay in the
    #        report -> the same book appears twice.
    # ISSUE: If the search terms are changed, old rows are NOT re-scanned
    #        (only the column names are checked) -> mixed results. Use --rescan.
    rows = [] if args.rescan else load_previous(args.output_dir)
    done = set(row.get("path") for row in rows if not row.get("error"))
    remaining = [path for path in files if path not in done]
    print("Previously completed: %d; books to scan or retry: %d." %
          (len(done), len(remaining)))

    # --- Scan the remaining books one by one ---
    for number, path in enumerate(remaining, 1):
        extension = os.path.splitext(path)[1][1:].lower()
        # Avoid printing filenames: the Windows console may not support all characters.
        # ISSUE: Because no file name is shown, you cannot see which book is slow
        #        or stuck.
        print("[%d/%d] Scanning %s." % (number, len(remaining), extension),
              file=sys.stderr)
        counts = dict((term, 0) for term in TERMS)
        samples = dict((term, []) for term in TERMS)
        error = ""
        try:
            chunks = pdf_chunks(path) if extension == "pdf" else epub_chunks(path)
            # Search each page/chapter separately and add up the counts.
            # ISSUE: A phrase split between two pages/chapters is not found.
            for chunk in chunks:
                for term, pattern in TERMS.items():
                    counts[term] += sum(1 for unused in pattern.finditer(chunk))
                    if len(samples[term]) < 3:
                        samples[term].extend(
                            excerpts(chunk, pattern, 3 - len(samples[term]))
                        )
        # If a book cannot be read, remember the error and continue with the next.
        # ISSUE: A failed book is marked relevant="no" although it was never
        #        really checked. If it failed half-way, the partial counts are
        #        kept and it can be marked "yes".
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)

        total = sum(counts.values())
        # ISSUE: "Relevant" = at least 1 match. One random mention counts the same
        #        as 200 mentions, and long books are not compared fairly with short ones.
        # ISSUE: Reports store full absolute paths, which only make sense on the
        #        computer where the scan was run; there is no stable book ID column.
        row = {
            "path": path,
            "format": extension,
            "relevant": "yes" if total else "no",
            "total_matches": total,
            "error": error,
        }
        row.update(counts)
        for term in TERMS:
            row["examples: " + term] = " || ".join(samples[term])

        # Replace any prior error row for this path, then checkpoint immediately.
        rows = [old for old in rows if old.get("path") != path]
        rows.append(row)
        save_reports(args.output_dir, rows)

    # Final summary: books recorded, books with matches, read errors.
    relevant_count = sum(1 for row in rows if row.get("relevant") == "yes")
    error_count = sum(1 for row in rows if row.get("error"))
    print("Done. Books recorded: %d; with keyword matches: %d; read errors: %d." %
          (len(rows), relevant_count, error_count))
    print("Reports saved in the selected output folder.")

if __name__ == "__main__":
    main()
