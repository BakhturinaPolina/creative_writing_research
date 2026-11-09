#!/usr/bin/env python3
"""Scan PDF and EPUB files for selected terms, one book at a time."""
# =============================================================================
# WHAT THIS SCRIPT DOES (first version of the scanner)
#   1. Finds every .pdf / .epub file in the folders you give it.
#   2. Pulls plain text out of each book (page by page / chapter by chapter).
#   3. Counts how often 3 English terms appear ("point of view", "omniscient",
#      "first-person") and keeps up to 3 short text snippets per term.
#   4. Writes 3 reports: all_books.csv, relevant_books.csv, relevant_books.txt.
#      A book is "relevant" if it has at least ONE match of ANY term.
#
# HOW TO READ THE COMMENTS
#   "# ..."        -> plain explanation of what the code does
#   "# ISSUE: ..." -> a known problem / limitation (see README.md)
#
# Needs: Python 3.7+, pdfplumber, beautifulsoup4, ebooklib.
# Later versions (english / german scripts) replaced these libraries so the
# code runs on an old Python 3.6 Jupyter kernel.
#
# ISSUE: The code is packed into very long one-line statements (many commands
#        on one line, no spaces). It works, but is hard to read and review.
# =============================================================================
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
from typing import Iterable
import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

# -----------------------------------------------------------------------------
# SEARCH TERMS: name shown in the report -> regular expression to search for.
# re.I = ignore upper/lower case. \b = word boundary (whole words only).
# -----------------------------------------------------------------------------
# ISSUE: "point of view" is a very common everyday phrase ("from a business
#        point of view") -> many books get marked relevant for the wrong reason.
# ISSUE: "omnisci(ent)" finds only "omniscient", NOT "omniscience".
#        The brackets (...) do nothing useful here.
# ISSUE: "first-person" only matches with a hyphen/dash. "first person"
#        written with a normal space is NOT found (very common spelling!).
# ISSUE: Missing obvious synonyms: POV, viewpoint, third person, second person,
#        narrator, perspective, focalization, free indirect ...
# ISSUE: The text is not cleaned before searching. PDFs often contain the
#        ligature character "ﬁ" (one symbol) instead of "fi", so "ﬁrst person"
#        is not found. Words broken by a line-end hyphen ("first-\nperson")
#        are also missed.
TERMS = {
    "point of view": re.compile(r"\bpoint\s+of\s+view\b", re.I),
    "omniscient": re.compile(r"\bomnisci(ent)\b", re.I),
    "first-person": re.compile(r"\bfirst[\u2010-\u2015-]person\b", re.I),
}

# -----------------------------------------------------------------------------
# PDF -> text. Gives back the text of one page at a time.
# -----------------------------------------------------------------------------
# ISSUE: No OCR. Scanned PDFs (pictures of pages) give empty text, so the book
#        silently gets 0 matches and looks "not relevant".
# ISSUE: The amount of extracted text is not saved, so in the report you cannot
#        tell "no text could be read" apart from "text read, no matches".
def pdf_text(path: Path) -> Iterable[str]:
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages: yield page.extract_text() or ""

# -----------------------------------------------------------------------------
# EPUB -> text. An EPUB is a zip of HTML files; gives back the plain text of
# one HTML file (usually one chapter) at a time.
# -----------------------------------------------------------------------------
# ISSUE: Table of contents, index and navigation pages are counted too, so a
#        term listed in the contents is counted as an extra "match".
def epub_text(path: Path) -> Iterable[str]:
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        html=item.get_content().decode("utf-8", errors="replace")
        yield BeautifulSoup(html,"html.parser").get_text(" ",strip=True)

# -----------------------------------------------------------------------------
# Cut short snippets around each match (100 chars before, 140 after) so a human
# can check in the CSV whether the match is really about narrative perspective.
# Stops after `limit` snippets.
# -----------------------------------------------------------------------------
def excerpts(text: str, pattern: re.Pattern, limit: int=3) -> list[str]:
    result=[]
    for m in pattern.finditer(text):
        result.append(" ".join(text[max(0,m.start()-100):min(len(text),m.end()+140)].split()))
        if len(result)>=limit: break
    return result

def main() -> int:
    # --- Read command line options: input files/folders and output folder ---
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path,nargs="+",help="One or more PDF/EPUB files or folders")
    parser.add_argument("--output-dir",type=Path,default=Path("book_scan_results"))
    args=parser.parse_args()
    if any(not p.exists() for p in args.inputs): parser.error("Every input file or folder must exist")
    args.output_dir.mkdir(parents=True,exist_ok=True)

    # --- Collect all .pdf/.epub files (folders are searched recursively) ---
    # ISSUE: No duplicate detection. The same book as PDF and EPUB (or two
    #        copies) is scanned and reported twice, often with different results.
    files=sorted({q for p in args.inputs for q in ([p] if p.is_file() else p.rglob("*")) if q.is_file() and q.suffix.lower() in {".pdf",".epub"}})
    rows=[]

    # --- Scan books one by one ---
    # ISSUE: Results are kept only in memory and written at the very end.
    #        If the script crashes on book 500, books 1-499 are lost.
    for i,path in enumerate(files,1):
        print(f"[{i}/{len(files)}] {path.name}",file=sys.stderr)
        counts,examples,error={t:0 for t in TERMS},{t:[] for t in TERMS},""
        try:
            chunks=pdf_text(path) if path.suffix.lower()==".pdf" else epub_text(path)
            # Search each page/chapter separately and add up the counts.
            # ISSUE: A phrase split between two pages/chapters is not found.
            for chunk in chunks:
                for term,pattern in TERMS.items():
                    counts[term]+=sum(1 for _ in pattern.finditer(chunk))
                    if len(examples[term])<3: examples[term].extend(excerpts(chunk,pattern,3-len(examples[term])))
        # If a book cannot be read, remember the error and continue with the next.
        # ISSUE: A book that failed is still marked relevant="no" (it was never
        #        really checked). If it failed half-way, the partial counts are
        #        kept and it can even be marked "yes".
        except Exception as exc: error=f"{type(exc).__name__}: {exc}"
        total=sum(counts.values())
        # ISSUE: "Relevant" = at least 1 match. One random mention counts the same
        #        as 200 mentions, and long books are not compared fairly with short ones.
        row={"path":str(path.resolve()),"format":path.suffix.lower()[1:],"relevant":"yes" if total else "no","total_matches":total,**counts,"error":error}
        row.update({f"examples: {t}":" || ".join(examples[t]) for t in TERMS})
        rows.append(row)

    # --- Write reports ---
    # all_books.csv      = every book
    # relevant_books.csv = only books with relevant == "yes"
    # relevant_books.txt = only the file paths of relevant books
    # ISSUE: Reports store full absolute paths (e.g. C:\Users\<name>\...), which
    #        only make sense on the computer where the scan was run. There is no
    #        stable book ID column (the MD5 hash in the file name could be used).
    columns=["path","format","relevant","total_matches",*TERMS,"error",*(f"examples: {t}" for t in TERMS)]
    relevant=[r for r in rows if r["relevant"]=="yes"]
    for name,selected in (("all_books.csv",rows),("relevant_books.csv",relevant)):
        with (args.output_dir/name).open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(selected)
    (args.output_dir/"relevant_books.txt").write_text("\n".join(r["path"] for r in relevant)+("\n" if relevant else ""),encoding="utf-8")
    print(f"Scanned {len(rows)} files; {len(relevant)} have at least one match.")
    print(f"Results: {args.output_dir.resolve()}")
    return 0
if __name__=="__main__": raise SystemExit(main())
