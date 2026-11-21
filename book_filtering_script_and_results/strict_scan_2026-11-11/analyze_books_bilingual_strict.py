"""Resumable PDF/EPUB scanner for English and German narratology terms.

Python 3.6 compatible. PDF support requires PyPDF2==1.26.0.
Checkpoints are written after every book.
"""
from __future__ import print_function
import argparse
import csv
import os
import re
import sys
import zipfile
from html.parser import HTMLParser

try:
    from PyPDF2 import PdfFileReader
except ImportError:
    PdfFileReader = None

# IDs are stable for selection rules; labels appear in the CSV.
GROUPS = [
    ("point_of_view", "Point of view / Erzaehlperspektive",
     re.compile(
         r"\b(?:point[\s-]+of[\s-]+view|viewpoint|"
         r"Erzählperspektiv\w*|Erzählstandpunkt\w*|Erzählerstandort\w*|"
         r"Blickpunkt\w*|Blickwinkel\w*|Fokalisierung\w*)\b", re.I)),
    ("omniscient", "Omniscient / allwissend / auktorial",
     re.compile(
         r"\b(?:omniscient\w*|allwissend\w*|auktorial\w*|"
         r"Nullfokalisierung\w*)\b", re.I)),
    ("first_person", "First person / Ich-Erzaehlung",
     re.compile(
         r"\b(?:first[-\u2010-\u2015]person|"
         r"Ich[-\u2010-\u2015 ]?Erzähl\w*|"
         r"Ich[-\u2010-\u2015 ]?(?:Perspektive\w*|Form\w*)|"
         r"homodiegetisch\w*)\b", re.I)),
]
GROUP_IDS = [item[0] for item in GROUPS]
GROUP_LABELS = dict((item[0], item[1]) for item in GROUPS)
PAIR_CHOICES = (
    "omniscient+point_of_view",
    "point_of_view+first_person",
    "omniscient+first_person",
)
CORPUS_NAMES = ("raw_de_cw_corp", "raw_eng_cw_corp")
COLUMNS = ["book", "corpus", "selection_status", "selected",
           "matched_categories", "total_matches"]
COLUMNS += [GROUP_LABELS[key] for key in GROUP_IDS]
COLUMNS += ["error"]
COLUMNS += ["examples: " + GROUP_LABELS[key] for key in GROUP_IDS]

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

def excerpts(text, pattern, limit=3):
    result = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 140)
        result.append(" ".join(text[start:end].split()))
        if len(result) >= limit:
            break
    return result

def corpus_root_for(path):
    """Find the named corpus root, even if the user passes one of its subfolders."""
    current = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    while True:
        if os.path.basename(current).lower() in CORPUS_NAMES:
            return current, os.path.basename(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    return root, os.path.basename(root)

def collect_files(inputs):
    # Mapping: stable relative key -> (absolute file, corpus label, relative book name)
    found = {}
    for item in inputs:
        if not os.path.exists(item):
            raise ValueError("Input path does not exist: " + item)
        root, corpus = corpus_root_for(item)
        candidates = [item] if os.path.isfile(item) else []
        if os.path.isdir(item):
            for current, dirs, names in os.walk(item):
                for name in names:
                    candidates.append(os.path.join(current, name))
        for path in candidates:
            if os.path.splitext(path)[1].lower() not in (".pdf", ".epub"):
                continue
            absolute = os.path.abspath(path)
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            key = corpus + "/" + relative
            found[key] = (absolute, corpus, relative)
    return found

def write_csv_atomic(path, rows):
    temp = path + ".tmp"
    with open(temp, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

def save_reports(outdir, rows):
    selected = [row for row in rows if row.get("selected") == "yes"]
    review = [row for row in rows if row.get("selection_status") == "review"]
    write_csv_atomic(os.path.join(outdir, "all_books.csv"), rows)
    write_csv_atomic(os.path.join(outdir, "relevant_books.csv"), selected)
    write_csv_atomic(os.path.join(outdir, "review_books.csv"), review)
    temp = os.path.join(outdir, "relevant_books.txt.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(row["corpus"] + "/" + row["book"] + "\n")
    os.replace(temp, os.path.join(outdir, "relevant_books.txt"))

def load_previous(outdir):
    path = os.path.join(outdir, "all_books.csv")
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(COLUMNS) - set(reader.fieldnames or []):
            raise ValueError(
                "This output folder contains results from a different script. "
                "Choose a new output folder."
            )
        return list(reader)

def category_presence(row):
    present = set()
    for key in GROUP_IDS:
        if int(row.get(GROUP_LABELS[key], 0) or 0) > 0:
            present.add(key)
    return present

def is_selected(present, min_categories, required_pair):
    if required_pair:
        needed = set(required_pair.split("+"))
        return needed.issubset(present)
    return len(present) >= min_categories

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+",
                        help="One or more corpus folders or individual PDF/EPUB files")
    parser.add_argument("--output-dir", default="book_scan_results_strict")
    parser.add_argument("--min-categories", type=int, choices=(1, 2, 3), default=2,
                        help="Select books with at least this many distinct cue groups (default: 2)")
    parser.add_argument("--require-pair", choices=PAIR_CHOICES, default=None,
                        help="Instead require this exact category pair")
    parser.add_argument("--rescan", action="store_true",
                        help="Ignore saved checkpoints and scan inputs again")
    args = parser.parse_args()

    files = collect_files(args.inputs)
    if PdfFileReader is None and any(item[0].lower().endswith(".pdf")
                                     for item in files.values()):
        parser.error("Install PyPDF2==1.26.0 in this Jupyter kernel before scanning PDFs")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    rows = [] if args.rescan else load_previous(args.output_dir)
    previous = dict((row.get("corpus", "") + "/" + row.get("book", ""), row)
                    for row in rows)
    if args.rescan:
        previous = {}
    pending = []
    for key, item in sorted(files.items()):
        prior = previous.get(key)
        if prior and not prior.get("error"):
            continue
        pending.append((key, item))

    print("Previously completed: %d; books to scan or retry: %d." %
          (sum(1 for row in rows if not row.get("error")), len(pending)))

    for number, pair in enumerate(pending, 1):
        key, item = pair
        path, corpus, book = item
        extension = os.path.splitext(path)[1][1:].lower()
        # Do not print filenames: Windows console encodings may not support every character.
        print("[%d/%d] Scanning %s." % (number, len(pending), extension), file=sys.stderr)
        counts = dict((group_id, 0) for group_id in GROUP_IDS)
        samples = dict((group_id, []) for group_id in GROUP_IDS)
        error = ""
        try:
            chunks = pdf_chunks(path) if extension == "pdf" else epub_chunks(path)
            for chunk in chunks:
                for group_id, label, pattern in GROUPS:
                    counts[group_id] += sum(1 for unused in pattern.finditer(chunk))
                    if len(samples[group_id]) < 3:
                        samples[group_id].extend(
                            excerpts(chunk, pattern, 3 - len(samples[group_id]))
                        )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)

        present = set(group_id for group_id in GROUP_IDS if counts[group_id] > 0)
        chosen = bool(present) and is_selected(
            present, args.min_categories, args.require_pair)
        if error:
            status = "read_error"
        elif chosen:
            status = "selected"
        elif present:
            status = "review"
        else:
            status = "no_match"

        row = {
            "book": book,
            "corpus": corpus,
            "selection_status": status,
            "selected": "yes" if chosen else "no",
            "matched_categories": "; ".join(
                GROUP_LABELS[group_id] for group_id in GROUP_IDS if group_id in present
            ),
            "total_matches": sum(counts.values()),
            "error": error,
        }
        for group_id, label, pattern in GROUPS:
            row[label] = counts[group_id]
            row["examples: " + label] = " || ".join(samples[group_id])

        rows = [old for old in rows
                if old.get("corpus", "") + "/" + old.get("book", "") != key]
        rows.append(row)
        save_reports(args.output_dir, rows)

    selected_count = sum(1 for row in rows if row.get("selected") == "yes")
    review_count = sum(1 for row in rows if row.get("selection_status") == "review")
    error_count = sum(1 for row in rows if row.get("error"))
    print("Done. Books recorded: %d; selected: %d; review: %d; read errors: %d." %
          (len(rows), selected_count, review_count, error_count))
    print("Reports saved in the selected output folder.")

if __name__ == "__main__":
    main()