"""HeBIS SRU client: query, paginate, save raw MARC-XML, parse into catalogue rows.

HeBIS is the Hessian union catalogue (includes Frankfurt university library holdings).
Endpoint needs no authentication. Up to 500 records per request.
Goal for now: look up known seed titles/ISBNs — not bulk discovery.
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import date

import pandas as pd
import requests
from lxml import etree

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_RAW
from src.common.schema import CATALOGUE_COLUMNS

RAW_DIR = DATA_RAW / "hebis"
OUT_DIR = DATA_INTERIM / "hebis"
HEBIS_URL = "https://sru.hebis.de/sru/DB=2.1"

# SRU wrapper + MARC21 slim namespaces commonly used by HeBIS.
NS = {
    "srw": "http://www.loc.gov/zing/srw/",
    "marc": "http://www.loc.gov/MARC21/slim",
}
SEP = " | "
# Extra column beyond the shared catalogue schema (Frankfurt / local note).
HEBIS_COLUMNS = CATALOGUE_COLUMNS + ["holding_note"]


def query_hebis(query, start_record=1, maximum_records=500):
    """Send one HeBIS SRU search and return raw XML bytes."""
    params = {
        "version": "1.1",
        "operation": "searchRetrieve",
        "query": query,
        "recordSchema": "marcxml",
        "startRecord": start_record,
        "maximumRecords": min(maximum_records, 500),
    }
    # 30s connect/read: HeBIS is sometimes unreachable from some networks.
    response = requests.get(HEBIS_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.content


def total_records(xml_bytes):
    """How many hits HeBIS reports for this query."""
    root = etree.fromstring(xml_bytes)
    n = root.xpath("string(//srw:numberOfRecords)", namespaces=NS)
    return int(n) if n else 0


def _subfields(datafield, codes):
    """Join subfield texts for the given code letters (e.g. 'a', 'b')."""
    texts = []
    for code in codes:
        for sf in datafield.xpath(f"./marc:subfield[@code='{code}']", namespaces=NS):
            if sf.text and sf.text.strip():
                texts.append(sf.text.strip())
    return texts


def _fields(record, tag):
    """All datafields with this MARC tag."""
    return record.xpath(f".//marc:datafield[@tag='{tag}']", namespaces=NS)


def _control(record, tag):
    """One control field value (e.g. 001)."""
    nodes = record.xpath(f".//marc:controlfield[@tag='{tag}']", namespaces=NS)
    return nodes[0].text.strip() if nodes and nodes[0].text else ""


def parse_hebis_marc(xml_bytes, query, retrieval_date=None):
    """Parse MARC-XML records into rows shaped like CATALOGUE_COLUMNS + holding_note."""
    root = etree.fromstring(xml_bytes)
    retrieval_date = retrieval_date or date.today().isoformat()
    rows = []

    for rec in root.xpath("//marc:record", namespaces=NS):
        # Title: 245 $a $b
        title_parts = []
        for df in _fields(rec, "245"):
            title_parts.extend(_subfields(df, "ab"))
        title = " ".join(title_parts).strip(" /:")

        # Authors: 100 / 700 $a
        authors = []
        for tag in ("100", "700"):
            for df in _fields(rec, tag):
                authors.extend(_subfields(df, "a"))

        # Publisher + year often in 260 or 264
        publishers, years = [], []
        for tag in ("260", "264"):
            for df in _fields(rec, tag):
                publishers.extend(_subfields(df, "b"))
                for ytext in _subfields(df, "c"):
                    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", ytext)
                    if m:
                        years.append(m.group(0))

        # ISBNs in 020 $a
        isbns = []
        for df in _fields(rec, "020"):
            for raw in _subfields(df, "a"):
                m = re.match(r"[\d\-Xx]+", raw)
                if m:
                    cleaned = re.sub(r"[^0-9Xx]", "", m.group(0)).upper()
                    if cleaned and cleaned not in isbns:
                        isbns.append(cleaned)

        # Language: 041 $a or 008 positions 35-37
        langs = []
        for df in _fields(rec, "041"):
            langs.extend(_subfields(df, "a"))
        f008 = _control(rec, "008")
        if len(f008) >= 38 and not langs:
            langs.append(f008[35:38])

        # Subjects: 650 $a
        subjects = []
        for df in _fields(rec, "650"):
            subjects.extend(_subfields(df, "a"))

        # Holdings / location hints: 852 $a $b $c $h (when present in the record)
        holdings = []
        for df in _fields(rec, "852"):
            holdings.append(" ".join(_subfields(df, "abchj")))
        for df in _fields(rec, "866"):
            holdings.extend(_subfields(df, "a"))

        record_id = _control(rec, "001")
        year = int(years[0]) if years else None
        rows.append(
            {
                "record_id": "",
                "source": "HeBIS",
                "source_record_id": record_id or None,
                "source_query": query,
                "title": title or None,
                "subtitle": None,
                "author": SEP.join(authors) if authors else None,
                "publisher": SEP.join(publishers) if publishers else None,
                "publication_year": year,
                "isbn": SEP.join(isbns) if isbns else None,
                "language": SEP.join(langs) if langs else None,
                "subjects": SEP.join(subjects) if subjects else None,
                "edition_statement": None,
                "format": None,
                "url": None,
                "retrieval_date": retrieval_date,
                "holding_note": SEP.join(h for h in holdings if h) or None,
            }
        )
    return pd.DataFrame(rows, columns=HEBIS_COLUMNS)


def _slug(text):
    return re.sub(r"\W+", "_", text).strip("_")[:60]


def retrieve_all_hebis(query, max_records=500, max_pages=None, pause=0.5, log=None):
    """Paginate a HeBIS query; save each XML page under data/raw/hebis/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    frames, start, page = [], 1, 1
    total = 0

    def say(msg):
        if log:
            log.info(msg)
        else:
            print(msg, flush=True)

    while True:
        say(f"HeBIS page {page}: startRecord={start} query={query!r}")
        try:
            xml = query_hebis(query, start_record=start, maximum_records=max_records)
        except requests.RequestException as exc:
            say(f"HTTP error on page {page}: {exc}")
            break
        raw_path = RAW_DIR / f"{_slug(query)}_{stamp}_p{page:04d}.xml"
        raw_path.write_bytes(xml)
        say(f"  saved raw XML → {raw_path}")
        total = total_records(xml)
        df = parse_hebis_marc(xml, query, stamp)
        if df.empty:
            say("  page returned 0 MARC records — stopping")
            break
        frames.append(df)
        say(f"  parsed {len(df)} rows (catalogue reports {total} total)")
        start += max_records
        page += 1
        if start > total or (max_pages and page > max_pages):
            break
        time.sleep(pause)

    n_retrieved = sum(len(f) for f in frames)
    say(f"{query!r}: {total if frames else 0} total hits, {n_retrieved} retrieved")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HEBIS_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default=None,
        help='CQL query, e.g. pica.tit="Writing Fiction" or pica.isb=9780226616551',
    )
    parser.add_argument(
        "--isbn",
        default=None,
        help="Shortcut: build an ISBN query (pica.isb=...). Overrides --query if both given.",
    )
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    if args.isbn:
        # HeBIS commonly indexes ISBN under pica.isb
        cleaned = re.sub(r"[^0-9Xx]", "", args.isbn)
        query = f"pica.isb={cleaned}"
    elif args.query:
        query = args.query
    else:
        raise SystemExit("Provide --query or --isbn")

    log = setup_logger(__file__)
    try:
        df = retrieve_all_hebis(query, max_pages=args.max_pages, pause=args.pause, log=log)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"hebis_hits_{_slug(query)}.csv"
        df.to_csv(out, index=False)
        log.info(f"saved {out} ({len(df)} rows)")
        if not df.empty:
            log.info("first rows:\n" + df.head().to_string())
        else:
            log.info("No hits (empty result is OK for a missing title).")
    finally:
        log.close()


if __name__ == "__main__":
    main()
