"""DNB SRU client: query, paginate, save raw XML, parse oai_dc into query-hit rows.

Each output row is one (record, query) hit. No deduplication happens here.

German National Library (Deutsche Nationalbibliothek) — best source for German books.
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

# Where we keep untouched XML pages and the working CSV tables.
RAW_DIR = DATA_RAW / "dnb"
OUT_DIR = DATA_INTERIM / "dnb"

# Public DNB SRU endpoint (no login needed).
DNB_URL = "https://services.dnb.de/sru/dnb"

# XML namespaces used in SRU / Dublin Core replies.
NS = {
    "srw": "http://www.loc.gov/zing/srw/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
XSI = "http://www.w3.org/2001/XMLSchema-instance"
SEP = " | "


def query_dnb(query, start_record=1, maximum_records=100, schema="oai_dc"):
    """Send one SRU search request and return the raw XML bytes."""
    params = {
        "version": "1.1",
        "operation": "searchRetrieve",
        "query": query,
        "recordSchema": schema,
        "startRecord": start_record,
        "maximumRecords": maximum_records,
    }
    response = requests.get(DNB_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.content


def total_records(xml_bytes):
    """Read how many hits the catalogue claims for this query (from the XML header)."""
    root = etree.fromstring(xml_bytes)
    n = root.xpath("string(//srw:numberOfRecords)", namespaces=NS)
    return int(n) if n else 0


def _texts(record, tag):
    """Collect all non-empty text values for one Dublin Core tag inside a record."""
    return [x.text.strip() for x in record.xpath(f".//dc:{tag}", namespaces=NS) if x.text]


def _typed_identifiers(record, type_suffix):
    """Identifiers whose xsi:type ends with e.g. 'ISBN' or 'IDN' (prefix varies: tel:, dnb:)."""
    out = []
    for x in record.xpath(".//dc:identifier", namespaces=NS):
        if x.text and x.get(f"{{{XSI}}}type", "").endswith(type_suffix):
            out.append(x.text.strip())
    return out


def parse_dnb(xml_bytes, query, retrieval_date=None):
    """Turn one XML page into a pandas DataFrame of catalogue hit rows."""
    root = etree.fromstring(xml_bytes)
    retrieval_date = retrieval_date or date.today().isoformat()
    rows = []
    for rec in root.xpath("//srw:record", namespaces=NS):
        dates = _texts(rec, "date")
        # First four-digit year that looks like a publication year.
        year = next(
            (m.group(0) for d in dates if (m := re.search(r"\b(1[5-9]\d\d|20\d\d)\b", d))),
            None,
        )
        # ISBN text may carry binding/price after the number — keep digits only.
        isbns = []
        for i in _typed_identifiers(rec, "ISBN"):
            m = re.match(r"[\d\-Xx]+", i)
            if m:
                isbns.append(re.sub(r"[^0-9Xx]", "", m.group(0)).upper())
        idns = _typed_identifiers(rec, "IDN")
        rows.append(
            {
                "source": "DNB",
                "source_record_id": idns[0] if idns else None,
                "source_query": query,
                "title": SEP.join(_texts(rec, "title")),
                "author": SEP.join(_texts(rec, "creator")),
                "publisher": SEP.join(_texts(rec, "publisher")),
                "publication_year": int(year) if year else None,
                "isbn": SEP.join(isbns),
                "language": SEP.join(_texts(rec, "language")),
                "subjects": SEP.join(_texts(rec, "subject")),
                "format": SEP.join(_texts(rec, "format")),
                "url": f"http://d-nb.info/{idns[0]}" if idns else None,
                "retrieval_date": retrieval_date,
            }
        )
    return pd.DataFrame(rows)


def _slug(text):
    """Make a short safe file-name fragment from a CQL query string."""
    return re.sub(r"\W+", "_", text).strip("_")[:60]


def retrieve_all_dnb(query, max_records=100, max_pages=None, pause=0.5, log=None):
    """Paginate through a query; raw XML pages are saved under data/raw/dnb/."""
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
        say(f"DNB page {page}: startRecord={start} query={query!r}")
        try:
            xml = query_dnb(query, start_record=start, maximum_records=max_records)
        except requests.RequestException as exc:
            say(f"HTTP error on page {page}: {exc}")
            break
        raw_path = RAW_DIR / f"{_slug(query)}_{stamp}_p{page:04d}.xml"
        raw_path.write_bytes(xml)
        say(f"  saved raw XML → {raw_path}")
        total = total_records(xml)
        df = parse_dnb(xml, query, stamp)
        if df.empty:
            say("  page returned 0 records — stopping")
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help='CQL, e.g. tit="kreatives schreiben"')
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (100 records each)")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to sleep between pages")
    args = parser.parse_args()

    log = setup_logger(__file__)
    try:
        df = retrieve_all_dnb(args.query, max_pages=args.max_pages, pause=args.pause, log=log)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"dnb_hits_{_slug(args.query)}.csv"
        df.to_csv(out, index=False)
        log.info(f"saved {out} ({len(df)} rows)")
        if not df.empty:
            log.info("first rows:\n" + df.head().to_string())
    finally:
        log.close()


if __name__ == "__main__":
    main()
