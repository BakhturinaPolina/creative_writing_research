"""Round 1 extraction: EPUB / MOBI / AZW3 / DOCX / page-text zips to per-book JSON.

For every manifest row in a round-1 format, reads the documents in reading
order, splits them into chapters (TOC entries and top-level headings), flags
front/back matter, and writes data/interim/fulltext/json/{lang}/{book_id}.json:

  {"book_id", "md5", "lang", "format", "title", "author",
   "extraction": {"round", "tools", "date"}, "n_words",
   "chapters": [{"idx", "title", "kind": "front|body|back", "n_words", "paragraphs": [...]}]}

Labels (corpus_role etc.) live only in the manifest; join on book_id.

Updates extraction_round / extraction_status / n_words / n_chapters in the
manifest and writes the spot-check report (statistics only; excerpts go to the
gitignored excerpts/ folder).

Usage:
  python -m src.fulltext.extract_ebook            # extract books without JSON yet
  python -m src.fulltext.extract_ebook --force    # re-extract everything
  python -m src.fulltext.extract_ebook --report-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.fulltext.books import (
    EBOOK_EXTS,
    EXCERPTS_DIR,
    FULLTEXT_DIR,
    JSON_DIR,
    MANIFEST_PATH,
    EncryptedBookError,
    ebook_documents,
    normalize_text,
    word_count,
)

REPORT_PATH = FULLTEXT_DIR / "spotcheck_round1.md"
SPOTCHECK_EXCERPTS = EXCERPTS_DIR / "spotcheck_round1.txt"

HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCKS = HEADINGS + ["p", "li", "blockquote", "pre", "dd", "dt", "td", "div"]
PAGE_NUMBER = re.compile(r"^(\[?\d{1,4}\]?|[ivxlcdm]{1,6})$", re.IGNORECASE)
# Whole-title matches only: "Literatur und Gesellschaft" is a body chapter, "Literatur" is back matter.
FRONT_TITLE = re.compile(
    r"(contents|table of contents|inhalt|inhaltsverzeichnis|copyright|copyright page|impressum|"
    r"title page|titel|titelseite|cover|umschlag|dedication|widmung|also by .*|weitere bücher|"
    r"about the (author|authors|book)|über (den|die) autor(in|en)?|zum buch|zur autorin|zum autor|"
    r"der autor|die autorin|praise for .*|epigraph|motto)[\W\d]*", re.IGNORECASE)
BACK_TITLE = re.compile(
    r"(index|register|personenregister|sachregister|notes|endnotes|anmerkungen|fußnoten|"
    r"bibliography|bibliographie|bibliografie|literaturverzeichnis|literatur|quellen|sources|"
    r"references|works cited|acknowledg(e)?ments?|danksagung|dank|glossar|glossary|permissions|credits|"
    r"mehr informationen|leseprobe|also available)[\W\d]*", re.IGNORECASE)
FILENAME_LABEL = re.compile(r"^(text|part|split|index_split|ch|chapter|section|c)?[\d_\-]+$|\.x?html?$",
                            re.IGNORECASE)
MAX_FRONT_WORDS = 3000
MOJIBAKE = re.compile(r"Ã[¤¶¼„–œŸ]|â€[™œ\x9d“”˜]")
MIN_BOOK_WORDS = 1000


def tool_versions() -> dict:
    tools = {}
    for package in ("beautifulsoup4", "lxml", "mobi"):
        try:
            tools[package] = version(package)
        except Exception:
            tools[package] = ""
    return tools


def doc_blocks(html: bytes) -> list[tuple[str, str]]:
    """(tag, text) for every innermost block element, in document order."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "nav"]):
        tag.decompose()
    body = soup.body or soup
    blocks = []
    for el in body.find_all(BLOCKS):
        if el.name not in HEADINGS and el.find(BLOCKS) is not None:
            continue  # container; its children are emitted instead
        if el.name in HEADINGS and el.find_parent(HEADINGS) is not None:
            continue
        text = normalize_text(el.get_text(" "))
        if text and not PAGE_NUMBER.match(text):
            blocks.append((el.name, text))
    if not blocks:
        text = normalize_text(body.get_text(" "))
        if text:
            blocks.append(("p", text))
    return blocks


def top_heading_level(docs_blocks: list[list[tuple[str, str]]]) -> str | None:
    """Highest heading level used at least 3 times in the book (chapter-level headings)."""
    counts = {h: 0 for h in HEADINGS[:3]}
    for blocks in docs_blocks:
        for tag, _ in blocks:
            if tag in counts:
                counts[tag] += 1
    return next((h for h in HEADINGS[:3] if counts[h] >= 3), None)


def classify(title: str, n_words: int, book_words: int, position: str) -> str:
    """A TOC label like 'Title Page' can cover the whole book, so size limits apply."""
    if FRONT_TITLE.fullmatch(title) and n_words <= MAX_FRONT_WORDS:
        return "front"
    if BACK_TITLE.fullmatch(title) and n_words <= book_words / 2:
        return "back"
    if position == "before_body" and n_words < 150:
        return "front"
    return "body"


def build_chapters(docs, toc) -> list[dict]:
    docs_blocks = [doc_blocks(html) for _, html in docs]
    level = top_heading_level(docs_blocks)
    # Without TOC or chapter headings, document / MOBI page-break boundaries are the best chapter guess.
    split_on_docs = not toc and level is None and len(docs) > 2
    chapters: list[dict] = []
    current = None

    def start(title: str):
        nonlocal current
        current = {"title": title, "paragraphs": []}
        chapters.append(current)

    for (name, _), blocks in zip(docs, docs_blocks):
        if not blocks:
            continue
        toc_title = toc.get(name, "")
        if FILENAME_LABEL.search(toc_title):
            toc_title = ""
        first_is_heading = blocks[0][0] in HEADINGS
        if current is None or toc_title or split_on_docs or (level and blocks[0][0] == level):
            start(toc_title or (blocks[0][1] if first_is_heading else ""))
        for tag, text in blocks:
            if level and tag == level and current["paragraphs"]:
                start(text)
                continue
            if tag in HEADINGS and not current["title"]:
                current["title"] = text
                continue
            if tag in HEADINGS and text == current["title"]:
                continue
            current["paragraphs"].append(text)

    out, seen_body = [], False
    book_words = sum(word_count(p) for c in chapters for p in c["paragraphs"])
    for chapter in chapters:
        if not chapter["paragraphs"] and not chapter["title"]:
            continue
        n_words = sum(word_count(p) for p in chapter["paragraphs"])
        kind = classify(chapter["title"], n_words, book_words, "after_body" if seen_body else "before_body")
        seen_body = seen_body or kind == "body"
        out.append({"idx": len(out), "title": chapter["title"][:300], "kind": kind,
                    "n_words": n_words, "paragraphs": chapter["paragraphs"]})
    return out


def extract_one(row: dict) -> dict:
    """Extract one book; returns the manifest fields to update."""
    path = ROOT / row["raw_path"]
    out_path = JSON_DIR / row["lang"] / f"{row['book_id']}.json"
    try:
        docs, toc = ebook_documents(path, row["format"])
        chapters = build_chapters(docs, toc)
    except EncryptedBookError as exc:
        return {"book_id": row["book_id"], "extraction_status": f"drm_encrypted: {exc}"[:200],
                "n_words": 0, "n_chapters": 0}
    except Exception as exc:
        return {"book_id": row["book_id"], "extraction_status": f"failed: {type(exc).__name__}: {exc}"[:200],
                "n_words": 0, "n_chapters": 0}
    body = [c for c in chapters if c["kind"] == "body"]
    n_words = sum(c["n_words"] for c in chapters)
    status = "ok" if n_words >= MIN_BOOK_WORDS else ("empty" if n_words == 0 else "short")
    record = {
        "book_id": row["book_id"], "md5": row["md5"], "lang": row["lang"], "format": row["format"],
        "title": row["title"] or row["file_title"], "author": row["author_name"] or row["file_author"],
        "extraction": {"round": 1, "tools": tool_versions(), "date": date.today().isoformat()},
        "n_words": n_words, "chapters": chapters,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    text = " ".join(p for c in chapters for p in c["paragraphs"])
    return {
        "book_id": row["book_id"], "extraction_status": status, "n_words": n_words,
        "n_chapters": len(body), "front_back_share": round(1 - sum(c["n_words"] for c in body) / n_words, 3)
        if n_words else 0,
        "replacement_chars": text.count("\ufffd"), "mojibake": len(MOJIBAKE.findall(text)),
    }


# ---------------------------------------------------------------------------
# Spot-check report
# ---------------------------------------------------------------------------

def write_report(manifest: pd.DataFrame, stats: pd.DataFrame, log) -> None:
    done = manifest[manifest["extraction_round"] == "1"].copy()
    done["n_words"] = pd.to_numeric(done["n_words"], errors="coerce").fillna(0).astype(int)
    done["n_chapters"] = pd.to_numeric(done["n_chapters"], errors="coerce").fillna(0).astype(int)
    stats = stats.set_index("book_id") if not stats.empty else stats
    lines = [
        "# Round 1 extraction spot-check",
        "",
        f"Generated by `python -m src.fulltext.extract_ebook` on {date.today().isoformat()}. "
        "Statistics only; text excerpts for manual reading are in the gitignored "
        "`excerpts/spotcheck_round1.txt`.",
        "",
        "## Status by language and format",
        "",
        done.groupby(["lang", "format", "extraction_status"]).size().rename("books").reset_index()
        .to_markdown(index=False),
        "",
        "## Words per book (status ok), by language and role",
        "",
        done[done["extraction_status"] == "ok"].groupby(["lang", "corpus_role"])["n_words"]
        .describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]].round(0).to_markdown(),
        "",
    ]
    failed = done[~done["extraction_status"].isin(["ok"])]
    lines += ["## Failed, empty or short books", "",
              failed[["book_id", "format", "extraction_status", "n_words", "file_title"]].to_markdown(index=False)
              if not failed.empty else "None.", ""]
    flat = done[(done["extraction_status"] == "ok") & (done["n_chapters"] <= 1)]
    lines += [f"## Books with 0-1 body chapters ({len(flat)})", "",
              "Chapter structure was not recoverable (single HTML file, no TOC, no headings). "
              "Text is still usable; chapter-level analysis is not.", "",
              flat[["book_id", "format", "n_words", "file_title"]].to_markdown(index=False) if not flat.empty else "None.",
              ""]
    if not stats.empty:
        enc = stats[(stats["replacement_chars"] > 20) | (stats["mojibake"] > 20)]
        lines += [f"## Encoding problems ({len(enc)} books with >20 replacement or mojibake characters)", "",
                  enc.join(done.set_index("book_id")["file_title"])[["replacement_chars", "mojibake", "file_title"]]
                  .to_markdown() if not enc.empty else "None.", ""]
        heavy = stats[stats["front_back_share"] > 0.4]
        lines += [f"## Large front/back matter share (>40% of words; {len(heavy)} books)", "",
                  "Usually long notes/bibliographies or a misclassified chapter title; check before analysis.", "",
                  heavy.join(done.set_index("book_id")["file_title"])[["front_back_share", "file_title"]]
                  .to_markdown() if not heavy.empty else "None.", ""]

    sample = spot_sample(done)
    lines += ["## Manual spot-check sample (10 EN + 10 DE, stratified by role)", "",
              "Read the excerpts file and fill in the last column.", "",
              "| book_id | role | chapters | words | first chapter titles | check |",
              "|---|---|---|---|---|---|"]
    excerpt_lines = []
    for row in sample.to_dict(orient="records"):
        record = json.loads((JSON_DIR / row["lang"] / f"{row['book_id']}.json").read_text(encoding="utf-8"))
        body = [c for c in record["chapters"] if c["kind"] == "body"]
        titles = "; ".join(c["title"][:30] for c in body[:4]).replace("|", "/")
        lines.append(f"| {row['book_id']} | {row['corpus_role']} | {row['n_chapters']} | {row['n_words']} | {titles} | |")
        excerpt_lines.append(f"===== {row['book_id']} | {row['file_title']} | {row['corpus_role']}")
        rng = random.Random(row["book_id"])
        for chapter in rng.sample(body, min(2, len(body))):
            excerpt_lines.append(f"--- chapter {chapter['idx']}: {chapter['title'][:80]}")
            excerpt_lines.extend(p[:600] for p in chapter["paragraphs"][:3])
        excerpt_lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    EXCERPTS_DIR.mkdir(parents=True, exist_ok=True)
    SPOTCHECK_EXCERPTS.write_text("\n".join(excerpt_lines), encoding="utf-8")
    log.info(f"wrote {REPORT_PATH} and {SPOTCHECK_EXCERPTS}")


def spot_sample(done: pd.DataFrame) -> pd.DataFrame:
    ok = done[done["extraction_status"] == "ok"]
    picks = []
    for lang in ("en", "de"):
        pool = ok[ok["lang"] == lang]
        quota = {"primary": 5, "comparison": 2, "negative": 2, "unreviewed": 1}
        for role, n in quota.items():
            group = pool[pool["corpus_role"] == role]
            picks.append(group.sample(min(n, len(group)), random_state=42))
        taken = sum(len(p) for p in picks[-len(quota):])
        if taken < 10:
            rest = pool.drop(pd.concat(picks[-len(quota):]).index)
            picks.append(rest.sample(min(10 - taken, len(rest)), random_state=42))
    return pd.concat(picks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-extract books that already have JSON")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    log = setup_logger(__file__)
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
    todo = manifest[manifest["format"].isin(EBOOK_EXTS)]
    if not args.force:
        todo = todo[[not (JSON_DIR / r.lang / f"{r.book_id}.json").is_file() for r in todo.itertuples()]]
    if args.report_only:
        todo = todo.iloc[0:0]
    log.info(f"{len(todo)} books to extract ({', '.join(sorted(EBOOK_EXTS))})")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(extract_one, todo.to_dict(orient="records"), chunksize=4), 1):
            results.append(result)
            if result["extraction_status"].startswith("failed"):
                log.error(f"{result['book_id']}: {result['extraction_status']}")
            if i % 50 == 0:
                log.info(f"  {i}/{len(todo)}")

    stats_path = FULLTEXT_DIR / "extraction_stats_round1.csv"
    stats = pd.DataFrame(results)
    if stats_path.is_file() and not args.force:
        old = pd.read_csv(stats_path, dtype={"book_id": str})
        stats = pd.concat([old[~old["book_id"].isin(stats.get("book_id", []))], stats])
    if not stats.empty:
        stats.to_csv(stats_path, index=False, encoding="utf-8")
        update = stats.set_index("book_id")
        idx = manifest["book_id"].isin(update.index)
        for col in ("extraction_status", "n_words", "n_chapters"):
            manifest.loc[idx, col] = manifest.loc[idx, "book_id"].map(update[col]).astype(str)
        manifest.loc[idx, "extraction_round"] = "1"
        manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
        log.info("extraction_status:\n" + manifest.loc[idx, "extraction_status"]
                 .str.replace(r":.*", "", regex=True).value_counts().to_string())
    write_report(manifest, pd.read_csv(stats_path) if stats_path.is_file() else pd.DataFrame(), log)
    log.close()


if __name__ == "__main__":
    main()
