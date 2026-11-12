"""Round 1b extraction: PDFs with a usable text layer to per-book JSON.

Reads every page with PyMuPDF plain text (no OCR, no layout dict), rejoins
hyphenated line breaks, splits into chapters from the PDF outline when present
or from heading-line cues (Chapter N / Kapitel N / Part …), and writes the same
JSON schema as round 1:

  data/interim/fulltext/json/{lang}/{book_id}.json

By default only rows with format=pdf and text_layer=good are extracted.
text_layer=none / poor are recorded as needs_ocr and left for round 2.
Already-extracted ebooks (round 1) are left untouched.

Usage:
  python -m src.fulltext.extract_pdf                 # good-text PDFs without JSON
  python -m src.fulltext.extract_pdf --lang en
  python -m src.fulltext.extract_pdf --include-poor
  python -m src.fulltext.extract_pdf --force
  python -m src.fulltext.extract_pdf --report-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from importlib.metadata import version

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.fulltext.books import (
    EXCERPTS_DIR,
    FULLTEXT_DIR,
    JSON_DIR,
    MANIFEST_PATH,
    PDF_TEXT_LAYERS_OK,
    normalize_text,
    pdf_pages,
    pdf_toc,
    word_count,
)

ROUND = "1b"
REPORT_PATH = FULLTEXT_DIR / "spotcheck_round1b.md"
STATS_PATH = FULLTEXT_DIR / "extraction_stats_round1b.csv"
SPOTCHECK_EXCERPTS = EXCERPTS_DIR / "spotcheck_round1b.txt"

# Whole-title matches (same spirit as extract_ebook); size limits still apply.
FRONT_TITLE = re.compile(
    r"(contents|table of contents|inhalt|inhaltsverzeichnis|copyright|copyright page|impressum|"
    r"title page|titel|titelseite|cover|umschlag|dedication|widmung|also by .*|weitere bücher|"
    r"about the (author|authors|book)|über (den|die) autor(in|en)?|zum buch|zur autorin|zum autor|"
    r"der autor|die autorin|praise for .*|epigraph|motto|acknowledg(e)?ments?|danksagung|"
    r"preface|vorwort|foreword)[\W\d]*", re.IGNORECASE)
BACK_TITLE = re.compile(
    r"(index|register|personenregister|sachregister|notes|endnotes|anmerkungen|fußnoten|"
    r"bibliography|bibliographie|bibliografie|literaturverzeichnis|literatur|quellen|sources|"
    r"references|works cited|glossar|glossary|permissions|credits|"
    r"mehr informationen|leseprobe|also available|about the author)[\W\d]*", re.IGNORECASE)

# Standalone heading lines used when the PDF has no outline.
# Numbered chapter/part only. A period after the number often starts a normal sentence
# ("Chapter 7. But presentation…"), so titles may follow only : or a dash.
CHAPTER_LINE = re.compile(
    r"^\s*(("
    r"(chapter|part|kapitel|teil|abschnitt|(ch|kap)\.?)"
    r"\s+[\divxlcdm0-9]+"
    r"(\s*[:\-–—]\s*.{1,80})?"
    r")|"
    r"(prologue|epilogue|preface|foreword|afterword|"
    r"prolog|epilog|vorwort|nachwort)"
    r")\s*$",
    re.IGNORECASE,
)
MAX_FRONT_WORDS = 3000
MIN_BOOK_WORDS = 1000
MOJIBAKE = re.compile(r"Ã[¤¶¼„–œŸ]|â€[™œ\x9d“”˜]")
MIN_CHAPTERS_FROM_HEADINGS = 3
# If heading cues fire too often, treat structure as unreliable and keep one body chapter.
MAX_HEADING_CHAPTERS = 60


def tool_versions() -> dict:
    try:
        return {"pymupdf": version("pymupdf")}
    except Exception:
        return {"pymupdf": ""}


def page_paragraphs(page_text: str) -> list[str]:
    """Split a page into paragraphs on blank lines; each paragraph is NFC-normalized."""
    blocks = re.split(r"\n\s*\n", page_text)
    out = []
    for block in blocks:
        text = normalize_text(block.replace("\n", " "))
        if text:
            out.append(text)
    if not out and page_text.strip():
        text = normalize_text(page_text.replace("\n", " "))
        if text:
            out.append(text)
    return out


def classify(title: str, n_words: int, book_words: int, position: str) -> str:
    if FRONT_TITLE.fullmatch(title) and n_words <= MAX_FRONT_WORDS:
        return "front"
    if BACK_TITLE.fullmatch(title) and n_words <= book_words / 2:
        return "back"
    if position == "before_body" and n_words < 150 and title:
        return "front"
    return "body"


def finalize_chapters(raw: list[dict]) -> list[dict]:
    book_words = sum(word_count(p) for c in raw for p in c["paragraphs"])
    out, seen_body = [], False
    for chapter in raw:
        if not chapter["paragraphs"] and not chapter["title"]:
            continue
        n_words = sum(word_count(p) for p in chapter["paragraphs"])
        kind = classify(chapter["title"], n_words, book_words,
                        "after_body" if seen_body else "before_body")
        seen_body = seen_body or kind == "body"
        out.append({"idx": len(out), "title": chapter["title"][:300], "kind": kind,
                    "n_words": n_words, "paragraphs": chapter["paragraphs"]})
    return out


def chapters_from_toc(pages: list[str], toc: list[tuple[int, str, int]]) -> list[dict] | None:
    """Split on level-1 outline entries (fallback to the shallowest level present)."""
    if not toc or not pages:
        return None
    levels = {level for level, _, _ in toc}
    target = 1 if 1 in levels else min(levels)
    starts: list[tuple[int, str]] = []  # 0-based page index, title
    for level, title, page in toc:
        if level != target:
            continue
        idx = max(0, min(page - 1, len(pages) - 1))
        if starts and starts[-1][0] == idx:
            # Same page as previous entry: keep the longer / later title.
            starts[-1] = (idx, title)
            continue
        starts.append((idx, title))
    if len(starts) < 2:
        return None
    # Anything before the first bookmark is a untitled front block.
    if starts[0][0] > 0:
        starts.insert(0, (0, ""))
    raw = []
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(pages)
        paras = [p for page in pages[start:end] for p in page_paragraphs(page)]
        raw.append({"title": title, "paragraphs": paras})
    return finalize_chapters(raw)


def heading_title(line: str) -> str | None:
    text = line.strip()
    if not text or len(text) > 120:
        return None
    if CHAPTER_LINE.match(text):
        return normalize_text(text)
    return None


def chapters_from_headings(pages: list[str]) -> list[dict] | None:
    """Scan early lines of each page for chapter-like headings.

    Restricting to the top of the page avoids mid-paragraph false hits and skips
    tables of contents (many chapter labels on one page).
    """
    hits: list[tuple[int, int, str]] = []  # page_idx, line_idx, title
    page_lines = [page.splitlines() for page in pages]
    for p_idx, lines in enumerate(page_lines):
        nonempty = [(l_idx, line) for l_idx, line in enumerate(lines) if line.strip()]
        page_hits = []
        for l_idx, line in nonempty[:8]:
            title = heading_title(line)
            if title:
                page_hits.append((p_idx, l_idx, title))
        # TOC pages list many chapters; chapter-start pages usually have one label.
        if 1 <= len(page_hits) <= 2:
            hits.extend(page_hits[:1])
    if len(hits) < MIN_CHAPTERS_FROM_HEADINGS:
        return None

    raw: list[dict] = []
    # Text before the first heading.
    first_p, first_l, _ = hits[0]
    pre = []
    for p_idx in range(first_p):
        pre.extend(page_paragraphs(pages[p_idx]))
    if first_l > 0:
        pre.extend(page_paragraphs("\n".join(page_lines[first_p][:first_l])))
    if pre:
        raw.append({"title": "", "paragraphs": pre})

    for i, (p_idx, l_idx, title) in enumerate(hits):
        if i + 1 < len(hits):
            np, nl, _ = hits[i + 1]
        else:
            np, nl = len(pages), 0
        chunk_lines = []
        if p_idx == np:
            chunk_lines = page_lines[p_idx][l_idx + 1:nl]
        else:
            chunk_lines = page_lines[p_idx][l_idx + 1:]
            for mid in range(p_idx + 1, np):
                chunk_lines.extend(page_lines[mid])
            if np < len(pages):
                chunk_lines.extend(page_lines[np][:nl])
        paras = page_paragraphs("\n".join(chunk_lines))
        raw.append({"title": title, "paragraphs": paras})
    return finalize_chapters(raw)


def chapters_flat(pages: list[str]) -> list[dict]:
    """One body chapter; text is still usable when structure cannot be recovered."""
    paras = [p for page in pages for p in page_paragraphs(page)]
    return finalize_chapters([{"title": "", "paragraphs": paras}])


def build_chapters(pages: list[str], toc: list[tuple[int, str, int]]) -> tuple[list[dict], str]:
    toc_chapters = chapters_from_toc(pages, toc)
    if toc_chapters and sum(1 for c in toc_chapters if c["kind"] == "body") >= 2:
        return toc_chapters, "toc"

    heading_chapters = chapters_from_headings(pages)
    if heading_chapters:
        n_body = sum(1 for c in heading_chapters if c["kind"] == "body")
        if 2 <= n_body <= MAX_HEADING_CHAPTERS:
            return heading_chapters, "headings"
    return chapters_flat(pages), "flat"


def extract_one(row: dict) -> dict:
    path = ROOT / row["raw_path"]
    out_path = JSON_DIR / row["lang"] / f"{row['book_id']}.json"
    layer = (row.get("text_layer") or "").strip()
    if layer and layer not in PDF_TEXT_LAYERS_OK and not row.get("_include_poor"):
        status = "needs_ocr" if layer in {"none", "poor"} else f"skipped_text_layer:{layer}"
        return {"book_id": row["book_id"], "extraction_status": status, "n_words": 0,
                "n_chapters": 0, "chapter_source": "", "front_back_share": 0,
                "replacement_chars": 0, "mojibake": 0, "n_pages_extracted": 0}

    try:
        pages = pdf_pages(path)
        toc = pdf_toc(path)
        chapters, source = build_chapters(pages, toc)
    except Exception as exc:
        return {"book_id": row["book_id"], "extraction_status": f"failed: {type(exc).__name__}: {exc}"[:200],
                "n_words": 0, "n_chapters": 0, "chapter_source": "", "front_back_share": 0,
                "replacement_chars": 0, "mojibake": 0, "n_pages_extracted": 0}

    body = [c for c in chapters if c["kind"] == "body"]
    n_words = sum(c["n_words"] for c in chapters)
    status = "ok" if n_words >= MIN_BOOK_WORDS else ("empty" if n_words == 0 else "short")
    record = {
        "book_id": row["book_id"], "md5": row["md5"], "lang": row["lang"], "format": "pdf",
        "title": row["title"] or row["file_title"], "author": row["author_name"] or row["file_author"],
        "extraction": {"round": ROUND, "tools": tool_versions(), "date": date.today().isoformat(),
                       "chapter_source": source, "n_pages": len(pages)},
        "n_words": n_words, "chapters": chapters,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    text = " ".join(p for c in chapters for p in c["paragraphs"])
    return {
        "book_id": row["book_id"], "extraction_status": status, "n_words": n_words,
        "n_chapters": len(body), "chapter_source": source,
        "front_back_share": round(1 - sum(c["n_words"] for c in body) / n_words, 3) if n_words else 0,
        "replacement_chars": text.count("\ufffd"), "mojibake": len(MOJIBAKE.findall(text)),
        "n_pages_extracted": len(pages),
    }


def write_report(manifest: pd.DataFrame, stats: pd.DataFrame, log) -> None:
    done = manifest[manifest["extraction_round"] == ROUND].copy()
    done["n_words"] = pd.to_numeric(done["n_words"], errors="coerce").fillna(0).astype(int)
    done["n_chapters"] = pd.to_numeric(done["n_chapters"], errors="coerce").fillna(0).astype(int)
    lines = [
        "# Round 1b PDF extraction spot-check",
        "",
        f"Generated by `python -m src.fulltext.extract_pdf` on {date.today().isoformat()}. "
        "Statistics only; text excerpts for manual reading are in the gitignored "
        "`excerpts/spotcheck_round1b.txt`.",
        "",
        "## Status by language and text_layer",
        "",
    ]
    if done.empty:
        lines += ["No round-1b rows in the manifest yet.", ""]
    else:
        lines += [
            done.groupby(["lang", "text_layer", "extraction_status"]).size()
            .rename("books").reset_index().to_markdown(index=False),
            "",
            "## Words per book (status ok), by language and role",
            "",
            done[done["extraction_status"] == "ok"].groupby(["lang", "corpus_role"])["n_words"]
            .describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]].round(0).to_markdown(),
            "",
        ]
        if not stats.empty and "chapter_source" in stats.columns:
            src = stats.merge(done[["book_id", "lang"]], on="book_id", how="left")
            lines += ["## Chapter recovery method", "",
                      src.groupby(["lang", "chapter_source"]).size().rename("books")
                      .reset_index().to_markdown(index=False), ""]
        failed = done[~done["extraction_status"].isin(["ok"])]
        lines += ["## Failed, empty, short or needs_ocr", "",
                  failed[["book_id", "text_layer", "extraction_status", "n_words", "file_title"]]
                  .to_markdown(index=False) if not failed.empty else "None.", ""]
        flat = done[(done["extraction_status"] == "ok") & (done["n_chapters"] <= 1)]
        lines += [f"## Books with 0-1 body chapters ({len(flat)})", "",
                  "No usable TOC/heading structure; full text is still in one body chapter.", "",
                  flat[["book_id", "n_words", "file_title"]].to_markdown(index=False)
                  if not flat.empty else "None.", ""]
        if not stats.empty:
            enc = stats[(stats.get("replacement_chars", 0) > 20) | (stats.get("mojibake", 0) > 20)]
            if "replacement_chars" in stats.columns:
                lines += [f"## Encoding problems ({len(enc)} books with >20 replacement/mojibake chars)", "",
                          enc.merge(done[["book_id", "file_title"]], on="book_id", how="left")
                          [["book_id", "replacement_chars", "mojibake", "file_title"]]
                          .to_markdown(index=False) if not enc.empty else "None.", ""]

        sample = spot_sample(done)
        lines += ["## Manual spot-check sample (up to 10 EN + 10 DE)", "",
                  "| book_id | role | chapters | words | first chapter titles | check |",
                  "|---|---|---|---|---|---|"]
        excerpt_lines = []
        for row in sample.to_dict(orient="records"):
            path = JSON_DIR / row["lang"] / f"{row['book_id']}.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            body = [c for c in record["chapters"] if c["kind"] == "body"]
            titles = "; ".join(c["title"][:30] for c in body[:4]).replace("|", "/")
            lines.append(
                f"| {row['book_id']} | {row['corpus_role']} | {row['n_chapters']} | "
                f"{row['n_words']} | {titles} | |"
            )
            excerpt_lines.append(f"===== {row['book_id']} | {row['file_title']} | {row['corpus_role']}")
            rng = random.Random(row["book_id"])
            for chapter in rng.sample(body, min(2, len(body))) if body else []:
                excerpt_lines.append(f"--- chapter {chapter['idx']}: {chapter['title'][:80]}")
                excerpt_lines.extend(p[:600] for p in chapter["paragraphs"][:3])
            excerpt_lines.append("")
        EXCERPTS_DIR.mkdir(parents=True, exist_ok=True)
        SPOTCHECK_EXCERPTS.write_text("\n".join(excerpt_lines), encoding="utf-8")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"wrote {REPORT_PATH}")


def spot_sample(done: pd.DataFrame) -> pd.DataFrame:
    ok = done[done["extraction_status"] == "ok"]
    picks = []
    for lang in ("en", "de"):
        pool = ok[ok["lang"] == lang]
        quota = {"primary": 5, "comparison": 2, "negative": 2, "unreviewed": 1}
        for role, n in quota.items():
            group = pool[pool["corpus_role"] == role]
            if not group.empty:
                picks.append(group.sample(min(n, len(group)), random_state=42))
        if not picks:
            continue
        taken = sum(len(p) for p in picks[-len(quota):]) if picks else 0
        # recount last lang's picks only
        lang_picks = [p for p in picks if (p["lang"] == lang).all()]
        taken = sum(len(p) for p in lang_picks)
        if taken < 10 and not pool.empty:
            already = pd.concat(lang_picks).index if lang_picks else []
            rest = pool.drop(index=already, errors="ignore")
            if not rest.empty:
                picks.append(rest.sample(min(10 - taken, len(rest)), random_state=42))
    return pd.concat(picks) if picks else done.iloc[0:0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-extract PDFs that already have JSON")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--include-poor", action="store_true",
                        help="also try text_layer=poor (still skips none)")
    parser.add_argument("--lang", choices=["en", "de", "other"],
                        help="restrict to one manifest language")
    parser.add_argument("--role", choices=["primary", "comparison", "negative", "unreviewed"],
                        help="restrict to one corpus_role")
    parser.add_argument("--workers", type=int, default=2,
                        help="process workers (default 2; keep low on small RAM machines)")
    parser.add_argument("--limit", type=int, default=0, help="extract at most N books (0 = all)")
    args = parser.parse_args()

    log = setup_logger(__file__)
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
    todo = manifest[manifest["format"] == "pdf"].copy()
    if args.lang:
        todo = todo[todo["lang"] == args.lang]
    if args.role:
        todo = todo[todo["corpus_role"] == args.role]
    allowed = set(PDF_TEXT_LAYERS_OK) | ({"poor"} if args.include_poor else set())
    todo = todo[todo["text_layer"].isin(allowed)]

    if not args.force:
        todo = todo[[not (JSON_DIR / r.lang / f"{r.book_id}.json").is_file()
                     for r in todo.itertuples()]]
    if args.limit and args.limit > 0:
        todo = todo.head(args.limit)
    if args.report_only:
        todo = todo.iloc[0:0]
    log.info(f"{len(todo)} PDFs to extract (round {ROUND}, workers={args.workers})")

    rows = todo.to_dict(orient="records")
    for row in rows:
        row["_include_poor"] = args.include_poor

    results = []
    if rows:
        # ProcessPool keeps peak RAM down versus loading many PDFs in one process.
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for i, result in enumerate(pool.map(extract_one, rows, chunksize=1), 1):
                results.append(result)
                status = result["extraction_status"]
                if status.startswith("failed"):
                    log.error(f"{result['book_id']}: {status}")
                if i % 25 == 0 or i == len(rows):
                    log.info(f"  {i}/{len(rows)}")

    stats = pd.DataFrame(results)
    if STATS_PATH.is_file() and not args.force and not stats.empty:
        old = pd.read_csv(STATS_PATH, dtype={"book_id": str})
        stats = pd.concat([old[~old["book_id"].isin(stats["book_id"])], stats], ignore_index=True)
    if not stats.empty:
        stats.to_csv(STATS_PATH, index=False, encoding="utf-8")
        update = stats.set_index("book_id")
        idx = manifest["book_id"].isin(update.index)
        for col in ("extraction_status", "n_words", "n_chapters"):
            manifest.loc[idx, col] = manifest.loc[idx, "book_id"].map(update[col]).astype(str)
        manifest.loc[idx, "extraction_round"] = ROUND
        manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
        log.info("extraction_status:\n" + manifest.loc[idx, "extraction_status"]
                 .str.replace(r":.*", "", regex=True).value_counts().to_string())
        if "chapter_source" in stats.columns:
            log.info("chapter_source:\n" + stats["chapter_source"].value_counts().to_string())

    write_report(manifest, pd.read_csv(STATS_PATH) if STATS_PATH.is_file() else pd.DataFrame(), log)
    log.close()


if __name__ == "__main__":
    main()
