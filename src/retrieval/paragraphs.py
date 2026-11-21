"""Stream body paragraphs for books in the analysis set (memory-safe)."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.textnorm import normalize_text, tokenize

JSON_ROOT = DATA_INTERIM / "fulltext" / "json"
MIN_PARA_CHARS = 40


@dataclass(slots=True)
class Paragraph:
    book_id: str
    lang: str
    corpus_role: str
    primary_group: str
    title: str
    author: str
    edition_year: str
    chapter_idx: int
    para_idx: int
    text: str
    prev_text: str = ""
    next_text: str = ""

    @property
    def passage_id(self) -> str:
        return f"{self.book_id}:c{self.chapter_idx}:p{self.para_idx}"

    def text_norm(self) -> str:
        return normalize_text(self.text)

    def tokens(self) -> list[str]:
        return tokenize(self.text)


@dataclass
class CorpusScanStats:
    expected_books: int = 0
    processed_books: int = 0
    skipped_missing: list[str] | None = None
    skipped_unreadable: list[str] | None = None
    n_paragraphs: int = 0

    def __post_init__(self) -> None:
        self.skipped_missing = self.skipped_missing or []
        self.skipped_unreadable = self.skipped_unreadable or []


def load_book_stats(lang: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{lang}_book_stats.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["in_analysis_set"] = df["in_analysis_set"].astype(str).str.lower().isin(["true", "1"])
    return df


def _field(book: pd.Series, *names: str) -> str:
    for n in names:
        if n in book.index and str(book.get(n, "")).strip():
            return str(book.get(n, "")).strip()
    return ""


def iter_body_paragraphs(
    lang: str,
    analysis_only: bool = True,
    *,
    keep_tokens: bool = False,
    with_neighbors: bool = False,
    scan_stats: CorpusScanStats | None = None,
) -> Iterator[Paragraph]:
    """Yield body paragraphs one at a time (one book JSON in memory).

    ``keep_tokens`` is accepted for API compatibility and ignored.
    Missing/unreadable JSON files are logged on ``scan_stats`` (never silent).
    """
    del keep_tokens
    stats = load_book_stats(lang)
    if analysis_only:
        stats = stats[stats["in_analysis_set"]]
    if scan_stats is not None:
        scan_stats.expected_books = len(stats)

    for _, book in stats.iterrows():
        bid = book["book_id"]
        path = JSON_ROOT / lang / f"{bid}.json"
        if not path.is_file():
            if scan_stats is not None:
                scan_stats.skipped_missing.append(bid)
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if scan_stats is not None:
                scan_stats.skipped_unreadable.append(f"{bid}:{type(exc).__name__}")
            continue

        if scan_stats is not None:
            scan_stats.processed_books += 1

        role = _field(book, "corpus_role")
        group = _field(book, "primary_group")
        title = _field(book, "title")
        author = _field(book, "author", "authors", "author_name")
        year = _field(book, "edition_year", "publication_year", "year")

        for c_idx, ch in enumerate(rec.get("chapters") or []):
            kind = (ch.get("kind") or "body").lower()
            if kind in ("front", "back"):
                continue
            paras = [(i, (p or "").strip()) for i, p in enumerate(ch.get("paragraphs") or [])]
            paras = [(i, t) for i, t in paras if len(t) >= MIN_PARA_CHARS]
            for j, (p_idx, text) in enumerate(paras):
                prev_t = paras[j - 1][1] if with_neighbors and j > 0 else ""
                next_t = paras[j + 1][1] if with_neighbors and j + 1 < len(paras) else ""
                if scan_stats is not None:
                    scan_stats.n_paragraphs += 1
                yield Paragraph(
                    book_id=bid,
                    lang=lang,
                    corpus_role=role,
                    primary_group=group,
                    title=title,
                    author=author,
                    edition_year=year,
                    chapter_idx=c_idx,
                    para_idx=p_idx,
                    text=text,
                    prev_text=prev_t,
                    next_text=next_t,
                )
        del rec


def count_paragraphs(lang: str, analysis_only: bool = True) -> int:
    return sum(1 for _ in iter_body_paragraphs(lang, analysis_only=analysis_only))
