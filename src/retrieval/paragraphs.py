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
    chapter_idx: int
    para_idx: int
    text: str

    def text_norm(self) -> str:
        return normalize_text(self.text)

    def tokens(self) -> list[str]:
        return tokenize(self.text)


def load_book_stats(lang: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{lang}_book_stats.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["in_analysis_set"] = df["in_analysis_set"].astype(str).str.lower().isin(["true", "1"])
    return df


def iter_body_paragraphs(
    lang: str,
    analysis_only: bool = True,
    *,
    keep_tokens: bool = False,
) -> Iterator[Paragraph]:
    """Yield body paragraphs one at a time (one book JSON in memory).

    ``keep_tokens`` is accepted for API compatibility and ignored; callers
    should tokenize ephemerally via ``Paragraph.tokens()``.
    """
    del keep_tokens  # unused; tokenize on demand
    stats = load_book_stats(lang)
    if analysis_only:
        stats = stats[stats["in_analysis_set"]]
    for _, book in stats.iterrows():
        bid = book["book_id"]
        path = JSON_ROOT / lang / f"{bid}.json"
        if not path.is_file():
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        role = book.get("corpus_role", "")
        group = book.get("primary_group", "")
        title = book.get("title", "")
        for c_idx, ch in enumerate(rec.get("chapters") or []):
            kind = (ch.get("kind") or "body").lower()
            if kind in ("front", "back"):
                continue
            for p_idx, para in enumerate(ch.get("paragraphs") or []):
                text = (para or "").strip()
                if len(text) < MIN_PARA_CHARS:
                    continue
                yield Paragraph(
                    book_id=bid,
                    lang=lang,
                    corpus_role=role,
                    primary_group=group,
                    title=title,
                    chapter_idx=c_idx,
                    para_idx=p_idx,
                    text=text,
                )
        del rec


def count_paragraphs(lang: str, analysis_only: bool = True) -> int:
    return sum(1 for _ in iter_body_paragraphs(lang, analysis_only=analysis_only))
