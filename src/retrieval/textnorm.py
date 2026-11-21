"""Shared text normalization for lexicon audit and BM25."""
from __future__ import annotations

import re
import unicodedata

_SOFT_HYPHEN = "\u00ad"
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f]")
_WS = re.compile(r"\s+")
_DEHYPHEN = re.compile(r"(\w)-\n(\w)")
_TOKEN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFKC, soft hyphens, dashes/quotes, line-break dehyphenation, casefold.

    Non-destructive w.r.t. source corpora: operates on a copy only.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_SOFT_HYPHEN, "")
    t = _DEHYPHEN.sub(r"\1\2", t)
    t = _DASHES.sub("-", t)
    t = _QUOTES.sub("'", t)
    t = _WS.sub(" ", t)
    return t.casefold().strip()


def tokenize(text: str) -> list[str]:
    """Whitespace-friendly tokens after normalize_text."""
    t = normalize_text(text)
    return _TOKEN.findall(t)


def phrase_in_text(phrase: str, text_norm: str) -> bool:
    """Literal phrase match on already-normalized text (word-ish boundaries).

    Prefer ``src.retrieval.match.phrase_in_text`` for bilingual variants + offsets.
    Kept for term-audit streaming matchers that already hold normalized text.
    """
    p = normalize_text(phrase)
    if not p:
        return False
    # Allow hyphen/space equivalence on the needle against a space-folded haystack.
    hay = text_norm.replace("-", " ")
    needle = p.replace("-", " ")
    if " " in needle:
        return needle in hay
    return re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text_norm) is not None
