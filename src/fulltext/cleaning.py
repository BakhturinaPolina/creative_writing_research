"""Shared fulltext cleaning: residual hyphen rejoining and running-header stripping.

Used by clean_fulltext (post-extract rewrite of JSON) and preprocess_qc (metrics).
Also available to extractors so new extracts get the same paragraph-level rules.
"""
from __future__ import annotations

import re
from collections import Counter

# Version recorded in JSON under extraction.cleaning.rules_version
RULES_VERSION = "2"  # v2: also rejoin hyphenation split across adjacent paragraphs

# Coordinated compounds / intentional hyphen+space: keep as-is (DE and EN).
# e.g. "Menschheits- und", "pre- and post-war", "either- or"
COORD_RIGHT = re.compile(
    r"^(und|oder|wie|bzw\.?|als|auch|beziehungsweise|and|or|to|as)\b",
    re.IGNORECASE,
)

# Residual line-break hyphen left after whitespace collapse: "gezwun- gen", "inci- dents".
# Right side must start with a lowercase letter (continuation of a word), not a capital (new word).
RESIDUAL_HYPHEN = re.compile(
    r"(\w+)-\s+([a-zäöüßàáâãåæçèéêëìíîïñòóôõøùúûüýÿœ][^\s-]*)",
    re.UNICODE,
)

# Analysis-aligned quote/dash normalize (see corpus_stats.clean_text).
_TRANSLATE = str.maketrans({
    "\u2019": "'",
    "\u2018": "'",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u00ad": None,
})

# Running header / footer heuristics
DEFAULT_MIN_REPEATS = 10
DEFAULT_MAX_HEADER_WORDS = 12


def normalize_quotes_dashes(text: str) -> str:
    """Curly quotes and Unicode hyphens → ASCII; soft hyphens dropped; whitespace collapsed."""
    return " ".join(text.translate(_TRANSLATE).split())


def count_residual_hyphens(text: str) -> tuple[int, int]:
    """Return (rejoinable_count, protected_coord_count) for residual word- word patterns."""
    rejoinable = protected = 0
    for match in RESIDUAL_HYPHEN.finditer(text):
        if COORD_RIGHT.match(match.group(2)):
            protected += 1
        else:
            rejoinable += 1
    return rejoinable, protected


def rejoin_residual_hyphens(text: str) -> tuple[str, int]:
    """Rejoin residual hyphenation; leave coordinated compounds untouched.

    Returns (cleaned_text, n_rejoins).
    """
    n = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal n
        if COORD_RIGHT.match(match.group(2)):
            return match.group(0)
        n += 1
        return match.group(1) + match.group(2)

    prev = None
    while prev != text:
        prev = text
        text = RESIDUAL_HYPHEN.sub(repl, text)
    return text, n


# Paragraph ending in a hyphenated stump (line-break split across HTML blocks).
_PARA_END_HYPHEN = re.compile(r"(\w+)-$", re.UNICODE)
_PARA_START_CONTINUATION = re.compile(
    r"^([a-zäöüßàáâãåæçèéêëìíîïñòóôõøùúûüýÿœ][^\s]*)",
    re.UNICODE,
)


def count_cross_paragraph_hyphens(paragraphs: list[str]) -> tuple[int, int]:
    """Return (rejoinable, protected) for word- | next-para splits."""
    rejoinable = protected = 0
    for i in range(1, len(paragraphs)):
        prev, cur = paragraphs[i - 1].rstrip(), paragraphs[i].lstrip()
        if not _PARA_END_HYPHEN.search(prev):
            continue
        m = _PARA_START_CONTINUATION.match(cur)
        if not m:
            continue
        if COORD_RIGHT.match(m.group(1)):
            protected += 1
        else:
            rejoinable += 1
    return rejoinable, protected


def rejoin_cross_paragraph_hyphens(paragraphs: list[str]) -> tuple[list[str], int]:
    """Merge '... gezwun-' + 'gen. ...' across adjacent paragraphs.

    Coordinated compounds ('Menschheits-' + 'und …') are left as separate paragraphs.
    """
    if not paragraphs:
        return [], 0
    out = [paragraphs[0]]
    n = 0
    for p in paragraphs[1:]:
        prev = out[-1]
        end = _PARA_END_HYPHEN.search(prev.rstrip())
        start = _PARA_START_CONTINUATION.match(p.lstrip())
        if end and start and not COORD_RIGHT.match(start.group(1)):
            # Drop trailing hyphen on previous; glue continuation without extra space.
            base = prev.rstrip()[:-1]
            out[-1] = base + p.lstrip()
            n += 1
        else:
            out.append(p)
    return out, n


def paragraph_key(text: str) -> str:
    return " ".join(text.lower().split())


def repeated_short_keys(
    paragraphs: list[str],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_words: int = DEFAULT_MAX_HEADER_WORDS,
) -> set[str]:
    """Keys of short paragraphs that repeat often enough to be running headers/footers."""
    keys = [paragraph_key(p) for p in paragraphs if p.strip()]
    counts = Counter(keys)
    drop: set[str] = set()
    for key, n in counts.items():
        if not key or n < min_repeats:
            continue
        if len(key.split()) > max_words:
            continue
        drop.add(key)
    return drop


def count_repeated_short_paragraphs(
    paragraphs: list[str],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_words: int = DEFAULT_MAX_HEADER_WORDS,
) -> tuple[int, int]:
    """Return (n_paragraphs_matching_drop_keys, n_distinct_drop_keys)."""
    drop = repeated_short_keys(paragraphs, min_repeats=min_repeats, max_words=max_words)
    if not drop:
        return 0, 0
    n = sum(1 for p in paragraphs if paragraph_key(p) in drop)
    return n, len(drop)


def strip_repeated_paragraphs(
    paragraphs: list[str],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_words: int = DEFAULT_MAX_HEADER_WORDS,
) -> tuple[list[str], int]:
    """Drop repeated short paragraphs (running headers/footers). Returns (kept, n_removed)."""
    drop = repeated_short_keys(paragraphs, min_repeats=min_repeats, max_words=max_words)
    if not drop:
        return list(paragraphs), 0
    out: list[str] = []
    removed = 0
    for p in paragraphs:
        if paragraph_key(p) in drop:
            removed += 1
            continue
        out.append(p)
    return out, removed


def clean_paragraph(text: str) -> tuple[str, int]:
    """Quote/dash normalize + residual hyphen rejoin for one paragraph.

    Returns (cleaned, n_hyphen_rejoins).
    """
    text = normalize_quotes_dashes(text)
    return rejoin_residual_hyphens(text)


def clean_book_paragraphs(
    paragraphs: list[str],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_words: int = DEFAULT_MAX_HEADER_WORDS,
) -> tuple[list[str], dict]:
    """Clean paragraphs of one book (any chapter order flattened or per-chapter).

    Applies paragraph-level cleaning first, then strips repeated short paragraphs.
    Returns (cleaned_paragraphs, stats).
    """
    cleaned: list[str] = []
    hyphen_joins = 0
    for p in paragraphs:
        text, n = clean_paragraph(p)
        if text:
            cleaned.append(text)
            hyphen_joins += n
    cleaned, headers_removed = strip_repeated_paragraphs(
        cleaned, min_repeats=min_repeats, max_words=max_words,
    )
    stats = {
        "hyphen_joins": hyphen_joins,
        "headers_removed": headers_removed,
        "paragraphs_in": len(paragraphs),
        "paragraphs_out": len(cleaned),
        "rules_version": RULES_VERSION,
    }
    return cleaned, stats
