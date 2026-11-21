"""Phrase matching with bilingual variants and character offsets."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval.textnorm import normalize_text

# Soft / typographic hyphens already folded by normalize_text; here we build
# flexible search patterns over lightly cleaned original text.


@dataclass(frozen=True, slots=True)
class MatchSpan:
    term: str
    start: int
    end: int
    matched_form: str
    tier: str


def _light_clean(text: str) -> str:
    """Preserve length approximately for offset mapping: NFKC-ish without casefold."""
    import unicodedata

    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u00ad", "")
    return t


def expand_phrase_variants(phrase: str, lang: str) -> list[str]:
    """Hyphen / space / compound variants; German gender / inflection stems."""
    p = phrase.strip()
    if not p:
        return []
    variants = {p, p.casefold()}
    # Hyphenated ↔ spaced
    if "-" in p:
        variants.add(p.replace("-", " "))
        variants.add(p.replace("-", ""))
    if " " in p:
        variants.add(p.replace(" ", "-"))
        variants.add(p.replace(" ", ""))
    # Apostrophe variants
    if "'" in p or "\u2019" in p:
        variants.add(p.replace("\u2019", "'"))
        variants.add(p.replace("'", "\u2019"))
        variants.add(p.replace("'", ""))
    if lang == "de":
        # Ich-Erzähler / Ich-Erzählerin / Icherzähler
        for v in list(variants):
            vf = v.casefold()
            if "erzähler" in vf and "erzählerin" not in vf:
                variants.add(re.sub(r"(?i)erzähler\b", "Erzählerin", v))
            if "erzählerin" in vf:
                variants.add(re.sub(r"(?i)erzählerin\b", "Erzähler", v))
            # heterodiegetischer Erzähler ↔ heterodiegetische Erzählerin
            m = re.match(
                r"(?i)^(hetero|homo|auto)diegetisch(?:er|e|es|en)?\s+erzähler(?:in)?$",
                v.strip(),
            )
            if m:
                stem = m.group(1)
                variants.update(
                    [
                        f"{stem}diegetischer Erzähler",
                        f"{stem}diegetische Erzählerin",
                        f"{stem}diegetisches Erzählen",
                    ]
                )
            # interne Fokalisierung ↔ intern fokalisiert
            if re.search(r"(?i)fokalisier", v):
                base = re.sub(r"(?i)\s*fokalisier\w*$", "", v).strip()
                if base:
                    variants.add(f"{base} fokalisiert")
                    variants.add(f"{base}e Fokalisierung")
            # auktoriales Erzählen ↔ auktorialer Erzähler
            if re.search(r"(?i)auktorial", v):
                variants.update(
                    [
                        "auktoriales Erzählen",
                        "auktorialer Erzähler",
                        "auktoriale Erzählerin",
                        "auktoriale Erzählsituation",
                    ]
                )
            if re.search(r"(?i)personal", v) and re.search(r"(?i)erzähl", v):
                variants.update(
                    [
                        "personales Erzählen",
                        "personaler Erzähler",
                        "personale Erzählerin",
                        "personale Erzählsituation",
                    ]
                )
    if lang == "en":
        # Simple plural for multi-word ending in narrator / character
        for v in list(variants):
            if re.search(r"(?i)\bnarrator$", v):
                variants.add(re.sub(r"(?i)narrator$", "narrators", v))
            if re.search(r"(?i)\bcharacter$", v):
                variants.add(re.sub(r"(?i)character$", "characters", v))
            # focalization / focalisation
            if "focalization" in v.casefold():
                variants.add(re.sub(r"(?i)focalization", "focalisation", v))
            if "focalisation" in v.casefold():
                variants.add(re.sub(r"(?i)focalisation", "focalization", v))
    # Deduplicate by casefold while keeping a readable form
    seen: dict[str, str] = {}
    for v in variants:
        key = normalize_text(v)
        if key and key not in seen:
            seen[key] = v
    return list(seen.values())


def _phrase_to_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary aware pattern; hyphen/space flexible between tokens."""
    norm = normalize_text(phrase)
    parts = re.split(r"[\s\-]+", norm)
    parts = [p for p in parts if p]
    if not parts:
        return re.compile(r"(?!)")
    if len(parts) == 1:
        body = re.escape(parts[0])
        return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)
    body = r"[\s\-]+".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def find_phrase_spans(phrase: str, text: str, lang: str, *, tier: str = "") -> list[MatchSpan]:
    """Find all non-destructive matches; offsets refer to the original ``text``."""
    spans: list[MatchSpan] = []
    for variant in expand_phrase_variants(phrase, lang):
        pat = _phrase_to_pattern(variant)
        for m in pat.finditer(text):
            spans.append(
                MatchSpan(
                    term=phrase,
                    start=m.start(),
                    end=m.end(),
                    matched_form=m.group(0),
                    tier=tier,
                )
            )
    return _dedupe_overlapping(spans)


def _dedupe_overlapping(spans: list[MatchSpan]) -> list[MatchSpan]:
    """Keep longest span per overlap cluster; overlapping variants = one cue."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[MatchSpan] = []
    cur = ordered[0]
    for s in ordered[1:]:
        if s.start < cur.end:  # overlap
            if (s.end - s.start) > (cur.end - cur.start):
                cur = s
            continue
        kept.append(cur)
        cur = s
    kept.append(cur)
    return kept


def phrase_in_text(phrase: str, text: str, lang: str) -> bool:
    return bool(find_phrase_spans(phrase, text, lang))


def cluster_independent_cues(spans: list[MatchSpan]) -> list[MatchSpan]:
    """Collapse overlapping evidence spans into independent cues."""
    return _dedupe_overlapping(spans)
