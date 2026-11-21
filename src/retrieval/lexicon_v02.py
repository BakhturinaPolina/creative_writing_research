"""Load PsyPer bilingual lexicon v0.2 draft for retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.common.paths import CONFIG

LEXICON_PATH = CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml"

# Ultra-ambiguous craft/context singles never used as solo high cues.
AMBIGUOUS_SOLO = {
    "en": {
        "perspective", "angle", "lens", "position", "standpoint", "view",
        "voice", "speaker", "telling", "narrator", "storyteller", "tone",
        "you", "address", "addressee", "he", "she", "they", "third", "pronoun",
        "i", "me", "we", "us", "first",
    },
    "de": {
        "perspektive", "sichtweise", "blick", "standpunkt", "blickrichtung",
        "position", "sicht", "stimme", "sprecher", "erzählen", "ton",
        "erzähler", "erzählerin", "du", "dich", "sie", "anrede",
        "ich", "mich", "wir", "uns", "pronomen", "er", "name",
    },
}


@dataclass
class ConceptTerms:
    concept_id: str
    academic: list[str] = field(default_factory=list)
    craft: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    def bm25_query_terms(self) -> list[str]:
        return list(self.academic) + list(self.craft) + list(self.paraphrases)


@dataclass
class LexiconV02:
    version: str
    concepts: dict[str, dict[str, ConceptTerms]]  # lang -> id -> terms
    example_regex: dict[str, dict[str, re.Pattern]]
    pilot_examples: list[dict]
    context_rules: list[str]
    path: Path


def load_lexicon(path: Path | None = None) -> LexiconV02:
    path = path or LEXICON_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    concepts: dict[str, dict[str, ConceptTerms]] = {"en": {}, "de": {}}
    for item in raw.get("concepts") or []:
        cid = item["id"]
        for lang in ("en", "de"):
            block = item.get(lang) or {}
            concepts[lang][cid] = ConceptTerms(
                concept_id=cid,
                academic=list(block.get("academic") or []),
                craft=list(block.get("craft") or []),
                paraphrases=list(block.get("paraphrases") or []),
                context=list(block.get("context") or []),
            )
    regexes: dict[str, dict[str, re.Pattern]] = {"en": {}, "de": {}}
    for lang, pats in (raw.get("example_regex") or {}).items():
        for name, pat in (pats or {}).items():
            regexes[lang][name] = re.compile(pat, re.IGNORECASE)
    return LexiconV02(
        version=str(raw.get("version", "")),
        concepts=concepts,
        example_regex=regexes,
        pilot_examples=list(raw.get("pilot_examples") or []),
        context_rules=list(raw.get("context_rules") or []),
        path=path,
    )


def iter_literal_terms(lex: LexiconV02, lang: str) -> list[tuple[str, str, str]]:
    """Yield (concept_id, tier, term) for academic and craft only (not paraphrases)."""
    out = []
    for cid, ct in lex.concepts[lang].items():
        for term in ct.academic:
            out.append((cid, "academic", term))
        for term in ct.craft:
            out.append((cid, "craft", term))
    return out


def classify_tier(lang: str, tier: str, term: str) -> str:
    """high | supporting | demoted | context_only — for audit / candidate rule."""
    t = term.casefold().strip()
    if tier == "academic":
        return "high"
    if t in AMBIGUOUS_SOLO.get(lang, set()) or " " not in t and "-" not in t and len(t) < 8:
        # short single-token craft → demoted for solo use
        if tier == "craft":
            return "demoted"
    if tier == "craft":
        return "supporting"
    return "demoted"
