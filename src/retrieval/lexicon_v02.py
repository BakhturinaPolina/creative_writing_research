"""Load PsyPer bilingual lexicon v0.2 draft for retrieval."""
from __future__ import annotations

import hashlib
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
        "i", "me", "we", "us", "first", "character", "characters",
    },
    "de": {
        "perspektive", "sichtweise", "blick", "standpunkt", "blickrichtung",
        "position", "sicht", "stimme", "sprecher", "erzählen", "ton",
        "erzähler", "erzählerin", "du", "dich", "sie", "anrede",
        "ich", "mich", "wir", "uns", "pronomen", "er", "name",
        "figur", "figuren",
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
    sha256: str = ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


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
        sha256=file_sha256(path),
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
    """Return evidence class: high | supporting | demoted | context_only.

    Corpus frequency is NOT used here — rarity does not imply quality.
    """
    t = term.casefold().strip()
    if tier == "context":
        return "context_only"
    if tier == "academic":
        # Bare general singles demoted even if listed under academic by mistake
        if t in AMBIGUOUS_SOLO.get(lang, set()):
            return "demoted"
        return "high"
    if tier == "paraphrase":
        return "supporting"
    if tier == "craft":
        if t in AMBIGUOUS_SOLO.get(lang, set()):
            return "demoted"
        # short single-token craft → demoted for solo / supporting use
        if " " not in t and "-" not in t and len(t) < 8:
            return "demoted"
        return "supporting"
    return "demoted"
