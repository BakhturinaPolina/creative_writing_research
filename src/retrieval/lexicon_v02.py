"""Load PsyPer bilingual lexicon (v0.2 / v0.3) for retrieval."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.common.paths import CONFIG

LEXICON_PATH = CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml"
LEXICON_PATH_V03 = CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml"

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

# Explicit evidence tiers independent of vocabulary origin.
EVIDENCE_TIERS = (
    "strong_academic",
    "strong_craft",
    "strong_paraphrase",
    "supporting",
    "context_only",
    "ambiguous_demoted",
)


@dataclass
class ConceptTerms:
    concept_id: str
    academic: list[str] = field(default_factory=list)
    craft: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    psyper: list[str] = field(default_factory=list)
    # casefolded term → explicit evidence tier override
    evidence_overrides: dict[str, str] = field(default_factory=dict)

    def bm25_query_terms(self) -> list[str]:
        return list(self.academic) + list(self.craft) + list(self.paraphrases)

    def psyper_tags_joined(self) -> str:
        return ";".join(self.psyper)


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


def _as_str_list(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def load_lexicon(path: Path | None = None) -> LexiconV02:
    path = path or LEXICON_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    concepts: dict[str, dict[str, ConceptTerms]] = {"en": {}, "de": {}}
    for item in raw.get("concepts") or []:
        cid = item["id"]
        psyper = _as_str_list(item.get("psyper"))
        for lang in ("en", "de"):
            block = item.get(lang) or {}
            overrides_raw = block.get("evidence_overrides") or {}
            overrides = {
                str(k).casefold().strip(): str(v).strip()
                for k, v in overrides_raw.items()
                if str(k).strip() and str(v).strip() in EVIDENCE_TIERS
            }
            concepts[lang][cid] = ConceptTerms(
                concept_id=cid,
                academic=list(block.get("academic") or []),
                craft=list(block.get("craft") or []),
                paraphrases=list(block.get("paraphrases") or []),
                context=list(block.get("context") or []),
                psyper=list(psyper),
                evidence_overrides=overrides,
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
    """Legacy class: high | supporting | demoted | context_only.

    Corpus frequency is NOT used here — rarity does not imply quality.
    Prefer ``resolve_evidence_tier`` for v0.3 explicit tiers.
    """
    t = term.casefold().strip()
    if tier == "context":
        return "context_only"
    if tier == "academic":
        if t in AMBIGUOUS_SOLO.get(lang, set()):
            return "demoted"
        return "high"
    if tier == "paraphrase":
        return "supporting"
    if tier == "craft":
        if t in AMBIGUOUS_SOLO.get(lang, set()):
            return "demoted"
        if " " not in t and "-" not in t and len(t) < 8:
            return "demoted"
        return "supporting"
    return "demoted"


def resolve_evidence_tier(
    lang: str,
    vocab_origin: str,
    term: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Return explicit evidence tier independent of vocabulary origin."""
    key = term.casefold().strip()
    overrides = overrides or {}
    if key in overrides:
        return overrides[key]

    if vocab_origin == "context":
        return "context_only"
    if key in AMBIGUOUS_SOLO.get(lang, set()):
        return "ambiguous_demoted"
    if vocab_origin == "academic":
        return "strong_academic"
    if vocab_origin == "craft":
        # short single-token craft → demoted unless overridden
        if " " not in key and "-" not in key and len(key) < 8:
            return "ambiguous_demoted"
        return "supporting"
    if vocab_origin == "paraphrase":
        return "supporting"
    return "ambiguous_demoted"
