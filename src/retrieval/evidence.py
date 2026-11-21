"""Shared lexical evidence tiers and candidate-selection rules.

Used by term audit, Baseline B/C retrieval, and regression tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.lexicon_v02 import ConceptTerms, LexiconV02, classify_tier
from src.retrieval.match import MatchSpan, cluster_independent_cues, find_phrase_spans
from src.retrieval.regex_map import (
    ROLE_AMBIGUOUS,
    ROLE_CONCEPT_SPECIFIC,
    ROLE_CONTEXT_ONLY,
    ROLE_SUPPORTING,
    RegexConceptMap,
    load_regex_map,
)

# Surfaces that look "high" but are only general POV context.
WEAK_POV_SURFACES = {
    "point of view",
    "pov",
    "povs",
    "viewpoint",
    "viewpoints",
}

# General narrative terms that never establish concept-specific relevance alone.
GENERAL_NARRATIVE_TERMS = {
    "en": {
        "character", "characters", "voice", "perspective", "narrator", "storyteller",
        "telling", "tone", "angle", "lens", "view", "speaker",
    },
    "de": {
        "figur", "figuren", "stimme", "perspektive", "erzähler", "erzählerin",
        "erzählung", "ton", "sicht", "blick", "sprecher",
    },
}


@dataclass
class EvidenceHit:
    term: str
    tier: str  # academic_high | supporting_craft | demoted | context_only | paraphrase | regex_*
    matched_form: str = ""
    start: int = -1
    end: int = -1
    regex_name: str = ""


@dataclass
class ConceptEvidence:
    concept_id: str
    high: list[EvidenceHit] = field(default_factory=list)
    supporting: list[EvidenceHit] = field(default_factory=list)
    demoted: list[EvidenceHit] = field(default_factory=list)
    context_only: list[EvidenceHit] = field(default_factory=list)
    paraphrase: list[EvidenceHit] = field(default_factory=list)
    regex: list[EvidenceHit] = field(default_factory=list)

    @property
    def matched_terms(self) -> list[str]:
        return [h.term for h in self.high + self.supporting + self.paraphrase]

    @property
    def matched_regex(self) -> list[str]:
        return [h.regex_name or h.term for h in self.regex]

    @property
    def matched_spans(self) -> list[dict]:
        out = []
        for h in self.high + self.supporting + self.paraphrase + self.regex + self.context_only:
            if h.start < 0:
                continue
            out.append(
                {
                    "term": h.term,
                    "matched_form": h.matched_form,
                    "start": h.start,
                    "end": h.end,
                    "tier": h.tier,
                }
            )
        return out

    def evidence_tiers_summary(self) -> dict[str, int]:
        return {
            "academic_high": len(self.high),
            "supporting_craft": len(self.supporting),
            "demoted": len(self.demoted),
            "context_only": len(self.context_only),
            "paraphrase": len(self.paraphrase),
            "regex": len(self.regex),
        }


def _surface(term: str) -> str:
    return term.casefold().split(":")[-1].strip()


def _is_general_narrative(lang: str, term: str) -> bool:
    return _surface(term) in GENERAL_NARRATIVE_TERMS.get(lang, set())


def _spans_to_hits(spans: list[MatchSpan], tier: str) -> list[EvidenceHit]:
    return [
        EvidenceHit(
            term=s.term,
            tier=tier,
            matched_form=s.matched_form,
            start=s.start,
            end=s.end,
        )
        for s in cluster_independent_cues(spans)
    ]


def collect_concept_evidence(
    text: str,
    concept_id: str,
    ct: ConceptTerms,
    lang: str,
    lex: LexiconV02,
    rmap: RegexConceptMap | None = None,
    *,
    include_paraphrases: bool = True,
) -> ConceptEvidence:
    """Concept-specific literal + mapped-regex evidence for one passage."""
    rmap = rmap or load_regex_map()
    ev = ConceptEvidence(concept_id=concept_id)

    # Academic → high (unless bare general narrative singleton)
    for term in ct.academic:
        spans = find_phrase_spans(term, text, lang, tier="academic_high")
        if not spans:
            continue
        if _is_general_narrative(lang, term) and " " not in term and "-" not in term:
            ev.demoted.extend(_spans_to_hits(spans, "demoted"))
        else:
            ev.high.extend(_spans_to_hits(spans, "academic_high"))

    # Craft → supporting | demoted | high (rare)
    for term in ct.craft:
        spans = find_phrase_spans(term, text, lang, tier="supporting_craft")
        if not spans:
            continue
        tier = classify_tier(lang, "craft", term)
        if tier == "high":
            ev.high.extend(_spans_to_hits(spans, "academic_high"))
        elif tier == "supporting":
            if _is_general_narrative(lang, term):
                ev.demoted.extend(_spans_to_hits(spans, "demoted"))
            else:
                ev.supporting.extend(_spans_to_hits(spans, "supporting_craft"))
        else:
            # demoted — never counts as supporting for candidate rules
            ev.demoted.extend(_spans_to_hits(spans, "demoted"))

    # Context → context_only only
    for term in ct.context:
        spans = find_phrase_spans(term, text, lang, tier="context_only")
        if spans:
            ev.context_only.extend(_spans_to_hits(spans, "context_only"))

    # Paraphrases → paraphrase tier (may count as supporting if multi-word)
    if include_paraphrases:
        for term in ct.paraphrases:
            spans = find_phrase_spans(term, text, lang, tier="paraphrase")
            if spans:
                ev.paraphrase.extend(_spans_to_hits(spans, "paraphrase"))

    # Regex — ONLY patterns mapped to this concept
    for name, pat in lex.example_regex.get(lang, {}).items():
        mapping = rmap.get(lang, name)
        if mapping is None:
            continue
        for m in pat.finditer(text):
            matched = m.group(0)
            if not mapping.applies_to(concept_id, matched):
                continue
            role = mapping.role
            hit = EvidenceHit(
                term=f"regex:{name}",
                tier=f"regex:{role}",
                matched_form=matched,
                start=m.start(),
                end=m.end(),
                regex_name=name,
            )
            if role == ROLE_CONCEPT_SPECIFIC:
                # Weak POV regex still not strong high
                if name == "general_pov":
                    ev.context_only.append(hit)
                else:
                    ev.high.append(hit)
                    ev.regex.append(hit)
            elif role == ROLE_SUPPORTING:
                ev.supporting.append(hit)
                ev.regex.append(hit)
            elif role == ROLE_CONTEXT_ONLY:
                ev.context_only.append(hit)
                ev.regex.append(hit)
            else:
                ev.demoted.append(hit)

    # Deduplicate overlapping high/supporting cues across tiers
    ev.high = _dedupe_hits(ev.high)
    ev.supporting = _dedupe_hits(ev.supporting)
    return ev


def _dedupe_hits(hits: list[EvidenceHit]) -> list[EvidenceHit]:
    spans = [
        MatchSpan(term=h.term, start=h.start, end=h.end, matched_form=h.matched_form, tier=h.tier)
        for h in hits
        if h.start >= 0
    ]
    no_span = [h for h in hits if h.start < 0]
    kept = cluster_independent_cues(spans)
    out = [
        EvidenceHit(term=s.term, tier=s.tier, matched_form=s.matched_form, start=s.start, end=s.end,
                    regex_name=s.term.replace("regex:", "") if s.term.startswith("regex:") else "")
        for s in kept
    ]
    # Keep no-span hits that are not duplicate terms
    seen = {_surface(h.term) for h in out}
    for h in no_span:
        if _surface(h.term) not in seen:
            out.append(h)
            seen.add(_surface(h.term))
    return out


def independent_supporting_count(ev: ConceptEvidence, lang: str) -> int:
    """Count genuinely independent supporting cues (excl. demoted/context/weak POV)."""
    cues: list[EvidenceHit] = []
    for h in ev.supporting + ev.paraphrase:
        if _surface(h.term) in WEAK_POV_SURFACES:
            continue
        if _is_general_narrative(lang, h.term):
            continue
        if h.tier.startswith("regex:") and "context" in h.tier:
            continue
        cues.append(h)
    # Overlapping variants of the same expression → one cue
    return len(_dedupe_hits(cues))


def strong_high_hits(ev: ConceptEvidence) -> list[EvidenceHit]:
    strong = []
    for h in ev.high:
        surf = _surface(h.term)
        if surf in WEAK_POV_SURFACES:
            continue
        if h.regex_name == "general_pov" or h.term.casefold() == "regex:general_pov":
            continue
        if h.tier.startswith("regex:context"):
            continue
        strong.append(h)
    return strong


def candidate_status(ev: ConceptEvidence, lang: str) -> tuple[bool, str]:
    """Return (qualifies, selection_rule).

    Demoted and context-only never independently qualify.
    """
    strong = strong_high_hits(ev)
    if strong:
        return True, "strong_high_or_concept_regex"

    weak_high = [h for h in ev.high if h not in strong]
    # Context-only regex/general POV treated as weak
    weak_ctx = [
        h for h in ev.context_only
        if h.regex_name == "general_pov" or _surface(h.term) in WEAK_POV_SURFACES
    ]
    n_supp = independent_supporting_count(ev, lang)
    if (weak_high or weak_ctx) and n_supp >= 1:
        return True, "weak_pov_plus_supporting"
    if n_supp >= 2:
        return True, "two_independent_supporting"
    return False, "rejected"


def qualifies(ev: ConceptEvidence, lang: str) -> bool:
    return candidate_status(ev, lang)[0]


def collect_all_concepts(
    text: str,
    lang: str,
    lex: LexiconV02,
    rmap: RegexConceptMap | None = None,
) -> dict[str, ConceptEvidence]:
    rmap = rmap or load_regex_map()
    out = {}
    for cid, ct in lex.concepts[lang].items():
        out[cid] = collect_concept_evidence(text, cid, ct, lang, lex, rmap)
    return out


def qualifying_concepts(
    text: str,
    lang: str,
    lex: LexiconV02,
    rmap: RegexConceptMap | None = None,
) -> dict[str, ConceptEvidence]:
    return {
        cid: ev
        for cid, ev in collect_all_concepts(text, lang, lex, rmap).items()
        if qualifies(ev, lang)
    }


# ---- Baseline A (v0.1) helpers -------------------------------------------------

@dataclass
class V01Evidence:
    dimension: str
    high: list[str] = field(default_factory=list)
    medium: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    regex: list[str] = field(default_factory=list)

    def qualifies(self, min_supporting: int = 2) -> bool:
        if self.high or self.regex:
            return True
        return (len(self.medium) + len(self.context)) >= min_supporting


def collect_v01_evidence(text: str, lang: str, dimensions: dict) -> dict[str, V01Evidence]:
    """Keyword evidence for lexicon v0.1 dimensions (Baseline A)."""
    import re as _re

    out: dict[str, V01Evidence] = {}
    for dim, block in dimensions.items():
        lang_block = (block or {}).get(lang) or {}
        ev = V01Evidence(dimension=dim)
        for term in lang_block.get("high") or []:
            if find_phrase_spans(term, text, lang):
                ev.high.append(term)
        for term in lang_block.get("medium") or []:
            if find_phrase_spans(term, text, lang):
                ev.medium.append(term)
        for term in lang_block.get("context") or []:
            if find_phrase_spans(term, text, lang):
                ev.context.append(term)
        for pat in (block.get("regex") or {}).get(lang) or []:
            if _re.search(pat, text, _re.IGNORECASE):
                ev.regex.append(pat)
        if ev.high or ev.medium or ev.context or ev.regex:
            out[dim] = ev
    return out
