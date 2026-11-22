"""Shared lexical evidence tiers and candidate-selection rules.

Used by term audit, Baseline B/C retrieval, and regression tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.lexicon_v02 import (
    ConceptTerms,
    LexiconV02,
    classify_tier,
    resolve_evidence_tier,
)
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

STRONG_TIERS = frozenset({"strong_academic", "strong_craft", "strong_paraphrase", "academic_high"})


@dataclass
class EvidenceHit:
    term: str
    tier: str  # explicit evidence tier or regex:* role
    matched_form: str = ""
    start: int = -1
    end: int = -1
    regex_name: str = ""
    vocab_origin: str = ""  # academic | craft | paraphrase | context | regex


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
                    "vocab_origin": h.vocab_origin,
                }
            )
        return out

    def evidence_tiers_summary(self) -> dict[str, int]:
        # Preserve legacy keys used by reports/tests; add explicit v0.3 keys.
        return {
            "academic_high": len([h for h in self.high if h.tier in ("academic_high", "strong_academic")]),
            "strong_academic": len([h for h in self.high if h.tier == "strong_academic"]),
            "strong_craft": len([h for h in self.high if h.tier == "strong_craft"]),
            "strong_paraphrase": len(
                [h for h in self.high + self.paraphrase if h.tier == "strong_paraphrase"]
            ),
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


def _spans_to_hits(
    spans: list[MatchSpan],
    tier: str,
    *,
    vocab_origin: str = "",
) -> list[EvidenceHit]:
    return [
        EvidenceHit(
            term=s.term,
            tier=tier,
            matched_form=s.matched_form,
            start=s.start,
            end=s.end,
            vocab_origin=vocab_origin,
        )
        for s in cluster_independent_cues(spans)
    ]


def _bucket_for_tier(ev: ConceptEvidence, tier: str, hits: list[EvidenceHit]) -> None:
    if tier in ("strong_academic", "strong_craft", "strong_paraphrase", "academic_high"):
        # Keep legacy "academic_high" label for un-overridden academic for summary compat
        for h in hits:
            if tier == "strong_academic":
                h.tier = "academic_high"  # legacy alias used by strong_high_hits / reports
            ev.high.append(h)
    elif tier == "supporting":
        ev.supporting.extend(hits)
        # paraphrases that remain supporting also appear in paraphrase list for matched_terms
        for h in hits:
            if h.vocab_origin == "paraphrase":
                ev.paraphrase.append(h)
    elif tier == "context_only":
        ev.context_only.extend(hits)
    else:
        # ambiguous_demoted / demoted
        for h in hits:
            h.tier = "demoted" if h.tier == "ambiguous_demoted" else h.tier
        ev.demoted.extend(hits)


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
    overrides = ct.evidence_overrides or {}

    # Academic
    for term in ct.academic:
        spans = find_phrase_spans(term, text, lang, tier="strong_academic")
        if not spans:
            continue
        tier = resolve_evidence_tier(lang, "academic", term, overrides)
        if _is_general_narrative(lang, term) and " " not in term and "-" not in term:
            tier = "ambiguous_demoted"
        _bucket_for_tier(
            ev,
            tier,
            _spans_to_hits(spans, tier, vocab_origin="academic"),
        )

    # Craft
    for term in ct.craft:
        spans = find_phrase_spans(term, text, lang, tier="supporting_craft")
        if not spans:
            continue
        tier = resolve_evidence_tier(lang, "craft", term, overrides)
        if tier == "supporting" and _is_general_narrative(lang, term):
            tier = "ambiguous_demoted"
        hits = _spans_to_hits(spans, tier, vocab_origin="craft")
        if tier in ("strong_craft", "strong_academic", "strong_paraphrase"):
            for h in hits:
                h.tier = tier
            ev.high.extend(hits)
        elif tier == "supporting":
            for h in hits:
                h.tier = "supporting_craft"
            ev.supporting.extend(hits)
        else:
            for h in hits:
                h.tier = "demoted"
            ev.demoted.extend(hits)

    # Context → context_only only
    for term in ct.context:
        spans = find_phrase_spans(term, text, lang, tier="context_only")
        if spans:
            ev.context_only.extend(
                _spans_to_hits(spans, "context_only", vocab_origin="context")
            )

    # Paraphrases
    if include_paraphrases:
        for term in ct.paraphrases:
            spans = find_phrase_spans(term, text, lang, tier="paraphrase")
            if not spans:
                continue
            tier = resolve_evidence_tier(lang, "paraphrase", term, overrides)
            hits = _spans_to_hits(spans, tier, vocab_origin="paraphrase")
            if tier == "strong_paraphrase":
                for h in hits:
                    h.tier = "strong_paraphrase"
                ev.high.extend(hits)
                ev.paraphrase.extend(hits)
            elif tier == "supporting":
                for h in hits:
                    h.tier = "paraphrase"
                ev.paraphrase.extend(hits)
            else:
                for h in hits:
                    h.tier = "demoted"
                ev.demoted.extend(hits)

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
                vocab_origin="regex",
            )
            if role == ROLE_CONCEPT_SPECIFIC:
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
    # Recover vocab_origin / regex_name from originals
    by_key = {(h.term, h.start, h.end): h for h in hits if h.start >= 0}
    out = []
    for s in kept:
        orig = by_key.get((s.term, s.start, s.end))
        out.append(
            EvidenceHit(
                term=s.term,
                tier=s.tier,
                matched_form=s.matched_form,
                start=s.start,
                end=s.end,
                regex_name=(
                    orig.regex_name
                    if orig
                    else (s.term.replace("regex:", "") if s.term.startswith("regex:") else "")
                ),
                vocab_origin=orig.vocab_origin if orig else "",
            )
        )
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
        if h.tier in STRONG_TIERS or h.tier == "strong_paraphrase":
            # already counted via high path; still ok as supporting complement
            pass
        if _surface(h.term) in WEAK_POV_SURFACES:
            continue
        if _is_general_narrative(lang, h.term):
            continue
        if h.tier.startswith("regex:") and "context" in h.tier:
            continue
        if h.tier in ("demoted", "ambiguous_demoted", "context_only"):
            continue
        cues.append(h)
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
        if h.tier in ("demoted", "ambiguous_demoted", "context_only"):
            continue
        strong.append(h)
    return strong


def candidate_status(ev: ConceptEvidence, lang: str) -> tuple[bool, str]:
    """Return (qualifies, selection_rule).

    Demoted and context-only never independently qualify.
    """
    strong = strong_high_hits(ev)
    if strong:
        # Distinguish craft/paraphrase/academic for auditability
        if any(h.tier == "strong_craft" for h in strong):
            return True, "strong_craft"
        if any(h.tier == "strong_paraphrase" for h in strong):
            return True, "strong_paraphrase"
        return True, "strong_high_or_concept_regex"

    weak_high = [h for h in ev.high if h not in strong]
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
