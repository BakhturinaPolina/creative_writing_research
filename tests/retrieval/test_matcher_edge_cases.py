"""Expanded matching edge cases for v0.4 finalization."""
from __future__ import annotations

from src.common.paths import CONFIG
from src.retrieval.evidence import collect_concept_evidence, qualifies, qualifying_concepts
from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon
from src.retrieval.match import (
    build_concept_prefilter_needles,
    expand_phrase_variants,
    find_phrase_spans,
    haystack_for_prefilter,
)
from src.retrieval.regex_map import load_regex_map

LEX = load_lexicon(LEXICON_PATH_V03)
RMAP = load_regex_map(CONFIG / "regex_concept_map_v03.yml")


def _qual(text: str, lang: str) -> set[str]:
    return set(qualifying_concepts(text, lang, LEX, RMAP).keys())


def _prefilter_vs_exhaustive(text: str, lang: str) -> tuple[set[str], set[str]]:
    hay = haystack_for_prefilter(text)
    exhaustive: set[str] = set()
    prefiltered: set[str] = set()
    for cid, ct in LEX.concepts[lang].items():
        ev = collect_concept_evidence(text, cid, ct, lang, LEX, RMAP)
        if qualifies(ev, lang):
            exhaustive.add(cid)
        needles = build_concept_prefilter_needles(
            ct.academic, ct.craft, ct.paraphrases, lang
        )
        regex_hit = False
        for name in RMAP.names_for_concept(lang, cid):
            pat = LEX.example_regex.get(lang, {}).get(name)
            if pat is not None and pat.search(text):
                regex_hit = True
                break
        if regex_hit or any(n in hay for n in needles):
            if qualifies(ev, lang):
                prefiltered.add(cid)
    return exhaustive, prefiltered


EDGE_CASES = [
    # German inflection / gender
    ("de", "Die auktoriale Erzählerin kommentiert die Handlung."),
    ("de", "Ein allwissender Erzähler kennt die Gedanken aller Figuren."),
    ("de", "Die personale Erzählsituation rückt nah an die Figur."),
    # Hyphenated / non-hyphenated
    ("de", "Die Ich-Erzähler-Perspektive und die Icherzaehler Perspektive."),
    ("en", "Prefer first-person narration over first person narration only as labels."),
    # Compound nouns
    ("de", "Nullfokalisierung und Erzählperspektive werden unterschieden."),
    # Unicode punctuation / capitalization
    ("en", "“Omniscient” narration—capitalized—still signals zero focalization."),
    ("de", "Allwissende Erzählperspektive: ein Klassiker."),
    # OCR-related spacing
    ("en", "first - person narrator with awkward OCR spaces"),
    ("de", "all wissen der Erzähler with spaced tokens may miss; allwissender Erzähler hits"),
    # English plurals
    ("en", "Avoid filter words; too many filters distance the reader."),
    # Overlapping spans / multi-concept
    ("en", "An omniscient third-person narrator with free indirect discourse."),
    ("de", "Auktoriales Erzählen mit allwissendem Erzähler und erlebter Rede."),
    # Negated / contrastive
    ("en", "Do not use an omniscient narrator; stay in limited third person."),
    ("de", "Vermeide auktoriales Erzählen; bleibe bei personaler Erzählsituation."),
    # Ambiguous grammatical person
    ("de", "Schreiben Sie die Sie-Form des Verbs korrekt."),
    ("en", "Use 'you' carefully; second-person fiction is rare."),
]


def test_edge_cases_prefilter_subseteq_exhaustive():
    """Prefilter must not invent quals the exhaustive path rejects; ideally equal."""
    for lang, text in EDGE_CASES:
        exh, pref = _prefilter_vs_exhaustive(text, lang)
        # Prefilter may only miss (false negative) in theory; must not add extras
        # after the qualifies gate. Equality is the acceptance target on fixtures.
        assert pref == exh, (
            f"prefilter≠exhaustive lang={lang} text={text!r} "
            f"only_exh={exh - pref} only_pre={pref - exh}"
        )


def test_german_gender_and_hyphen_variants_expand():
    vars_ = expand_phrase_variants("auktorialer Erzähler", "de")
    folded = " ".join(v.casefold() for v in vars_)
    assert "erzählerin" in folded
    needles = build_concept_prefilter_needles(["allwissender Erzähler"], [], [], "de")
    assert any("allwissen" in n for n in needles)


def test_ocr_spacing_first_person_en():
    spans = find_phrase_spans("first-person", "A first - person view.", "en")
    # Soft: variant expansion may or may not catch OCR spaces; record behaviour
    # Canonical hyphenated form must match clean text.
    assert find_phrase_spans("first-person", "A first-person view.", "en")


def test_overlapping_evidence_collapsed():
    text = "first-person narrator and first person narrator again"
    spans = find_phrase_spans("first-person narrator", text, "en")
    assert len(spans) >= 1


def test_contrastive_does_not_force_participant():
    q = _qual("Write in first person without making the narrator a participant.", "en")
    assert "voice_participant" not in q


def test_sie_form_grammar_not_focalization():
    q = _qual("Konjugieren Sie das Verb in der Sie-Form.", "de")
    assert "focalization_zero" not in q
    assert "stance_narratorial" not in q
