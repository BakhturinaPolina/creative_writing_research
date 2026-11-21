"""Concept-specific bilingual regression tests for PsyPer lexical evidence.

≥60 synthetic cases covering all 36 concepts. Critical regressions:

1. First-person narrator ↛ voice_heterodiegetic
2. First-person narrator ↛ filter_words
3. Erzählperspektive alone ↛ every PsyPer concept
4. Economic point of view ↛ literary perspective without support
5. Overlapping variants ≠ two independent supporting cues
6. Demoted terms must not bypass candidate-selection rules
"""
from __future__ import annotations

import pytest

from src.retrieval.evidence import (
    candidate_status,
    collect_all_concepts,
    collect_concept_evidence,
    independent_supporting_count,
    qualifies,
    qualifying_concepts,
)
from src.retrieval.lexicon_v02 import load_lexicon
from src.retrieval.match import expand_phrase_variants, find_phrase_spans
from src.retrieval.queries import QueryConfig, build_concept_query
from src.retrieval.regex_map import load_regex_map

LEX = load_lexicon()
RMAP = load_regex_map()
ALL_CONCEPTS = list(LEX.concepts["en"].keys())
assert len(ALL_CONCEPTS) == 36


def _ev(text: str, lang: str, concept_id: str):
    return collect_concept_evidence(
        text, concept_id, LEX.concepts[lang][concept_id], lang, LEX, RMAP
    )


def _qualifying(text: str, lang: str) -> set[str]:
    return set(qualifying_concepts(text, lang, LEX, RMAP).keys())


# ---------------------------------------------------------------------------
# Critical regressions
# ---------------------------------------------------------------------------


def test_first_person_not_heterodiegetic_en():
    text = "Use a first-person narrator who recounts events after they happen."
    assert "voice_heterodiegetic" not in _qualifying(text, "en")
    assert qualifies(_ev(text, "en", "grammatical_first_person"), "en")


def test_first_person_not_filter_words_en():
    text = "A first-person narrator can feel intimate to the reader."
    assert "filter_words" not in _qualifying(text, "en")


def test_first_person_not_heterodiegetic_de():
    text = "Der Ich-Erzähler berichtet aus der Rückschau."
    q = _qualifying(text, "de")
    assert "voice_heterodiegetic" not in q
    assert "grammatical_first_person" in q or any(
        "ich" in t.casefold() for t in _ev(text, "de", "grammatical_first_person").matched_terms
        + _ev(text, "de", "grammatical_first_person").matched_regex
    ) or qualifies(_ev(text, "de", "grammatical_first_person"), "de")


def test_erzaehlperspektive_not_all_concepts():
    text = "Die Erzählperspektive ist ein zentrales Thema dieses Kapitels."
    q = _qualifying(text, "de")
    assert len(q) < 10, f"too many concepts from general POV: {q}"
    assert "filter_words" not in q
    assert "voice_heterodiegetic" not in q
    assert "reliability" not in q


def test_economic_pov_not_literary():
    text = "From an economic point of view, the contract is advantageous."
    assert _qualifying(text, "en") == set()


def test_financial_sicht_not_literary():
    text = "Aus finanzieller Sicht hat die Entscheidung Vorteile."
    assert _qualifying(text, "de") == set()


def test_overlapping_variants_one_cue():
    # Both hyphenated and spaced forms of the same expression
    text = "Choose a first-person narrator; the first person narrator sees events closely."
    # Force supporting-only path via a concept that uses craft paraphrases carefully:
    # Count independent supporting for a synthetic overlap on academic terms.
    spans_a = find_phrase_spans("first-person narrator", text, "en")
    spans_b = find_phrase_spans("first person narrator", text, "en")
    from src.retrieval.match import cluster_independent_cues

    merged = cluster_independent_cues(spans_a + spans_b)
    # Two separate occurrences in the sentence → up to 2 clusters, not 4
    assert len(merged) <= 2
    # Single occurrence with both variant needles on same span → 1
    text2 = "The first-person narrator is limited."
    merged2 = cluster_independent_cues(
        find_phrase_spans("first-person narrator", text2, "en")
        + find_phrase_spans("first person narrator", text2, "en")
    )
    assert len(merged2) == 1


def test_demoted_terms_do_not_bypass():
    # Bare "voice" / "perspective" must not qualify narrative_style_voice alone
    text = "The voice and perspective matter in branding."
    assert "narrative_style_voice" not in _qualifying(text, "en")
    assert "perspective_general" not in _qualifying(text, "en")
    assert "voice_general" not in _qualifying(text, "en")


def test_regex_not_applied_across_concepts():
    # Grammatical person regex must not high-qualify filter_words
    text = "Avoid filter words when writing in first-person."
    ev_fw = _ev(text, "en", "filter_words")
    assert not any(h.regex_name == "grammatical_person" for h in ev_fw.high)
    # first-person should map only to grammatical_first_person among person concepts
    ev_1 = _ev(text, "en", "grammatical_first_person")
    ev_2 = _ev(text, "en", "grammatical_second_person")
    assert any(h.regex_name == "grammatical_person" for h in ev_1.high + ev_1.regex)
    assert not any(h.regex_name == "grammatical_person" for h in ev_2.high + ev_2.regex)


# ---------------------------------------------------------------------------
# Positive concept coverage (EN + DE) — ensure all 36 appear at least once
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    # EN — academic high terms from the v0.2 lexicon
    ("en", "perspective_general", "Discuss narrative perspective before drafting."),
    ("en", "voice_general", "Narrative voice shapes reader trust."),
    ("en", "grammatical_first_person", "A first-person narrator recounts the events."),
    ("en", "grammatical_second_person", "Second-person narration addresses the reader as you."),
    ("en", "grammatical_third_person", "Third-person narration keeps some distance."),
    ("en", "voice_heterodiegetic", "A heterodiegetic narrator stands outside the story world."),
    ("en", "voice_homodiegetic", "The homodiegetic narrator belongs to the story world."),
    ("en", "voice_eyewitness", "An eyewitness narrator reports what they saw."),
    ("en", "voice_participant", "A participant narrator takes part in the action."),
    ("en", "voice_protagonist", "An autodiegetic narrator is also the protagonist."),
    ("en", "stance_narratorial", "Narratorial stance allows commentary on events."),
    ("en", "stance_figural", "Figural narration filters events through a reflector."),
    ("en", "focalization_general", "Focalization determines whose perception guides the scene."),
    ("en", "focalization_zero", "Zero focalization grants broad narrative knowledge."),
    ("en", "focalization_internal", "Internal focalization limits knowledge to one mind."),
    ("en", "focalization_external", "External focalization shows only observable behaviour."),
    ("en", "focalization_fixed", "Fixed focalization stays with one character throughout."),
    ("en", "focalization_variable", "Variable focalization shifts among several characters."),
    ("en", "pov_subject", "Treat the character as the perceiving subject of the scene."),
    ("en", "pov_object", "The secondary figure remains an object of perception only."),
    ("en", "mode_narrative", "Narrative summary compresses long stretches of time."),
    ("en", "mode_dramatic", "Dramatic mode prefers scenic showing."),
    ("en", "mode_balance", "On the narrative-dramatic continuum, alternate scene and summary."),
    ("en", "consciousness_access", "Access to consciousness lets readers hear private thoughts."),
    ("en", "thought_representation", "Free indirect discourse blends narrator and character thought."),
    ("en", "distance_depth", "Psychic distance can move from close to remote."),
    ("en", "switching_consistency", "Head hopping confuses the reader mid-scene."),
    ("en", "multiple_focalization_same_event", "Multiple focalization revisits the same event."),
    ("en", "information_control", "Information control decides what readers learn when."),
    ("en", "reliability", "An unreliable narrator withholds or distorts the truth."),
    ("en", "narrator_level_time", "An extradiegetic narrator frames the embedded tale."),
    ("en", "filter_words", "Perception verbs such as felt and noticed distance the reader."),
    ("en", "reader_effect", "Reader identification grows with shared perception."),
    ("en", "speech_dialogue", "Direct discourse and dialogue tags shape the scene."),
    ("en", "narrative_style_voice", "Narratorial style creates coherence across chapters."),
    ("en", "choice_tradeoffs", "Narrative strategy involves clear perspectival tradeoffs."),
    # DE
    ("de", "perspective_general", "Die Erzählperspektive steuert die Wahrnehmung."),
    ("de", "voice_general", "Die Erzählstimme prägt den Ton der Darstellung."),
    ("de", "grammatical_first_person", "Der Ich-Erzähler berichtet aus eigener Sicht."),
    ("de", "grammatical_second_person", "Die Du-Perspektive spricht die Lesenden direkt an."),
    ("de", "grammatical_third_person", "Die Er-Erzählung hält Distanz zur Figur."),
    ("de", "voice_heterodiegetic", "Ein heterodiegetischer Erzähler steht außerhalb der erzählten Welt."),
    ("de", "voice_homodiegetic", "Der homodiegetische Erzähler gehört zur erzählten Welt."),
    ("de", "voice_eyewitness", "Der Zeugenerzähler schildert als Beobachter."),
    ("de", "voice_participant", "Ein teilnehmender Erzähler wirkt am Geschehen mit."),
    ("de", "voice_protagonist", "Ein autodiegetischer Erzähler ist zugleich Protagonist."),
    ("de", "stance_narratorial", "Narratoriale Erzählhaltung erlaubt Kommentare."),
    ("de", "stance_figural", "Figurale Erzählhaltung filtert durch eine Reflektorfigur."),
    ("de", "focalization_general", "Fokalisierung legt fest, wessen Wahrnehmung gilt."),
    ("de", "focalization_zero", "Nullfokalisierung entspricht auktorialer Allwissenheit."),
    ("de", "focalization_internal", "Interne Fokalisierung bleibt im Bewusstsein einer Figur."),
    ("de", "focalization_external", "Externe Fokalisierung zeigt nur äußeres Verhalten."),
    ("de", "focalization_fixed", "Feste Fokalisierung bleibt bei einer Figur."),
    ("de", "focalization_variable", "Bei variabler Fokalisierung wechseln die Wissenshorizonte."),
    ("de", "pov_subject", "Das wahrnehmendes Subjekt strukturiert die Szene."),
    ("de", "pov_object", "Die Figur bleibt bloßes Wahrnehmungsobjekt."),
    ("de", "mode_narrative", "Der narrativer Modus verdichtet lange Handlungsstrecken."),
    ("de", "mode_dramatic", "Der dramatischer Modus bevorzugt szenisches Erzählen."),
    ("de", "mode_balance", "Das Verhältnis von Bericht und Szene muss stimmen."),
    ("de", "consciousness_access", "Zugang zum Bewusstsein gewährt Einblick in Gedanken."),
    ("de", "thought_representation", "Erlebte Rede mischt Erzähler- und Figurenrede."),
    ("de", "distance_depth", "Die narrative Distanz kann nah oder fern gewählt werden."),
    ("de", "switching_consistency", "Ein Perspektivenwechsel mitten in der Szene irritiert."),
    ("de", "multiple_focalization_same_event", "Mehrfache Fokalisierung zeigt dasselbe Ereignis erneut."),
    ("de", "information_control", "Informationssteuerung regelt, was Leser wissen."),
    ("de", "reliability", "Ein unzuverlässiger Erzähler verzerrt die Wahrheit."),
    ("de", "narrator_level_time", "Ein extradiegetischer Erzähler rahmt die Binnenerzählung."),
    ("de", "filter_words", "Wahrnehmungsverben wie fühlte schaffen Distanz."),
    ("de", "reader_effect", "Leseridentifikation wächst mit geteilter Wahrnehmung."),
    ("de", "speech_dialogue", "Direkte Rede und Dialoggestaltung prägen die Szene."),
    ("de", "narrative_style_voice", "Der Erzählstil schafft Kohärenz."),
    ("de", "choice_tradeoffs", "Die Erzählstrategie birgt klare Vor- und Nachteile."),
]


@pytest.mark.parametrize("lang,concept_id,text", POSITIVE_CASES)
def test_positive_concept_qualifies(lang, concept_id, text):
    ev = _ev(text, lang, concept_id)
    ok, rule = candidate_status(ev, lang)
    assert ok, f"{lang}/{concept_id} should qualify ({rule}); tiers={ev.evidence_tiers_summary()} terms={ev.matched_terms} regex={ev.matched_regex}"


def test_all_36_concepts_have_positive_case():
    covered = {c for _, c, _ in POSITIVE_CASES}
    missing = set(ALL_CONCEPTS) - covered
    assert not missing, f"missing positive cases for {missing}"


# ---------------------------------------------------------------------------
# Negative / ambiguous / everyday
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    ("en", "The marketing voice of the brand should stay consistent."),
    ("en", "From a legal point of view the clause is void."),
    ("en", "She has a unique voice as a singer."),
    ("en", "Character development takes time in any story."),
    ("de", "Die Perspektive der Anleger ist entscheidend."),
    ("de", "Seine Stimme klang heiser."),
    ("de", "Die Figur auf dem Schachbrett ist ein Springer."),
    ("de", "Aus Sicht der Bank ist das Risiko hoch."),
]


@pytest.mark.parametrize("lang,text", NEGATIVE_CASES)
def test_everyday_non_literary_negative(lang, text):
    q = _qualifying(text, lang)
    # Allow empty or at most weak general hits that still fail candidate rule
    assert q == set(), f"unexpected concepts {q} for everyday text: {text}"


# ---------------------------------------------------------------------------
# Ambiguous / person / diegesis / focalization contrasts
# ---------------------------------------------------------------------------


def test_third_person_not_internal_focalization_alone():
    text = "Write in third-person and keep sentences short."
    q = _qualifying(text, "en")
    assert "focalization_internal" not in q
    assert "consciousness_access" not in q


def test_heterodiegetic_positive_not_homodiegetic():
    text = "A heterodiegetic narrator stands outside the story world entirely."
    q = _qualifying(text, "en")
    assert "voice_heterodiegetic" in q
    assert "voice_homodiegetic" not in q


def test_internal_vs_external_de():
    text = "Interne Fokalisierung bleibt im Bewusstsein; externe Fokalisierung zeigt nur Verhalten."
    q = _qualifying(text, "de")
    assert "focalization_internal" in q
    assert "focalization_external" in q


def test_german_inflection_ich_erzaehlerin():
    text = "Die Ich-Erzählerin kommentiert die Ereignisse ironisch."
    assert qualifies(_ev(text, "de", "grammatical_first_person"), "de")


def test_german_compound_icherzaehler():
    spans = find_phrase_spans("Ich-Erzähler", "Der Icherzähler bleibt nah.", "de")
    assert spans, "Icherzähler should match Ich-Erzähler variants"


def test_auktorial_variants():
    text = "Auktoriales Erzählen und der auktoriale Erzähler erlauben Überblick."
    assert qualifies(_ev(text, "de", "stance_narratorial"), "de") or qualifies(
        _ev(text, "de", "focalization_zero"), "de"
    )


def test_hyphen_space_first_person():
    assert find_phrase_spans("first-person narrator", "the first person narrator spoke", "en")
    assert find_phrase_spans("first person narrator", "the first-person narrator spoke", "en")


def test_missing_concept_does_not_crash():
    # Empty craft/academic still returns structured evidence
    ev = _ev("Completely unrelated gardening tips.", "en", "filter_words")
    assert not qualifies(ev, "en")


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_query_dedupes_repeated_tokens():
    q = build_concept_query(
        "x",
        academic=["first-person narrator", "first person narrator"],
        craft=["narrator"],
        paraphrases=["first-person narrator"],
        lang="en",
        cfg=QueryConfig(dedupe_tokens=True, allow_repeated_token_weighting=False),
    )
    assert len(q.expressions) == 3  # hyphenated + spaced + craft singleton
    # 'first' and 'person' and 'narrator' appear once each
    assert q.tokens.count("first") <= 1
    assert q.tokens.count("person") <= 1
    assert q.tokens.count("narrator") <= 1


def test_query_drops_stopwords():
    q = build_concept_query(
        "x",
        academic=["point of view"],
        craft=[],
        paraphrases=[],
        lang="en",
        cfg=QueryConfig(drop_stopwords=True),
    )
    assert "of" not in q.tokens
    assert "point" in q.tokens
    assert "view" in q.tokens


def test_expand_german_gender_variants():
    vars_ = expand_phrase_variants("Ich-Erzähler", "de")
    folded = {v.casefold() for v in vars_}
    assert any("erzählerin" in v for v in folded)


# ---------------------------------------------------------------------------
# Regex map integrity
# ---------------------------------------------------------------------------


def test_regex_map_covers_lexicon_patterns():
    for lang in ("en", "de"):
        for name in LEX.example_regex[lang]:
            assert RMAP.get(lang, name) is not None, f"missing map for {lang}/{name}"


def test_suite_case_count():
    # POSITIVE (72) + NEGATIVE (8) + named tests ≈ well over 60
    n = len(POSITIVE_CASES) + len(NEGATIVE_CASES)
    assert n >= 60
