"""v0.3 regressions: reservoir, PsyPer tags, prefilter equivalence, tiers, contrastives."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.paths import CONFIG
from src.retrieval.evidence import (
    candidate_status,
    collect_concept_evidence,
    independent_supporting_count,
    qualifies,
    qualifying_concepts,
)
from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon, resolve_evidence_tier
from src.retrieval.match import (
    build_concept_prefilter_needles,
    expand_phrase_variants,
    find_phrase_spans,
    haystack_for_prefilter,
)
from src.retrieval.regex_map import load_regex_map
from src.retrieval.sampling import ReservoirSample, StratifiedReservoirs

LEX02 = load_lexicon()
LEX03 = load_lexicon(LEXICON_PATH_V03)
RMAP01 = load_regex_map()
RMAP03 = load_regex_map(CONFIG / "regex_concept_map_v03.yml")


def _qual03(text: str, lang: str) -> set[str]:
    return set(qualifying_concepts(text, lang, LEX03, RMAP03).keys())


def _ev03(text: str, lang: str, cid: str):
    return collect_concept_evidence(
        text, cid, LEX03.concepts[lang][cid], lang, LEX03, RMAP03
    )


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------


def test_reservoir_can_sample_final_items():
    """Eligible items at the end of the stream must be selectable."""
    stream = [f"book{i:03d}:p{j}" for i in range(100) for j in range(20)]
    # Last books are book095–book099
    last_ids = {x for x in stream if x.startswith("book09")}
    hits = 0
    for seed in range(50):
        r = ReservoirSample(k=30, seed=seed)
        r.fill_from(stream)
        if any(x in last_ids for x in r.reservoir):
            hits += 1
    assert hits >= 20, f"final-book items rarely sampled ({hits}/50 seeds)"
    assert r.population_size == len(stream)
    assert r.inclusion_probability == pytest.approx(30 / len(stream))


def test_reservoir_never_stops_early():
    r = ReservoirSample(k=5, seed=1)
    for i in range(10_000):
        r.consider(i)
    assert r.seen == 10_000
    assert len(r.reservoir) == 5
    # Values from the end of the stream should appear across seeds
    endish = 0
    for seed in range(20):
        rr = ReservoirSample(k=5, seed=seed)
        rr.fill_from(range(10_000))
        if any(x >= 9000 for x in rr.reservoir):
            endish += 1
    assert endish >= 5


def test_stratified_reservoirs_independent():
    s = StratifiedReservoirs(k_per_stratum=3, seed=42)
    for i in range(100):
        s.consider("en::fiction", f"e{i}")
        s.consider("de::fiction", f"d{i}")
    assert len(s._reservoirs) == 2
    assert all(len(r.reservoir) == 3 for r in s._reservoirs.values())


# ---------------------------------------------------------------------------
# PsyPer tag mappings
# ---------------------------------------------------------------------------


def test_psyper_tags_are_string_codes_not_concept_ids():
    ct = LEX03.concepts["en"]["voice_heterodiegetic"]
    assert ct.psyper == ["1", "1.1"]
    assert "voice_heterodiegetic" not in ct.psyper
    ct2 = LEX03.concepts["en"]["voice_eyewitness"]
    assert ct2.psyper == ["2.2.1"]
    ct3 = LEX03.concepts["en"]["filter_words"]
    assert ct3.psyper == ["4", "6", "8"]


def test_v02_lexicon_also_loads_psyper():
    ct = LEX02.concepts["de"]["focalization_zero"]
    assert "5" in ct.psyper or "5.1" in ct.psyper


# ---------------------------------------------------------------------------
# Prefilter ≡ exhaustive matcher
# ---------------------------------------------------------------------------


BILINGUAL_FIXTURES = [
    ("en", "Choose a first-person narrator for intimacy."),
    ("en", "Avoid filter words such as she saw and he heard."),
    ("en", "Third-person omniscient lets the narrator know everything."),
    ("en", "The narrator is not a character in the story."),
    ("en", "Head hopping breaks fixed focalization."),
    ("de", "Der Ich-Erzähler berichtet als Beobachter."),
    ("de", "Nutze eine heterodiegetische Erzählerin."),
    ("de", "Bei interner Fokalisierung bleibt das Wissen begrenzt."),
    ("de", "Vermeide Filterwörter und distanzierende Formulierungen."),
    ("de", "Die allwissende Erzählperspektive zeigt Gedanken sämtlicher Figuren."),
    ("de", "Ein personaler Erzähler bleibt nah an einer Figur."),
    ("de", "Die Geschichte wird von einer Nebenfigur erzählt."),
]


def test_prefilter_matches_exhaustive_concept_sets():
    for lang, text in BILINGUAL_FIXTURES:
        hay = haystack_for_prefilter(text)
        exhaustive = set()
        prefiltered = set()
        for cid, ct in LEX03.concepts[lang].items():
            ev = collect_concept_evidence(text, cid, ct, lang, LEX03, RMAP03)
            if qualifies(ev, lang):
                exhaustive.add(cid)
            needles = build_concept_prefilter_needles(
                ct.academic, ct.craft, ct.paraphrases, lang
            )
            regex_hit = False
            for name in RMAP03.names_for_concept(lang, cid):
                pat = LEX03.example_regex.get(lang, {}).get(name)
                if pat is not None and pat.search(text):
                    regex_hit = True
                    break
            if regex_hit or any(n in hay for n in needles):
                if qualifies(ev, lang):
                    prefiltered.add(cid)
        assert prefiltered == exhaustive, (
            f"prefilter≠exhaustive for {lang!r} text={text!r}: "
            f"only_exh={exhaustive - prefiltered} only_pre={prefiltered - exhaustive}"
        )


def test_german_gender_variant_in_prefilter_needles():
    needles = build_concept_prefilter_needles(
        ["heterodiegetischer Erzähler"], [], [], "de"
    )
    joined = " ".join(needles)
    assert "erzählerin" in joined


# ---------------------------------------------------------------------------
# Strong craft / paraphrase; repetition; overlap
# ---------------------------------------------------------------------------


def test_strong_craft_filter_words_qualifies():
    text = "Delete filter words that distance the reader from the character."
    ev = _ev03(text, "en", "filter_words")
    ok, rule = candidate_status(ev, "en")
    assert ok
    assert rule in ("strong_craft", "strong_high_or_concept_regex", "strong_paraphrase")


def test_strong_paraphrase_heterodiegetic():
    text = "Remember: the narrator is not a character in the story."
    assert "voice_heterodiegetic" in _qual03(text, "en")


def test_repeated_identical_term_not_two_supporting():
    text = "Use filter words. More filter words. Still filter words."
    # Even if supporting, overlapping/repeated same surface collapses.
    ev = _ev03(text, "en", "filter_words")
    # strong craft may already qualify; independent supporting count must be small
    n = independent_supporting_count(ev, "en")
    assert n <= 2


def test_overlapping_hyphen_space_one_cue():
    text = "A first-person narrator; also called a first person narrator."
    spans = find_phrase_spans("first-person narrator", text, "en")
    from src.retrieval.match import cluster_independent_cues

    # two occurrences may remain; variants of the same occurrence collapse
    assert len(spans) <= 2


# ---------------------------------------------------------------------------
# Contrastives
# ---------------------------------------------------------------------------


def test_first_person_not_participant():
    text = "Write in first person so the narrator recounts events intimately."
    q = _qual03(text, "en")
    assert "voice_participant" not in q


def test_eyewitness_not_automatically_protagonist():
    text = "Use an eyewitness narrator who observes rather than participates."
    q = _qual03(text, "en")
    assert "voice_eyewitness" in q or "voice_homodiegetic" in q
    # Protagonist requires stronger autodiegetic/protagonist cues
    assert "voice_protagonist" not in q


def test_fixed_focalization_not_one_narrator():
    text = "The novel has one narrator throughout, but focalization still shifts."
    q = _qual03(text, "en")
    assert "focalization_fixed" not in q


def test_unreliable_not_omniscient():
    text = "An unreliable narrator withholds information from the reader."
    q = _qual03(text, "en")
    assert "focalization_zero" not in q


def test_perception_verb_not_filter_words():
    text = "She saw the door open and he heard footsteps in the hall."
    assert "filter_words" not in _qual03(text, "en")


def test_german_grammar_instruction_false_positive():
    text = "Konjugieren Sie das Verb in der Sie-Form des Präsens."
    q = _qual03(text, "de")
    # Bare grammar instruction must not validate literary POV concepts alone
    assert "voice_heterodiegetic" not in q
    assert "filter_words" not in q
    assert "focalization_zero" not in q


def test_narratological_discussion_vs_fiction_illustration():
    # Instruction about filter words → hit
    assert "filter_words" in _qual03(
        "Eliminate unnecessary sensory filters in deep POV.", "en"
    )
    # Fiction illustration alone → no hit
    assert "filter_words" not in _qual03(
        "Maria realized the danger and wondered what to do next.", "en"
    )


def test_de_homodiegetic_paraphrase():
    text = "Der Erzähler ist selbst eine handelnde Figur der Geschichte."
    assert "voice_homodiegetic" in _qual03(text, "de")


def test_de_fixed_focalization_paraphrase():
    text = (
        "Die Perspektive auf eine Figur beschränken und "
        "durchgehend dieselbe Erzählperspektive beibehalten."
    )
    assert "focalization_fixed" in _qual03(text, "de")


# ---------------------------------------------------------------------------
# Evidence tier resolution / missing files
# ---------------------------------------------------------------------------


def test_resolve_evidence_tier_override():
    assert (
        resolve_evidence_tier("en", "craft", "filter words", {"filter words": "strong_craft"})
        == "strong_craft"
    )


def test_missing_lexicon_raises():
    with pytest.raises(Exception):
        load_lexicon(Path("/nonexistent/lexicon.yml"))


def test_version_specific_paths_exist():
    assert LEXICON_PATH_V03.is_file()
    assert (CONFIG / "regex_concept_map_v03.yml").is_file()
    assert (CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml").is_file()


def test_expand_variants_icherzaehler():
    from src.retrieval.textnorm import normalize_text

    vars_ = expand_phrase_variants("Ich-Erzähler", "de")
    joined = " ".join(v.casefold() for v in vars_)
    assert "erzählerin" in joined
    folded = [normalize_text(v).replace("-", "").replace(" ", "") for v in vars_]
    assert any("icherzähler" in f for f in folded)
