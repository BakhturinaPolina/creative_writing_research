"""Integrity tests for narrative-perspective analysis v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.analysis_v2 import (
    assert_integrity,
    calculate_dispersion_all,
    calculate_family_frequencies_v2,
    calculate_frequencies_v2,
    category_and_standardized_frequencies,
    compare_methods_v2,
    cooccurrence_v2,
    eligible_word_denominators,
    frequency_ratio,
    language_denominators,
)
from src.analysis.inventory import file_sha256
from src.analysis.narrative_perspective import frequency_ratio as fr_legacy
from src.analysis.narrative_perspective import load_concept_ids
from src.common.paths import DATA_PROCESSED, ROOT


def _tiny_corpus() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = pd.DataFrame(
        {
            "book_id": ["en_a", "en_b", "de_a", "de_b"],
            "language": ["en", "en", "de", "de"],
            "in_analysis_set": ["true", "true", "true", "true"],
            "number_of_words": [10000, 20000, 8000, 12000],
            "number_of_eligible_paragraphs": [10, 20, 8, 12],
            "primary_group": [
                "fiction_manual",
                "contrast",
                "fiction_manual",
                "contrast",
            ],
            "qc_notes": ["", "", "", ""],
            "publication_year": ["2000", "2010", "", "1990"],
            "author": ["A", "B", "C", "D"],
            "title": ["t1", "t2", "t3", "t4"],
        }
    )
    passages = pd.DataFrame(
        {
            "passage_id": [
                "en_a:c0:p0",
                "en_a:c0:p1",
                "en_b:c0:p0",
                "de_a:c0:p0",
                "de_b:c0:p0",
            ],
            "book_id": ["en_a", "en_a", "en_b", "de_a", "de_b"],
            "language": ["en", "en", "en", "de", "de"],
            "word_count": [4000, 5000, 18000, 7000, 11000],
            "character_count": [20000, 25000, 90000, 35000, 55000],
            "segmentation_status": ["ok"] * 5,
            "qc_flags": ["", "", "", "", ""],
            "primary_group": [
                "fiction_manual",
                "fiction_manual",
                "contrast",
                "fiction_manual",
                "contrast",
            ],
        }
    )
    lexical = pd.DataFrame(
        {
            "passage_id": ["en_a:c0:p0", "en_a:c0:p0", "en_b:c0:p0", "de_a:c0:p0"],
            "book_id": ["en_a", "en_a", "en_b", "de_a"],
            "language": ["en", "en", "en", "de"],
            "retrieval_concept_id": [
                "filter_words",
                "stance_narratorial",
                "filter_words",
                "filter_words",
            ],
            "method": ["lexical_v03"] * 4,
            "matched_terms": ["filter", "narrator", "filter", "filter"],
            "matched_regex": ["", "", "", ""],
            "matched_spans": [
                '[{"start":0,"end":6}]',
                '[{"start":10,"end":18}]',
                "[]",
                "[]",
            ],
            "evidence_tiers": [
                '{"academic_high":1}',
                '{"strong_craft":1}',
                '{"context_only":1}',
                '{"academic_high":1}',
            ],
        }
    )
    return corpus, passages, lexical


def test_file_sha256_stable():
    path = DATA_PROCESSED / "corpus_quality_v1_candidate.csv"
    assert path.is_file()
    a = file_sha256(path)
    b = file_sha256(path)
    assert a == b
    assert len(a) == 64


def test_frequency_ratio_zero_denominator():
    assert frequency_ratio(0, 0) is None
    assert frequency_ratio(1.5, 0) == float("inf")
    assert frequency_ratio(4, 2) == 2.0
    assert fr_legacy(3, 0) == float("inf")


def test_eligible_denominators_and_zeros():
    corpus, passages, lexical = _tiny_corpus()
    books = eligible_word_denominators(corpus, passages)
    den = language_denominators(books)
    assert set(den["language"]) == {"en", "de"}
    en_elig = float(den.loc[den.language == "en", "eligible_words"].iloc[0])
    assert en_elig == 4000 + 5000 + 18000
    concepts = ["filter_words", "stance_narratorial", "reader_effect"]
    # Patch load by passing concept list
    freq = calculate_frequencies_v2(lexical, books, concepts)
    assert len(freq) == 6  # 3 × 2
    # Zero concept present
    zero = freq[(freq.concept_id == "reader_effect") & (freq.language == "en")]
    assert int(zero["n_assignments"].iloc[0]) == 0
    assert int(zero["n_analysis_books"].iloc[0]) == 2


def test_family_dedup():
    corpus, passages, lexical = _tiny_corpus()
    books = eligible_word_denominators(corpus, passages)
    # Restrict to one EN passage with two consciousness-family concepts.
    lex2 = pd.DataFrame(
        {
            "passage_id": ["en_a:c0:p0", "en_a:c0:p0"],
            "book_id": ["en_a", "en_a"],
            "language": ["en", "en"],
            "retrieval_concept_id": ["filter_words", "thought_representation"],
        }
    )
    fam = calculate_family_frequencies_v2(lex2, books)
    row = fam[
        (fam.concept_family == "consciousness_and_techniques") & (fam.language == "en")
    ]
    assert int(row["n_unique_passages"].iloc[0]) == 1
    assert int(row["n_assignments"].iloc[0]) == 2
    assert "deduplicated" in row["note"].iloc[0].lower()


def test_standardization_hand_computed():
    corpus, passages, lexical = _tiny_corpus()
    books = eligible_word_denominators(corpus, passages)
    concepts = ["filter_words"]
    composition, weights, std, cat = category_and_standardized_frequencies(
        lexical, books, concepts
    )
    # EN fiction: 9000 words, 1 assign → 1/9000*10000
    # EN contrast: 18000 words, 1 assign → 1/18000*10000
    en_fiction = cat[
        (cat.language == "en")
        & (cat.primary_group == "fiction_manual")
        & (cat.concept_id == "filter_words")
    ].iloc[0]
    assert abs(en_fiction.frequency_per_10000_eligible_words - (1 / 9000 * 10000)) < 1e-9
    # Weights are mean of EN/DE shares.
    assert abs(float(weights["common_weight"].sum()) - 1.0) < 1e-9
    assert len(std) == 1


def test_dispersion_includes_zero_books():
    corpus, passages, lexical = _tiny_corpus()
    books = eligible_word_denominators(corpus, passages)
    disp, book_level = calculate_dispersion_all(lexical, books, ["filter_words"])
    de = disp[(disp.concept_id == "filter_words") & (disp.language == "de")].iloc[0]
    # de_b has zero matches → pct_books_zero > 0
    assert de.pct_books_zero == 50.0
    assert de.n_books_with_concept == 1
    assert len(book_level[book_level.language == "de"]) == 2


def test_cooccurrence_jaccard():
    corpus, passages, lexical = _tiny_corpus()
    # Force two concepts on same passage with min_support 1 via direct call internals
    lex = lexical.copy()
    # Only EN passage with two concepts
    mat, pairs = cooccurrence_v2(lex, "en", min_support=1)
    assert mat.loc["filter_words", "stance_narratorial"] == 1
    pair = pairs[
        ((pairs.concept_a == "filter_words") & (pairs.concept_b == "stance_narratorial"))
        | ((pairs.concept_b == "filter_words") & (pairs.concept_a == "stance_narratorial"))
    ]
    assert len(pair) == 1
    assert abs(float(pair.iloc[0].jaccard) - 0.5) < 1e-6  # 1/(2+1-1)=0.5


def test_set_algebra_l_bq_bnew():
    lexical = pd.DataFrame(
        {
            "language": ["en", "en"],
            "passage_id": ["p1", "p2"],
            "retrieval_concept_id": ["c1", "c1"],
        }
    )
    bnew = pd.DataFrame(
        {
            "language": ["en", "en"],
            "passage_id": ["p2", "p3"],
            "retrieval_concept_id": ["c2", "c1"],
        }
    )
    bq = pd.DataFrame(
        {
            "language": ["en"],
            "passage_id": ["p1"],
            "retrieval_concept_id": ["c1"],
        }
    )
    # Add empty DE
    for df in (lexical, bnew, bq):
        pass
    lexical = pd.concat(
        [lexical, lexical.assign(language="de", passage_id="d1")], ignore_index=True
    )
    bnew = pd.concat(
        [bnew, bnew.iloc[0:0]], ignore_index=True
    )
    bq = pd.concat([bq, bq.iloc[0:0]], ignore_index=True)
    out = compare_methods_v2(lexical, bnew, bq)
    en = out[out.language == "en"].iloc[0]
    assert en.n_L_passages == 2
    assert en.n_genuinely_new_bm25_passages == 1  # p3
    assert en.n_known_passages_with_new_bm25_concepts == 1  # p2
    assert bool(en.braw_available) is False


def test_assert_integrity_rejects_bad_freq():
    corpus, passages, lexical = _tiny_corpus()
    books = eligible_word_denominators(corpus, passages)
    concepts = load_concept_ids()[:3]
    freq = calculate_frequencies_v2(lexical, books, concepts)
    family = calculate_family_frequencies_v2(lexical, books)
    # Drop a row to break expected length
    bad = freq.iloc[:-1]
    with pytest.raises(AssertionError):
        assert_integrity(corpus, passages, lexical, concepts, bad, family)


def test_inventory_hash_agreement_file_exists():
    inv = DATA_PROCESSED / "input_inventory_v2.json"
    assert inv.is_file()
    data = json.loads(inv.read_text(encoding="utf-8"))
    assert data["book_stats_hash_agreement"]["en"]["matches_lexical_v03"] is True
    assert data["book_stats_hash_agreement"]["de"]["matches_snapshot"] is True
