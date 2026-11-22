"""Regression tests for mutually exclusive v0.5 evaluation sampling."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.common.paths import ROOT
from src.retrieval.eval_sample import (
    PRIMARY_STRATA_V05,
    _agg_retrieval_membership,
    _srs_without_replacement,
    assert_strata_pairwise_exclusive,
    assign_primary_stratum,
    redistribute_quotas_v05,
)
from src.retrieval.sampling import derive_stratum_seed


def test_multi_rank_passage_assigned_to_high_only():
    """Passage ranked highly for one concept and poorly for another → high only."""
    s = assign_primary_stratum(
        in_lexical=True,
        in_qualified_bm25=True,
        in_bm25_expansion=False,
        best_bm25_rank=34,  # also has rank 257; best is 34
    )
    assert s == "high_ranked_qualified_bm25"
    # Aggregation keeps min rank and exclusive stratum.
    lexical = pd.DataFrame(
        {
            "passage_id": ["p1", "p1"],
            "book_id": ["b1", "b1"],
            "language": ["en", "en"],
            "primary_group": ["fiction_manual", "fiction_manual"],
            "concept_id": ["c_a", "c_b"],
            "retrieval_concept_id": ["c_a", "c_b"],
            "rank": ["", ""],
            "title": ["t", "t"],
            "author": ["a", "a"],
            "edition_year": ["", ""],
            "corpus_role": ["primary", "primary"],
            "chapter_idx": ["0", "0"],
            "para_idx": ["1", "1"],
        }
    )
    qualified = pd.DataFrame(
        {
            "passage_id": ["p1", "p1"],
            "book_id": ["b1", "b1"],
            "language": ["en", "en"],
            "primary_group": ["fiction_manual", "fiction_manual"],
            "concept_id": ["c_a", "c_b"],
            "retrieval_concept_id": ["c_a", "c_b"],
            "rank": [34, 257],
            "title": ["t", "t"],
            "author": ["a", "a"],
            "edition_year": ["", ""],
            "corpus_role": ["primary", "primary"],
            "chapter_idx": ["0", "0"],
            "para_idx": ["1", "1"],
        }
    )
    out = _agg_retrieval_membership(lexical, qualified, pd.DataFrame())
    assert len(out) == 1
    assert out.iloc[0]["best_bm25_rank"] == 34
    assert out.iloc[0]["primary_stratum"] == "high_ranked_qualified_bm25"


def test_lexical_and_bm25_expansion_known():
    s = assign_primary_stratum(
        in_lexical=True,
        in_qualified_bm25=False,
        in_bm25_expansion=True,
        best_bm25_rank=None,
    )
    assert s == "bm25_expansion_known"


def test_purely_lexical_passage():
    s = assign_primary_stratum(
        in_lexical=True,
        in_qualified_bm25=False,
        in_bm25_expansion=False,
        best_bm25_rank=None,
    )
    assert s == "lexical_only"


def test_genuinely_new_bm25_passage():
    s = assign_primary_stratum(
        in_lexical=False,
        in_qualified_bm25=False,
        in_bm25_expansion=True,
        best_bm25_rank=12,
    )
    assert s == "new_bm25_passages"


def test_non_retrieved_passage():
    s = assign_primary_stratum(
        in_lexical=False,
        in_qualified_bm25=False,
        in_bm25_expansion=False,
        best_bm25_rank=None,
    )
    assert s == "non_retrieved"


def test_lower_ranked_qualified():
    s = assign_primary_stratum(
        in_lexical=True,
        in_qualified_bm25=True,
        in_bm25_expansion=True,
        best_bm25_rank=80,
    )
    assert s == "lower_ranked_qualified_bm25"


def test_small_stratum_census_pi_equals_one():
    df = pd.DataFrame({"passage_id": ["a", "b", "c"], "x": [1, 2, 3]})
    sampled, meta = _srs_without_replacement(df, n=10, seed=123)
    assert meta["N_h"] == 3
    assert meta["n_h"] == 3
    assert meta["inclusion_probability"] == pytest.approx(1.0)
    assert meta["sampling_weight"] == pytest.approx(1.0)
    assert set(sampled["passage_id"]) == {"a", "b", "c"}


def test_empty_stratum():
    df = pd.DataFrame(columns=["passage_id"])
    sampled, meta = _srs_without_replacement(df, n=5, seed=1)
    assert len(sampled) == 0
    assert meta["N_h"] == 0
    assert meta["n_h"] == 0
    assert meta["inclusion_probability"] is None


def test_quota_redistribution_from_empty_stratum():
    pops = {
        "high_ranked_qualified_bm25": 0,
        "lower_ranked_qualified_bm25": 100,
        "lexical_only": 50,
        "bm25_expansion_known": 10,
        "new_bm25_passages": 40,
        "non_retrieved": 1000,
    }
    targets = {
        "high_ranked_qualified_bm25": 20,
        "lower_ranked_qualified_bm25": 20,
        "lexical_only": 20,
        "bm25_expansion_known": 15,
        "new_bm25_passages": 20,
        "non_retrieved": 30,
    }
    q = redistribute_quotas_v05(pops, targets)
    assert q["high_ranked_qualified_bm25"] == 0
    # Unused 20 from high redistributed to later strata with capacity.
    assert sum(q.values()) == sum(targets.values()) - 0  # all unused placed
    assert sum(q.values()) == sum(targets.values())
    assert q["lower_ranked_qualified_bm25"] >= 20


def test_pairwise_exclusive_populations():
    ids = {
        "high_ranked_qualified_bm25": {"p1", "p2"},
        "lower_ranked_qualified_bm25": {"p3"},
        "lexical_only": {"p4"},
        "bm25_expansion_known": {"p5"},
        "new_bm25_passages": {"p6"},
        "non_retrieved": {"p7", "p8"},
    }
    assert_strata_pairwise_exclusive(ids)
    with pytest.raises(AssertionError, match="overlap"):
        assert_strata_pairwise_exclusive(
            {
                "high_ranked_qualified_bm25": {"p1", "p2"},
                "lexical_only": {"p2", "p3"},
            }
        )


def test_synthetic_population_fully_reconciled():
    """Synthetic L/Bq/Bnew mix → exclusive strata covering all passages."""
    lexical = pd.DataFrame(
        {
            "passage_id": ["high1", "low1", "lex1", "exp1"],
            "book_id": ["b"] * 4,
            "language": ["en"] * 4,
            "primary_group": ["fiction_manual"] * 4,
            "concept_id": ["c1", "c2", "c3", "c4"],
            "retrieval_concept_id": ["c1", "c2", "c3", "c4"],
            "rank": [""] * 4,
            "title": ["t"] * 4,
            "author": ["a"] * 4,
            "edition_year": [""] * 4,
            "corpus_role": ["primary"] * 4,
            "chapter_idx": ["0"] * 4,
            "para_idx": ["0", "1", "2", "3"],
        }
    )
    qualified = pd.DataFrame(
        {
            "passage_id": ["high1", "high1", "low1"],
            "book_id": ["b"] * 3,
            "language": ["en"] * 3,
            "primary_group": ["fiction_manual"] * 3,
            "concept_id": ["c1", "c9", "c2"],
            "retrieval_concept_id": ["c1", "c9", "c2"],
            "rank": [10, 200, 90],
            "title": ["t"] * 3,
            "author": ["a"] * 3,
            "edition_year": [""] * 3,
            "corpus_role": ["primary"] * 3,
            "chapter_idx": ["0"] * 3,
            "para_idx": ["0", "0", "1"],
        }
    )
    expansion = pd.DataFrame(
        {
            "passage_id": ["exp1", "new1"],
            "book_id": ["b", "b"],
            "language": ["en", "en"],
            "primary_group": ["fiction_manual", "fiction_manual"],
            "concept_id": ["c8", "c7"],
            "retrieval_concept_id": ["c8", "c7"],
            "rank": [5, 3],
            "title": ["t", "t"],
            "author": ["a", "a"],
            "edition_year": ["", ""],
            "corpus_role": ["primary", "primary"],
            "chapter_idx": ["0", "0"],
            "para_idx": ["3", "4"],
        }
    )
    out = _agg_retrieval_membership(lexical, qualified, expansion)
    # Add a synthetic non-retrieved row as the builder would.
    nr = pd.DataFrame(
        [
            {
                "passage_id": "nr1",
                "book_id": "b",
                "language": "en",
                "primary_group": "fiction_manual",
                "in_lexical": False,
                "in_qualified_bm25": False,
                "in_bm25_expansion": False,
                "has_new_concept_assignment": False,
                "best_bm25_rank": np.nan,
                "n_lexically_retrieved_concepts": 0,
                "n_bm25_retrieved_concepts": 0,
                "primary_stratum": "non_retrieved",
                "title": "t",
                "author": "a",
                "edition_year": "",
                "corpus_role": "primary",
                "chapter_idx": "0",
                "para_idx": "5",
            }
        ]
    )
    out = pd.concat([out, nr], ignore_index=True)
    by_stratum = {
        s: set(out.loc[out["primary_stratum"] == s, "passage_id"]) for s in PRIMARY_STRATA_V05
    }
    assert_strata_pairwise_exclusive(by_stratum)
    assert by_stratum["high_ranked_qualified_bm25"] == {"high1"}
    assert by_stratum["lower_ranked_qualified_bm25"] == {"low1"}
    assert by_stratum["lexical_only"] == {"lex1"}
    assert by_stratum["bm25_expansion_known"] == {"exp1"}
    assert by_stratum["new_bm25_passages"] == {"new1"}
    assert by_stratum["non_retrieved"] == {"nr1"}
    assert set().union(*by_stratum.values()) == set(out["passage_id"])


def test_assign_reproducible_across_pythonhashseed():
    script = textwrap.dedent(
        """
        import json
        from src.retrieval.eval_sample import assign_primary_stratum, redistribute_quotas_v05
        from src.retrieval.sampling import derive_stratum_seed
        cases = [
            assign_primary_stratum(in_lexical=True, in_qualified_bm25=True, in_bm25_expansion=False, best_bm25_rank=12),
            assign_primary_stratum(in_lexical=True, in_qualified_bm25=False, in_bm25_expansion=True, best_bm25_rank=None),
            assign_primary_stratum(in_lexical=False, in_qualified_bm25=False, in_bm25_expansion=True, best_bm25_rank=3),
            assign_primary_stratum(in_lexical=False, in_qualified_bm25=False, in_bm25_expansion=False, best_bm25_rank=None),
        ]
        seeds = [derive_stratum_seed(20261124, s) for s in (
            "en::high_ranked_qualified_bm25",
            "de::non_retrieved::fiction_manual",
        )]
        q = redistribute_quotas_v05(
            {"high_ranked_qualified_bm25": 5, "lexical_only": 100, "non_retrieved": 1000},
            {"high_ranked_qualified_bm25": 20, "lexical_only": 20, "non_retrieved": 30},
        )
        print(json.dumps({"cases": cases, "seeds": seeds, "q": q}))
        """
    )
    env_base = {**os.environ, "PYTHONPATH": str(ROOT)}
    results = []
    for hs in ("0", "1", "42"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**env_base, "PYTHONHASHSEED": hs},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    for r in results[1:]:
        assert r == results[0]
