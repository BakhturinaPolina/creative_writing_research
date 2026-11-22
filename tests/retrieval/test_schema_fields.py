"""Schema: retrieval_concept_id vs PsyPer-related vs human-validated tags."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.paths import DATA_PROCESSED
from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon


REQUIRED_V03 = {
    "retrieval_concept_id",
    "psyper_related_tags",
    "human_validated_psyper_tags",
}


@pytest.mark.parametrize(
    "stem",
    [
        "pov_baseline_b_v03_lexical_de",
        "pov_baseline_b_v03_lexical_en",
        "pov_baseline_c_v03_bm25_de",
        "pov_baseline_c_v03_bm25_only_unvalidated_de",
    ],
)
def test_v03_outputs_keep_three_tag_fields(stem: str):
    path = DATA_PROCESSED / f"{stem}.csv"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    df = pd.read_csv(path, nrows=50, dtype=str).fillna("")
    for col in REQUIRED_V03:
        assert col in df.columns, f"{stem} missing {col}"
    # Human-validated must be empty in automatic retrieval outputs
    assert (df["human_validated_psyper_tags"] == "").all()
    # retrieval_concept_id should match concept_id when both present
    if "concept_id" in df.columns and len(df):
        assert (df["retrieval_concept_id"] == df["concept_id"]).all()


def test_hierarchical_psyper_codes_remain_strings():
    lex = load_lexicon(LEXICON_PATH_V03)
    ct = lex.concepts["en"]["voice_eyewitness"]
    assert ct.psyper == ["2.2.1"]
    assert all(isinstance(x, str) for x in ct.psyper)
    # Joined tags must preserve dots
    joined = ct.psyper_tags_joined()
    assert "2.2.1" in joined


def test_automatic_retrieval_is_not_gold_annotation():
    """psyper_related_tags are lexicon links, not human validation."""
    path = DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv"
    if not path.is_file():
        pytest.skip("missing de lexical v03")
    df = pd.read_csv(path, nrows=20, dtype=str).fillna("")
    assert "human_validated_psyper_tags" in df.columns
    assert (df["human_validated_psyper_tags"] == "").all()
    # Related tags may be non-empty but must not equal concept id alone as a fake gold
    if len(df) and df["psyper_related_tags"].iloc[0]:
        assert df["psyper_related_tags"].iloc[0] != df["retrieval_concept_id"].iloc[0]
