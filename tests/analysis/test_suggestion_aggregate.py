"""Tests for terminology suggestion aggregation."""
from __future__ import annotations

import json
from pathlib import Path

from src.lexicon_review.aggregate_suggestions import aggregate


def test_aggregate_preserves_conflicts(tmp_path: Path):
    a = {
        "schema_version": "1.0",
        "lexicon_version": "0.3-draft",
        "corpus_snapshot_id": "snap",
        "reviewer_id": "rev-a",
        "suggestions": [
            {
                "suggestion_id": "s1",
                "concept_id": "stance_figural",
                "language": "de",
                "action": "add_term",
                "proposed_expression": "personales Erzählen",
                "expression_type": "academic",
                "rationale": "common term",
                "alternative_concept_id": None,
                "ambiguity_notes": "",
                "source_reference": "",
                "confidence": "medium",
                "status": "pending_review",
            }
        ],
    }
    b = {
        "schema_version": "1.0",
        "lexicon_version": "0.3-draft",
        "corpus_snapshot_id": "snap",
        "reviewer_id": "rev-b",
        "suggestions": [
            {
                "suggestion_id": "s2",
                "concept_id": "stance_figural",
                "language": "de",
                "action": "remove_term",
                "proposed_expression": "personales Erzählen",
                "expression_type": "academic",
                "rationale": "too ambiguous",
                "alternative_concept_id": "focalization_internal",
                "ambiguity_notes": "Stanzel overlap",
                "source_reference": "",
                "confidence": "high",
                "status": "pending_review",
            }
        ],
    }
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    df = aggregate([pa, pb])
    assert len(df) == 2
    assert (df["conflict_group"].astype(str).str.startswith("conflict::")).all()
    assert set(df["reviewer_id"]) == {"rev-a", "rev-b"}
