"""Merge multiple reviewer JSON suggestion exports into one review table.

Conflicting suggestions are preserved as separate rows. The active lexicon is
never modified by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, ROOT

SUGGESTION_COLUMNS = [
    "source_file",
    "schema_version",
    "lexicon_version",
    "corpus_snapshot_id",
    "reviewer_id",
    "suggestion_id",
    "concept_id",
    "language",
    "action",
    "proposed_expression",
    "expression_type",
    "rationale",
    "alternative_concept_id",
    "ambiguity_notes",
    "source_reference",
    "example_sentence",
    "confidence",
    "status",
    "conflict_group",
]


def load_suggestion_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")
    suggestions = data.get("suggestions") or []
    rows = []
    resolved = path.resolve()
    try:
        source_file = str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        source_file = str(path)
    for s in suggestions:
        rows.append(
            {
                "source_file": source_file,
                "schema_version": data.get("schema_version", ""),
                "lexicon_version": data.get("lexicon_version", ""),
                "corpus_snapshot_id": data.get("corpus_snapshot_id", ""),
                "reviewer_id": data.get("reviewer_id", s.get("reviewer_id", "")),
                "suggestion_id": s.get("suggestion_id", ""),
                "concept_id": s.get("concept_id", ""),
                "language": s.get("language", ""),
                "action": s.get("action", ""),
                "proposed_expression": s.get("proposed_expression", ""),
                "expression_type": s.get("expression_type", ""),
                "rationale": s.get("rationale", ""),
                "alternative_concept_id": s.get("alternative_concept_id"),
                "ambiguity_notes": s.get("ambiguity_notes", ""),
                "source_reference": s.get("source_reference", ""),
                "example_sentence": s.get("example_sentence", ""),
                "confidence": s.get("confidence", ""),
                "status": s.get("status", "pending_review"),
            }
        )
    return rows


def conflict_key(row: dict) -> str:
    """Group by concept + language + expression (action differences = conflict)."""
    return "|".join(
        [
            str(row.get("concept_id") or ""),
            str(row.get("language") or ""),
            str(row.get("proposed_expression") or "").casefold().strip(),
        ]
    )


def aggregate(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        rows.extend(load_suggestion_file(p))
    if not rows:
        return pd.DataFrame(columns=SUGGESTION_COLUMNS)

    # Detect conflicts: same key, differing action details / reviewers / rationales.
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(conflict_key(r), []).append(r)

    for key, group in by_key.items():
        reviewers = {g.get("reviewer_id") for g in group}
        actions = {g.get("action") for g in group}
        exprs = {str(g.get("proposed_expression") or "").casefold() for g in group}
        # Mark conflict when multiple reviewers propose competing actions on same concept+lang+expression
        # or same expression with different actions / alternative concepts.
        alts = {str(g.get("alternative_concept_id")) for g in group}
        is_conflict = len(group) > 1 and (len(reviewers) > 1 or len(actions) > 1 or len(alts) > 1)
        label = f"conflict::{key}" if is_conflict else ""
        for g in group:
            g["conflict_group"] = label

    # Preserve all rows, including conflicts.
    return pd.DataFrame(rows)[SUGGESTION_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more reviewer suggestion JSON files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_INTERIM / "retrieval" / "terminology_review_consolidated.csv",
    )
    args = parser.parse_args()
    log = setup_logger(__file__)
    df = aggregate([p.resolve() for p in args.inputs])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_conflict = int((df["conflict_group"].astype(str).str.len() > 0).sum()) if len(df) else 0
    log.info(f"Wrote {args.out} rows={len(df)} conflict_rows={n_conflict}")


if __name__ == "__main__":
    main()
