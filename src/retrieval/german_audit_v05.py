"""Clarify German lexical audit as heuristic development data (v0.5).

Does not re-sample passages. Transforms the frozen v0.4 review table:
- rename assessment_source → heuristic_provisional_assessment
- add empty human-review fields
- document Wissensvorsprung ambiguity (do not auto-mark unrelated)
- label dataset as lexicon development, not independent evaluation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM
from src.retrieval.run_meta import base_run_meta, write_json

OUT = DATA_INTERIM / "retrieval"
V04 = OUT / "german_audit_review_v04.csv"

HUMAN_FIELDS = [
    "human_category",
    "human_error_type",
    "human_ambiguity",
    "human_notes",
    "reviewer_id",
    "review_date",
]


def build_german_audit_v05(log) -> pd.DataFrame:
    if not V04.is_file():
        raise SystemExit(f"Missing frozen v0.4 audit: {V04}")
    df = pd.read_csv(V04, dtype=str).fillna("")

    # Rename assessment source; keep provisional_* columns as heuristic_* aliases.
    df["assessment_source"] = "heuristic_provisional_assessment"
    df = df.rename(
        columns={
            "provisional_category": "heuristic_provisional_category",
            "provisional_error_type": "heuristic_provisional_error_type",
            "provisional_ambiguity": "heuristic_provisional_ambiguity",
        }
    )
    # Concept-assignment level ambiguity/error placeholders (empty for humans).
    df["concept_assignment_error_type"] = ""
    df["concept_assignment_ambiguity"] = ""
    for col in HUMAN_FIELDS:
        df[col] = ""

    df["heuristic_provisional_notes"] = ""
    # Wissensvorsprung: do not auto-classify as unrelated — flag for dual relevance.
    mask = df["trigger"].astype(str).str.contains("wissensvorsprung", case=False, na=False)
    df.loc[mask, "concept_assignment_ambiguity"] = (
        "wissensvorsprung_may_relate_to_information_control_or_reader_effect;"
        "focalization_relevance_uncertain;do_not_auto_mark_unrelated"
    )
    df.loc[mask, "heuristic_provisional_notes"] = (
        "Surface hits often discuss dramatic irony, suspense, audience knowledge "
        "or information asymmetry. Relevance to focalization may be uncertain; "
        "may be highly relevant to information_control or reader_effect."
    )

    df["dataset_role"] = "lexicon_development_audit"
    df["not_independent_evaluation"] = "true"
    df["not_precision_estimate"] = "true"

    ptr_cols = [c for c in df.columns if c not in ("passage", "prev_passage", "next_passage")]
    out = OUT / "german_audit_review_v05.csv"
    df[ptr_cols].to_csv(out, index=False)

    # Heuristic category counts — explicitly NOT precision.
    cat_col = "heuristic_provisional_category"
    vc = df[cat_col].value_counts().to_dict() if cat_col in df.columns else {}
    n_genuine = int(vc.get("genuine_narratological_discussion", 0))
    n = len(df)

    md = [
        "# German lexical audit clarification (v0.5)",
        "",
        f"**Rows:** {n} (frozen from v0.4 sample; development held-out books only).",
        "",
        "## Critical interpretation warning",
        "",
        f"The heuristic labelled approximately **{n_genuine}/{n}** rows as "
        "`genuine_narratological_discussion`.",
        "",
        "**This proportion must NOT be interpreted as retrieval precision.**",
        "",
        "Reasons:",
        "",
        "1. Categories are `heuristic_provisional_assessment` (surface heuristics).",
        "2. The heuristic partly uses the triggering lexical expressions themselves "
        "as evidence of relevance.",
        "3. This table is a **lexicon development** dataset, not the independent "
        "probability-based evaluation sample.",
        "4. Human-review fields are present but **empty** — no human gold labels here.",
        "",
        "## Wissensvorsprung",
        "",
        "Several passages discuss dramatic irony, suspense, audience knowledge and "
        "information asymmetry. Although relevance to focalization may be uncertain, "
        "they may be highly relevant to `information_control` or `reader_effect`.",
        "They are **not** automatically classified as unrelated.",
        "",
        "## Heuristic category counts (not precision)",
        "",
    ]
    for k, v in sorted(vc.items(), key=lambda x: -x[1]):
        md.append(f"- `{k}`: {v}")
    md += [
        "",
        "## Human review fields (empty)",
        "",
        ", ".join(f"`{c}`" for c in HUMAN_FIELDS),
        "",
        "Preserve v0.4 artifacts unchanged. Keep this audit separate from "
        "`manual_eval_sample_prob_v05`.",
        "",
    ]
    (OUT / "german_lexical_audit_v05.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(
        OUT / "german_audit_review_v05.meta.json",
        base_run_meta(
            kind="german_audit_review_v05",
            n_rows=n,
            dataset_role="lexicon_development_audit",
            assessment_source="heuristic_provisional_assessment",
            heuristic_category_counts=vc,
            note=(
                f"Heuristic {n_genuine}/{n} is NOT precision. "
                "Human fields empty. v0.4 preserved."
            ),
        ),
    )
    log.info(f"wrote {out} ({n} rows); german_lexical_audit_v05.md")
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    log = setup_logger(__file__)
    build_german_audit_v05(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
