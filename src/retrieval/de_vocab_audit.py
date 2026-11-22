"""German lexical audit for v0.3 growth concepts + provenance table.

AI assessments are provisional and must not be treated as human ground truth.
Samples use lexicon-development held-out books only (not the independent eval set).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED
from src.retrieval.eval_sample import load_held_out_books
from src.retrieval.paragraphs import iter_body_paragraphs
from src.retrieval.run_meta import base_run_meta, write_json
from src.retrieval.sampling import ReservoirSample, derive_stratum_seed

OUT = DATA_INTERIM / "retrieval"
LEX_V03 = CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml"
LEX_V02 = CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml"
SEED = 20261123

FOCUS_CONCEPTS = [
    "focalization_zero",
    "stance_narratorial",
    "stance_figural",
    "grammatical_third_person",
]

DE_ZERO = [
    "focalization_fixed",
    "voice_eyewitness",
    "voice_participant",
    "voice_homodiegetic",
]

EN_LOW = [
    "pov_object",
    "pov_subject",
    "voice_participant",
    "multiple_focalization_same_event",
]

TRIGGER_PATTERNS = {
    "allwissen": re.compile(r"allwissen\w*", re.I),
    "auktorial": re.compile(r"auktorial\w*", re.I),
    "personal_erzaehl": re.compile(
        r"personal\w*\s+(?:erzähl|perspektiv)\w*|personal\w*\s+erzähl\w*", re.I
    ),
    "wissensvorsprung": re.compile(r"wissensvorsprung\w*", re.I),
    "nullfokalisier": re.compile(r"null[\s-]*fokalisier\w*", re.I),
}


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _provisional_assess(text: str, trigger: str, concept: str) -> dict:
    """Heuristic provisional AI assessment — NOT human gold."""
    t = text.casefold()
    ambiguity = []
    error_type = ""
    category = "unclear"
    needs_human = True

    instructional = any(
        x in t
        for x in (
            "erzähler",
            "erzähl",
            "perspektive",
            "fokalis",
            "auktorial",
            "personal",
            "allwissen",
            "genette",
            "stanzel",
        )
    )
    fictionish = any(
        x in t for x in (" sagte ", " dachte ", " plötzlich ", " kapitel ")
    )
    grammar = "sie-form" in t or "konjug" in t or "grammatik" in t

    if grammar and not instructional:
        category = "grammatical_construction"
        error_type = "likely_false_positive"
    elif instructional and trigger in ("auktorial", "personal_erzaehl", "allwissen"):
        category = "genuine_narratological_discussion"
        # Conceptual caution for Genette/Stanzel
        if concept == "focalization_zero" and "auktorial" in t and "allwissen" not in t:
            ambiguity.append("auktorial_vs_zero_focalization")
            category = "conceptually_ambiguous"
        elif concept == "stance_figural" and "fokalis" in t:
            ambiguity.append("figural_vs_internal_focalization")
        error_type = "possible_true_positive"
    elif fictionish and not instructional:
        category = "fictional_example"
        error_type = "possible_false_positive"
    elif instructional:
        category = "general_literary_discussion"
        error_type = "needs_review"
    else:
        category = "unrelated_or_unclear"
        error_type = "needs_review"

    if "ocr" in t or "  " in text:
        ambiguity.append("possible_ocr_or_segmentation")

    return {
        "provisional_category": category,
        "provisional_error_type": error_type,
        "provisional_ambiguity": ";".join(ambiguity),
        "needs_human_review": needs_human,
        "assessment_source": "provisional_ai_assessment",
    }


def attribute_gains(log) -> pd.DataFrame:
    """Attribute v0.2 → algo_fix → v0.3 DE gains using matched terms/regex."""
    b02 = _load(DATA_PROCESSED / "pov_baseline_b_v02_lexical_de.csv")
    bfix = _load(DATA_PROCESSED / "pov_baseline_b_v02_algo_fix_de.csv")
    b03 = _load(DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv")
    delta = _load(OUT / "retrieval_v02_v03_concept_delta.csv")

    rows = []
    for cid in FOCUS_CONCEPTS:
        k02 = set(zip(b02["passage_id"], b02["concept_id"])) if len(b02) else set()
        kfix = set(zip(bfix["passage_id"], bfix["concept_id"])) if len(bfix) else set()
        k03 = set(zip(b03["passage_id"], b03["concept_id"])) if len(b03) else set()
        # Filter to concept
        k02c = {k for k in k02 if k[1] == cid}
        kfixc = {k for k in kfix if k[1] == cid}
        k03c = {k for k in k03 if k[1] == cid}
        algo_new = kfixc - k02c
        vocab_new = k03c - kfixc

        def _cause_counts(keys, df):
            causes = Counter()
            sub = df[df["concept_id"] == cid] if len(df) else df
            keyset = keys
            for _, r in sub.iterrows():
                key = (r["passage_id"], r["concept_id"])
                if key not in keyset:
                    continue
                terms = (r.get("matched_terms") or "").casefold()
                regex = (r.get("matched_regex") or "").casefold()
                tiers = r.get("evidence_tiers") or ""
                cause = []
                if regex.strip():
                    cause.append("regex")
                if "allwissen" in terms or "allwissen" in regex:
                    cause.append("dict_allwissen")
                if "auktorial" in terms or "auktorial" in regex:
                    cause.append("dict_auktorial")
                if "personal" in terms or "personal" in regex:
                    cause.append("dict_personal")
                if "strong" in tiers.casefold():
                    cause.append("strong_override")
                if not cause:
                    cause.append("other_dict_or_morph")
                causes["+".join(sorted(set(cause)))] += 1
            return dict(causes)

        drow = {}
        if len(delta):
            hit = delta[(delta["language"] == "de") & (delta["concept_id"] == cid)]
            if len(hit):
                drow = hit.iloc[0].to_dict()

        rows.append(
            {
                "concept_id": cid,
                "n_v02": len(k02c),
                "n_algo_fix": len(kfixc),
                "n_v03": len(k03c),
                "gained_algo_fix": len(algo_new),
                "gained_vocab_or_tiers": len(vocab_new),
                "algo_new_causes": json.dumps(
                    _cause_counts(algo_new, bfix), ensure_ascii=False
                ),
                "vocab_new_causes": json.dumps(
                    _cause_counts(vocab_new, b03), ensure_ascii=False
                ),
                "delta_gained_algo": drow.get("gained_algo_fix", ""),
                "delta_gained_vocab": drow.get("gained_vocab_or_tiers", ""),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "german_gain_attribution_v04.csv", index=False)
    log.info(f"wrote german_gain_attribution_v04.csv ({len(out)} concepts)")
    return out


def sample_trigger_passages(log, *, per_trigger: int = 25) -> pd.DataFrame:
    """Sample from development held-out books only."""
    held = load_held_out_books()
    de_held = {b for b in held if b.startswith("de_")}
    if not de_held:
        log.warning("No DE held-out books; refusing to sample from eval population")
        return pd.DataFrame()

    b03 = _load(DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv")
    focus = b03[b03["concept_id"].isin(FOCUS_CONCEPTS)].copy()
    focus = focus[focus["book_id"].isin(de_held)]
    preds_by_pid: dict[str, list[str]] = defaultdict(list)
    terms_by_pid: dict[str, list[str]] = defaultdict(list)
    regex_by_pid: dict[str, list[str]] = defaultdict(list)
    for _, r in focus.iterrows():
        pid = r["passage_id"]
        preds_by_pid[pid].append(r["concept_id"])
        if r.get("matched_terms"):
            terms_by_pid[pid].append(r["matched_terms"])
        if r.get("matched_regex"):
            regex_by_pid[pid].append(r["matched_regex"])

    by_trigger: dict[str, list] = defaultdict(list)
    for para in iter_body_paragraphs("de", with_neighbors=True):
        if para.book_id not in de_held:
            continue
        for tname, pat in TRIGGER_PATTERNS.items():
            if not pat.search(para.text):
                continue
            pid = para.passage_id
            by_trigger[tname].append(
                {
                    "passage_id": pid,
                    "book_id": para.book_id,
                    "trigger": tname,
                    "matching_rule": "surface_regex_scan",
                    "predicted_concepts": ";".join(sorted(set(preds_by_pid.get(pid, [])))),
                    "matched_terms": ";".join(terms_by_pid.get(pid, [])),
                    "matched_regex": ";".join(regex_by_pid.get(pid, [])),
                    "passage": para.text,
                    "prev_passage": para.prev_text,
                    "next_passage": para.next_text,
                }
            )

    rows = []
    seen_passages: set[str] = set()
    for tname, cands in by_trigger.items():
        uniq = {c["passage_id"]: c for c in cands}
        cands = sorted(uniq.values(), key=lambda x: x["passage_id"])
        res = ReservoirSample(
            k=min(per_trigger, len(cands)),
            seed=derive_stratum_seed(SEED, f"de_audit::{tname}"),
        )
        for c in cands:
            res.consider(c)
        for c in res.reservoir:
            if c["passage_id"] in seen_passages:
                continue
            seen_passages.add(c["passage_id"])
            pred = (c["predicted_concepts"] or "unknown").split(";")[0]
            assess = _provisional_assess(c["passage"], c["trigger"], pred)
            rows.append({**c, **assess})

    out_df = pd.DataFrame(rows)
    ptr_cols = [
        c
        for c in out_df.columns
        if c not in ("passage", "prev_passage", "next_passage")
    ]
    if len(out_df):
        out_df[ptr_cols].to_csv(OUT / "german_audit_review_v04.csv", index=False)
        out_df.to_csv(OUT / "german_audit_review_v04_passages.csv", index=False)
    else:
        pd.DataFrame(columns=ptr_cols).to_csv(
            OUT / "german_audit_review_v04.csv", index=False
        )
    log.info(
        f"wrote german_audit_review_v04.csv ({len(out_df)} rows); "
        "passages sidecar gitignored"
    )
    return out_df


def low_coverage_bm25(log) -> pd.DataFrame:
    rows = []
    only_de = _load(DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_de.csv")
    only_en = _load(DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_en.csv")
    lex_de = _load(DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv")
    lex_en = _load(DATA_PROCESSED / "pov_baseline_b_v03_lexical_en.csv")

    for cid in DE_ZERO:
        n_lex = int((lex_de["concept_id"] == cid).sum()) if len(lex_de) else 0
        n_bm25 = int((only_de["concept_id"] == cid).sum()) if len(only_de) else 0
        rows.append(
            {
                "language": "de",
                "concept_id": cid,
                "n_lexical_v03": n_lex,
                "n_bm25_only_v03": n_bm25,
                "interpretation": (
                    "possible_retrieval_failure_or_rare_wording"
                    if n_bm25 > 0
                    else "possible_genuine_absence_or_missing_lexicon"
                ),
            }
        )
    for cid in EN_LOW:
        n_lex = int((lex_en["concept_id"] == cid).sum()) if len(lex_en) else 0
        n_bm25 = int((only_en["concept_id"] == cid).sum()) if len(only_en) else 0
        rows.append(
            {
                "language": "en",
                "concept_id": cid,
                "n_lexical_v03": n_lex,
                "n_bm25_only_v03": n_bm25,
                "interpretation": (
                    "low_lexical_hit_with_exploratory_bm25_candidates"
                    if n_bm25 > 0
                    else "low_lexical_and_low_bm25"
                ),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "low_coverage_concepts_v04.csv", index=False)
    log.info(f"wrote low_coverage_concepts_v04.csv ({len(df)} rows)")
    return df


def build_provenance_table(log) -> pd.DataFrame:
    """Versioned term provenance — distinguish corpus-based vs AI/provisional."""
    raw = yaml.safe_load(LEX_V03.read_text(encoding="utf-8")) or {}
    # Load v0.2 term sets for provenance typing
    raw02 = yaml.safe_load(LEX_V02.read_text(encoding="utf-8")) or {}

    def _term_sets(raw_lex):
        out = defaultdict(set)
        for block in raw_lex.get("concepts") or []:
            cid = block.get("id")
            for lang in ("en", "de"):
                node = (block.get(lang) or {})
                for term_type in ("academic", "craft", "paraphrases", "context"):
                    for term in node.get(term_type) or []:
                        out[(cid, lang, term_type)].add(str(term).casefold())
                for term, tier in (node.get("evidence_overrides") or {}).items():
                    out[(cid, lang, "override")].add(str(term).casefold())
        return out

    v02_terms = _term_sets(raw02)
    rows = []
    for block in raw.get("concepts") or []:
        cid = block.get("id")
        for lang in ("en", "de"):
            node = block.get(lang) or {}
            for term_type, key in (
                ("academic", "academic"),
                ("craft", "craft"),
                ("paraphrase", "paraphrases"),
                ("context", "context"),
            ):
                for term in node.get(key) or []:
                    cf = str(term).casefold()
                    in_v02 = cf in v02_terms.get((cid, lang, key if key != "paraphrases" else "paraphrases"), set())
                    # v0.3 expansions were seeded from task brief / BM25 discovery — provisional
                    if in_v02:
                        prov = "prior_lexicon_v02"
                        status = "inherited"
                    elif term_type == "paraphrase":
                        prov = "provisional_expansion_v03_ai_or_brief"
                        status = "needs_human_review"
                    else:
                        prov = "v03_addition_unverified_corpus_origin"
                        status = "needs_human_review"
                    tier = (node.get("evidence_overrides") or {}).get(cf, "")
                    rows.append(
                        {
                            "concept_id": cid,
                            "language": lang,
                            "term": term,
                            "term_type": term_type,
                            "evidence_tier": tier or "",
                            "provenance_type": prov,
                            "source_reference": (
                                "config/pov_lexicon_psyper_en_de_v0.2.yml"
                                if in_v02
                                else "config/pov_lexicon_psyper_en_de_v0.3.yml"
                            ),
                            "development_passage_id": "",
                            "review_status": status,
                            "notes": (
                                "Do not claim corpus-based origin without supporting passage."
                                if not in_v02
                                else ""
                            ),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lexicon_term_provenance_v04.csv", index=False)
    log.info(f"wrote lexicon_term_provenance_v04.csv ({len(df)} rows)")
    return df


def write_audit_report(log, gains, review, lowcov) -> Path:
    md = [
        "# German lexical audit (v0.4)",
        "",
        "**All AI assessments are provisional** and are not human ground truth.",
        "Independent evaluation labels were not modified.",
        "",
        "## Gain attribution (DE focus concepts)",
        "",
        "| concept | v0.2 | algo_fix | v0.3 | +algo | +vocab |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in gains.iterrows():
        md.append(
            f"| `{r['concept_id']}` | {r['n_v02']} | {r['n_algo_fix']} | "
            f"{r['n_v03']} | {r['gained_algo_fix']} | {r['gained_vocab_or_tiers']} |"
        )
    md.append("")
    md.append("Cause breakdowns (vocab-new) are in `german_gain_attribution_v04.csv`.")
    md.append("")
    md.append("## Conceptual distinctions (human decisions required)")
    md.append("")
    md.append(
        "- Do **not** equate auktoriales Erzählen with zero focalization "
        "(regex map already routes `auktorial` → `stance_narratorial`)."
    )
    md.append(
        "- Do **not** equate personales Erzählen with internal focalization automatically."
    )
    md.append(
        "- Lexicon still lists some overlapping academic cues "
        "(e.g. `auktoriale Erzählsituation` under `focalization_zero`) — "
        "**human decision** whether to demote/remove."
    )
    md.append(
        "- Prefer additional evidence requirements for ambiguous terms over silent deletion."
    )
    md.append("")
    md.append("## Trigger sample")
    md.append("")
    if len(review):
        md.append(f"- Annotation-ready rows: {len(review)} (`german_audit_review_v04.csv`)")
        if "provisional_category" in review.columns:
            vc = review["provisional_category"].value_counts().to_dict()
            md.append(f"- Provisional category counts: `{vc}`")
    else:
        md.append("- No review rows generated (check held-out books / fulltext availability).")
    md.append("")
    md.append("## Low-coverage concepts")
    md.append("")
    for _, r in lowcov.iterrows():
        md.append(
            f"- `{r['language']}` `{r['concept_id']}`: lexical={r['n_lexical_v03']}, "
            f"bm25_only={r['n_bm25_only_v03']} → {r['interpretation']}"
        )
    md.append("")
    md.append("Preserve zero/low-coverage concepts as diagnostic cases for semantic retrieval.")
    md.append("")
    md.append("## Proposed rule changes")
    md.append("")
    md.append(
        "1. **Human decision:** whether `auktoriale Erzählsituation` should remain "
        "under `focalization_zero` academic terms or move to stance-only."
    )
    md.append(
        "2. **Mechanical (optional):** demote bare `allwissen*` hits outside "
        "narratological collocates — needs human confirmation before v0.4 lexicon edit."
    )
    md.append(
        "3. No automatic conceptual remaps applied in this finalization pass."
    )
    md.append("")
    md.append("## Known segmentation anomaly")
    md.append("")
    md.append(
        "- `de_26856f179d5e:c7:p0` ≈ 102k characters (single paragraph chapter) — "
        "flag in corpus QC; do not silently truncate."
    )
    path = OUT / "german_lexical_audit_v04.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-trigger", type=int, default=25)
    parser.add_argument("--skip-samples", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gains = attribute_gains(log)
    lowcov = low_coverage_bm25(log)
    prov = build_provenance_table(log)
    if args.skip_samples:
        review = pd.DataFrame()
    else:
        review = sample_trigger_passages(log, per_trigger=args.per_trigger)
    write_audit_report(log, gains, review, lowcov)
    write_json(
        OUT / "german_lexical_audit_v04.meta.json",
        base_run_meta(
            kind="german_lexical_audit_v04",
            duration_sec=round(time.time() - t0, 1),
            n_review_rows=len(review),
            n_provenance_rows=len(prov),
            note="AI assessments provisional; eval labels untouched.",
        ),
    )
    log.info("Log finished.")


if __name__ == "__main__":
    main()
