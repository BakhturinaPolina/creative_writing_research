"""Machine-readable input inventory and book-coverage audit for analysis v2.

Read-only over frozen corpus and retrieval artifacts. Does not modify sources.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.narrative_perspective import (
    LANGUAGE_COLORS,
    LANG_LABEL,
    file_sha256,
    load_corpus_manifest,
    load_retrieval_results,
    source_commit,
)
from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED, ROOT

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures" / "02_narrative_perspective"
SNAPSHOT_ID = "corpus_snapshot_analysis_v1_candidate_20261122"
ANALYSIS_VERSION = "v2"
RETRIEVAL_EXECUTION_ID = "v03_lexical_bm25_20261122"


def _meta_for(path: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if path.suffix == ".csv":
        alt = path.with_suffix(".meta.json")
        if alt.is_file():
            meta_path = alt
    if not meta_path.is_file():
        # also try path.stem.meta.json for .yml
        alt2 = path.parent / f"{path.name}.meta.json"
        if alt2.is_file():
            meta_path = alt2
        else:
            return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_records(path: Path) -> int | None:
    if not path.is_file():
        return None
    if path.suffix == ".csv":
        # Fast line count minus header.
        with path.open("rb") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    if path.suffix in {".yml", ".yaml"}:
        return None
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict) and "n_entries" in data:
                return int(data["n_entries"])
        except Exception:
            return None
    if path.suffix == ".md":
        return None
    if path.suffix == ".py":
        return None
    return None


def _inventory_entry(
    rel: str,
    *,
    version: str = "",
    language: str = "",
    relationship: str = "",
    record_count: int | None = None,
) -> dict:
    path = ROOT / rel
    meta = _meta_for(path) if path.is_file() else {}
    n = record_count if record_count is not None else _count_records(path)
    if n is None and "n_candidate_rows" in meta:
        n = meta.get("n_candidate_rows")
    if n is None and "n_passages" in meta:
        n = meta.get("n_passages")
    if n is None and "n_books" in meta:
        n = meta.get("n_books")
    if n is None and "n_rows" in meta:
        n = meta.get("n_rows")
    if n is None and "n_entries" in meta:
        n = meta.get("n_entries")
    gen = (
        meta.get("built")
        or meta.get("execution_timestamp")
        or meta.get("generated")
        or ""
    )
    snap = meta.get("corpus_snapshot_id") or (
        SNAPSHOT_ID if "v1_candidate" in rel or "v03" in rel else ""
    )
    return {
        "file_path": rel,
        "exists": path.is_file(),
        "sha256": file_sha256(path) if path.is_file() else "",
        "git_commit": meta.get("git_commit") or meta.get("source_commit") or "",
        "version": version
        or str(meta.get("lexicon_version") or meta.get("version") or meta.get("kind") or ""),
        "corpus_snapshot_id": snap,
        "n_records": n,
        "language": language or meta.get("lang") or "",
        "generation_date": gen,
        "relationship": relationship,
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def build_input_inventory() -> dict:
    """Enumerate substantive pipeline inputs for analysis v2."""
    items = []

    # Corpus
    for lang in ("en", "de"):
        items.append(
            _inventory_entry(
                f"data/processed/{lang}_book_stats.csv",
                version="book_stats",
                language=lang,
                relationship="Upstream membership and exclusion_step; feeds snapshot manifests",
            )
        )
        items.append(
            _inventory_entry(
                f"data/processed/corpus_manifest_{lang}_v1_candidate.csv",
                version=SNAPSHOT_ID,
                language=lang,
                relationship="Book-level denominator and inclusion for analysis v2",
            )
        )
        items.append(
            _inventory_entry(
                f"data/processed/passage_manifest_{lang}_v1_candidate.csv",
                version=SNAPSHOT_ID,
                language=lang,
                relationship="Eligible searchable passages; word-count denominators",
            )
        )

    items.append(
        _inventory_entry(
            "data/processed/corpus_quality_v1_candidate.csv",
            version=SNAPSHOT_ID,
            relationship="QC summary copied into reports",
        )
    )
    for lang in ("en", "de"):
        items.append(
            _inventory_entry(
                f"data/processed/corpus_quality_extreme_passages_{lang}_v1_candidate.csv",
                version=SNAPSHOT_ID,
                language=lang,
                relationship="Extreme-length passage list for S3 and diagnostics",
            )
        )

    # Lexical retrieval executions
    for stem, ver, rel in (
        ("pov_baseline_b_v02_lexical", "v0.2", "Historical lexical baseline"),
        ("pov_baseline_b_v02_algo_fix", "v0.2-algo-fix", "Historical lexical with shared prefilter"),
        ("pov_baseline_b_v03_lexical", "v0.3", "Primary lexical execution for analysis v2"),
    ):
        for lang in ("en", "de"):
            items.append(
                _inventory_entry(
                    f"data/processed/{stem}_{lang}.csv",
                    version=ver,
                    language=lang,
                    relationship=rel,
                )
            )

    # BM25
    for stem, ver, rel in (
        ("pov_baseline_c_v03_bm25", "v0.3-Bq", "Qualified BM25 (Bq)"),
        ("pov_baseline_c_v03_bm25_only_unvalidated", "v0.3-Bnew", "BM25-only after subtracting L (Bnew); Braw not on disk"),
        ("pov_baseline_c_v03_bm25_top50", "v0.3-top50", "Review slice of qualified BM25"),
    ):
        for lang in ("en", "de"):
            items.append(
                _inventory_entry(
                    f"data/processed/{stem}_{lang}.csv",
                    version=ver,
                    language=lang,
                    relationship=rel,
                )
            )

    # Interim retrieval / eval / provenance
    interim = [
        ("data/interim/retrieval/retrieval_set_algebra_v05.csv", "v0.5", "BM25 pool terminology summary; no new hits"),
        ("data/interim/retrieval/lexicon_term_provenance_v05.csv", "v0.5", "Dictionary provenance for concept cards"),
        ("data/interim/retrieval/german_audit_review_v05.csv", "v0.5", "Heuristic German development audit"),
        ("data/interim/retrieval/eval_passage_membership_v05.csv", "v0.5", "Exclusive eval strata membership"),
        ("data/interim/retrieval/manual_eval_sample_prob_v05.csv", "v0.5", "Unlabelled probability sample frame"),
        ("data/interim/retrieval/manual_eval_sample_v04_v05_reconciliation.csv", "v0.5", "Sample reconciliation"),
    ]
    for rel, ver, relationship in interim:
        items.append(_inventory_entry(rel, version=ver, relationship=relationship))

    # Config
    for rel, ver, relationship in (
        ("config/pov_lexicon_psyper_en_de_v0.3.yml", "0.3-draft", "Active retrieval lexicon"),
        ("config/regex_concept_map_v03.yml", "0.3", "Regex map for lexical matcher"),
        ("config/concept_families_v05.yml", "0.5", "Analytical family grouping + S4 rules"),
    ):
        items.append(_inventory_entry(rel, version=ver, relationship=relationship))

    # Analysis scripts
    for rel in (
        "src/analysis/narrative_perspective.py",
        "src/analysis/run_np_exploration.py",
        "src/analysis/inventory.py",
    ):
        items.append(
            _inventory_entry(
                rel,
                version=ANALYSIS_VERSION,
                relationship="Analysis code checksummed for reproducibility",
            )
        )

    # Verify book-stats hashes match lexical v03 metas
    hash_check = {}
    for lang in ("en", "de"):
        stats_path = DATA_PROCESSED / f"{lang}_book_stats.csv"
        lex_meta = _meta_for(DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv")
        snap_meta = _meta_for(DATA_PROCESSED / f"corpus_manifest_{lang}_v1_candidate.csv")
        actual = file_sha256(stats_path) if stats_path.is_file() else ""
        expected_lex = (lex_meta.get("book_stats_hashes") or {}).get(lang, "")
        expected_snap = (snap_meta.get("book_stats_hashes") or {}).get(lang, "") or (
            snap_meta.get("book_stats_sha256") or ""
        )
        hash_check[lang] = {
            "book_stats_sha256": actual,
            "lexical_v03_expected": expected_lex,
            "snapshot_expected": expected_snap,
            "matches_lexical_v03": bool(expected_lex) and actual == expected_lex,
            "matches_snapshot": bool(expected_snap) and actual == expected_snap,
        }

    return {
        "schema_version": "1.0",
        "analysis_version": ANALYSIS_VERSION,
        "corpus_snapshot_id": SNAPSHOT_ID,
        "lexicon_version": "0.3-draft",
        "retrieval_execution_id": RETRIEVAL_EXECUTION_ID,
        "retrieval_algorithm_version": "v0.3",
        "evaluation_infrastructure_version": "v0.5",
        "source_commit": source_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "book_stats_hash_agreement": hash_check,
        "note": (
            "v0.4/v0.5 did not regenerate lexical or BM25 assignment CSVs. "
            "Analysis v2 uses frozen v0.3 retrieval execution against the v1 candidate snapshot. "
            "Braw (pre-subtraction BM25 heap) is not on disk."
        ),
        "inputs": items,
    }


def _recommended_action(exclusion_step: str, processing_status: str, in_set: bool) -> str:
    if in_set:
        return "retain_in_analysis"
    if exclusion_step == "needs_ocr" or processing_status == "missing_file":
        return "obtain_ocr_or_readable_fulltext_then_reprocess"
    if exclusion_step == "short_text":
        return "keep_excluded_unless_research_threshold_changes"
    if exclusion_step in {"surplus_duplicate", "role_conflict"}:
        return "keep_excluded_confirmed_duplicate_or_role_conflict"
    if exclusion_step:
        return f"review_exclusion_rule:{exclusion_step}"
    return "investigate_unexpected_exclusion"


def _status_vocab(
    *,
    in_set: bool,
    exclusion_step: str,
    processing_status: str,
    n_lexical_passages: int,
    possible_duplicate: bool,
) -> str:
    if processing_status in {"missing_file", "unreadable"} or exclusion_step == "needs_ocr":
        return "missing_source"
    if exclusion_step in {"surplus_duplicate", "role_conflict"}:
        return "confirmed_duplicate"
    if not in_set and exclusion_step:
        return "intentional_exclusion"
    if not in_set:
        return "preprocessing_failed"
    if possible_duplicate:
        # Still in analysis set but flagged.
        if n_lexical_passages > 0:
            return "potential_duplicate"
        return "potential_duplicate"
    if n_lexical_passages > 0:
        return "processed_with_matches"
    return "processed_zero_matches"


def build_book_coverage_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per listed book; funnel summary by language."""
    corpus = load_corpus_manifest()
    frames = []
    for lang in ("en", "de"):
        stats = pd.read_csv(DATA_PROCESSED / f"{lang}_book_stats.csv", dtype=str).fillna("")
        stats["language"] = lang
        frames.append(stats)
    stats = pd.concat(frames, ignore_index=True)

    # Eligible passage word sums from passage manifests (column subset).
    pass_words = {}
    pass_counts = {}
    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv"
        pm = pd.read_csv(path, usecols=["book_id", "word_count"], dtype=str)
        pm["word_count"] = pd.to_numeric(pm["word_count"], errors="coerce").fillna(0)
        pass_words[lang] = pm.groupby("book_id")["word_count"].sum().to_dict()
        pass_counts[lang] = pm.groupby("book_id").size().to_dict()

    lexical = load_retrieval_results("lexical_v03")
    bm25_q = load_retrieval_results("bm25_qualified_v03")
    bm25_new = load_retrieval_results("bm25_only_v03")

    lex_pass = (
        lexical.groupby(["language", "book_id"])["passage_id"].nunique().to_dict()
        if len(lexical)
        else {}
    )
    bm25_books = set()
    if len(bm25_q):
        bm25_books |= set(zip(bm25_q["language"], bm25_q["book_id"]))
    if len(bm25_new):
        bm25_books |= set(zip(bm25_new["language"], bm25_new["book_id"]))

    # Join corpus manifest with book_stats for exclusion_step / n_words / body_words.
    merged = corpus.merge(
        stats[
            [
                "book_id",
                "language",
                "n_words",
                "body_words",
                "exclusion_step",
                "needs_ocr",
                "possible_duplicate",
                "in_analysis_set",
            ]
        ],
        on=["book_id", "language"],
        how="left",
        suffixes=("", "_stats"),
    )
    # Prefer stats in_analysis_set if present.
    if "in_analysis_set_stats" in merged.columns:
        merged["in_analysis_set"] = merged["in_analysis_set_stats"].fillna(
            merged["in_analysis_set"]
        )

    rows = []
    for _, r in merged.iterrows():
        lang = r["language"]
        bid = r["book_id"]
        in_set = str(r.get("in_analysis_set", "")).lower() in {"true", "1"}
        exclusion_step = str(r.get("exclusion_step") or "").strip()
        processing = str(r.get("processing_status") or "").strip()
        n_words = pd.to_numeric(r.get("n_words"), errors="coerce")
        body = pd.to_numeric(r.get("body_words"), errors="coerce")
        if pd.isna(body):
            body = pd.to_numeric(r.get("number_of_words"), errors="coerce")
        eligible_w = float(pass_words.get(lang, {}).get(bid, 0) or 0)
        eligible_n = int(pass_counts.get(lang, {}).get(bid, 0) or 0)
        body_f = float(body) if pd.notna(body) else 0.0
        raw_f = float(n_words) if pd.notna(n_words) else 0.0
        excl_w = max(0.0, body_f - eligible_w)
        n_lex_pass = int(lex_pass.get((lang, bid), 0) or 0)
        possible_dup = str(r.get("possible_duplicate", "")).lower() in {"true", "1"} or (
            "possible_duplicate" in str(r.get("qc_notes", "")).lower()
        )
        status = _status_vocab(
            in_set=in_set,
            exclusion_step=exclusion_step,
            processing_status=processing,
            n_lexical_passages=n_lex_pass,
            possible_duplicate=possible_dup and in_set,
        )
        # Override: if in set and possible dup, keep match status as primary with note.
        if in_set and n_lex_pass > 0:
            inclusion_note = "included"
        elif in_set:
            inclusion_note = "included_zero_matches"
        else:
            inclusion_note = "excluded"
        if in_set and possible_dup:
            # Flag only; book remains in the analytical population.
            inclusion_note = inclusion_note  # kept for clarity; pipeline_status carries potential_duplicate

        bm25_status = "processed" if (lang, bid) in bm25_books else (
            "eligible_no_bm25_hit" if in_set else "not_in_retrieval"
        )
        lex_status = (
            "processed_with_matches"
            if n_lex_pass > 0
            else ("processed_zero_matches" if in_set else "not_in_retrieval")
        )

        rows.append(
            {
                "book_id": bid,
                "work_id": r.get("work_id", ""),
                "language": lang,
                "title": r.get("title", ""),
                "author": r.get("author", ""),
                "publication_year": r.get("publication_year", "") or r.get("edition_year", ""),
                "primary_group": r.get("primary_group", ""),
                "source_file_availability": (
                    "missing"
                    if processing == "missing_file" or exclusion_step == "needs_ocr"
                    else "available"
                ),
                "text_extraction_status": processing or ("ok" if in_set else exclusion_step),
                "raw_word_count": raw_f,
                "cleaned_body_word_count": body_f,
                "eligible_passage_word_count": eligible_w,
                "excluded_word_count": excl_w,
                "eligible_paragraph_count": eligible_n,
                "lexical_retrieval_status": lex_status,
                "bm25_processing_status": bm25_status,
                "n_retrieved_lexical_passages": n_lex_pass,
                "inclusion_status": inclusion_note,
                "pipeline_status": status,
                "exclusion_step": exclusion_step,
                "recommended_action": _recommended_action(exclusion_step, processing, in_set),
            }
        )

    audit = pd.DataFrame(rows)

    # Unexpected: readable text, empty exclusion, not in analysis set.
    unexpected = audit[
        (audit["inclusion_status"] == "excluded")
        & (audit["exclusion_step"] == "")
        & (audit["source_file_availability"] == "available")
    ]
    if len(unexpected):
        raise RuntimeError(
            f"Found {len(unexpected)} books with available text and empty exclusion_step "
            f"but not in analysis set: {unexpected['book_id'].tolist()[:20]}"
        )

    funnel_rows = []
    for lang in ("en", "de"):
        sub = audit[audit["language"] == lang]
        listed = len(sub)
        text_ok = int((sub["source_file_availability"] == "available").sum())
        preprocessed = int(
            (
                (sub["inclusion_status"].isin(["included", "included_zero_matches"]))
                | (sub["exclusion_step"].isin(["short_text", "surplus_duplicate", "role_conflict"]))
            ).sum()
        )
        # Books that reached extraction successfully enough for stats.
        in_set_mask = sub["inclusion_status"].isin(["included", "included_zero_matches"])
        in_retrieval = int(in_set_mask.sum())
        with_matches = int((in_set_mask & (sub["n_retrieved_lexical_passages"] > 0)).sum())
        zero_matches = int((in_set_mask & (sub["n_retrieved_lexical_passages"] == 0)).sum())
        excl = int((sub["inclusion_status"] == "excluded").sum())
        by_reason = sub.loc[sub["inclusion_status"] == "excluded", "exclusion_step"].value_counts()
        n_potential_dup_in_set = int(
            (in_set_mask & (sub["pipeline_status"] == "potential_duplicate")).sum()
        )
        funnel_rows.append(
            {
                "language": lang,
                "language_label": LANG_LABEL[lang],
                "listed_books": listed,
                "books_with_available_text": text_ok,
                "successfully_preprocessed_or_ruled": preprocessed,
                "books_included_in_retrieval": in_retrieval,
                "books_with_at_least_one_match": with_matches,
                "books_without_matches": zero_matches,
                "books_excluded": excl,
                "analysis_books_flagged_potential_duplicate": n_potential_dup_in_set,
                "excluded_needs_ocr": int(by_reason.get("needs_ocr", 0)),
                "excluded_short_text": int(by_reason.get("short_text", 0)),
                "excluded_surplus_duplicate": int(by_reason.get("surplus_duplicate", 0)),
                "excluded_role_conflict": int(by_reason.get("role_conflict", 0)),
                "total_eligible_passage_words": float(sub.loc[
                    sub["inclusion_status"].isin(["included", "included_zero_matches"]),
                    "eligible_passage_word_count",
                ].sum()),
                "total_cleaned_body_words": float(sub.loc[
                    sub["inclusion_status"].isin(["included", "included_zero_matches"]),
                    "cleaned_body_word_count",
                ].sum()),
            }
        )
    funnel = pd.DataFrame(funnel_rows)
    return audit, funnel


def plot_pipeline_funnel(funnel: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        ("listed_books", "Listed"),
        ("books_with_available_text", "Text available"),
        ("books_included_in_retrieval", "In retrieval"),
        ("books_with_at_least_one_match", "With matches"),
        ("books_without_matches", "Zero matches"),
        ("books_excluded", "Excluded"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, lang in zip(axes, ("en", "de")):
        row = funnel[funnel["language"] == lang].iloc[0]
        labels = [s[1] for s in stages]
        vals = [int(row[s[0]]) for s in stages]
        color = LANGUAGE_COLORS[LANG_LABEL[lang]]
        ax.barh(labels[::-1], vals[::-1], color=color)
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Books")
        for i, v in enumerate(vals[::-1]):
            ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    fig.suptitle("Corpus pipeline coverage by language")
    fig.tight_layout()
    fig.savefig(out_dir / "01_pipeline_funnel.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_pipeline_funnel.svg", bbox_inches="tight")
    plt.close(fig)


def write_coverage_audit_doc(inventory: dict, funnel: pd.DataFrame, audit: pd.DataFrame) -> Path:
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / "corpus_coverage_audit.md"
    hash_ok = inventory["book_stats_hash_agreement"]
    lines = [
        "# Corpus coverage audit (analysis v2)",
        "",
        f"Generated: `{inventory['generated_at']}`",
        f"Source commit: `{inventory['source_commit']}`",
        f"Corpus snapshot: `{inventory['corpus_snapshot_id']}`",
        f"Lexicon version: `{inventory['lexicon_version']}`",
        f"Retrieval execution: `{inventory['retrieval_execution_id']}`",
        f"Evaluation infrastructure: `{inventory['evaluation_infrastructure_version']}`",
        "",
        "## Snapshot vs retrieval agreement",
        "",
        "Lexical and BM25 v0.3 hit tables were executed against the same analysis-set book-stats",
        "hashes that the v1 candidate snapshot freezes. v0.4 and v0.5 did not re-run retrieval.",
        "",
        "| Language | book_stats sha256 matches lexical v0.3 | matches snapshot meta |",
        "|---|---|---|",
    ]
    for lang in ("en", "de"):
        h = hash_ok[lang]
        lines.append(
            f"| {lang} | {h['matches_lexical_v03']} | {h['matches_snapshot']} |"
        )
    # Simple markdown table without requiring the tabulate package.
    cols = list(funnel.columns)
    funnel_md = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in funnel.iterrows():
        funnel_md.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines += [
        "",
        "## Pipeline funnel",
        "",
        *funnel_md,
        "",
        "## Exclusion rules",
        "",
        "Exclusions come from `exclusion_step` in `{en,de}_book_stats.csv` (first match wins):",
        "`needs_ocr`, `short_text` (<10,000 words), `role_conflict`, `surplus_duplicate`.",
        "Books with zero lexical matches remain in the analysis population.",
        "Fulltext catalogue `negative` roles were never part of this analysis corpus.",
        "",
        f"Audit rows: {len(audit)}. Unexpected empty-exclusion available-text books: 0.",
        "",
        "Machine-readable inventory: `data/processed/input_inventory_v2.json`.",
        "Book table: `reports/tables/02_book_coverage_audit.csv`.",
        "Funnel table: `reports/tables/02_pipeline_funnel.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    log = setup_logger(__file__)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    log.info("Building input inventory")
    inventory = build_input_inventory()
    inv_path = DATA_PROCESSED / "input_inventory_v2.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    log.info(f"Wrote {inv_path} ({len(inventory['inputs'])} inputs)")

    for lang, h in inventory["book_stats_hash_agreement"].items():
        if not h["matches_lexical_v03"]:
            raise RuntimeError(f"{lang}: book_stats hash does not match lexical v0.3 meta")
        if not h["matches_snapshot"]:
            raise RuntimeError(f"{lang}: book_stats hash does not match snapshot meta")
    log.info("Book-stats hashes agree with lexical v0.3 and snapshot")

    log.info("Building book coverage audit")
    audit, funnel = build_book_coverage_audit()
    audit.to_csv(TABLES / "02_book_coverage_audit.csv", index=False)
    funnel.to_csv(TABLES / "02_pipeline_funnel.csv", index=False)
    plot_pipeline_funnel(funnel, FIGURES)
    doc = write_coverage_audit_doc(inventory, funnel, audit)
    log.info(f"Wrote coverage tables, funnel figure, and {doc}")
    en = funnel.loc[funnel.language == "en"].iloc[0]
    de = funnel.loc[funnel.language == "de"].iloc[0]
    log.info(
        f"EN listed={int(en.listed_books)} analysis={int(en.books_included_in_retrieval)} "
        f"excl={int(en.books_excluded)} | DE listed={int(de.listed_books)} "
        f"analysis={int(de.books_included_in_retrieval)} excl={int(de.books_excluded)}"
    )


if __name__ == "__main__":
    main()
