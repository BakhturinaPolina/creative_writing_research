"""Comparative retrieval report across baselines A/B/C."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED, ROOT
from src.retrieval.paragraphs import load_book_stats
from src.retrieval.run_meta import base_run_meta, write_json

OUT_DIR = DATA_INTERIM / "retrieval"


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _concept_stats(df: pd.DataFrame, body_words: dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in df.groupby("concept_id"):
        n_assign = len(g)
        n_pass = g["passage_id"].nunique() if "passage_id" in g.columns else 0
        n_books = g["book_id"].nunique()
        by_group = g["primary_group"].value_counts().to_dict() if "primary_group" in g.columns else {}
        lang = g["language"].iloc[0] if "language" in g.columns else g.get("lang", pd.Series([""])).iloc[0]
        bw = body_words.get(str(lang), 0.0)
        per_10k = (n_assign / bw * 10000.0) if bw else 0.0
        rows.append(
            {
                "concept_id": cid,
                "language": lang,
                "n_assignments": n_assign,
                "n_unique_passages": n_pass,
                "n_books": n_books,
                "per_10k_body_words": round(per_10k, 4),
                "by_primary_group": json.dumps(by_group, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_report(log) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body_words = {}
    corpus_rows = []
    for lang in ("en", "de"):
        stats = load_book_stats(lang)
        a = stats[stats["in_analysis_set"]]
        bw = pd.to_numeric(a.get("body_words"), errors="coerce").fillna(0).sum()
        body_words[lang] = float(bw)
        for grp, g in a.groupby("primary_group"):
            corpus_rows.append(
                {
                    "language": lang,
                    "primary_group": grp,
                    "n_books": len(g),
                    "body_words": int(pd.to_numeric(g.get("body_words"), errors="coerce").fillna(0).sum()),
                }
            )
        corpus_rows.append(
            {
                "language": lang,
                "primary_group": "ALL",
                "n_books": len(a),
                "body_words": int(bw),
            }
        )
    pd.DataFrame(corpus_rows).to_csv(OUT_DIR / "retrieval_corpus_denominator.csv", index=False)

    baselines = {
        "A_v01": {
            "en": DATA_PROCESSED / "pov_baseline_a_v01_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_a_v01_de.csv",
        },
        "B_v02_lexical": {
            "en": DATA_PROCESSED / "pov_baseline_b_v02_lexical_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_b_v02_lexical_de.csv",
        },
        "C_bm25": {
            "en": DATA_PROCESSED / "pov_baseline_c_bm25_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_c_bm25_de.csv",
        },
        "C_bm25_only_unvalidated": {
            "en": DATA_PROCESSED / "pov_baseline_c_bm25_only_unvalidated_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_c_bm25_only_unvalidated_de.csv",
        },
    }

    concept_tables = []
    summary_rows = []
    for name, paths in baselines.items():
        for lang, path in paths.items():
            df = _read(path)
            if "language" not in df.columns and not df.empty:
                df["language"] = lang
            if "passage_id" not in df.columns and not df.empty and "book_id" in df.columns:
                df["passage_id"] = (
                    df["book_id"] + ":c" + df["chapter_idx"].astype(str) + ":p" + df["para_idx"].astype(str)
                )
            cs = _concept_stats(df, body_words)
            if not cs.empty:
                cs["baseline"] = name
                concept_tables.append(cs)
            summary_rows.append(
                {
                    "baseline": name,
                    "language": lang,
                    "n_rows": len(df),
                    "n_unique_passages": int(df["passage_id"].nunique()) if len(df) and "passage_id" in df else 0,
                    "n_concepts": int(df["concept_id"].nunique()) if len(df) else 0,
                    "path": str(path.relative_to(ROOT)) if path.is_file() else "",
                }
            )

    concept_df = pd.concat(concept_tables, ignore_index=True) if concept_tables else pd.DataFrame()
    concept_df.to_csv(OUT_DIR / "retrieval_concept_stats.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "retrieval_baseline_summary.csv", index=False)

    # Deduplicated passage-level table linked to Baseline B passage–concept rows
    for lang in ("en", "de"):
        b = _read(baselines["B_v02_lexical"][lang])
        if b.empty:
            continue
        if "passage_id" not in b.columns:
            b["passage_id"] = (
                b["book_id"] + ":c" + b["chapter_idx"].astype(str) + ":p" + b["para_idx"].astype(str)
            )
        concepts = (
            b.groupby("passage_id")["concept_id"]
            .agg(lambda s: "; ".join(sorted(set(s))))
            .rename("concept_ids")
        )
        n_c = b.groupby("passage_id")["concept_id"].nunique().rename("n_concepts")
        meta_cols = [
            c
            for c in (
                "passage_id",
                "book_id",
                "language",
                "title",
                "author",
                "edition_year",
                "corpus_role",
                "primary_group",
                "chapter_idx",
                "para_idx",
            )
            if c in b.columns
        ]
        passages = b[meta_cols].drop_duplicates("passage_id").set_index("passage_id")
        passages = passages.join(concepts).join(n_c).reset_index()
        passages.to_csv(DATA_PROCESSED / f"pov_baseline_b_passages_{lang}.csv", index=False)
        log.info(f"wrote passage-level table for {lang}: {len(passages)} unique passages")

    # Overlap B ∩ C (passage_id, concept_id)
    overlap_rows = []
    for lang in ("en", "de"):
        b = _read(baselines["B_v02_lexical"][lang])
        c = _read(baselines["C_bm25"][lang])
        if b.empty or c.empty:
            continue
        if "passage_id" not in b.columns:
            b["passage_id"] = b["book_id"] + ":c" + b["chapter_idx"].astype(str) + ":p" + b["para_idx"].astype(str)
        if "passage_id" not in c.columns:
            c["passage_id"] = c["book_id"] + ":c" + c["chapter_idx"].astype(str) + ":p" + c["para_idx"].astype(str)
        kb = set(zip(b["passage_id"], b["concept_id"]))
        kc = set(zip(c["passage_id"], c["concept_id"]))
        overlap_rows.append(
            {
                "language": lang,
                "n_b": len(kb),
                "n_c": len(kc),
                "intersection": len(kb & kc),
                "b_only": len(kb - kc),
                "c_only": len(kc - kb),
                "jaccard": round(len(kb & kc) / len(kb | kc), 4) if (kb | kc) else 0.0,
            }
        )
    pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "retrieval_baseline_overlap.csv", index=False)

    # Concepts per passage (Baseline B)
    conc_rows = []
    for lang in ("en", "de"):
        b = _read(baselines["B_v02_lexical"][lang])
        if b.empty:
            continue
        if "passage_id" not in b.columns:
            b["passage_id"] = b["book_id"] + ":c" + b["chapter_idx"].astype(str) + ":p" + b["para_idx"].astype(str)
        vc = b.groupby("passage_id")["concept_id"].nunique()
        conc_rows.append(
            {
                "language": lang,
                "max_concepts_per_passage": int(vc.max()) if len(vc) else 0,
                "mean_concepts_per_passage": round(float(vc.mean()), 3) if len(vc) else 0.0,
                "passages_with_ge_5_concepts": int((vc >= 5).sum()),
                "passages_with_ge_10_concepts": int((vc >= 10).sum()),
            }
        )
    pd.DataFrame(conc_rows).to_csv(OUT_DIR / "retrieval_concepts_per_passage.csv", index=False)

    # Selection-rule distribution (B + C)
    rule_rows = []
    for name in ("B_v02_lexical", "C_bm25"):
        for lang, path in baselines[name].items():
            df = _read(path)
            if df.empty or "selection_rule" not in df.columns:
                continue
            for rule, n in df["selection_rule"].value_counts().items():
                rule_rows.append({"baseline": name, "language": lang, "selection_rule": rule, "n": int(n)})
    pd.DataFrame(rule_rows).to_csv(OUT_DIR / "retrieval_selection_rules.csv", index=False)

    # Book concentration
    conc_book = []
    for lang in ("en", "de"):
        b = _read(baselines["B_v02_lexical"][lang])
        if b.empty:
            continue
        top = b["book_id"].value_counts().head(10)
        for bid, n in top.items():
            conc_book.append({"language": lang, "book_id": bid, "n_assignments": int(n)})
    pd.DataFrame(conc_book).to_csv(OUT_DIR / "retrieval_book_concentration_top10.csv", index=False)

    # Markdown report
    md = []
    md.append("# Comparative PsyPer lexical retrieval report")
    md.append("")
    md.append("Lexical baseline only — **not** manually validated. Do not treat candidate counts as concept prevalence.")
    md.append("")
    md.append("## Corpus denominators")
    md.append("")
    md.append("| language | primary_group | n_books | body_words |")
    md.append("|---|---|---:|---:|")
    for r in corpus_rows:
        md.append(f"| {r['language']} | {r['primary_group']} | {r['n_books']} | {r['body_words']:,} |")
    md.append("")
    md.append("## Baseline summary")
    md.append("")
    md.append("| baseline | language | rows | unique passages | concepts |")
    md.append("|---|---|---:|---:|---:|")
    for r in summary_rows:
        md.append(
            f"| {r['baseline']} | {r['language']} | {r['n_rows']} | {r['n_unique_passages']} | {r['n_concepts']} |"
        )
    md.append("")
    md.append("## Overlap Baseline B ∩ C")
    md.append("")
    for r in overlap_rows:
        md.append(
            f"- **{r['language']}**: Jaccard={r['jaccard']} "
            f"(B={r['n_b']}, C={r['n_c']}, ∩={r['intersection']})"
        )
    md.append("")
    md.append("## Concepts per passage (Baseline B)")
    md.append("")
    for r in conc_rows:
        md.append(
            f"- **{r['language']}**: max={r['max_concepts_per_passage']}, "
            f"mean={r['mean_concepts_per_passage']}, "
            f"≥10 concepts: {r['passages_with_ge_10_concepts']}"
        )
    md.append("")
    md.append("## Likely lexical-insufficient concepts")
    md.append("")
    md.append(
        "Concepts with zero or very few Baseline-B hits relative to corpus size "
        "are priority targets for later semantic retrieval."
    )
    if not concept_df.empty:
        bdf = concept_df[concept_df["baseline"] == "B_v02_lexical"]
        for lang in ("en", "de"):
            sub = bdf[bdf["language"] == lang].sort_values("n_assignments")
            low = sub.head(8)
            md.append(f"### {lang} (lowest assignment counts)")
            for _, row in low.iterrows():
                md.append(
                    f"- `{row['concept_id']}`: {row['n_assignments']} assignments, "
                    f"{row['n_unique_passages']} passages, {row['per_10k_body_words']}/10k words"
                )
            md.append("")
    md.append("## Before vs after (reference commit 83706dd)")
    md.append("")
    md.append(
        "Prior run (commit family around streaming BM25 fix): EN/DE each had exactly "
        "1,800 candidate rows (50×36) with heavy regex-only high evidence leakage. "
        "This report replaces that with concept-specific evidence and uncapped Baseline B."
    )
    md.append("")
    md.append("## Methodological limitations")
    md.append("")
    md.append("- Lexical hits ≠ validated concept discussion; manual review is required.")
    md.append("- BM25 scores are not comparable across languages or concepts.")
    md.append("- Demoted/zero-hit terms remain in the master lexicon for semantic retrieval.")
    md.append("- Full passages are local/private; public CSVs are pointers only.")
    md.append("")

    report_path = OUT_DIR / "retrieval_comparative_report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(OUT_DIR / "retrieval_comparative_report.meta.json", base_run_meta(kind="comparative_report"))
    log.info(f"wrote {report_path}")
    return report_path


def build_report_v03(log) -> Path:
    """Compare v0.2 vs algo-fix vs v0.3 lexical/BM25 baselines."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body_words = {}
    corpus_rows = []
    for lang in ("en", "de"):
        stats = load_book_stats(lang)
        a = stats[stats["in_analysis_set"]]
        bw = pd.to_numeric(a.get("body_words"), errors="coerce").fillna(0).sum()
        body_words[lang] = float(bw)
        for grp, g in a.groupby("primary_group"):
            corpus_rows.append(
                {
                    "language": lang,
                    "primary_group": grp,
                    "n_books": len(g),
                    "body_words": int(
                        pd.to_numeric(g.get("body_words"), errors="coerce").fillna(0).sum()
                    ),
                }
            )
        corpus_rows.append(
            {
                "language": lang,
                "primary_group": "ALL",
                "n_books": len(a),
                "body_words": int(bw),
            }
        )

    baselines = {
        "B_v02_lexical": {
            "en": DATA_PROCESSED / "pov_baseline_b_v02_lexical_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_b_v02_lexical_de.csv",
        },
        "B_v02_algo_fix": {
            "en": DATA_PROCESSED / "pov_baseline_b_v02_algo_fix_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_b_v02_algo_fix_de.csv",
        },
        "B_v03_lexical": {
            "en": DATA_PROCESSED / "pov_baseline_b_v03_lexical_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv",
        },
        "C_v03_bm25": {
            "en": DATA_PROCESSED / "pov_baseline_c_v03_bm25_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_c_v03_bm25_de.csv",
        },
        "C_v03_bm25_only": {
            "en": DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_en.csv",
            "de": DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_de.csv",
        },
    }

    def _keys(df: pd.DataFrame) -> set[tuple[str, str]]:
        if df.empty:
            return set()
        if "passage_id" not in df.columns:
            df = df.copy()
            df["passage_id"] = (
                df["book_id"]
                + ":c"
                + df["chapter_idx"].astype(str)
                + ":p"
                + df["para_idx"].astype(str)
            )
        return set(zip(df["passage_id"], df["concept_id"]))

    concept_tables = []
    summary_rows = []
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for name, paths in baselines.items():
        loaded[name] = {}
        for lang, path in paths.items():
            df = _read(path)
            if "language" not in df.columns and not df.empty:
                df["language"] = lang
            if "passage_id" not in df.columns and not df.empty and "book_id" in df.columns:
                df["passage_id"] = (
                    df["book_id"]
                    + ":c"
                    + df["chapter_idx"].astype(str)
                    + ":p"
                    + df["para_idx"].astype(str)
                )
            loaded[name][lang] = df
            cs = _concept_stats(df, body_words)
            if not cs.empty:
                cs["baseline"] = name
                concept_tables.append(cs)
            summary_rows.append(
                {
                    "baseline": name,
                    "language": lang,
                    "n_rows": len(df),
                    "n_unique_passages": int(df["passage_id"].nunique())
                    if len(df) and "passage_id" in df
                    else 0,
                    "n_concepts": int(df["concept_id"].nunique()) if len(df) else 0,
                    "path": str(path.relative_to(ROOT)) if path.is_file() else "",
                }
            )

    concept_df = pd.concat(concept_tables, ignore_index=True) if concept_tables else pd.DataFrame()
    concept_df.to_csv(OUT_DIR / "retrieval_concept_stats_v03.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "retrieval_baseline_summary_v03.csv", index=False)

    # Per-concept delta v0.2 → algo_fix → v0.3
    delta_rows = []
    for lang in ("en", "de"):
        b02 = loaded["B_v02_lexical"][lang]
        bfix = loaded["B_v02_algo_fix"][lang]
        b03 = loaded["B_v03_lexical"][lang]
        concepts = sorted(
            set(b02["concept_id"] if len(b02) else [])
            | set(bfix["concept_id"] if len(bfix) else [])
            | set(b03["concept_id"] if len(b03) else [])
        )
        # Also include lexicon concepts with zero hits
        from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon

        lex = load_lexicon(LEXICON_PATH_V03)
        concepts = sorted(set(concepts) | set(lex.concepts[lang].keys()))

        k02 = _keys(b02)
        kfix = _keys(bfix)
        k03 = _keys(b03)

        def _count(df, cid):
            if df.empty:
                return 0, 0, 0
            g = df[df["concept_id"] == cid]
            return len(g), int(g["passage_id"].nunique()), int(g["book_id"].nunique())

        for cid in concepts:
            n02, p02, bk02 = _count(b02, cid)
            nfix, pfix, bkfix = _count(bfix, cid)
            n03, p03, bk03 = _count(b03, cid)
            gained_algo = len({k for k in kfix if k[1] == cid} - {k for k in k02 if k[1] == cid})
            lost_algo = len({k for k in k02 if k[1] == cid} - {k for k in kfix if k[1] == cid})
            gained_vocab = len({k for k in k03 if k[1] == cid} - {k for k in kfix if k[1] == cid})
            lost_vocab = len({k for k in kfix if k[1] == cid} - {k for k in k03 if k[1] == cid})
            bw = body_words.get(lang, 0.0)
            delta_rows.append(
                {
                    "concept_id": cid,
                    "language": lang,
                    "n_v02": n02,
                    "n_algo_fix": nfix,
                    "n_v03": n03,
                    "passages_v02": p02,
                    "passages_v03": p03,
                    "books_v02": bk02,
                    "books_v03": bk03,
                    "gained_algo_fix": gained_algo,
                    "lost_algo_fix": lost_algo,
                    "gained_vocab_or_tiers": gained_vocab,
                    "lost_vocab_or_tiers": lost_vocab,
                    "per_10k_v02": round((n02 / bw * 10000.0) if bw else 0.0, 4),
                    "per_10k_v03": round((n03 / bw * 10000.0) if bw else 0.0, 4),
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "retrieval_v02_v03_concept_delta.csv", index=False)

    # Coverage audit (lexicon structure)
    from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon, resolve_evidence_tier
    from src.retrieval.regex_map import load_regex_map
    from src.common.paths import CONFIG

    lex = load_lexicon(LEXICON_PATH_V03)
    rmap = load_regex_map(CONFIG / "regex_concept_map_v03.yml")
    cov_rows = []
    for lang in ("en", "de"):
        for cid, ct in lex.concepts[lang].items():
            n_strong = n_supp = 0
            for origin, terms in (
                ("academic", ct.academic),
                ("craft", ct.craft),
                ("paraphrase", ct.paraphrases),
            ):
                for term in terms:
                    tier = resolve_evidence_tier(lang, origin, term, ct.evidence_overrides)
                    if tier.startswith("strong"):
                        n_strong += 1
                    elif tier == "supporting":
                        n_supp += 1
            hits = 0
            if not concept_df.empty:
                hit_rows = concept_df[
                    (concept_df["baseline"] == "B_v03_lexical")
                    & (concept_df["language"] == lang)
                    & (concept_df["concept_id"] == cid)
                ]
                hits = int(hit_rows["n_assignments"].sum()) if len(hit_rows) else 0
            cov_rows.append(
                {
                    "concept_id": cid,
                    "language": lang,
                    "n_academic": len(ct.academic),
                    "n_craft": len(ct.craft),
                    "n_paraphrases": len(ct.paraphrases),
                    "n_context": len(ct.context),
                    "n_regex": len(rmap.names_for_concept(lang, cid)),
                    "n_strong_cues": n_strong,
                    "n_supporting_cues": n_supp,
                    "n_overrides": len(ct.evidence_overrides),
                    "corpus_hits_v03": hits,
                    "psyper_related_tags": ct.psyper_tags_joined(),
                }
            )
    pd.DataFrame(cov_rows).to_csv(OUT_DIR / "retrieval_coverage_audit_v03.csv", index=False)

    # Overlaps
    overlap_rows = []
    for lang in ("en", "de"):
        pairs = [
            ("B_v02_lexical", "B_v02_algo_fix"),
            ("B_v02_algo_fix", "B_v03_lexical"),
            ("B_v02_lexical", "B_v03_lexical"),
            ("B_v03_lexical", "C_v03_bm25"),
        ]
        for a, b in pairs:
            ka, kb = _keys(loaded[a][lang]), _keys(loaded[b][lang])
            union = ka | kb
            overlap_rows.append(
                {
                    "language": lang,
                    "a": a,
                    "b": b,
                    "n_a": len(ka),
                    "n_b": len(kb),
                    "intersection": len(ka & kb),
                    "a_only": len(ka - kb),
                    "b_only": len(kb - ka),
                    "jaccard": round(len(ka & kb) / len(union), 4) if union else 0.0,
                }
            )
    pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "retrieval_baseline_overlap_v03.csv", index=False)

    # Multi-concept passages
    multi_rows = []
    for lang in ("en", "de"):
        b = loaded["B_v03_lexical"][lang]
        if b.empty:
            continue
        vc = b.groupby("passage_id")["concept_id"].nunique()
        multi_rows.append(
            {
                "language": lang,
                "max_concepts_per_passage": int(vc.max()) if len(vc) else 0,
                "mean_concepts_per_passage": round(float(vc.mean()), 3) if len(vc) else 0.0,
                "passages_with_ge_10_concepts": int((vc >= 10).sum()),
            }
        )

    # Markdown
    md = []
    md.append("# Comparative PsyPer retrieval report (v0.2 vs v0.3)")
    md.append("")
    md.append(
        "Lexical baseline only — **not** manually validated. "
        "Increased candidate counts are not evidence of improved recall without relevance judgments."
    )
    md.append("")
    md.append("## Corpus denominators")
    md.append("")
    md.append("| language | primary_group | n_books | body_words |")
    md.append("|---|---|---:|---:|")
    for r in corpus_rows:
        md.append(
            f"| {r['language']} | {r['primary_group']} | {r['n_books']} | {r['body_words']:,} |"
        )
    md.append("")
    md.append("## Baseline summary")
    md.append("")
    md.append("| baseline | language | rows | unique passages | concepts |")
    md.append("|---|---|---:|---:|---:|")
    for r in summary_rows:
        md.append(
            f"| {r['baseline']} | {r['language']} | {r['n_rows']} | "
            f"{r['n_unique_passages']} | {r['n_concepts']} |"
        )
    md.append("")
    md.append("## Algorithm vs vocabulary contributions")
    md.append("")
    md.append(
        "`B_v02_algo_fix` uses the v0.2 vocabulary with the corrected shared prefilter/evidence "
        "pipeline. Gains over `B_v02_lexical` are algorithmic; further gains in `B_v03_lexical` "
        "reflect vocabulary/tier changes."
    )
    md.append("")
    for lang in ("en", "de"):
        s = {r["baseline"]: r for r in summary_rows if r["language"] == lang}
        md.append(f"### {lang}")
        for key in ("B_v02_lexical", "B_v02_algo_fix", "B_v03_lexical"):
            if key in s:
                md.append(
                    f"- `{key}`: {s[key]['n_rows']} assignments, "
                    f"{s[key]['n_unique_passages']} passages, {s[key]['n_concepts']} concepts"
                )
        md.append("")

    md.append("## Concepts with largest absolute gains (v0.2 → v0.3)")
    md.append("")
    if not delta_df.empty:
        for lang in ("en", "de"):
            sub = delta_df[delta_df["language"] == lang].copy()
            sub["delta"] = sub["n_v03"] - sub["n_v02"]
            top = sub.sort_values("delta", ascending=False).head(10)
            md.append(f"### {lang}")
            for _, row in top.iterrows():
                md.append(
                    f"- `{row['concept_id']}`: {int(row['n_v02'])} → {int(row['n_v03'])} "
                    f"(algo +{int(row['gained_algo_fix'])}/−{int(row['lost_algo_fix'])}, "
                    f"vocab +{int(row['gained_vocab_or_tiers'])}/−{int(row['lost_vocab_or_tiers'])})"
                )
            md.append("")

    md.append("## Zero-hit / low-hit audit (v0.3 lexical)")
    md.append("")
    if not delta_df.empty:
        for lang in ("en", "de"):
            sub = delta_df[delta_df["language"] == lang].sort_values("n_v03")
            zeros = sub[sub["n_v03"] == 0]
            md.append(f"### {lang}: {len(zeros)} concepts with zero lexical hits")
            for _, row in zeros.iterrows():
                md.append(f"- `{row['concept_id']}`")
            low = sub[(sub["n_v03"] > 0)].head(8)
            md.append(f"Lowest non-zero ({lang}):")
            for _, row in low.iterrows():
                md.append(
                    f"- `{row['concept_id']}`: {int(row['n_v03'])} assignments "
                    f"({row['per_10k_v03']}/10k)"
                )
            md.append("")

    md.append("## Overlaps")
    md.append("")
    for r in overlap_rows:
        md.append(
            f"- **{r['language']}** {r['a']} ∩ {r['b']}: Jaccard={r['jaccard']} "
            f"(∩={r['intersection']}, a_only={r['a_only']}, b_only={r['b_only']})"
        )
    md.append("")
    md.append("## Multi-concept passages (v0.3 lexical)")
    md.append("")
    for r in multi_rows:
        md.append(
            f"- **{r['language']}**: max={r['max_concepts_per_passage']}, "
            f"mean={r['mean_concepts_per_passage']}, "
            f"≥10 concepts: {r['passages_with_ge_10_concepts']}"
        )
    md.append("")
    md.append("## Cross-linguistic coverage caveat")
    md.append("")
    md.append(
        "Before interpreting EN–DE frequency differences as literary or cultural, "
        "consult `retrieval_coverage_audit_v03.csv`. Unequal lexical coverage "
        "(especially paraphrases and DE morphology) may explain residual gaps."
    )
    md.append("")
    md.append("## Evaluation protocol (future)")
    md.append("")
    md.append(
        "Compare on the held-out manual sample (plus later semantic-only additions): "
        "A v0.1 · B v0.2 · B v0.3 · BM25 · dense · RRF · cross-encoder rerank. "
        "Do not claim scientific validation until judgments are collected."
    )
    md.append("")
    md.append("## Methodological limitations")
    md.append("")
    md.append("- Lexical hits ≠ validated concept discussion.")
    md.append("- Stratified candidate samples do not estimate absolute corpus-wide recall.")
    md.append("- Provisional lexicon additions require human confirmation.")
    md.append("- Full passages remain local/private; public CSVs are pointers only.")
    md.append("")

    report_path = OUT_DIR / "retrieval_comparative_report_v03.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "retrieval_comparative_report_v03.meta.json",
        base_run_meta(kind="comparative_report_v03"),
    )
    # Also refresh the interim pointer used by the open file if desired — keep v0.2 report intact
    log.info(f"wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v02", "v03"], default="v03")
    args = parser.parse_args()
    log = setup_logger(__file__)
    if args.version == "v03":
        build_report_v03(log)
    else:
        build_report(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()

