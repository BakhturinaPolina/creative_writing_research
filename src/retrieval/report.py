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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    log = setup_logger(__file__)
    build_report(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
