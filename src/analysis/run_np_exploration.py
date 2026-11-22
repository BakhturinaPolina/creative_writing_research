"""Generate tables and figures for the narrative-perspective exploration notebook.

Memory-conscious: avoids keeping both full passage manifests in RAM; uses
column subsets; frees large frames after each stage.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.narrative_perspective import (
    LANGUAGE_COLORS,
    LANG_LABEL,
    attach_concept_families,
    calculate_book_dispersion,
    calculate_concept_cooccurrence,
    calculate_family_frequencies,
    calculate_normalized_frequencies,
    compare_retrieval_methods,
    load_concept_ids,
    load_corpus_manifest,
    load_passage_manifest,
    load_retrieval_results,
    reproducibility_record,
    run_sensitivity_analysis,
    save_table,
    validate_retrieval_keys,
)
from src.common.logging_util import setup_logger
from src.common.paths import DATA_PROCESSED, ROOT

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures" / "01_narrative_perspective"


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--edition",
        choices=("v1", "v2"),
        default="v1",
        help="v1 writes reports/01_*; v2 delegates to run_np_v2 (reports/02_*).",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=0,
        help="Skip stages before this number (0=all). Use 11 to resume after heatmap.",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    args = parser.parse_args()
    if args.edition == "v2":
        from src.analysis.run_np_v2 import main as main_v2
        import sys

        # Rebuild argv for the v2 runner.
        sys.argv = [sys.argv[0]]
        if args.skip_bootstrap:
            sys.argv.append("--skip-bootstrap")
        if args.bootstrap_iters != 1000:
            sys.argv.extend(["--bootstrap-iters", str(args.bootstrap_iters)])
        main_v2()
        return
    log = setup_logger(__file__)
    _style()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stage = args.from_stage

    corpus = load_corpus_manifest()
    lex = load_retrieval_results("lexical_v03")
    concepts = load_concept_ids()
    a = corpus[corpus["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))].copy()
    a["number_of_words"] = pd.to_numeric(a["number_of_words"], errors="coerce").fillna(0)
    a["number_of_eligible_paragraphs"] = pd.to_numeric(
        a["number_of_eligible_paragraphs"], errors="coerce"
    ).fillna(0)

    if stage <= 1:
        overview_rows = []
        for lang in ("en", "de"):
            cl = corpus[corpus["language"] == lang]
            al = a[a["language"] == lang]
            overview_rows.append(
                {
                    "language": LANG_LABEL[lang],
                    "listed_books": len(cl),
                    "analysis_set_books": len(al),
                    "excluded_books": len(cl) - len(al),
                    "n_authors": al["author"].nunique(),
                    "words": int(al["number_of_words"].sum()),
                    "eligible_paragraphs": int(al["number_of_eligible_paragraphs"].sum()),
                }
            )
        overview = pd.DataFrame(overview_rows)
        save_table(overview, "corpus_overview", TABLES)
        comp = a.groupby(["language", "primary_group"], as_index=False)["number_of_words"].sum()
        comp["pct_words"] = comp.groupby("language")["number_of_words"].transform(
            lambda s: 100 * s / s.sum()
        )
        save_table(comp, "corpus_composition_by_group", TABLES)
        qc = pd.read_csv(DATA_PROCESSED / "corpus_quality_v1_candidate.csv")
        save_table(qc, "corpus_quality", TABLES)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        x = np.arange(len(overview))
        ax.bar(x - 0.2, overview["listed_books"], 0.4, label="Listed", color="#888888")
        ax.bar(
            x + 0.2,
            overview["analysis_set_books"],
            0.4,
            label="Analysis set",
            color=[LANGUAGE_COLORS[l] for l in overview["language"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(overview["language"])
        ax.set_ylabel("Number of books")
        ax.set_title("Corpus size by language")
        ax.legend()
        save_fig(fig, "01_corpus_size")

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        pivot = a.groupby(["language", "primary_group"]).size().unstack(fill_value=0)
        pivot.index = [LANG_LABEL[i] for i in pivot.index]
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Books")
        ax.set_title("Books by language and subcategory")
        ax.legend(title="primary_group", fontsize=8)
        plt.xticks(rotation=0)
        save_fig(fig, "02_books_by_subcategory")

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        wp = comp.pivot(index="primary_group", columns="language", values="pct_words").fillna(0)
        wp.columns = [LANG_LABEL[c] for c in wp.columns]
        wp.plot(
            kind="bar",
            ax=ax,
            color=[LANGUAGE_COLORS["English"], LANGUAGE_COLORS["German"]],
        )
        ax.set_ylabel("% of analysis-set words")
        ax.set_title("Word-based corpus composition")
        plt.xticks(rotation=25, ha="right")
        save_fig(fig, "03_word_composition")

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for lang, label in LANG_LABEL.items():
            vals = a.loc[a["language"] == lang, "number_of_words"]
            ax.hist(vals, bins=25, alpha=0.55, label=label, color=LANGUAGE_COLORS[label])
        ax.set_xlabel("Book length (words)")
        ax.set_ylabel("Number of books")
        ax.set_title("Book-length distributions")
        ax.legend()
        save_fig(fig, "04_book_length")

        # Paragraph lengths: load one language at a time, word_count only.
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for lang, label in LANG_LABEL.items():
            pm = load_passage_manifest(lang, usecols=["passage_id", "word_count"])
            vals = pd.to_numeric(pm["word_count"], errors="coerce").dropna()
            vals = vals[vals > 0]
            ax.hist(
                np.log10(vals),
                bins=35,
                alpha=0.55,
                label=label,
                color=LANGUAGE_COLORS[label],
            )
            del pm, vals
            gc.collect()
        ax.set_xlabel("log10(paragraph word count)")
        ax.set_ylabel("Number of paragraphs")
        ax.set_title("Paragraph-length distributions (log10 scale)")
        ax.legend()
        save_fig(fig, "05_paragraph_length_log")

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for lang, label in LANG_LABEL.items():
            al = a[a["language"] == lang]
            years = pd.to_numeric(al["publication_year"], errors="coerce")
            missing = int(years.isna().sum())
            known = years.dropna()
            ax.hist(
                known,
                bins=18,
                alpha=0.55,
                label=f"{label} dated n={len(known)}; missing={missing}",
                color=LANGUAGE_COLORS[label],
            )
        ax.set_xlabel("Publication year")
        ax.set_ylabel("Books")
        ax.set_title("Known publication years")
        ax.legend(fontsize=7)
        save_fig(fig, "06_publication_years")
        log.info("stage 1 corpus figures done")

    if stage <= 7:
        freq = attach_concept_families(
            calculate_normalized_frequencies(lex, corpus, concept_ids=concepts)
        )
        save_table(freq, "concept_frequencies", TABLES)
        fam_freq = calculate_family_frequencies(lex, corpus)
        save_table(fam_freq, "concept_family_frequencies", TABLES)

        fig, ax = plt.subplots(figsize=(9, 8))
        top = (
            freq.groupby("concept_id")["frequency_per_10000_words"].max().nlargest(20).index
        )
        sub = freq[freq["concept_id"].isin(top)]
        sns.barplot(
            data=sub,
            y="concept_id",
            x="frequency_per_10000_words",
            hue="language_label",
            palette={k: LANGUAGE_COLORS[k] for k in ("English", "German")},
            ax=ax,
        )
        ax.set_xlabel("Assignments per 10,000 words")
        ax.set_title("Normalized concept frequencies (top 20)")
        save_fig(fig, "07_concept_frequencies_top20")

        fig, axes = plt.subplots(1, 2, figsize=(11, 9))
        for ax, lang, label in zip(axes, ("en", "de"), ("English", "German")):
            s = freq[freq["language"] == lang].sort_values(
                "frequency_per_10000_words", ascending=True
            )
            ax.barh(s["concept_id"], s["frequency_per_10000_words"], color=LANGUAGE_COLORS[label])
            ax.set_title(label)
            ax.set_xlabel("per 10,000 words")
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Ranked frequency profiles (all 36 concepts)")
        fig.tight_layout()
        save_fig(fig, "08_ranked_profiles")

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(
            data=fam_freq,
            y="concept_family_label",
            x="frequency_per_10000_words",
            hue="language_label",
            palette={k: LANGUAGE_COLORS[k] for k in ("English", "German")},
            ax=ax,
        )
        ax.set_xlabel("Unique passages per 10,000 words (family-deduplicated)")
        ax.set_title("Concept-family frequency profiles")
        save_fig(fig, "09_family_frequencies")

        piv = freq.pivot(
            index="concept_id", columns="language", values="frequency_per_10000_words"
        )
        piv["RR"] = piv.apply(
            lambda r: (r["en"] / r["de"])
            if r["de"] > 0
            else (np.nan if r["en"] == 0 else np.nan),
            axis=1,
        )
        # Finite RR only; zeros handled explicitly in table (no pseudocounts).
        ordered = piv.dropna(subset=["RR"]).sort_values("RR")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(ordered["RR"], range(len(ordered)), color="#333333", s=18)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index, fontsize=7)
        ax.axvline(1.0, color="red", ls="--", label="RR=1")
        ax.set_xscale("log")
        ax.set_xlabel("RR = f_EN / f_DE (log; finite only)")
        ax.set_title("Cross-language frequency ratios")
        ax.legend()
        save_fig(fig, "10_frequency_ratios")
        log.info("stage 7 frequency figures done")
    else:
        freq = pd.read_csv(TABLES / "concept_frequencies.csv")
        fam_freq = pd.read_csv(TABLES / "concept_family_frequencies.csv")

    if stage <= 11:
        # Heatmap: vectorized group frequencies for top concepts only.
        top_c = (
            freq.groupby("concept_id")["frequency_per_10000_words"].max().nlargest(15).index
        )
        heat_rows = []
        for lang in ("en", "de"):
            for grp in ("fiction_manual", "pedagogy_genre_craft", "contrast"):
                books = a[(a["language"] == lang) & (a["primary_group"] == grp)]
                words = float(books["number_of_words"].sum())
                bids = set(books["book_id"])
                rsub = lex[
                    (lex["language"] == lang)
                    & (lex["book_id"].isin(bids))
                    & (lex["retrieval_concept_id"].isin(top_c))
                ]
                vc = rsub["retrieval_concept_id"].value_counts()
                for cid in top_c:
                    n = int(vc.get(cid, 0))
                    heat_rows.append(
                        {
                            "panel": f"{LANG_LABEL[lang][:2]}·{grp[:7]}",
                            "concept_id": cid,
                            "freq": (n / words * 10000) if words else 0.0,
                        }
                    )
        heat_p = pd.DataFrame(heat_rows).pivot(
            index="concept_id", columns="panel", values="freq"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_p, cmap="YlOrBr", ax=ax)
        ax.set_title("Concept frequency by language × category (top 15)")
        save_fig(fig, "11_heatmap_lang_group")
        log.info("stage 11 heatmap done")

    if stage <= 12:
        disp_rows = []
        book_level_rows = []
        focus_books = ("focalization_zero", "filter_words", "stance_narratorial")
        for lang in ("en", "de"):
            for cid in concepts:
                d = calculate_book_dispersion(lex, corpus, cid, lang)
                disp_rows.append({k: v for k, v in d.items() if k != "book_table"})
                if cid in focus_books:
                    bt = d["book_table"][
                        ["book_id", "language", "n_assign", "freq_per_10k", "number_of_words"]
                    ].copy()
                    bt["concept_id"] = cid
                    book_level_rows.append(bt)
                del d
        dispersion = pd.DataFrame(disp_rows)
        save_table(dispersion, "concept_dispersion", TABLES)
        save_table(pd.concat(book_level_rows, ignore_index=True), "book_level_frequencies", TABLES)

        m = dispersion.merge(
            freq[["concept_id", "language", "frequency_per_10000_words"]],
            on=["concept_id", "language"],
        )
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for lang, label in LANG_LABEL.items():
            s = m[m["language"] == lang]
            ax.scatter(
                s["frequency_per_10000_words"],
                s["pct_books_with_concept"],
                alpha=0.65,
                label=label,
                color=LANGUAGE_COLORS[label],
                s=28,
            )
        ax.set_xlabel("Frequency per 10,000 words")
        ax.set_ylabel("% of books with ≥1 assignment")
        ax.set_title("Concept frequency versus book prevalence")
        ax.legend()
        save_fig(fig, "12_freq_vs_prevalence")
        del dispersion, m, book_level_rows
        gc.collect()
        log.info("stage 12 dispersion done")

    if stage <= 13:
        for lang in ("en", "de"):
            counts, pairs = calculate_concept_cooccurrence(lex, lang, min_support=5)
            counts.to_csv(TABLES / f"concept_cooccurrence_{lang}.csv")
            pairs.to_csv(TABLES / f"concept_cooccurrence_pairs_{lang}.csv", index=False)
            support = (
                lex.loc[lex["language"] == lang]
                .groupby("retrieval_concept_id")["passage_id"]
                .nunique()
            )
            keep = [c for c in concepts if int(support.get(c, 0)) >= 10][:18]
            if keep:
                fig, ax = plt.subplots(figsize=(8, 7))
                sns.heatmap(np.log1p(counts.loc[keep, keep]), ax=ax, cmap="Blues")
                ax.set_title(
                    f"{LANG_LABEL[lang]} co-occurrence (log1p; concepts with ≥10 passages)"
                )
                ax.tick_params(labelsize=7)
                save_fig(fig, f"13_cooccurrence_{lang}")
            del counts, pairs
            gc.collect()
        log.info("stage 13 cooccurrence done")

    if stage <= 14:
        bnew = load_retrieval_results("bm25_only_v03")
        bq = load_retrieval_results("bm25_qualified_v03")
        comp_m = compare_retrieval_methods(lex, bnew, bq)
        save_table(comp_m, "retrieval_complementarity", TABLES)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        x = np.arange(len(comp_m))
        ax.bar(
            x - 0.2,
            comp_m["n_lexical_passages"],
            0.4,
            label="Lexical passages",
            color="#3976C6",
        )
        ax.bar(
            x + 0.2,
            comp_m["n_genuinely_new_bm25_passages"],
            0.4,
            label="Genuinely new BM25",
            color="#CC8C42",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(comp_m["language_label"])
        ax.set_ylabel("Unique passages")
        ax.set_title("Lexical vs genuinely new BM25 passages")
        ax.legend()
        save_fig(fig, "14_complementarity")
        del bnew, bq
        gc.collect()
        log.info("stage 14 complementarity done")

    if stage <= 15:
        lex02 = load_retrieval_results("lexical_v02")
        lex_fix = load_retrieval_results("lexical_algo_fix")

        def _keys(df, lang):
            s = df[df["language"] == lang]
            return set(zip(s["passage_id"], s["retrieval_concept_id"]))

        delta_rows = []
        for lang in ("en", "de"):
            k02, kfix, k03 = _keys(lex02, lang), _keys(lex_fix, lang), _keys(lex, lang)
            delta_rows.append(
                {
                    "language": LANG_LABEL[lang],
                    "n_v02": len(k02),
                    "n_algo_fix": len(kfix),
                    "n_v03": len(k03),
                    "preserved_v02_to_v03": len(k02 & k03),
                    "added_v02_to_v03": len(k03 - k02),
                    "removed_v02_to_v03": len(k02 - k03),
                    "gained_algo_fix": len(kfix - k02),
                    "gained_vocab_after_fix": len(k03 - kfix),
                }
            )
        deltas = pd.DataFrame(delta_rows)
        save_table(deltas, "version_deltas", TABLES)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        x = np.arange(len(deltas))
        ax.plot(x, deltas["n_v02"], "o-", label="v0.2")
        ax.plot(x, deltas["n_algo_fix"], "s-", label="algo_fix")
        ax.plot(x, deltas["n_v03"], "^-", label="v0.3")
        ax.set_xticks(x)
        ax.set_xticklabels(deltas["language"])
        ax.set_ylabel("Passage–concept assignments")
        ax.set_title("Retrieval coverage across lexical versions")
        ax.legend()
        save_fig(fig, "15_version_coverage")
        del lex02, lex_fix
        gc.collect()
        log.info("stage 15 versions done")

    if stage <= 16:
        # Extreme IDs from small QC files — not full manifests.
        extreme_frames = []
        for lang in ("en", "de"):
            p = DATA_PROCESSED / f"corpus_quality_extreme_passages_{lang}_v1_candidate.csv"
            if p.is_file():
                extreme_frames.append(pd.read_csv(p, dtype=str))
        passages_lite = (
            pd.concat(extreme_frames, ignore_index=True)
            if extreme_frames
            else pd.DataFrame(columns=["passage_id", "language", "word_count", "segmentation_status", "qc_flags"])
        )
        # Sensitivity needs word_count on extremes + flags; synthesize minimal frame.
        if "segmentation_status" not in passages_lite.columns:
            passages_lite["segmentation_status"] = "extreme_length"
        if "qc_flags" not in passages_lite.columns:
            passages_lite["qc_flags"] = "extreme_length"
        if "word_count" not in passages_lite.columns:
            passages_lite["word_count"] = "0"
        # For S3 denominator adjustment we need extreme word counts — load only those rows'
        # word counts from manifests via filtered read if missing.
        if passages_lite["word_count"].astype(str).isin(["0", ""]).all():
            for lang in ("en", "de"):
                ids = set(
                    passages_lite.loc[passages_lite.get("language", pd.Series(dtype=str)) == lang, "passage_id"]
                ) if "language" in passages_lite.columns else set(passages_lite["passage_id"])
                if not ids:
                    # extreme files have language? check
                    ids = set(passages_lite["passage_id"])
                pm = load_passage_manifest(lang, usecols=["passage_id", "word_count", "language"])
                pm = pm[pm["passage_id"].isin(ids)]
                passages_lite = passages_lite.drop(columns=["word_count"], errors="ignore")
                passages_lite = passages_lite.merge(
                    pm[["passage_id", "word_count"]], on="passage_id", how="left"
                )
                del pm
                gc.collect()

        sens = run_sensitivity_analysis(lex, corpus, passages_lite)
        save_table(sens, "sensitivity_analysis", TABLES)
        fig, ax = plt.subplots(figsize=(8, 4))
        for cid in ("focalization_zero", "filter_words", "stance_narratorial"):
            s = sens[sens["concept_id"] == cid].copy()
            s["rr"] = pd.to_numeric(s["frequency_ratio_en_over_de"], errors="coerce")
            ax.plot(s["scenario"], s["rr"], "o-", label=cid)
        ax.axhline(1.0, color="grey", ls="--")
        ax.set_ylabel("RR = f_EN / f_DE")
        ax.set_title("Sensitivity of selected frequency ratios")
        ax.legend(fontsize=8)
        save_fig(fig, "16_sensitivity_rr")
        log.info("stage 16 sensitivity done")

    # Coverage + meta
    cov_rows = []
    for lang in ("en", "de"):
        sub = lex[lex["language"] == lang]
        words = float(a.loc[a["language"] == lang, "number_of_words"].sum())
        paras = float(a.loc[a["language"] == lang, "number_of_eligible_paragraphs"].sum())
        n_pass = sub["passage_id"].nunique()
        n_assign = len(sub)
        cov_rows.append(
            {
                "language": LANG_LABEL[lang],
                "retrieved_passages": int(n_pass),
                "assignments": int(n_assign),
                "pct_eligible_paragraphs": 100 * n_pass / paras if paras else 0,
                "passages_per_10k_words": n_pass / words * 10000 if words else 0,
                "assignments_per_10k_words": n_assign / words * 10000 if words else 0,
                "avg_concepts_per_retrieved_passage": n_assign / n_pass if n_pass else 0,
                "concepts_nonzero": int(sub["retrieval_concept_id"].nunique()),
            }
        )
    save_table(pd.DataFrame(cov_rows), "retrieval_coverage", TABLES)

    # Lightweight validation without full manifests: use passage_id sets from lexical only.
    val = {}
    for lang in ("en", "de"):
        pm = load_passage_manifest(lang, usecols=["passage_id"])
        val[lang] = validate_retrieval_keys(lex[lex.language == lang], pm)
        del pm
        gc.collect()

    meta = reproducibility_record(
        [
            DATA_PROCESSED / "corpus_manifest_en_v1_candidate.csv",
            DATA_PROCESSED / "corpus_manifest_de_v1_candidate.csv",
            DATA_PROCESSED / "pov_baseline_b_v03_lexical_en.csv",
            DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv",
        ]
    )
    meta["validation"] = val
    meta["n_concepts"] = len(concepts)
    (REPORTS / "01_narrative_perspective_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info(f"wrote tables under {TABLES} and figures under {FIGURES}")
    log.info("Log finished.")


if __name__ == "__main__":
    main()
