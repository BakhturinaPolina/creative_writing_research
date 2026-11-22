"""Generate analysis v2 tables and figures under reports/tables/02_* and figures/02_*.

Preserves reports/01_* and frozen v0.3 retrieval artifacts.
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

from src.analysis.analysis_v2 import (
    ANALYSIS_VERSION,
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    FOCUS_CONCEPTS,
    RETRIEVAL_VERSION,
    assert_integrity,
    assignments_long,
    bootstrap_concept_freqs,
    calculate_dispersion_all,
    calculate_family_frequencies_v2,
    calculate_frequencies_v2,
    category_and_standardized_frequencies,
    classify_robustness,
    compare_methods_v2,
    cooccurrence_v2,
    coverage_table,
    eligible_word_denominators,
    historical_overview,
    load_gate_decision,
    reproducibility_record_v2,
    run_sensitivity_v2,
)
from src.analysis.narrative_perspective import (
    LANGUAGE_COLORS,
    LANG_LABEL,
    load_concept_ids,
    load_corpus_manifest,
    load_passage_manifest,
    load_retrieval_results,
)
from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_PROCESSED, ROOT

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures" / "02_narrative_perspective"


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


def save_t(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"02_{name}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS)
    args = parser.parse_args()
    log = setup_logger(__file__)
    _style()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    gate = load_gate_decision()
    if gate.get("resegment_gate_fires"):
        raise RuntimeError(
            "Resegment gate fired but v2 resegment snapshot/retrieval rerun is not present. "
            "Complete resegment before analysis."
        )
    log.info(f"Gate decision: {gate.get('decision')}")

    concept_ids = load_concept_ids()
    corpus = load_corpus_manifest()
    # Attach qc_notes / possible_duplicate from book_stats for S5.
    for lang in ("en", "de"):
        stats = pd.read_csv(DATA_PROCESSED / f"{lang}_book_stats.csv", dtype=str).fillna("")
        stats = stats[["book_id", "possible_duplicate"]]
        stats["language"] = lang
        corpus = corpus.merge(stats, on=["book_id", "language"], how="left", suffixes=("", "_y"))

    passages = load_passage_manifest(
        usecols=[
            "passage_id",
            "book_id",
            "language",
            "word_count",
            "character_count",
            "segmentation_status",
            "qc_flags",
            "primary_group",
        ]
    )
    books = eligible_word_denominators(corpus, passages)
    # Keep publication year on books for historical.
    for col in ("publication_year", "edition_year", "primary_group", "qc_notes", "title", "author"):
        if col in corpus.columns and col not in books.columns:
            books = books.merge(
                corpus[["book_id", "language", col]].drop_duplicates(),
                on=["book_id", "language"],
                how="left",
                suffixes=("", "_c"),
            )

    lexical = load_retrieval_results("lexical_v03")
    bm25_q = load_retrieval_results("bm25_qualified_v03")
    bm25_new = load_retrieval_results("bm25_only_v03")

    # --- Corpus overview ---
    overview_rows = []
    for lang in ("en", "de"):
        listed = corpus[corpus["language"] == lang]
        a = books[books["language"] == lang]
        overview_rows.append(
            {
                "language": lang,
                "language_label": LANG_LABEL[lang],
                "listed_books": len(listed),
                "analysis_set_books": len(a),
                "excluded_books": len(listed) - len(a),
                "n_authors": int(a["author"].nunique()) if "author" in a.columns else "",
                "eligible_words": float(a["eligible_words"].sum()),
                "body_words": float(a["body_words"].sum()),
                "eligible_paragraphs": int(a["n_elig_paras"].sum()),
            }
        )
    overview = pd.DataFrame(overview_rows)
    save_t(overview, "corpus_overview")

    # Book length / paragraph figures
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, lang in zip(axes, ("en", "de")):
        vals = books.loc[books["language"] == lang, "eligible_words"].clip(lower=1)
        ax.hist(np.log10(vals), bins=30, color=LANGUAGE_COLORS[LANG_LABEL[lang]], alpha=0.85)
        ax.set_title(f"{LANG_LABEL[lang]} book length")
        ax.set_xlabel("log10(eligible words)")
    fig.tight_layout()
    save_fig(fig, "03_book_length")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, lang in zip(axes, ("en", "de")):
        pw = passages.loc[passages["language"] == lang, "word_count"]
        pw = pd.to_numeric(pw, errors="coerce").fillna(0).clip(lower=1)
        ax.hist(np.log10(pw), bins=50, color=LANGUAGE_COLORS[LANG_LABEL[lang]], alpha=0.85)
        ax.set_title(f"{LANG_LABEL[lang]} paragraph length")
        ax.set_xlabel("log10(passage words)")
    fig.tight_layout()
    save_fig(fig, "04_paragraph_length")

    # --- Frequencies ---
    log.info("Frequencies")
    freq = calculate_frequencies_v2(lexical, books, concept_ids)
    save_t(freq, "concept_frequencies")
    family = calculate_family_frequencies_v2(lexical, books)
    save_t(family, "concept_family_frequencies")
    cov = coverage_table(lexical, books, concept_ids)
    save_t(cov, "retrieval_coverage")

    validation = assert_integrity(corpus, passages, lexical, concept_ids, freq, family)
    log.info(f"Integrity OK: {validation}")

    # Top concepts bar
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, lang in zip(axes, ("en", "de")):
        sub = freq[freq["language"] == lang].nlargest(15, "frequency_per_10000_eligible_words")
        ax.barh(
            sub["concept_id"][::-1],
            sub["frequency_per_10000_eligible_words"][::-1],
            color=LANGUAGE_COLORS[LANG_LABEL[lang]],
        )
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Assignments / 10k eligible words")
    fig.suptitle("Top 15 concept frequencies")
    fig.tight_layout()
    save_fig(fig, "05_concept_frequencies_top15")

    # Family frequencies
    fig, ax = plt.subplots(figsize=(9, 4.5))
    wide = family.pivot(index="concept_family_label", columns="language_label", values="frequency_per_10000_eligible_words")
    wide.plot(kind="barh", ax=ax, color=[LANGUAGE_COLORS["English"], LANGUAGE_COLORS["German"]])
    ax.set_xlabel("Unique passages / 10k eligible words (deduplicated)")
    ax.set_title("Concept family frequencies")
    fig.tight_layout()
    save_fig(fig, "06_family_frequencies")

    # --- Dispersion ---
    log.info("Dispersion")
    disp, book_level = calculate_dispersion_all(lexical, books, concept_ids)
    save_t(disp, "concept_dispersion")
    save_t(book_level, "book_level_frequencies")

    # Frequency vs prevalence scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    merged = freq.merge(disp, on=["concept_id", "language"])
    for lang, label in (("en", "English"), ("de", "German")):
        sub = merged[merged["language"] == lang]
        ax.scatter(
            sub["pct_books_with_concept"],
            sub["frequency_per_10000_eligible_words"],
            c=LANGUAGE_COLORS[label],
            label=label,
            alpha=0.75,
            s=40,
        )
        # Label outliers: high frequency
        top = sub.nlargest(3, "frequency_per_10000_eligible_words")
        for _, r in top.iterrows():
            ax.annotate(r["concept_id"], (r["pct_books_with_concept"], r["frequency_per_10000_eligible_words"]), fontsize=7)
    # Exploratory zones (visual only)
    xmax = max(merged["pct_books_with_concept"].max(), 1)
    ymax = max(merged["frequency_per_10000_eligible_words"].max(), 0.01)
    ax.axvline(xmax / 2, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(ymax / 2, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Book prevalence (%)")
    ax.set_ylabel("Frequency per 10k eligible words")
    ax.set_title("Frequency vs prevalence (zones are exploratory)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "07_freq_vs_prevalence")

    # --- Standardization ---
    log.info("Standardization")
    composition, weights, std_freq, cat_freq = category_and_standardized_frequencies(
        lexical, books, concept_ids
    )
    save_t(composition, "corpus_composition_by_group")
    save_t(weights, "standardization_weights")
    save_t(std_freq, "standardized_frequencies")
    save_t(cat_freq, "category_concept_frequencies")

    fig, ax = plt.subplots(figsize=(8, 4))
    comp_p = composition.pivot(index="primary_group", columns="language", values="pct_eligible_words")
    comp_p.plot(kind="bar", ax=ax, color=[LANGUAGE_COLORS["English"], LANGUAGE_COLORS["German"]])
    ax.set_ylabel("% of eligible words")
    ax.set_title("Corpus composition by category")
    fig.tight_layout()
    save_fig(fig, "08_composition_by_group")

    # --- Co-occurrence ---
    log.info("Co-occurrence")
    extreme_ids = set(
        passages.loc[
            passages["qc_flags"].astype(str).str.contains("extreme_length", case=False, na=False),
            "passage_id",
        ]
    )
    for lang in ("en", "de"):
        mat, pairs = cooccurrence_v2(lexical, lang, min_support=5)
        mat.to_csv(TABLES / f"02_concept_cooccurrence_{lang}.csv")
        save_t(pairs, f"concept_cooccurrence_pairs_{lang}")
        # Heatmap of top concepts by support
        support = np.diag(mat.to_numpy()) if len(mat) else np.array([])
        order = np.argsort(-support)[:18]
        labels = [mat.index[i] for i in order]
        subm = mat.loc[labels, labels].astype(float)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(np.log1p(subm), ax=ax, cmap="Blues" if lang == "en" else "Oranges")
        ax.set_title(f"Co-occurrence log1p counts ({LANG_LABEL[lang]})")
        fig.tight_layout()
        save_fig(fig, f"09_cooccurrence_{lang}")

        mat2, pairs2 = cooccurrence_v2(
            lexical, lang, min_support=5, exclude_passage_ids=extreme_ids
        )
        save_t(pairs2, f"concept_cooccurrence_pairs_{lang}_no_extreme")

    # --- Complementarity ---
    log.info("Complementarity")
    comp = compare_methods_v2(lexical, bm25_new, bm25_q)
    save_t(comp, "retrieval_complementarity")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, comp["n_L_passages"], 0.4, label="Lexical passages", color=LANGUAGE_COLORS["English"])
    ax.bar(x + 0.2, comp["n_Bnew_passages"], 0.4, label="Bnew passages", color=LANGUAGE_COLORS["German"])
    ax.set_xticks(x)
    ax.set_xticklabels(comp["language_label"])
    ax.set_ylabel("Unique passages")
    ax.set_title("Lexical vs BM25-new passage pools")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "10_complementarity")

    # --- Sensitivity ---
    log.info("Sensitivity (all concepts)")
    # Use extreme passages table as fallback for S3 word adjustment
    extreme_frames = []
    for lang in ("en", "de"):
        p = DATA_PROCESSED / f"corpus_quality_extreme_passages_{lang}_v1_candidate.csv"
        if p.is_file():
            extreme_frames.append(pd.read_csv(p, dtype=str))
    extreme_tab = pd.concat(extreme_frames, ignore_index=True) if extreme_frames else passages
    sens = run_sensitivity_v2(lexical, books, passages if "qc_flags" in passages.columns else extreme_tab, concept_ids)
    save_t(sens, "sensitivity_analysis")
    robust_rows = [
        {"concept_id": cid, "robustness": classify_robustness(sens, cid)} for cid in concept_ids
    ]
    save_t(pd.DataFrame(robust_rows), "robustness_labels")

    focus_sens = sens[sens["concept_id"].isin(FOCUS_CONCEPTS)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for cid in FOCUS_CONCEPTS:
        sub = focus_sens[focus_sens["concept_id"] == cid]
        ax.plot(sub["scenario"], sub["frequency_ratio_en_over_de"].replace([np.inf, -np.inf], np.nan), marker="o", label=cid)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("EN/DE frequency ratio")
    ax.set_title("Sensitivity of focus-concept frequency ratios")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, "11_sensitivity_rr")

    # --- Bootstrap ---
    if not args.skip_bootstrap:
        log.info(f"Bootstrap {args.bootstrap_iters} iters")
        boot = bootstrap_concept_freqs(
            lexical, books, FOCUS_CONCEPTS, n_iter=args.bootstrap_iters, seed=BOOTSTRAP_SEED
        )
        save_t(boot, "bootstrap_intervals")
    else:
        log.info("Skipping bootstrap")

    # --- Historical ---
    hist = historical_overview(books)
    save_t(hist, "historical_overview")
    fig, ax = plt.subplots(figsize=(9, 4))
    for lang, label in (("en", "English"), ("de", "German")):
        sub = hist[(hist["language"] == lang) & (hist["decade"] != "unknown") & (hist["primary_group"] != "all")]
        by_dec = sub.groupby("decade")["n_books"].sum().sort_index()
        ax.plot(by_dec.index, by_dec.values, marker="o", label=label, color=LANGUAGE_COLORS[label])
    ax.set_ylabel("Books with known year")
    ax.set_title("Books per decade (missing years excluded from trend)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "12_publication_decades")

    # --- Long assignments ---
    long = assignments_long(lexical, bm25_new, bm25_q)
    save_t(long, "assignments_long")

    # QC copy
    qc = pd.read_csv(DATA_PROCESSED / "corpus_quality_v1_candidate.csv")
    save_t(qc, "corpus_quality")

    # Meta
    input_paths = [
        DATA_PROCESSED / "corpus_manifest_en_v1_candidate.csv",
        DATA_PROCESSED / "corpus_manifest_de_v1_candidate.csv",
        DATA_PROCESSED / "passage_manifest_en_v1_candidate.csv",
        DATA_PROCESSED / "passage_manifest_de_v1_candidate.csv",
        DATA_PROCESSED / "pov_baseline_b_v03_lexical_en.csv",
        DATA_PROCESSED / "pov_baseline_b_v03_lexical_de.csv",
        DATA_PROCESSED / "pov_baseline_c_v03_bm25_en.csv",
        DATA_PROCESSED / "pov_baseline_c_v03_bm25_de.csv",
        DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_en.csv",
        DATA_PROCESSED / "pov_baseline_c_v03_bm25_only_unvalidated_de.csv",
        CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml",
        CONFIG / "concept_families_v05.yml",
        ROOT / "src" / "analysis" / "analysis_v2.py",
        ROOT / "src" / "analysis" / "run_np_v2.py",
    ]
    meta = reproducibility_record_v2(
        input_paths,
        validation,
        gate,
        bootstrap_iterations=0 if args.skip_bootstrap else args.bootstrap_iters,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    meta_path = REPORTS / "02_narrative_perspective_run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"Wrote {meta_path}")
    log.info(f"Analysis v2 complete ({len(concept_ids)} concepts, edition {ANALYSIS_VERSION})")


if __name__ == "__main__":
    main()
