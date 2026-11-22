"""Narrative-perspective analysis v2: integrity, rates, standardization, sensitivity.

Primary denominator: eligible searchable words (sum of passage word_count).
Supplementary denominator: cleaned body words (manifest number_of_words / body_words).
Retrieval execution: frozen v0.3 unless a resegment gate produced a newer snapshot.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from src.analysis.narrative_perspective import (
    LANGUAGE_COLORS,
    LANG_LABEL,
    SNAPSHOT_ID,
    attach_concept_families,
    file_sha256,
    frequency_ratio,
    load_concept_families,
    load_concept_ids,
    load_corpus_manifest,
    load_passage_manifest,
    load_retrieval_results,
    source_commit,
)
from src.common.paths import CONFIG, DATA_PROCESSED, ROOT

ANALYSIS_VERSION = "v2"
RETRIEVAL_METHOD = "lexical"
RETRIEVAL_VERSION = "v03"
BOOTSTRAP_SEED = 20261122
BOOTSTRAP_ITERS = 1000
FOCUS_CONCEPTS = [
    "focalization_zero",
    "stance_narratorial",
    "stance_figural",
    "grammatical_third_person",
    "filter_words",
    "reader_effect",
]


def load_gate_decision() -> dict:
    path = DATA_PROCESSED / "extreme_passage_gate_v2.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"resegment_gate_fires": False, "decision": "gate file missing; default retain v03"}


def eligible_word_denominators(corpus: pd.DataFrame, passages: pd.DataFrame) -> pd.DataFrame:
    """Per-language and per language×group eligible words + body words + book counts."""
    a = corpus[corpus["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))].copy()
    a["body_words"] = pd.to_numeric(a["number_of_words"], errors="coerce").fillna(0)
    pm = passages.copy()
    pm["word_count"] = pd.to_numeric(pm["word_count"], errors="coerce").fillna(0)
    # Restrict passages to analysis books.
    pm = pm[pm["book_id"].isin(set(a["book_id"]))]
    book_elig = pm.groupby("book_id")["word_count"].sum().rename("eligible_words")
    a = a.merge(book_elig, left_on="book_id", right_index=True, how="left")
    a["eligible_words"] = a["eligible_words"].fillna(0)
    a["n_elig_paras"] = (
        pm.groupby("book_id").size().reindex(a["book_id"]).fillna(0).to_numpy()
    )
    return a


def language_denominators(books: pd.DataFrame) -> pd.DataFrame:
    return books.groupby("language", as_index=False).agg(
        eligible_words=("eligible_words", "sum"),
        body_words=("body_words", "sum"),
        n_books=("book_id", "nunique"),
        eligible_paragraphs=("n_elig_paras", "sum"),
    )


def assert_integrity(
    corpus: pd.DataFrame,
    passages: pd.DataFrame,
    lexical: pd.DataFrame,
    concept_ids: list[str],
    freq: pd.DataFrame,
    family: pd.DataFrame,
) -> dict:
    """Fail-fast integrity checks. Returns a validation dict for the run meta."""
    errors: list[str] = []
    a = corpus[corpus["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))]
    if a["book_id"].duplicated().any():
        errors.append("duplicate book_id in analysis corpus")
    if passages["passage_id"].duplicated().any():
        errors.append("duplicate passage_id in passage manifest")
    missing = (~lexical["passage_id"].isin(set(passages["passage_id"]))).sum()
    if missing:
        errors.append(f"{missing} lexical passage_ids missing from passage manifest")
    dup_assign = lexical.duplicated(subset=["passage_id", "retrieval_concept_id"]).sum()
    if dup_assign:
        errors.append(f"{dup_assign} duplicate passage–concept lexical assignments")
    langs = set(a["language"])
    if langs != {"en", "de"}:
        errors.append(f"incomplete language coverage in analysis set: {langs}")
    expected = len(concept_ids) * 2
    if len(freq) != expected:
        errors.append(f"freq rows {len(freq)} != n_concepts×2 ({expected})")
    if set(freq["concept_id"]) != set(concept_ids):
        errors.append("concept_id set in frequencies does not match lexicon")
    if set(freq["language"]) != {"en", "de"}:
        errors.append("frequency table missing a language")
    # Zero-hit books must remain in n_analysis_books.
    for lang in ("en", "de"):
        n_books = int(a.loc[a["language"] == lang, "book_id"].nunique())
        n_in_freq = int(freq.loc[freq["language"] == lang, "n_analysis_books"].iloc[0])
        if n_books != n_in_freq:
            errors.append(f"{lang}: n_analysis_books {n_in_freq} != {n_books}")
    # Family note present.
    if len(family) and "deduplicated" not in str(family["note"].iloc[0]).lower():
        errors.append("family frequencies missing deduplication note")
    if errors:
        raise AssertionError("Integrity checks failed:\n- " + "\n- ".join(errors))
    return {
        "n_analysis_books_en": int(a.loc[a["language"] == "en", "book_id"].nunique()),
        "n_analysis_books_de": int(a.loc[a["language"] == "de", "book_id"].nunique()),
        "n_concepts": len(concept_ids),
        "n_freq_rows": len(freq),
        "n_lexical_assignments": len(lexical),
        "n_lexical_passages": int(lexical["passage_id"].nunique()),
        "n_missing_passage_keys": int(missing),
        "n_duplicate_assignments": int(dup_assign),
    }


def calculate_frequencies_v2(
    retrieval: pd.DataFrame,
    books: pd.DataFrame,
    concept_ids: list[str] | None = None,
    *,
    method: str = RETRIEVAL_METHOD,
    version: str = RETRIEVAL_VERSION,
) -> pd.DataFrame:
    concept_ids = concept_ids or load_concept_ids()
    den = language_denominators(books)
    rows = []
    for lang in ("en", "de"):
        drow = den[den["language"] == lang]
        elig = float(drow["eligible_words"].iloc[0]) if len(drow) else 0.0
        body = float(drow["body_words"].iloc[0]) if len(drow) else 0.0
        paras = float(drow["eligible_paragraphs"].iloc[0]) if len(drow) else 0.0
        n_books = int(drow["n_books"].iloc[0]) if len(drow) else 0
        sub = retrieval[retrieval["language"] == lang] if len(retrieval) else retrieval
        book_ids = books.loc[books["language"] == lang, "book_id"]
        book_elig = books.loc[books["language"] == lang].set_index("book_id")["eligible_words"]
        for cid in concept_ids:
            hit = sub[sub["retrieval_concept_id"] == cid] if len(sub) else sub
            n_assign = len(hit)
            n_pass = int(hit["passage_id"].nunique()) if n_assign else 0
            n_b = int(hit["book_id"].nunique()) if n_assign else 0
            counts = hit.groupby("book_id").size() if n_assign else pd.Series(dtype=float)
            freqs = []
            for bid in book_ids:
                w = float(book_elig.get(bid, 0) or 0)
                c = float(counts.get(bid, 0))
                freqs.append((c / w * 10000.0) if w > 0 else 0.0)
            rows.append(
                {
                    "concept_id": cid,
                    "concept_label": cid.replace("_", " "),
                    "language": lang,
                    "language_label": LANG_LABEL[lang],
                    "n_assignments": n_assign,
                    "n_unique_passages": n_pass,
                    "n_books": n_b,
                    "eligible_words": elig,
                    "body_words": body,
                    "eligible_paragraphs": paras,
                    "n_analysis_books": n_books,
                    "frequency_per_10000_eligible_words": (n_assign / elig * 10000.0) if elig else 0.0,
                    "frequency_per_10000_body_words": (n_assign / body * 10000.0) if body else 0.0,
                    "frequency_per_1000_paragraphs": (n_assign / paras * 1000.0) if paras else 0.0,
                    "percentage_of_books": (100.0 * n_b / n_books) if n_books else 0.0,
                    "median_book_frequency": float(np.median(freqs)) if freqs else 0.0,
                    "method": method,
                    "version": version,
                    "denominator_note": "primary=eligible_passage_words; supplementary=body_words",
                }
            )
    freq = attach_concept_families(pd.DataFrame(rows))
    # Cross-language ratios.
    piv = freq.pivot(index="concept_id", columns="language", values="frequency_per_10000_eligible_words")
    abs_diff = []
    ratios = []
    notes = []
    for cid in freq["concept_id"].unique():
        en = float(piv.loc[cid, "en"]) if "en" in piv.columns else 0.0
        de = float(piv.loc[cid, "de"]) if "de" in piv.columns else 0.0
        rr = frequency_ratio(en, de)
        abs_diff.append(en - de)
        if rr is None:
            ratios.append(np.nan)
            notes.append("both_zero")
        elif rr == float("inf"):
            ratios.append(np.inf)
            notes.append("de_zero_en_positive_ratio_undefined_infinite")
        else:
            ratios.append(rr)
            notes.append("")
    # Attach only once per concept onto both language rows via map.
    ratio_map = dict(zip(freq["concept_id"].unique(), ratios))
    diff_map = dict(zip(freq["concept_id"].unique(), abs_diff))
    note_map = dict(zip(freq["concept_id"].unique(), notes))
    # Rebuild maps correctly: unique preserves order matching ratios list.
    concepts_u = list(dict.fromkeys(freq["concept_id"]))
    ratio_map = dict(zip(concepts_u, ratios))
    diff_map = dict(zip(concepts_u, abs_diff))
    note_map = dict(zip(concepts_u, notes))
    freq["frequency_ratio_en_over_de"] = freq["concept_id"].map(ratio_map)
    freq["abs_diff_en_minus_de_per_10k"] = freq["concept_id"].map(diff_map)
    freq["ratio_note"] = freq["concept_id"].map(note_map)
    return freq


def calculate_family_frequencies_v2(
    retrieval: pd.DataFrame,
    books: pd.DataFrame,
    *,
    method: str = RETRIEVAL_METHOD,
    version: str = RETRIEVAL_VERSION,
) -> pd.DataFrame:
    fam = load_concept_families()
    den = language_denominators(books)
    rows = []
    for block in fam.get("families") or []:
        cids = set(block.get("concepts") or [])
        for lang in ("en", "de"):
            sub = retrieval[
                (retrieval["language"] == lang)
                & (retrieval["retrieval_concept_id"].isin(cids))
            ]
            drow = den[den["language"] == lang]
            elig = float(drow["eligible_words"].iloc[0]) if len(drow) else 0.0
            n_pass = int(sub["passage_id"].nunique()) if len(sub) else 0
            rows.append(
                {
                    "concept_family": block["id"],
                    "concept_family_label": block["label"],
                    "language": lang,
                    "language_label": LANG_LABEL[lang],
                    "n_assignments": len(sub),
                    "n_unique_passages": n_pass,
                    "eligible_words": elig,
                    "frequency_per_10000_eligible_words": (n_pass / elig * 10000.0) if elig else 0.0,
                    "note": "numerator = unique passages in family (deduplicated); passages may belong to several families",
                    "method": method,
                    "version": version,
                }
            )
    return pd.DataFrame(rows)


def calculate_dispersion_all(
    retrieval: pd.DataFrame,
    books: pd.DataFrame,
    concept_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concept_ids = concept_ids or load_concept_ids()
    disp_rows = []
    book_rows = []
    for lang in ("en", "de"):
        b = books[books["language"] == lang].copy()
        n = len(b)
        top10_n = max(1, int(np.ceil(0.1 * n))) if n else 1
        for cid in concept_ids:
            hit = retrieval[
                (retrieval["language"] == lang) & (retrieval["retrieval_concept_id"] == cid)
            ]
            counts = hit.groupby("book_id").size() if len(hit) else pd.Series(dtype=float)
            b2 = b.copy()
            b2["n_assign"] = b2["book_id"].map(counts).fillna(0)
            b2["freq_per_10k"] = np.where(
                b2["eligible_words"] > 0,
                b2["n_assign"] / b2["eligible_words"] * 10000.0,
                0.0,
            )
            total = float(b2["n_assign"].sum())
            top5 = b2.nlargest(5, "n_assign")
            top10 = b2.nlargest(top10_n, "n_assign")
            disp_rows.append(
                {
                    "concept_id": cid,
                    "language": lang,
                    "n_books_with_concept": int((b2["n_assign"] > 0).sum()),
                    "pct_books_with_concept": 100.0 * float((b2["n_assign"] > 0).mean()) if n else 0.0,
                    "mean_book_frequency": float(b2["freq_per_10k"].mean()) if n else 0.0,
                    "median_book_frequency": float(b2["freq_per_10k"].median()) if n else 0.0,
                    "q1_book_frequency": float(b2["freq_per_10k"].quantile(0.25)) if n else 0.0,
                    "q3_book_frequency": float(b2["freq_per_10k"].quantile(0.75)) if n else 0.0,
                    "iqr_book_frequency": float(
                        b2["freq_per_10k"].quantile(0.75) - b2["freq_per_10k"].quantile(0.25)
                    )
                    if n
                    else 0.0,
                    "pct_books_zero": 100.0 * float((b2["n_assign"] == 0).mean()) if n else 0.0,
                    "top5_books_contribution": (float(top5["n_assign"].sum()) / total) if total else 0.0,
                    "top10pct_books_contribution": (float(top10["n_assign"].sum()) / total) if total else 0.0,
                    "top10pct_n_books": top10_n,
                }
            )
            for _, br in b2.iterrows():
                book_rows.append(
                    {
                        "book_id": br["book_id"],
                        "language": lang,
                        "concept_id": cid,
                        "n_assign": int(br["n_assign"]),
                        "freq_per_10k_eligible": float(br["freq_per_10k"]),
                        "eligible_words": float(br["eligible_words"]),
                        "primary_group": br.get("primary_group", ""),
                    }
                )
    return pd.DataFrame(disp_rows), pd.DataFrame(book_rows)


def category_and_standardized_frequencies(
    retrieval: pd.DataFrame,
    books: pd.DataFrame,
    concept_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Category-specific rates + direct standardization with shared EN/DE mean weights."""
    concept_ids = concept_ids or load_concept_ids()
    groups = sorted(books["primary_group"].dropna().unique())
    # Category word shares.
    cat_rows = []
    for lang in ("en", "de"):
        b = books[books["language"] == lang]
        total = float(b["eligible_words"].sum())
        for g in groups:
            w = float(b.loc[b["primary_group"] == g, "eligible_words"].sum())
            cat_rows.append(
                {
                    "language": lang,
                    "primary_group": g,
                    "eligible_words": w,
                    "pct_eligible_words": (100.0 * w / total) if total else 0.0,
                    "n_books": int(b.loc[b["primary_group"] == g, "book_id"].nunique()),
                }
            )
    composition = pd.DataFrame(cat_rows)
    shares = composition.pivot(index="primary_group", columns="language", values="pct_eligible_words").fillna(0) / 100.0
    weights = shares.mean(axis=1)
    weight_df = weights.rename("common_weight").reset_index()

    # Category-specific concept frequencies.
    cat_freq_rows = []
    for lang in ("en", "de"):
        for g in groups:
            b = books[(books["language"] == lang) & (books["primary_group"] == g)]
            elig = float(b["eligible_words"].sum())
            bids = set(b["book_id"])
            sub = retrieval[(retrieval["language"] == lang) & (retrieval["book_id"].isin(bids))]
            for cid in concept_ids:
                hit = sub[sub["retrieval_concept_id"] == cid]
                n = len(hit)
                cat_freq_rows.append(
                    {
                        "concept_id": cid,
                        "language": lang,
                        "primary_group": g,
                        "n_assignments": n,
                        "eligible_words": elig,
                        "frequency_per_10000_eligible_words": (n / elig * 10000.0) if elig else 0.0,
                    }
                )
    cat_freq = pd.DataFrame(cat_freq_rows)

    # Standardized frequencies.
    std_rows = []
    for cid in concept_ids:
        rates = {}
        std = {}
        for lang in ("en", "de"):
            f_g = {}
            for g in groups:
                row = cat_freq[
                    (cat_freq.concept_id == cid)
                    & (cat_freq.language == lang)
                    & (cat_freq.primary_group == g)
                ]
                f_g[g] = float(row["frequency_per_10000_eligible_words"].iloc[0]) if len(row) else 0.0
            # Original overall rate from books denominators.
            b = books[books["language"] == lang]
            elig = float(b["eligible_words"].sum())
            n = len(
                retrieval[
                    (retrieval["language"] == lang) & (retrieval["retrieval_concept_id"] == cid)
                ]
            )
            rates[lang] = (n / elig * 10000.0) if elig else 0.0
            std[lang] = float(sum(float(weights.get(g, 0.0)) * f_g.get(g, 0.0) for g in groups))
        orig_diff = rates["en"] - rates["de"]
        std_diff = std["en"] - std["de"]
        sign_changed = (orig_diff > 0 and std_diff < 0) or (orig_diff < 0 and std_diff > 0)
        rr_orig = frequency_ratio(rates["en"], rates["de"])
        rr_std = frequency_ratio(std["en"], std["de"])
        std_rows.append(
            {
                "concept_id": cid,
                "freq_en_original": rates["en"],
                "freq_de_original": rates["de"],
                "freq_en_standardized": std["en"],
                "freq_de_standardized": std["de"],
                "ratio_en_over_de_original": rr_orig if rr_orig is not None else np.nan,
                "ratio_en_over_de_standardized": rr_std if rr_std is not None else np.nan,
                "sign_of_difference_changed": sign_changed,
                "note": "Weights = mean of EN and DE category eligible-word shares; does not remove dictionary bias",
            }
        )
    return composition, weight_df, pd.DataFrame(std_rows), cat_freq


def cooccurrence_v2(
    retrieval: pd.DataFrame,
    lang: str,
    *,
    min_support: int = 5,
    exclude_passage_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concepts = load_concept_ids()
    sub = retrieval.loc[retrieval["language"] == lang, ["passage_id", "book_id", "retrieval_concept_id"]].copy()
    if exclude_passage_ids:
        sub = sub[~sub["passage_id"].isin(exclude_passage_ids)]
    by_pass: dict[str, set[str]] = defaultdict(set)
    pass_book: dict[str, str] = {}
    for pid, bid, cid in zip(sub["passage_id"], sub["book_id"], sub["retrieval_concept_id"]):
        by_pass[pid].add(cid)
        pass_book[pid] = bid
    sets = {c: set() for c in concepts}
    for pid, cs in by_pass.items():
        for c in cs:
            if c in sets:
                sets[c].add(pid)
    n = len(concepts)
    counts = np.zeros((n, n), dtype=np.int32)
    sizes = {c: len(sets[c]) for c in concepts}
    pairs = []
    for i, a in enumerate(concepts):
        sa = sets[a]
        for j in range(i, n):
            b = concepts[j]
            sb = sets[b]
            inter = len(sa & sb) if i != j else sizes[a]
            counts[i, j] = counts[j, i] = inter
            if i < j and inter >= min_support:
                union = sizes[a] + sizes[b] - inter
                books = {pass_book[p] for p in (sa & sb)}
                pairs.append(
                    {
                        "language": lang,
                        "concept_a": a,
                        "concept_b": b,
                        "cooccurrence": inter,
                        "support_a": sizes[a],
                        "support_b": sizes[b],
                        "jaccard": (inter / union) if union else 0.0,
                        "conditional_a_given_b": (inter / sizes[b]) if sizes[b] else 0.0,
                        "conditional_b_given_a": (inter / sizes[a]) if sizes[a] else 0.0,
                        "n_books": len(books),
                    }
                )
    count_df = pd.DataFrame(counts, index=concepts, columns=concepts)

    # Evidence overlap flag using matched_spans when available.
    if "matched_spans" in retrieval.columns and pairs:
        # Build passage → concept → span intervals
        span_map: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        sub2 = retrieval[retrieval["language"] == lang]
        if exclude_passage_ids:
            sub2 = sub2[~sub2["passage_id"].isin(exclude_passage_ids)]
        for _, row in sub2.iterrows():
            try:
                spans = json.loads(row.get("matched_spans") or "[]")
            except Exception:
                spans = []
            for sp in spans:
                try:
                    span_map[(row["passage_id"], row["retrieval_concept_id"])].append(
                        (int(sp.get("start", 0)), int(sp.get("end", 0)))
                    )
                except Exception:
                    continue

        def _overlap(a_spans, b_spans) -> bool:
            for s1, e1 in a_spans:
                for s2, e2 in b_spans:
                    if s1 < e2 and s2 < e1:
                        return True
            return False

        for p in pairs:
            shared_passages = sets[p["concept_a"]] & sets[p["concept_b"]]
            independent = 0
            overlapping = 0
            for pid in shared_passages:
                sa = span_map.get((pid, p["concept_a"]), [])
                sb = span_map.get((pid, p["concept_b"]), [])
                if not sa or not sb:
                    continue
                if _overlap(sa, sb):
                    overlapping += 1
                else:
                    independent += 1
            p["n_passages_disjoint_triggers"] = independent
            p["n_passages_overlapping_evidence"] = overlapping

    return count_df, pd.DataFrame(pairs)


def compare_methods_v2(
    lexical: pd.DataFrame,
    bm25_only: pd.DataFrame,
    qualified: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for lang in ("en", "de"):
        L = set(
            zip(
                lexical.loc[lexical["language"] == lang, "passage_id"],
                lexical.loc[lexical["language"] == lang, "retrieval_concept_id"],
            )
        )
        Bq = set(
            zip(
                qualified.loc[qualified["language"] == lang, "passage_id"],
                qualified.loc[qualified["language"] == lang, "retrieval_concept_id"],
            )
        )
        Bnew = set(
            zip(
                bm25_only.loc[bm25_only["language"] == lang, "passage_id"],
                bm25_only.loc[bm25_only["language"] == lang, "retrieval_concept_id"],
            )
        )
        Lp = {p for p, _ in L}
        Bqp = {p for p, _ in Bq}
        Bnew_p = {p for p, _ in Bnew}
        shared_p = Lp & Bnew_p
        lexical_only_p = Lp - Bnew_p
        bm25_only_p = Bnew_p - Lp
        U_p = Lp | Bnew_p
        # Assignment-level
        shared_a = L & Bnew
        lex_only_a = L - Bnew
        bnew_only_a = Bnew - L
        new_on_known = {p for p, _ in Bnew if p in Lp}
        j_p = len(Lp & Bnew_p) / len(Lp | Bnew_p) if (Lp | Bnew_p) else 0.0
        j_a = len(L & Bnew) / len(L | Bnew) if (L | Bnew) else 0.0
        rows.append(
            {
                "language": lang,
                "language_label": LANG_LABEL[lang],
                "n_L_assignments": len(L),
                "n_L_passages": len(Lp),
                "n_Bq_assignments": len(Bq),
                "n_Bq_passages": len(Bqp),
                "n_Bnew_assignments": len(Bnew),
                "n_Bnew_passages": len(Bnew_p),
                "n_lexical_only_passages": len(lexical_only_p),
                "n_bm25_only_passages": len(bm25_only_p),
                "n_shared_passages": len(shared_p),
                "n_genuinely_new_bm25_passages": len(bm25_only_p),
                "n_known_passages_with_new_bm25_concepts": len(new_on_known),
                "n_combined_unique_passages_U": len(U_p),
                "n_lexical_only_assignments": len(lex_only_a),
                "n_bm25_only_assignments": len(bnew_only_a),
                "n_shared_assignments": len(shared_a),
                "relative_passage_expansion": (len(U_p) / len(Lp) - 1.0) if Lp else 0.0,
                "jaccard_passages_L_vs_Bnew": j_p,
                "jaccard_assignments_L_vs_Bnew": j_a,
                "braw_available": False,
                "pool_note": "Braw not on disk; Bnew from bm25_only_unvalidated ≈ B_combined\\\\L; not independent recall",
                "method": "lexical_v03_vs_bm25_v03",
                "version": RETRIEVAL_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _high_confidence_mask(r: pd.DataFrame) -> pd.Series:
    """Keep rows with strong evidence tiers; drop context_only-only matches."""
    tiers = r.get("evidence_tiers", pd.Series([""] * len(r))).astype(str)
    # Parse JSON-ish dict string; keep if any strong/academic/regex count > 0.
    keep = []
    for t in tiers:
        try:
            d = json.loads(t.replace("'", '"')) if t else {}
        except Exception:
            d = {}
        strong = (
            int(d.get("academic_high", 0) or 0)
            + int(d.get("strong_academic", 0) or 0)
            + int(d.get("strong_craft", 0) or 0)
            + int(d.get("strong_paraphrase", 0) or 0)
            + int(d.get("regex", 0) or 0)
        )
        keep.append(strong > 0)
    return pd.Series(keep, index=r.index)


def run_sensitivity_v2(
    lexical: pd.DataFrame,
    books: pd.DataFrame,
    passages: pd.DataFrame,
    concept_ids: list[str] | None = None,
) -> pd.DataFrame:
    concept_ids = concept_ids or load_concept_ids()
    fam = load_concept_families()
    s4 = fam.get("sensitivity_s4_exclusions") or {}
    # Extreme ids
    if "qc_flags" in passages.columns:
        extreme_ids = set(
            passages.loc[
                passages["qc_flags"].astype(str).str.contains("extreme_length", case=False, na=False)
                | passages["segmentation_status"].astype(str).str.contains("extreme", case=False, na=False),
                "passage_id",
            ]
        )
    else:
        extreme_ids = set(passages["passage_id"]) if "passage_id" in passages.columns else set()

    dup_books = set(
        books.loc[
            books.get("qc_notes", pd.Series("", index=books.index))
            .astype(str)
            .str.contains("possible_duplicate", case=False, na=False),
            "book_id",
        ]
    )
    # Also from possible_duplicate column if present on books via merge.
    if "possible_duplicate" in books.columns:
        dup_books |= set(
            books.loc[
                books["possible_duplicate"].astype(str).str.lower().isin(("true", "1")),
                "book_id",
            ]
        )

    scenarios = [
        ("S1", "Complete analysis-set corpus"),
        ("S2", "Exclude contrast materials"),
        ("S3", "Exclude extreme-length passages"),
        ("S4", "Conservative DE allwissen/focalization_zero exclusions"),
        ("S5", "Exclude possible_duplicate books"),
        ("S6", "High-confidence lexical evidence only"),
        ("S7", "Exclude ambiguous German stance triggers (auktorial/personal)"),
        ("S8", "Drop five most productive books per concept"),
        ("S9", "Fiction manuals only"),
        ("S10", "Fiction manuals + pedagogy (exclude contrast)"),
        ("S11", "Body-word denominator (supplementary)"),
    ]

    def _freq(r: pd.DataFrame, b: pd.DataFrame, cid: str, lang: str, use_body: bool = False) -> float:
        sub = r[(r["language"] == lang) & (r["retrieval_concept_id"] == cid)]
        col = "body_words" if use_body else "eligible_words"
        words = float(b.loc[b["language"] == lang, col].sum()) if len(b) else 0.0
        if not use_body and "eligible_words" not in b.columns:
            words = float(b.loc[b["language"] == lang, "body_words"].sum())
        return (len(sub) / words * 10000.0) if words else 0.0

    rows = []
    for scenario, desc in scenarios:
        r = lexical.copy()
        b = books.copy()
        use_body = False
        if scenario == "S2":
            b = b[b["primary_group"] != "contrast"]
            r = r[r["book_id"].isin(set(b["book_id"]))]
        elif scenario == "S3":
            r = r[~r["passage_id"].isin(extreme_ids)]
            # Subtract extreme words from eligible denominators at book level approximately via passages.
            pw = passages.copy()
            pw["word_count"] = pd.to_numeric(pw["word_count"], errors="coerce").fillna(0)
            drop_w = (
                pw[pw["passage_id"].isin(extreme_ids)]
                .groupby("book_id")["word_count"]
                .sum()
            )
            b = b.copy()
            b["eligible_words"] = b.apply(
                lambda row: max(0.0, float(row["eligible_words"]) - float(drop_w.get(row["book_id"], 0))),
                axis=1,
            )
        elif scenario == "S4":
            drop = pd.Series(False, index=r.index)
            lang = s4.get("language", "de")
            cids = set(s4.get("exclude_concept_ids_when_trigger_overlap") or [])
            for needle in s4.get("exclude_if_matched_regex_contains") or []:
                drop = drop | (
                    (r["language"] == lang)
                    & r["matched_regex"].astype(str).str.contains(needle, case=False, na=False)
                    & (r["retrieval_concept_id"].isin(cids) if cids else True)
                )
            for needle in s4.get("exclude_if_matched_terms_contains") or []:
                drop = drop | (
                    (r["language"] == lang)
                    & r["matched_terms"].astype(str).str.contains(needle, case=False, na=False)
                    & (r["retrieval_concept_id"].isin(cids) if cids else True)
                )
            r = r[~drop]
        elif scenario == "S5":
            if dup_books:
                b = b[~b["book_id"].isin(dup_books)]
                r = r[r["book_id"].isin(set(b["book_id"]))]
        elif scenario == "S6":
            r = r[_high_confidence_mask(r)]
        elif scenario == "S7":
            # Ambiguous German stance triggers.
            drop = (
                (r["language"] == "de")
                & (r["retrieval_concept_id"].isin(["stance_narratorial", "stance_figural"]))
                & (
                    r["matched_terms"].astype(str).str.contains(
                        r"auktorial|personales?\s+erzäh|personale\s+erzähl",
                        case=False,
                        na=False,
                        regex=True,
                    )
                    | r["matched_regex"].astype(str).str.contains(
                        r"auktorial|personal", case=False, na=False
                    )
                )
            )
            r = r[~drop]
        elif scenario == "S8":
            # Handled per-concept below.
            pass
        elif scenario == "S9":
            b = b[b["primary_group"] == "fiction_manual"]
            r = r[r["book_id"].isin(set(b["book_id"]))]
        elif scenario == "S10":
            b = b[b["primary_group"] != "contrast"]
            r = r[r["book_id"].isin(set(b["book_id"]))]
        elif scenario == "S11":
            use_body = True

        for cid in concept_ids:
            rr = r
            bb = b
            if scenario == "S8":
                # Drop top-5 books by assignment count for this concept in each language.
                drop_books = set()
                for lang in ("en", "de"):
                    hit = r[(r["language"] == lang) & (r["retrieval_concept_id"] == cid)]
                    top = hit.groupby("book_id").size().nlargest(5).index.tolist()
                    drop_books.update(top)
                rr = r[~r["book_id"].isin(drop_books)]
                bb = b[~b["book_id"].isin(drop_books)]
            freqs = {
                lang: _freq(rr, bb, cid, lang, use_body=use_body) for lang in ("en", "de")
            }
            ratio = frequency_ratio(freqs["en"], freqs["de"])
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_description": desc,
                    "concept_id": cid,
                    "freq_en_per_10k": freqs["en"],
                    "freq_de_per_10k": freqs["de"],
                    "frequency_ratio_en_over_de": ratio if ratio is not None else np.nan,
                    "method": RETRIEVAL_METHOD,
                    "version": RETRIEVAL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def classify_robustness(sens: pd.DataFrame, concept_id: str) -> str:
    """Label robustness of EN/DE ratio direction/magnitude across scenarios."""
    sub = sens[sens["concept_id"] == concept_id].copy()
    if sub.empty:
        return "unstable_or_insufficient_evidence"
    # Use finite ratios only.
    ratios = []
    for v in sub["frequency_ratio_en_over_de"]:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            ratios.append(None)
        else:
            ratios.append(float(v))
    finite = [r for r in ratios if r is not None]
    if len(finite) < 3:
        return "unstable_or_insufficient_evidence"
    signs = [np.sign(np.log(r)) if r > 0 else 0 for r in finite]
    # Compare to S1
    s1 = sub.loc[sub["scenario"] == "S1", "frequency_ratio_en_over_de"]
    s1v = float(s1.iloc[0]) if len(s1) else None
    # Scenario-specific sensitivity
    s4 = sub.loc[sub["scenario"] == "S4", "frequency_ratio_en_over_de"]
    s2 = sub.loc[sub["scenario"] == "S2", "frequency_ratio_en_over_de"]
    s8 = sub.loc[sub["scenario"] == "S8", "frequency_ratio_en_over_de"]
    s7 = sub.loc[sub["scenario"] == "S7", "frequency_ratio_en_over_de"]

    def _close(a, b, tol=0.25):
        if a is None or b is None or np.isinf(a) or np.isinf(b) or np.isnan(a) or np.isnan(b):
            return False
        if a == 0:
            return abs(b) < tol
        return abs(a - b) / max(abs(a), 1e-9) < tol

    if s1v is not None and len(s4) and not _close(s1v, float(s4.iloc[0]), tol=0.5):
        if concept_id == "focalization_zero":
            return "sensitive_to_particular_lexical_rules"
    if s1v is not None and len(s7) and not _close(s1v, float(s7.iloc[0]), tol=0.5):
        if concept_id in {"stance_narratorial", "stance_figural"}:
            return "sensitive_to_particular_lexical_rules"
    if s1v is not None and len(s2) and not _close(s1v, float(s2.iloc[0]), tol=0.4):
        return "sensitive_to_corpus_composition"
    if s1v is not None and len(s8) and not _close(s1v, float(s8.iloc[0]), tol=0.4):
        return "sensitive_to_individual_books"
    if len(set(signs)) > 1:
        return "unstable_or_insufficient_evidence"
    # Magnitude variability
    if finite and (max(finite) / min(finite) > 2.0 if min(finite) > 0 else True):
        return "stable_direction_variable_magnitude"
    return "stable_across_tested_scenarios"


def bootstrap_concept_freqs(
    retrieval: pd.DataFrame,
    books: pd.DataFrame,
    concepts: list[str] | None = None,
    *,
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    concepts = concepts or FOCUS_CONCEPTS
    rng = np.random.default_rng(seed)
    # Stratify by language × primary_group.
    rows = []
    for cid in concepts:
        samples = {"en": [], "de": [], "ratio": []}
        for _ in range(n_iter):
            freqs = {}
            for lang in ("en", "de"):
                parts = []
                b_lang = books[books["language"] == lang]
                for g, grp in b_lang.groupby("primary_group"):
                    idx = grp.index.to_numpy()
                    if len(idx) == 0:
                        continue
                    draw = rng.choice(idx, size=len(idx), replace=True)
                    parts.append(books.loc[draw])
                boot_books = pd.concat(parts) if parts else b_lang.iloc[0:0]
                elig = float(boot_books["eligible_words"].sum())
                # Resample assignments by selected books (with multiplicity).
                counts = boot_books["book_id"].value_counts()
                hit = retrieval[
                    (retrieval["language"] == lang) & (retrieval["retrieval_concept_id"] == cid)
                ]
                n_assign = 0
                if len(hit) and len(counts):
                    hc = hit.groupby("book_id").size()
                    for bid, mult in counts.items():
                        n_assign += int(hc.get(bid, 0)) * int(mult)
                freqs[lang] = (n_assign / elig * 10000.0) if elig else 0.0
                samples[lang].append(freqs[lang])
            rr = frequency_ratio(freqs["en"], freqs["de"])
            samples["ratio"].append(rr if rr is not None and rr != float("inf") else np.nan)
        for lang in ("en", "de"):
            arr = np.asarray(samples[lang], dtype=float)
            rows.append(
                {
                    "concept_id": cid,
                    "language": lang,
                    "estimate_type": "frequency_per_10k_eligible",
                    "point_mean": float(np.nanmean(arr)),
                    "ci_low": float(np.nanpercentile(arr, 2.5)),
                    "ci_high": float(np.nanpercentile(arr, 97.5)),
                    "n_iter": n_iter,
                    "seed": seed,
                    "note": "Book-level stratified bootstrap; does not capture dictionary bias",
                }
            )
        arr = np.asarray(samples["ratio"], dtype=float)
        rows.append(
            {
                "concept_id": cid,
                "language": "en_over_de",
                "estimate_type": "frequency_ratio",
                "point_mean": float(np.nanmean(arr)),
                "ci_low": float(np.nanpercentile(arr, 2.5)),
                "ci_high": float(np.nanpercentile(arr, 97.5)),
                "n_iter": n_iter,
                "seed": seed,
                "note": "Book-level stratified bootstrap; does not capture dictionary bias",
            }
        )
    return pd.DataFrame(rows)


def historical_overview(books: pd.DataFrame) -> pd.DataFrame:
    rows = []
    b = books.copy()
    # Prefer publication_year then edition_year.
    year = pd.to_numeric(b.get("publication_year", pd.Series(dtype=str)), errors="coerce")
    if "edition_year" in b.columns:
        year = year.fillna(pd.to_numeric(b["edition_year"], errors="coerce"))
    b["year"] = year
    for lang in ("en", "de"):
        sub = b[b["language"] == lang]
        known = sub[sub["year"].notna()]
        unknown = sub[sub["year"].isna()]
        rows.append(
            {
                "language": lang,
                "decade": "unknown",
                "n_books": len(unknown),
                "eligible_words": float(unknown["eligible_words"].sum()),
                "primary_group": "all",
                "note": "missing publication/edition year; not treated as zero",
            }
        )
        if len(known):
            known = known.copy()
            known["decade"] = (known["year"] // 10 * 10).astype(int).astype(str) + "s"
            for (decade, g), grp in known.groupby(["decade", "primary_group"]):
                rows.append(
                    {
                        "language": lang,
                        "decade": decade,
                        "n_books": len(grp),
                        "eligible_words": float(grp["eligible_words"].sum()),
                        "primary_group": g,
                        "note": "",
                    }
                )
    return pd.DataFrame(rows)


def assignments_long(
    lexical: pd.DataFrame,
    bm25_only: pd.DataFrame,
    qualified: pd.DataFrame,
    snapshot_id: str = SNAPSHOT_ID,
) -> pd.DataFrame:
    frames = []
    for df, method, version in (
        (lexical, "lexical", "v03"),
        (qualified, "bm25_qualified", "v03"),
        (bm25_only, "bm25_new", "v03"),
    ):
        if df is None or not len(df):
            continue
        out = pd.DataFrame(
            {
                "language": df["language"],
                "book_id": df["book_id"],
                "passage_id": df["passage_id"],
                "concept_id": df["retrieval_concept_id"],
                "retrieval_method": method,
                "retrieval_version": version,
                "score": df["bm25_score"] if "bm25_score" in df.columns else "",
                "rank": df["rank"] if "rank" in df.columns else "",
                "query_id": "",
                "corpus_snapshot_id": snapshot_id,
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage_table(
    lexical: pd.DataFrame,
    books: pd.DataFrame,
    concept_ids: list[str],
) -> pd.DataFrame:
    rows = []
    den = language_denominators(books)
    for lang in ("en", "de"):
        sub = lexical[lexical["language"] == lang]
        d = den[den["language"] == lang].iloc[0]
        n_pass = int(sub["passage_id"].nunique()) if len(sub) else 0
        n_assign = len(sub)
        n_books = int(sub["book_id"].nunique()) if len(sub) else 0
        elig_w = float(d["eligible_words"])
        elig_p = float(d["eligible_paragraphs"])
        n_abooks = int(d["n_books"])
        nonzero = sum(
            1
            for cid in concept_ids
            if len(sub[sub["retrieval_concept_id"] == cid]) > 0
        )
        rows.append(
            {
                "language": lang,
                "language_label": LANG_LABEL[lang],
                "retrieved_passages": n_pass,
                "assignments": n_assign,
                "books_with_matches": n_books,
                "pct_books_with_matches": 100.0 * n_books / n_abooks if n_abooks else 0.0,
                "pct_eligible_paragraphs": 100.0 * n_pass / elig_p if elig_p else 0.0,
                "passages_per_10k_eligible_words": n_pass / elig_w * 10000.0 if elig_w else 0.0,
                "assignments_per_10k_eligible_words": n_assign / elig_w * 10000.0 if elig_w else 0.0,
                "avg_concepts_per_retrieved_passage": n_assign / n_pass if n_pass else 0.0,
                "concepts_nonzero": nonzero,
                "n_concepts_configured": len(concept_ids),
                "eligible_words": elig_w,
                "eligible_paragraphs": elig_p,
                "n_analysis_books": n_abooks,
            }
        )
    return pd.DataFrame(rows)


def reproducibility_record_v2(
    input_paths: Iterable[Path],
    validation: dict,
    gate: dict,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    paths = [Path(p) for p in input_paths]
    dirty = False
    try:
        import subprocess

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        dirty = True
    return {
        "source_commit": source_commit(),
        "working_tree_dirty": dirty,
        "corpus_snapshot_id": SNAPSHOT_ID,
        "lexicon_version": "0.3-draft",
        "retrieval_algorithm_version": "v0.3",
        "retrieval_execution_id": "v03_lexical_bm25_20261122",
        "analysis_version": ANALYSIS_VERSION,
        "evaluation_infrastructure_version": "v0.5",
        "braw_available": False,
        "resegment_gate": gate,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file_checksums": {
            str(p.relative_to(ROOT)) if p.is_absolute() and str(p).startswith(str(ROOT)) else str(p): file_sha256(p)
            for p in paths
            if p.is_file()
        },
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "validation": validation,
        "n_concepts": len(load_concept_ids()),
        "primary_denominator": "eligible_passage_word_count",
        "supplementary_denominator": "cleaned_body_word_count",
    }
