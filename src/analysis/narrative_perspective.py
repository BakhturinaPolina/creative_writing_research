"""Reusable helpers for bilingual narrative-perspective retrieval analysis."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED, ROOT

LANGUAGE_COLORS = {"English": "#3976C6", "German": "#CC8C42", "en": "#3976C6", "de": "#CC8C42"}
LANG_LABEL = {"en": "English", "de": "German"}
SNAPSHOT_ID = "corpus_snapshot_analysis_v1_candidate_20261122"
FAMILIES_PATH = CONFIG / "concept_families_v05.yml"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_corpus_manifest(lang: str | None = None) -> pd.DataFrame:
    frames = []
    langs = ("en", "de") if lang is None else (lang,)
    for lg in langs:
        path = DATA_PROCESSED / f"corpus_manifest_{lg}_v1_candidate.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        df["language"] = lg
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_passage_manifest(
    lang: str | None = None,
    *,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load passage manifests. Prefer ``usecols`` to limit memory on large files."""
    default_cols = [
        "passage_id",
        "book_id",
        "work_id",
        "language",
        "word_count",
        "character_count",
        "segmentation_status",
        "qc_flags",
        "primary_group",
        "corpus_role",
    ]
    frames = []
    langs = ("en", "de") if lang is None else (lang,)
    for lg in langs:
        path = DATA_PROCESSED / f"passage_manifest_{lg}_v1_candidate.csv"
        cols = usecols
        if cols is None:
            # Read header to intersect with defaults (memory-safe).
            header = pd.read_csv(path, nrows=0).columns.tolist()
            cols = [c for c in default_cols if c in header]
        df = pd.read_csv(path, dtype=str, usecols=cols).fillna("")
        df["language"] = lg
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _concept_col(df: pd.DataFrame) -> str:
    if "retrieval_concept_id" in df.columns and df["retrieval_concept_id"].astype(str).str.len().gt(0).any():
        return "retrieval_concept_id"
    return "concept_id"


def load_retrieval_results(
    method: str = "lexical_v03",
    lang: str | None = None,
) -> pd.DataFrame:
    """Load versioned retrieval pointer CSVs.

    method: lexical_v02 | lexical_algo_fix | lexical_v03 | bm25_qualified_v03 | bm25_only_v03
    """
    stems = {
        "lexical_v02": "pov_baseline_b_v02_lexical_{lang}.csv",
        "lexical_algo_fix": "pov_baseline_b_v02_algo_fix_{lang}.csv",
        "lexical_v03": "pov_baseline_b_v03_lexical_{lang}.csv",
        "bm25_qualified_v03": "pov_baseline_c_v03_bm25_{lang}.csv",
        "bm25_only_v03": "pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv",
    }
    if method not in stems:
        raise ValueError(f"Unknown method {method}; choose from {list(stems)}")
    frames = []
    langs = ("en", "de") if lang is None else (lang,)
    for lg in langs:
        path = DATA_PROCESSED / stems[method].format(lang=lg)
        if not path.is_file():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df["language"] = lg
        df["method"] = method
        df["retrieval_concept_id"] = df[_concept_col(df)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_concept_ids() -> list[str]:
    raw = yaml.safe_load((CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml").read_text(encoding="utf-8"))
    return [c["id"] for c in (raw.get("concepts") or [])]


def load_concept_families() -> dict:
    return yaml.safe_load(FAMILIES_PATH.read_text(encoding="utf-8"))


def validate_retrieval_keys(
    retrieval: pd.DataFrame, passage_manifest: pd.DataFrame
) -> dict:
    pids = set(passage_manifest["passage_id"])
    missing = (~retrieval["passage_id"].isin(pids)).sum() if len(retrieval) else 0
    return {
        "n_assignments": len(retrieval),
        "n_unique_passages": int(retrieval["passage_id"].nunique()) if len(retrieval) else 0,
        "n_missing_from_passage_manifest": int(missing),
    }


def analysis_denominators(corpus: pd.DataFrame) -> pd.DataFrame:
    """Word and paragraph denominators by language (and optional primary_group)."""
    a = corpus[corpus["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))].copy()
    a["number_of_words"] = pd.to_numeric(a["number_of_words"], errors="coerce").fillna(0)
    a["number_of_eligible_paragraphs"] = pd.to_numeric(
        a["number_of_eligible_paragraphs"], errors="coerce"
    ).fillna(0)
    return a


def calculate_normalized_frequencies(
    retrieval: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    per_words: float = 10000.0,
    concept_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Concept × language frequency table including zero-count concepts."""
    concept_ids = concept_ids or load_concept_ids()
    a = analysis_denominators(corpus)
    den = (
        a.groupby("language", as_index=False)
        .agg(
            corpus_words=("number_of_words", "sum"),
            eligible_paragraphs=("number_of_eligible_paragraphs", "sum"),
            n_books=("book_id", "nunique"),
        )
    )
    rows = []
    for lang in ("en", "de"):
        sub = retrieval[retrieval["language"] == lang] if len(retrieval) else retrieval
        drow = den[den["language"] == lang]
        words = float(drow["corpus_words"].iloc[0]) if len(drow) else 0.0
        paras = float(drow["eligible_paragraphs"].iloc[0]) if len(drow) else 0.0
        n_books = int(drow["n_books"].iloc[0]) if len(drow) else 0
        for cid in concept_ids:
            hit = sub[sub["retrieval_concept_id"] == cid] if len(sub) else sub
            n_assign = len(hit)
            n_pass = int(hit["passage_id"].nunique()) if n_assign else 0
            n_b = int(hit["book_id"].nunique()) if n_assign else 0
            # Book-level frequencies for median (include zeros).
            book_ids = a.loc[a["language"] == lang, "book_id"]
            counts = hit.groupby("book_id").size() if n_assign else pd.Series(dtype=float)
            book_words = a.loc[a["language"] == lang].set_index("book_id")["number_of_words"]
            freqs = []
            for bid in book_ids:
                w = float(book_words.get(bid, 0) or 0)
                c = float(counts.get(bid, 0))
                freqs.append((c / w * per_words) if w > 0 else 0.0)
            rows.append(
                {
                    "concept_id": cid,
                    "concept_label": cid.replace("_", " "),
                    "language": lang,
                    "language_label": LANG_LABEL[lang],
                    "n_assignments": n_assign,
                    "n_unique_passages": n_pass,
                    "n_books": n_b,
                    "corpus_words": words,
                    "eligible_paragraphs": paras,
                    "n_analysis_books": n_books,
                    "frequency_per_10000_words": (n_assign / words * per_words) if words else 0.0,
                    "frequency_per_1000_paragraphs": (n_assign / paras * 1000.0) if paras else 0.0,
                    "percentage_of_books": (100.0 * n_b / n_books) if n_books else 0.0,
                    "median_book_frequency": float(np.median(freqs)) if freqs else 0.0,
                    "method": retrieval["method"].iloc[0] if len(retrieval) and "method" in retrieval.columns else "",
                    "version": "v03",
                }
            )
    return pd.DataFrame(rows)


def attach_concept_families(freq: pd.DataFrame) -> pd.DataFrame:
    fam = load_concept_families()
    mapping = {}
    labels = {}
    for block in fam.get("families") or []:
        for cid in block.get("concepts") or []:
            mapping[cid] = block["id"]
            labels[cid] = block["label"]
    out = freq.copy()
    out["concept_family"] = out["concept_id"].map(mapping).fillna("unassigned")
    out["concept_family_label"] = out["concept_id"].map(labels).fillna("Unassigned")
    return out


def calculate_family_frequencies(
    retrieval: pd.DataFrame, corpus: pd.DataFrame
) -> pd.DataFrame:
    """Family-level frequencies with passage deduplication within family."""
    fam = load_concept_families()
    a = analysis_denominators(corpus)
    den = a.groupby("language", as_index=False).agg(
        corpus_words=("number_of_words", "sum"),
        eligible_paragraphs=("number_of_eligible_paragraphs", "sum"),
    )
    rows = []
    for block in fam.get("families") or []:
        cids = set(block.get("concepts") or [])
        for lang in ("en", "de"):
            sub = retrieval[
                (retrieval["language"] == lang)
                & (retrieval["retrieval_concept_id"].isin(cids))
            ]
            drow = den[den["language"] == lang]
            words = float(drow["corpus_words"].iloc[0]) if len(drow) else 0.0
            n_pass = int(sub["passage_id"].nunique()) if len(sub) else 0
            n_assign = len(sub)
            rows.append(
                {
                    "concept_family": block["id"],
                    "concept_family_label": block["label"],
                    "language": lang,
                    "language_label": LANG_LABEL[lang],
                    "n_assignments": n_assign,
                    "n_unique_passages": n_pass,
                    "frequency_per_10000_words": (n_pass / words * 10000.0) if words else 0.0,
                    "note": "numerator = unique passages in family (deduplicated)",
                    "method": "lexical_v03",
                    "version": "v03",
                }
            )
    return pd.DataFrame(rows)


def calculate_book_dispersion(
    retrieval: pd.DataFrame, corpus: pd.DataFrame, concept_id: str, lang: str
) -> dict:
    a = analysis_denominators(corpus)
    books = a[a["language"] == lang].copy()
    hit = retrieval[
        (retrieval["language"] == lang) & (retrieval["retrieval_concept_id"] == concept_id)
    ]
    counts = hit.groupby("book_id").size() if len(hit) else pd.Series(dtype=float)
    books["n_assign"] = books["book_id"].map(counts).fillna(0)
    books["freq_per_10k"] = np.where(
        books["number_of_words"] > 0,
        books["n_assign"] / books["number_of_words"] * 10000.0,
        0.0,
    )
    n = len(books)
    top5 = books.nlargest(5, "n_assign")
    top5_share = (
        float(top5["n_assign"].sum() / books["n_assign"].sum())
        if books["n_assign"].sum() > 0
        else 0.0
    )
    return {
        "concept_id": concept_id,
        "language": lang,
        "n_books_with_concept": int((books["n_assign"] > 0).sum()),
        "pct_books_with_concept": 100.0 * float((books["n_assign"] > 0).mean()) if n else 0.0,
        "mean_book_frequency": float(books["freq_per_10k"].mean()) if n else 0.0,
        "median_book_frequency": float(books["freq_per_10k"].median()) if n else 0.0,
        "q1_book_frequency": float(books["freq_per_10k"].quantile(0.25)) if n else 0.0,
        "q3_book_frequency": float(books["freq_per_10k"].quantile(0.75)) if n else 0.0,
        "pct_books_zero": 100.0 * float((books["n_assign"] == 0).mean()) if n else 0.0,
        "top5_books_contribution": top5_share,
        "book_table": books,
    }


def calculate_concept_cooccurrence(
    retrieval: pd.DataFrame, lang: str, min_support: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Passage-level co-occurrence counts and Jaccard matrix for one language."""
    concepts = load_concept_ids()
    sub = retrieval.loc[
        retrieval["language"] == lang, ["passage_id", "retrieval_concept_id"]
    ]
    by_pass: dict[str, set[str]] = defaultdict(set)
    for pid, cid in zip(sub["passage_id"].to_numpy(), sub["retrieval_concept_id"].to_numpy()):
        by_pass[pid].add(cid)
    sets = {c: set() for c in concepts}
    for pid, cs in by_pass.items():
        for c in cs:
            if c in sets:
                sets[c].add(pid)
    del by_pass
    n = len(concepts)
    counts = np.zeros((n, n), dtype=np.int32)
    jacc = np.zeros((n, n), dtype=np.float32)
    sizes = {c: len(sets[c]) for c in concepts}
    for i, a in enumerate(concepts):
        sa = sets[a]
        for j in range(i, n):
            b = concepts[j]
            sb = sets[b]
            inter = len(sa & sb) if i != j else sizes[a]
            union = sizes[a] + sizes[b] - inter if i != j else sizes[a]
            counts[i, j] = counts[j, i] = inter
            val = (inter / union) if union else 0.0
            jacc[i, j] = jacc[j, i] = val
    count_df = pd.DataFrame(counts, index=concepts, columns=concepts)
    jacc_df = pd.DataFrame(jacc, index=concepts, columns=concepts)
    pairs = []
    for i, a in enumerate(concepts):
        for j in range(i + 1, n):
            b = concepts[j]
            if counts[i, j] >= min_support:
                pairs.append(
                    {
                        "language": lang,
                        "concept_a": a,
                        "concept_b": b,
                        "cooccurrence": int(counts[i, j]),
                        "support_a": sizes[a],
                        "support_b": sizes[b],
                        "jaccard": float(jacc[i, j]),
                    }
                )
    return count_df, pd.DataFrame(pairs)


def compare_retrieval_methods(
    lexical: pd.DataFrame, bm25_only: pd.DataFrame, qualified: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Complementarity using (passage_id, concept) and passage_id units."""
    rows = []
    for lang in ("en", "de"):
        L = set(
            zip(
                lexical.loc[lexical["language"] == lang, "passage_id"],
                lexical.loc[lexical["language"] == lang, "retrieval_concept_id"],
            )
        )
        Bnew = set(
            zip(
                bm25_only.loc[bm25_only["language"] == lang, "passage_id"],
                bm25_only.loc[bm25_only["language"] == lang, "retrieval_concept_id"],
            )
        )
        Lp = {p for p, _ in L}
        Bnew_p = {p for p, _ in Bnew}
        new_passages = Bnew_p - Lp
        new_on_known = {p for p, _ in Bnew if p in Lp}
        U_p = Lp | Bnew_p
        j_pass = len(Lp & Bnew_p) / len(Lp | Bnew_p) if (Lp | Bnew_p) else 0.0
        rows.append(
            {
                "language": lang,
                "language_label": LANG_LABEL[lang],
                "n_lexical_assignments": len(L),
                "n_lexical_passages": len(Lp),
                "n_bnew_assignments": len(Bnew),
                "n_genuinely_new_bm25_passages": len(new_passages),
                "n_known_passages_with_new_bm25_concepts": len(new_on_known),
                "n_combined_unique_passages": len(U_p),
                "jaccard_passages_L_vs_Bnew": j_pass,
                "pool_note": "Bnew from bm25_only_unvalidated file (≈ Braw\\\\L); not independent top-K",
                "method": "lexical_v03_vs_bm25_only_v03",
                "version": "v03",
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    lexical: pd.DataFrame,
    corpus: pd.DataFrame,
    passages: pd.DataFrame,
) -> pd.DataFrame:
    """Scenarios S1–S5 for selected concept frequency ratios."""
    fam = load_concept_families()
    s4 = fam.get("sensitivity_s4_exclusions") or {}
    if "segmentation_status" in passages.columns or "qc_flags" in passages.columns:
        seg = passages["segmentation_status"].astype(str) if "segmentation_status" in passages.columns else pd.Series("", index=passages.index)
        qcf = passages["qc_flags"].astype(str) if "qc_flags" in passages.columns else pd.Series("", index=passages.index)
        extreme_ids = set(
            passages.loc[
                seg.str.contains("extreme", case=False, na=False)
                | qcf.str.contains("extreme_length", case=False, na=False),
                "passage_id",
            ]
        )
    else:
        # Caller may pass only the extreme-passage QC table.
        extreme_ids = set(passages["passage_id"]) if "passage_id" in passages.columns else set()
    # Possible duplicate books from corpus qc_notes.
    dup_books = set(
        corpus.loc[
            corpus["qc_notes"].astype(str).str.contains("possible_duplicate", case=False),
            "book_id",
        ]
    )

    def _filter(scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        c = corpus.copy()
        r = lexical.copy()
        if scenario == "S1":
            pass
        elif scenario == "S2":
            c = c[c["primary_group"] != "contrast"]
            r = r[r["book_id"].isin(set(c["book_id"]))]
        elif scenario == "S3":
            r = r[~r["passage_id"].isin(extreme_ids)]
        elif scenario == "S4":
            r = r.copy()
            if s4.get("language"):
                lang = s4["language"]
                cids = set(s4.get("exclude_concept_ids_when_trigger_overlap") or [])
                drop = pd.Series(False, index=r.index)
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
                c = c[~c["book_id"].isin(dup_books)]
                r = r[r["book_id"].isin(set(c["book_id"]))]
        return r, c

    focus = [
        "focalization_zero",
        "stance_narratorial",
        "stance_figural",
        "grammatical_third_person",
        "filter_words",
        "reader_effect",
    ]
    rows = []
    for scenario, desc in (
        ("S1", "Complete analysis-set corpus"),
        ("S2", "Primary creative writing manuals only (exclude contrast)"),
        ("S3", "Exclude extreme-length passages"),
        ("S4", "Conservative DE ambiguous allwissen/focalization_zero exclusions"),
        ("S5", "Exclude books flagged possible_duplicate"),
    ):
        r, c = _filter(scenario)
        # Adjust denominators for S3: subtract extreme passage word counts.
        if scenario == "S3":
            c = analysis_denominators(c)
            pw = passages.copy()
            pw["word_count"] = pd.to_numeric(pw["word_count"], errors="coerce").fillna(0)
            extreme_words = (
                pw[pw["passage_id"].isin(extreme_ids)]
                .groupby("language")["word_count"]
                .sum()
                .to_dict()
            )
        else:
            c = analysis_denominators(c)
            extreme_words = {}

        for cid in focus:
            freqs = {}
            for lang in ("en", "de"):
                sub = r[(r["language"] == lang) & (r["retrieval_concept_id"] == cid)]
                words = float(
                    c.loc[c["language"] == lang, "number_of_words"].sum()
                )
                if scenario == "S3":
                    words = max(0.0, words - float(extreme_words.get(lang, 0)))
                freqs[lang] = (len(sub) / words * 10000.0) if words else 0.0
            rr = (
                freqs["en"] / freqs["de"]
                if freqs["de"] > 0
                else (np.inf if freqs["en"] > 0 else np.nan)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_description": desc,
                    "concept_id": cid,
                    "freq_en_per_10k": freqs["en"],
                    "freq_de_per_10k": freqs["de"],
                    "frequency_ratio_en_over_de": rr,
                    "method": "lexical_v03",
                    "version": "v03",
                }
            )
    return pd.DataFrame(rows)


def frequency_ratio(f_en: float, f_de: float) -> float | None:
    if f_de == 0 and f_en == 0:
        return None
    if f_de == 0:
        return float("inf")
    return f_en / f_de


def reproducibility_record(input_paths: Iterable[Path]) -> dict:
    paths = [Path(p) for p in input_paths]
    return {
        "source_commit": source_commit(),
        "corpus_snapshot_id": SNAPSHOT_ID,
        "lexicon_version": "0.3-draft",
        "retrieval_baseline_version": "v0.3",
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file_checksums": {
            str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else str(p): file_sha256(p)
            for p in paths
            if p.is_file()
        },
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def save_table(df: pd.DataFrame, name: str, reports_tables: Path) -> Path:
    reports_tables.mkdir(parents=True, exist_ok=True)
    path = reports_tables / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
