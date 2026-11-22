"""Stratified bilingual manual-evaluation sample (no gold labels assigned).

v0.2 builder preserved as ``build_sample`` (legacy stems).
v0.3 builder uses reservoir sampling and versioned outputs (frozen).
v0.4 splits probability-based (A) and book-diversity diagnostic (B) designs.
v0.5 builds mutually exclusive stratum *populations* before sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.paragraphs import iter_body_paragraphs
from src.retrieval.run_meta import base_run_meta, write_json
from src.retrieval.sampling import ReservoirSample, StratifiedReservoirs, derive_stratum_seed

OUT = DATA_INTERIM / "retrieval"
SEED = 20261121
SEED_V03 = 20261122
SEED_V04 = 20261123
SEED_V05 = 20261124
TARGET = 260  # within 200–300
HELD_OUT_DEFAULT = OUT / "lexicon_dev_held_out_books_v03.txt"

# Design A quotas per language (probability-based; no book-diversity step).
PROB_QUOTAS_V04 = {
    "high_rank_bm25": 25,
    "mid_low_rank_bm25": 25,
    "lexical_only": 25,
    "bm25_only": 25,
    "random_non_retrieved": 30,
}

# v0.5 quotas per language. Populations are exclusive *before* sampling.
# Target ≈ 125/lang ≈ 250 total; unused quota redistributed (see redistribute_quotas_v05).
PROB_QUOTAS_V05 = {
    "high_ranked_qualified_bm25": 20,
    "lower_ranked_qualified_bm25": 20,
    "lexical_only": 20,
    "bm25_expansion_known": 15,
    "new_bm25_passages": 20,
    "non_retrieved": 30,
}

PRIMARY_STRATA_V05 = (
    "high_ranked_qualified_bm25",
    "lower_ranked_qualified_bm25",
    "lexical_only",
    "bm25_expansion_known",
    "new_bm25_passages",
    "non_retrieved",
)

# Order for unused-quota redistribution among retrieved strata, then non-retrieved.
QUOTA_REDISTRIBUTE_ORDER_V05 = (
    "high_ranked_qualified_bm25",
    "lower_ranked_qualified_bm25",
    "lexical_only",
    "bm25_expansion_known",
    "new_bm25_passages",
    "non_retrieved",
)

REVIEW_FIELDS = [
    "relevant_to_narrative_perspective",
    "correct_concept_ids",
    "concept_relevance",
    "explicit_or_implicit_discussion",
    "instruction_or_fictional_example",
    "discussion_purpose",
    "false_positive_reason",
    "reviewer_notes",
]

# Annotator-facing columns hide retrieval method / rank / predicted concept.
ANNOTATOR_COLS = [
    "sample_id",
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
    "sample_stratum",
    *REVIEW_FIELDS,
]

HIDDEN_META_COLS = [
    "sample_id",
    "passage_id",
    "concept_id",
    "retrieval_method",
    "bm25_score",
    "rank",
    "matched_terms",
    "matched_regex",
    "candidate_status",
    "selection_rule",
    "sample_stratum",
    "sample_reason",
    "evidence_kind",
]


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _ensure_pid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "passage_id" not in df.columns:
        df = df.copy()
        df["passage_id"] = (
            df["book_id"] + ":c" + df["chapter_idx"].astype(str) + ":p" + df["para_idx"].astype(str)
        )
    return df


def build_sample(log, *, seed: int = SEED, target: int = TARGET) -> Path:
    """Legacy v0.2 sample builder (keeps original early-exit behaviour for reproducibility)."""
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for lang in ("en", "de"):
        b = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_b_v02_lexical_{lang}.csv"))
        c = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_c_bm25_{lang}.csv"))
        only = _ensure_pid(
            _load(DATA_PROCESSED / f"pov_baseline_c_bm25_only_unvalidated_{lang}.csv")
        )
        for label, df in (
            ("lexical_B", b),
            ("bm25_C", c),
            ("bm25_only_unvalidated", only),
        ):
            if df.empty:
                continue
            d = df.copy()
            d["sample_stratum"] = label
            d["language"] = lang
            frames.append(d)

    if not frames:
        raise SystemExit("No baseline outputs found; run retrieval first.")

    pool = pd.concat(frames, ignore_index=True)

    def _ev_tag(row) -> str:
        tiers = row.get("evidence_tiers") or ""
        regex = row.get("matched_regex") or ""
        status = row.get("candidate_status") or ""
        if status == "bm25_only_unvalidated":
            return "bm25_only"
        if regex.strip():
            return "regex"
        try:
            t = json.loads(tiers) if tiers.startswith("{") else {}
        except Exception:
            t = {}
        if int(t.get("academic_high") or 0) > 0:
            return "academic"
        if int(t.get("supporting_craft") or 0) > 0:
            return "craft"
        return "other"

    pool["evidence_kind"] = pool.apply(_ev_tag, axis=1)

    selected_ids: set[str] = set()
    selected_rows: list[dict] = []

    def take(df: pd.DataFrame, n: int, reason: str) -> None:
        nonlocal selected_rows
        if df.empty or n <= 0:
            return
        df = df.drop_duplicates("passage_id")
        df = df[~df["passage_id"].isin(selected_ids)]
        if df.empty:
            return
        n = min(n, len(df))
        idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        for i in idx:
            row = df.loc[i].to_dict()
            row["sample_reason"] = reason
            selected_ids.add(row["passage_id"])
            selected_rows.append(row)

    for lang in ("en", "de"):
        c = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "bm25_C")]
        if c.empty:
            continue
        c = c.copy()
        c["rank_num"] = pd.to_numeric(c["rank"], errors="coerce")
        take(c[c["rank_num"] <= 20], 25, "high_rank_bm25")
        take(c[(c["rank_num"] > 20) & (c["rank_num"] <= 100)], 20, "mid_rank_bm25")
        take(c[c["rank_num"] > 100], 15, "low_rank_bm25")

    for lang in ("en", "de"):
        b = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "lexical_B")]
        take(b[b["evidence_kind"] == "academic"], 20, "academic_term")
        take(b[b["evidence_kind"] == "craft"], 15, "craft_term")
        take(b[b["evidence_kind"] == "regex"], 15, "regex_match")

    for lang in ("en", "de"):
        o = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "bm25_only_unvalidated")]
        take(o, 15, "bm25_only_unvalidated")

    for lang in ("en", "de"):
        for grp in ("fiction_manual", "pedagogy_genre_craft", "contrast"):
            sub = pool[(pool["language"] == lang) & (pool["primary_group"] == grp)]
            take(sub, 8, f"subgroup_{grp}")

    weak = (
        pool[pool.get("selection_rule", pd.Series(dtype=str)).isin(
            ["weak_pov_plus_supporting", "two_independent_supporting"]
        )]
        if "selection_rule" in pool.columns
        else pd.DataFrame()
    )
    take(weak, 20, "potential_false_positive")

    retrieved = selected_ids | set(pool["passage_id"].unique())
    for lang in ("en", "de"):
        non = []
        for para in iter_body_paragraphs(lang):
            if para.passage_id in retrieved:
                continue
            non.append(para)
            if len(non) >= 5000:
                break
        if not non:
            continue
        pick = rng.choice(len(non), size=min(20, len(non)), replace=False)
        for i in pick:
            p = non[int(i)]
            if p.passage_id in selected_ids:
                continue
            selected_ids.add(p.passage_id)
            selected_rows.append(
                {
                    "passage_id": p.passage_id,
                    "book_id": p.book_id,
                    "language": lang,
                    "title": p.title,
                    "author": p.author,
                    "edition_year": p.edition_year,
                    "corpus_role": p.corpus_role,
                    "primary_group": p.primary_group,
                    "chapter_idx": p.chapter_idx,
                    "para_idx": p.para_idx,
                    "concept_id": "",
                    "retrieval_method": "random_non_retrieved",
                    "bm25_score": "",
                    "rank": "",
                    "matched_terms": "",
                    "matched_regex": "",
                    "candidate_status": "non_retrieved",
                    "sample_stratum": "random_non_retrieved",
                    "sample_reason": "random_non_retrieved",
                    "evidence_kind": "none",
                }
            )

    seen = set()
    uniq = []
    for r in selected_rows:
        pid = r["passage_id"]
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(r)
    if len(uniq) > target:
        uniq = list(rng.choice(uniq, size=target, replace=False))

    out_df = pd.DataFrame(uniq)
    for f in REVIEW_FIELDS:
        if f not in out_df.columns:
            out_df[f] = ""
        else:
            out_df[f] = ""

    pointer_cols = [
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
        "concept_id",
        "retrieval_method",
        "bm25_score",
        "rank",
        "matched_terms",
        "matched_regex",
        "candidate_status",
        "sample_stratum",
        "sample_reason",
        "evidence_kind",
        *REVIEW_FIELDS,
    ]
    cols = [c for c in pointer_cols if c in out_df.columns]
    out_path = OUT / "manual_eval_sample.csv"
    out_df[cols].to_csv(out_path, index=False)

    need = set(out_df["passage_id"])
    texts = {}
    for lang in ("en", "de"):
        for para in iter_body_paragraphs(lang, with_neighbors=True):
            if para.passage_id in need:
                texts[para.passage_id] = (para.text, para.prev_text, para.next_text)
            if len(texts) >= len(need):
                break
    side = out_df[cols].copy()
    side["passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[0])
    side["prev_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[1])
    side["next_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[2])
    side_path = OUT / "manual_eval_sample_passages.csv"
    side.to_csv(side_path, index=False)

    meta = base_run_meta(
        kind="manual_eval_sample",
        seed=seed,
        target=target,
        n_unique_passages=len(out_df),
        strata=out_df["sample_reason"].value_counts().to_dict() if len(out_df) else {},
        languages=out_df["language"].value_counts().to_dict() if len(out_df) else {},
        review_fields=REVIEW_FIELDS,
        note="No gold labels assigned. Keep separate from rule-tuning data.",
    )
    write_json(OUT / "manual_eval_sample.meta.json", meta)
    log.info(f"wrote {out_path} ({len(out_df)} passages); sidecar {side_path}")
    return out_path


def build_sample_v03(
    log,
    *,
    seed: int = SEED_V03,
    target: int = TARGET,
    held_out_books: set[str] | None = None,
) -> Path:
    """v0.3 evaluation sample with reservoir-sampled non-retrieved passages.

    Mutually exclusive strata (priority order):
      1. high_rank_bm25
      2. mid_low_rank_bm25
      3. lexical_only
      4. bm25_only
      5. random_non_retrieved (reservoir over full eligible corpus)

    Target ≈ 130 EN + 130 DE. Held-out development books are excluded when provided.
    """
    rng = np.random.default_rng(seed)
    held_out_books = held_out_books or set()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for lang in ("en", "de"):
        b = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv"))
        c = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_{lang}.csv"))
        only = _ensure_pid(
            _load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv")
        )
        for label, df in (
            ("lexical_B", b),
            ("bm25_C", c),
            ("bm25_only_unvalidated", only),
        ):
            if df.empty:
                continue
            d = df.copy()
            d["sample_stratum"] = label
            d["language"] = lang
            if held_out_books and "book_id" in d.columns:
                d = d[~d["book_id"].isin(held_out_books)]
            frames.append(d)

    if not frames:
        raise SystemExit("No v0.3 baseline outputs found; run retrieval first.")

    pool = pd.concat(frames, ignore_index=True)

    # Lexical-only = in B but not in C for same passage_id
    b_ids = set(pool.loc[pool["sample_stratum"] == "lexical_B", "passage_id"])
    c_ids = set(pool.loc[pool["sample_stratum"] == "bm25_C", "passage_id"])
    only_ids = set(pool.loc[pool["sample_stratum"] == "bm25_only_unvalidated", "passage_id"])
    lexical_only_ids = b_ids - c_ids - only_ids

    selected_ids: set[str] = set()
    selected_rows: list[dict] = []

    def take(df: pd.DataFrame, n: int, reason: str, stratum: str) -> None:
        nonlocal selected_rows
        if df.empty or n <= 0:
            return
        df = df.drop_duplicates("passage_id")
        df = df[~df["passage_id"].isin(selected_ids)]
        if df.empty:
            return
        n = min(n, len(df))
        idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        for i in idx:
            row = df.loc[i].to_dict()
            row["sample_reason"] = reason
            row["sample_stratum"] = stratum
            selected_ids.add(row["passage_id"])
            selected_rows.append(row)

    # Per-language quotas from plan §14
    quotas = {
        "high_rank_bm25": 25,
        "mid_low_rank_bm25": 25,
        "lexical_only": 25,
        "bm25_only": 25,
        "random_non_retrieved": 30,
    }

    stratum_meta: dict = {}

    for lang in ("en", "de"):
        c = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "bm25_C")].copy()
        if not c.empty:
            c["rank_num"] = pd.to_numeric(c["rank"], errors="coerce")
            take(c[c["rank_num"] <= 50], quotas["high_rank_bm25"], "high_rank_bm25", "high_rank_bm25")
            take(
                c[c["rank_num"] > 50],
                quotas["mid_low_rank_bm25"],
                "mid_low_rank_bm25",
                "mid_low_rank_bm25",
            )

        b = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "lexical_B")]
        take(
            b[b["passage_id"].isin(lexical_only_ids)],
            quotas["lexical_only"],
            "lexical_only",
            "lexical_only",
        )

        o = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "bm25_only_unvalidated")]
        take(o, quotas["bm25_only"], "bm25_only", "bm25_only")

        # Reservoir over ALL non-retrieved eligible passages (never stop early).
        retrieved_all = set(pool.loc[pool["language"] == lang, "passage_id"]) | selected_ids
        # Oversample per primary_group then downselect to quota for book diversity.
        reservoirs = StratifiedReservoirs(k_per_stratum=40, seed=seed + (0 if lang == "en" else 1))
        for para in iter_body_paragraphs(lang):
            if para.passage_id in retrieved_all:
                continue
            if held_out_books and para.book_id in held_out_books:
                continue
            stratum = f"{lang}::{para.primary_group or 'unknown'}"
            reservoirs.consider(
                stratum,
                {
                    "passage_id": para.passage_id,
                    "book_id": para.book_id,
                    "language": lang,
                    "title": para.title,
                    "author": para.author,
                    "edition_year": para.edition_year,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "concept_id": "",
                    "retrieval_method": "random_non_retrieved",
                    "bm25_score": "",
                    "rank": "",
                    "matched_terms": "",
                    "matched_regex": "",
                    "candidate_status": "non_retrieved",
                    "sample_stratum": "random_non_retrieved",
                    "sample_reason": "random_non_retrieved",
                    "evidence_kind": "none",
                },
            )
        stratum_meta[lang] = reservoirs.meta()
        candidates = reservoirs.all_items()
        # Prefer diversity across books
        by_book: dict[str, list] = {}
        for row in candidates:
            by_book.setdefault(row["book_id"], []).append(row)
        picks: list[dict] = []
        books = list(by_book.keys())
        rng.shuffle(books)
        while len(picks) < quotas["random_non_retrieved"] and books:
            still = []
            for bid in books:
                if not by_book[bid]:
                    continue
                row = by_book[bid].pop()
                if row["passage_id"] in selected_ids:
                    continue
                picks.append(row)
                if by_book[bid]:
                    still.append(bid)
                if len(picks) >= quotas["random_non_retrieved"]:
                    break
            books = still
            if not still:
                break
        for row in picks:
            selected_ids.add(row["passage_id"])
            selected_rows.append(row)

    seen = set()
    uniq = []
    for r in selected_rows:
        pid = r["passage_id"]
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(r)

    # Trim toward target while preserving language balance if possible
    if len(uniq) > target:
        en = [r for r in uniq if r.get("language") == "en"]
        de = [r for r in uniq if r.get("language") == "de"]
        half = target // 2
        en = list(rng.choice(en, size=min(half, len(en)), replace=False)) if en else []
        de = list(rng.choice(de, size=min(target - len(en), len(de)), replace=False)) if de else []
        uniq = list(en) + list(de)

    out_df = pd.DataFrame(uniq)
    out_df["sample_id"] = [f"v03-{i:04d}" for i in range(1, len(out_df) + 1)]
    for f in REVIEW_FIELDS:
        out_df[f] = ""

    # Full meta table (includes retrieval cues — not for annotators)
    meta_cols = [c for c in HIDDEN_META_COLS + [
        "book_id", "language", "title", "author", "edition_year",
        "corpus_role", "primary_group", "chapter_idx", "para_idx",
    ] + REVIEW_FIELDS if c in out_df.columns]
    out_path = OUT / "manual_eval_sample_v03.csv"
    out_df[meta_cols].to_csv(out_path, index=False)

    # Annotator view — hide method/rank/predicted concept
    ann_cols = [c for c in ANNOTATOR_COLS if c in out_df.columns]
    ann_path = OUT / "manual_eval_sample_v03_annotator.csv"
    out_df[ann_cols].to_csv(ann_path, index=False)

    # Passage sidecar
    need = set(out_df["passage_id"])
    texts = {}
    for lang in ("en", "de"):
        for para in iter_body_paragraphs(lang, with_neighbors=True):
            if para.passage_id in need:
                texts[para.passage_id] = (para.text, para.prev_text, para.next_text)
            if len(texts) >= len(need):
                break
    side = out_df[meta_cols].copy()
    side["passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[0])
    side["prev_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[1])
    side["next_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[2])
    side_path = OUT / "manual_eval_sample_v03_passages.csv"
    side.to_csv(side_path, index=False)

    meta = base_run_meta(
        kind="manual_eval_sample_v03",
        seed=seed,
        target=target,
        n_unique_passages=len(out_df),
        strata=out_df["sample_stratum"].value_counts().to_dict() if len(out_df) else {},
        languages=out_df["language"].value_counts().to_dict() if len(out_df) else {},
        review_fields=REVIEW_FIELDS,
        reservoir_meta=stratum_meta,
        held_out_books=sorted(held_out_books),
        mutually_exclusive_priority=[
            "high_rank_bm25",
            "mid_low_rank_bm25",
            "lexical_only",
            "bm25_only",
            "random_non_retrieved",
        ],
        note=(
            "No gold labels. Reservoir sampling over full non-retrieved corpus. "
            "Annotator CSV hides retrieval method/rank/predicted concept. "
            "Not an unbiased estimate of absolute corpus-wide recall."
        ),
    )
    write_json(OUT / "manual_eval_sample_v03.meta.json", meta)
    log.info(
        f"wrote {out_path} ({len(out_df)} passages); annotator {ann_path}; sidecar {side_path}"
    )
    return out_path


def load_held_out_books(path: Path | None = None) -> set[str]:
    path = path or HELD_OUT_DEFAULT
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _assignment_key_col(df: pd.DataFrame) -> str:
    if "retrieval_concept_id" in df.columns and df["retrieval_concept_id"].astype(str).str.len().gt(0).any():
        return "retrieval_concept_id"
    return "concept_id"


def _srs_without_replacement(
    df: pd.DataFrame, n: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Uniform SRS without replacement; records N_h, n_h, pi, weight."""
    N = len(df)
    meta = {
        "N_h": N,
        "n_h": 0,
        "inclusion_probability": None,
        "sampling_weight": None,
        "seed": seed,
        "design": "srs_without_replacement",
    }
    if N == 0 or n <= 0:
        return df.iloc[0:0].copy(), meta
    n_take = min(n, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n_take, replace=False)
    out = df.loc[idx].copy()
    pi = n_take / N
    meta.update(
        {
            "n_h": n_take,
            "inclusion_probability": pi,
            "sampling_weight": (N / n_take) if n_take else None,
        }
    )
    return out, meta


def _load_v03_pools(held_out_books: set[str]) -> pd.DataFrame:
    frames = []
    for lang in ("en", "de"):
        b = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv"))
        c = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_{lang}.csv"))
        only = _ensure_pid(
            _load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv")
        )
        for label, df in (
            ("lexical_B", b),
            ("bm25_C", c),
            ("bm25_only_unvalidated", only),
        ):
            if df.empty:
                continue
            d = df.copy()
            d["sample_stratum"] = label
            d["language"] = lang
            if held_out_books and "book_id" in d.columns:
                d = d[~d["book_id"].isin(held_out_books)]
            frames.append(d)
    if not frames:
        raise SystemExit("No v0.3 baseline outputs found; run retrieval first.")
    return pd.concat(frames, ignore_index=True)


def _write_sample_bundle(
    out_df: pd.DataFrame,
    *,
    stem: str,
    sample_id_prefix: str,
    meta: dict,
    log,
    with_passages: bool = True,
) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out_df = out_df.copy()
    out_df["sample_id"] = [f"{sample_id_prefix}-{i:04d}" for i in range(1, len(out_df) + 1)]
    for f in REVIEW_FIELDS:
        out_df[f] = ""

    meta_cols = [
        c
        for c in HIDDEN_META_COLS
        + [
            "book_id",
            "language",
            "title",
            "author",
            "edition_year",
            "corpus_role",
            "primary_group",
            "chapter_idx",
            "para_idx",
            "inclusion_probability",
            "sampling_weight",
            "stratum_population_N",
            "stratum_sample_n",
        ]
        + REVIEW_FIELDS
        if c in out_df.columns
    ]
    out_path = OUT / f"{stem}.csv"
    out_df[meta_cols].to_csv(out_path, index=False)

    ann_cols = [c for c in ANNOTATOR_COLS if c in out_df.columns]
    ann_path = OUT / f"{stem}_annotator.csv"
    out_df[ann_cols].to_csv(ann_path, index=False)

    side_path = OUT / f"{stem}_passages.csv"
    if with_passages:
        need = set(out_df["passage_id"])
        texts: dict[str, tuple[str, str, str]] = {}
        for lang in ("en", "de"):
            for para in iter_body_paragraphs(lang, with_neighbors=True):
                if para.passage_id in need:
                    texts[para.passage_id] = (para.text, para.prev_text, para.next_text)
                if len(texts) >= len(need):
                    break
        side = out_df[meta_cols].copy()
        side["passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[0])
        side["prev_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[1])
        side["next_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[2])
        side.to_csv(side_path, index=False)

    write_json(OUT / f"{stem}.meta.json", meta)
    log.info(f"wrote {out_path} ({len(out_df)} passages); annotator {ann_path}")
    return out_path


def build_sample_prob_v04(
    log,
    *,
    seed: int = SEED_V04,
    held_out_books: set[str] | None = None,
    with_passages: bool = True,
) -> Path:
    """Design A: probability-based evaluation sample (no book-diversity step).

    Estimates refer to the **independent evaluation population** (analysis-set
    books minus lexicon-development held-out books), within each named stratum
    population — not the entire unfiltered corpus.
    """
    held_out_books = held_out_books if held_out_books is not None else load_held_out_books()
    pool = _load_v03_pools(held_out_books)

    b_ids = set(pool.loc[pool["sample_stratum"] == "lexical_B", "passage_id"])
    c_ids = set(pool.loc[pool["sample_stratum"] == "bm25_C", "passage_id"])
    only_ids = set(pool.loc[pool["sample_stratum"] == "bm25_only_unvalidated", "passage_id"])
    lexical_only_ids = b_ids - c_ids - only_ids

    selected_rows: list[dict] = []
    stratum_meta: dict = {"master_seed": seed, "languages": {}}
    selected_ids: set[str] = set()

    for lang in ("en", "de"):
        lang_meta: dict = {}
        # --- candidate strata (passage-level unique within mutually exclusive buckets)
        c = pool[(pool["language"] == lang) & (pool["sample_stratum"] == "bm25_C")].copy()
        if not c.empty:
            c["rank_num"] = pd.to_numeric(c["rank"], errors="coerce")

        def _pop_unique(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            return df.drop_duplicates("passage_id")

        strata_pops = {
            "high_rank_bm25": _pop_unique(c[c["rank_num"] <= 50]) if not c.empty else c,
            "mid_low_rank_bm25": _pop_unique(c[c["rank_num"] > 50]) if not c.empty else c,
            "lexical_only": _pop_unique(
                pool[
                    (pool["language"] == lang)
                    & (pool["sample_stratum"] == "lexical_B")
                    & (pool["passage_id"].isin(lexical_only_ids))
                ]
            ),
            "bm25_only": _pop_unique(
                pool[
                    (pool["language"] == lang)
                    & (pool["sample_stratum"] == "bm25_only_unvalidated")
                ]
            ),
        }

        for stratum, pop in strata_pops.items():
            # Exclude already selected (mutually exclusive assignment)
            pop = pop[~pop["passage_id"].isin(selected_ids)] if not pop.empty else pop
            s_seed = derive_stratum_seed(seed, f"{lang}::{stratum}")
            sampled, sm = _srs_without_replacement(pop, PROB_QUOTAS_V04[stratum], s_seed)
            sm["population_definition"] = (
                f"Independent eval population; retrieval-candidate pool stratum={stratum}"
            )
            lang_meta[stratum] = sm
            for _, row in sampled.iterrows():
                d = row.to_dict()
                d["sample_reason"] = stratum
                d["sample_stratum"] = stratum
                d["inclusion_probability"] = sm["inclusion_probability"]
                d["sampling_weight"] = sm["sampling_weight"]
                d["stratum_population_N"] = sm["N_h"]
                d["stratum_sample_n"] = sm["n_h"]
                selected_ids.add(d["passage_id"])
                selected_rows.append(d)

        # --- non-retrieved: reservoir with k = final n_h so π = n_h/N_h is exact
        retrieved_all = set(pool.loc[pool["language"] == lang, "passage_id"]) | selected_ids
        k_nr = PROB_QUOTAS_V04["random_non_retrieved"]
        # Stratify non-retrieved by primary_group with equal share of quota.
        groups = ("fiction_manual", "pedagogy_genre_craft", "contrast")
        per_group = max(1, k_nr // len(groups))
        leftovers = k_nr - per_group * len(groups)
        group_ks = {g: per_group for g in groups}
        group_ks["fiction_manual"] += leftovers

        nr_rows: list[dict] = []
        nr_meta: dict = {}
        for grp in groups:
            stratum = f"{lang}::non_retrieved::{grp}"
            s_seed = derive_stratum_seed(seed, stratum)
            res = ReservoirSample(k=group_ks[grp], seed=s_seed)
            for para in iter_body_paragraphs(lang):
                if para.passage_id in retrieved_all:
                    continue
                if held_out_books and para.book_id in held_out_books:
                    continue
                if (para.primary_group or "unknown") != grp:
                    continue
                res.consider(
                    {
                        "passage_id": para.passage_id,
                        "book_id": para.book_id,
                        "language": lang,
                        "title": para.title,
                        "author": para.author,
                        "edition_year": para.edition_year,
                        "corpus_role": para.corpus_role,
                        "primary_group": para.primary_group,
                        "chapter_idx": para.chapter_idx,
                        "para_idx": para.para_idx,
                        "concept_id": "",
                        "retrieval_method": "random_non_retrieved",
                        "bm25_score": "",
                        "rank": "",
                        "matched_terms": "",
                        "matched_regex": "",
                        "candidate_status": "non_retrieved",
                        "sample_stratum": "random_non_retrieved",
                        "sample_reason": "random_non_retrieved",
                        "evidence_kind": "none",
                        "inclusion_probability": None,
                        "sampling_weight": None,
                        "stratum_population_N": None,
                        "stratum_sample_n": None,
                    }
                )
            sm = {
                "N_h": res.population_size,
                "n_h": len(res.reservoir),
                "inclusion_probability": res.inclusion_probability,
                "sampling_weight": res.sampling_weight,
                "seed": res.seed,
                "design": "reservoir_k_equals_final_n",
                "population_definition": (
                    "Independent eval population; non-retrieved eligible body paragraphs; "
                    f"primary_group={grp}"
                ),
            }
            nr_meta[grp] = sm
            for row in res.reservoir:
                row = dict(row)
                row["inclusion_probability"] = sm["inclusion_probability"]
                row["sampling_weight"] = sm["sampling_weight"]
                row["stratum_population_N"] = sm["N_h"]
                row["stratum_sample_n"] = sm["n_h"]
                if row["passage_id"] in selected_ids:
                    continue
                selected_ids.add(row["passage_id"])
                nr_rows.append(row)
                selected_rows.append(row)
        lang_meta["random_non_retrieved"] = nr_meta
        stratum_meta["languages"][lang] = lang_meta

    out_df = pd.DataFrame(selected_rows)
    meta = base_run_meta(
        kind="manual_eval_sample_prob_v04",
        sampling_design="A_probability_based",
        seed=seed,
        master_seed=seed,
        n_unique_passages=len(out_df),
        strata=out_df["sample_stratum"].value_counts().to_dict() if len(out_df) else {},
        languages=out_df["language"].value_counts().to_dict() if len(out_df) else {},
        review_fields=REVIEW_FIELDS,
        stratum_meta=stratum_meta,
        held_out_books=sorted(held_out_books),
        population_scope=(
            "independent_evaluation_population "
            "(analysis_set minus lexicon_dev held-out books); "
            "stratum-specific candidate or non-retrieved pools"
        ),
        estimate_refers_to=(
            "Stratum populations within the independent evaluation population. "
            "Not an unbiased estimate of absolute corpus-wide prevalence."
        ),
        note=(
            "Design A: uniform SRS / reservoir with k=final n_h. "
            "No book-diversity downselection. Weights valid only within stratum."
        ),
    )
    return _write_sample_bundle(
        out_df,
        stem="manual_eval_sample_prob_v04",
        sample_id_prefix="v04p",
        meta=meta,
        log=log,
        with_passages=with_passages,
    )


def build_sample_diversity_v04(
    log,
    *,
    seed: int = SEED_V04 + 1,
    held_out_books: set[str] | None = None,
    target_per_lang: int = 80,
    with_passages: bool = True,
) -> Path:
    """Design B: book-diversity diagnostic sample (qualitative only; no weights)."""
    held_out_books = held_out_books if held_out_books is not None else load_held_out_books()
    pool = _load_v03_pools(held_out_books)
    rng = np.random.default_rng(seed)
    selected_rows: list[dict] = []

    for lang in ("en", "de"):
        sub = pool[pool["language"] == lang].drop_duplicates("passage_id")
        if sub.empty:
            continue
        by_book: dict[str, list] = {}
        for _, row in sub.iterrows():
            by_book.setdefault(str(row["book_id"]), []).append(row.to_dict())
        books = list(by_book.keys())
        rng.shuffle(books)
        picks: list[dict] = []
        while len(picks) < target_per_lang and books:
            still = []
            for bid in books:
                if not by_book[bid]:
                    continue
                row = by_book[bid].pop()
                row["sample_stratum"] = "book_diversity_diagnostic"
                row["sample_reason"] = "book_diversity_diagnostic"
                row["inclusion_probability"] = ""
                row["sampling_weight"] = ""
                row["stratum_population_N"] = ""
                row["stratum_sample_n"] = ""
                picks.append(row)
                if by_book[bid]:
                    still.append(bid)
                if len(picks) >= target_per_lang:
                    break
            books = still
            if not still:
                break
        selected_rows.extend(picks)

    out_df = pd.DataFrame(selected_rows)
    meta = base_run_meta(
        kind="manual_eval_sample_diversity_v04",
        sampling_design="B_book_diversity_diagnostic",
        seed=seed,
        master_seed=seed,
        n_unique_passages=len(out_df),
        strata=out_df["sample_stratum"].value_counts().to_dict() if len(out_df) else {},
        languages=out_df["language"].value_counts().to_dict() if len(out_df) else {},
        review_fields=REVIEW_FIELDS,
        held_out_books=sorted(held_out_books),
        population_scope="independent_evaluation_population (qualitative diagnostic)",
        estimate_refers_to=(
            "Not for corpus-wide prevalence or recall. "
            "No probability weights — selection probabilities unknown after diversity prioritization."
        ),
        note=(
            "Design B: prioritizes passages from different books for error discovery. "
            "Do not attach inclusion probabilities."
        ),
    )
    return _write_sample_bundle(
        out_df,
        stem="manual_eval_sample_diversity_v04",
        sample_id_prefix="v04d",
        meta=meta,
        log=log,
        with_passages=with_passages,
    )


def assign_primary_stratum(
    *,
    in_lexical: bool,
    in_qualified_bm25: bool,
    in_bm25_expansion: bool,
    best_bm25_rank: float | int | None,
) -> str:
    """Assign exactly one primary stratum using the documented priority rule.

    Membership indicators may remain true for multiple retrieval methods; the
    primary stratum is exclusive.
    """
    rank = best_bm25_rank
    if rank is not None and (isinstance(rank, float) and np.isnan(rank)):
        rank = None
    if in_qualified_bm25 and rank is not None and float(rank) <= 50:
        return "high_ranked_qualified_bm25"
    if in_qualified_bm25:
        return "lower_ranked_qualified_bm25"
    if in_lexical and not in_qualified_bm25 and not in_bm25_expansion:
        return "lexical_only"
    if in_lexical and in_bm25_expansion:
        return "bm25_expansion_known"
    if (not in_lexical) and in_bm25_expansion:
        return "new_bm25_passages"
    # Exploratory/qualified flags false and not lexical → non-retrieved.
    # Also covers edge: expansion flag without lexical already handled above.
    if not in_lexical and not in_qualified_bm25 and not in_bm25_expansion:
        return "non_retrieved"
    # Fallback: lexical with neither qualified nor expansion already returned;
    # any residual combination treated as non_retrieved only if nothing else fits.
    return "non_retrieved"


def redistribute_quotas_v05(
    population_sizes: dict[str, int],
    targets: dict[str, int] | None = None,
    order: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Redistribute unused sampling quotas when N_h < target.

    Procedure: compute shortfall from strata where N_h < target; add unused
    slots in ``order`` to strata that still have capacity (N_h > current
    target). Leftover unused quota is dropped if every stratum is at census.
    """
    targets = dict(targets or PROB_QUOTAS_V05)
    if order is None:
        # Prefer documented primary order; append any extra keys deterministically.
        order_list = [k for k in QUOTA_REDISTRIBUTE_ORDER_V05 if k in population_sizes]
        extras = sorted(k for k in population_sizes if k not in order_list)
        order_list.extend(extras)
    else:
        order_list = [k for k in order if k in population_sizes]
        order_list.extend(sorted(k for k in population_sizes if k not in order_list))
    assigned = {k: min(int(targets.get(k, 0)), int(population_sizes.get(k, 0))) for k in order_list}
    unused = sum(int(targets.get(k, 0)) for k in order_list) - sum(assigned.values())
    if unused <= 0:
        return assigned
    for k in order_list:
        if unused <= 0:
            break
        capacity = int(population_sizes.get(k, 0)) - assigned[k]
        if capacity <= 0:
            continue
        take = min(capacity, unused)
        assigned[k] += take
        unused -= take
    return assigned


def assert_strata_pairwise_exclusive(stratum_to_ids: dict[str, set[str]]) -> None:
    """Raise AssertionError if any pair of primary stratum populations overlap."""
    names = sorted(stratum_to_ids.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = stratum_to_ids[a] & stratum_to_ids[b]
            if inter:
                raise AssertionError(
                    f"Stratum populations overlap: {a} ∩ {b} = {len(inter)} passages "
                    f"(e.g. {next(iter(inter))})"
                )


def _concept_col(df: pd.DataFrame) -> str:
    return _assignment_key_col(df)


def _meta_first(df: pd.DataFrame) -> pd.DataFrame:
    """One metadata row per passage_id (first occurrence)."""
    cols = [
        c
        for c in (
            "passage_id",
            "book_id",
            "language",
            "primary_group",
            "title",
            "author",
            "edition_year",
            "corpus_role",
            "chapter_idx",
            "para_idx",
        )
        if c in df.columns
    ]
    return df[cols].drop_duplicates("passage_id")


def _agg_retrieval_membership(
    lexical: pd.DataFrame,
    qualified: pd.DataFrame,
    expansion: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate assignment CSVs to passage-level membership indicators."""
    meta_parts = []
    for df in (lexical, qualified, expansion):
        if not df.empty:
            meta_parts.append(_meta_first(df))
    meta = (
        pd.concat(meta_parts, ignore_index=True).drop_duplicates("passage_id")
        if meta_parts
        else pd.DataFrame(columns=["passage_id"])
    )

    lex_n = pd.Series(dtype=int)
    if not lexical.empty:
        ccol = _concept_col(lexical)
        lex_n = lexical.groupby("passage_id")[ccol].nunique()

    bq_n = pd.Series(dtype=int)
    best_rank = pd.Series(dtype=float)
    if not qualified.empty:
        ccol = _concept_col(qualified)
        q = qualified.copy()
        q["rank_num"] = pd.to_numeric(q["rank"], errors="coerce")
        bq_n = q.groupby("passage_id")[ccol].nunique()
        best_rank = q.groupby("passage_id")["rank_num"].min()

    exp_n = pd.Series(dtype=int)
    if not expansion.empty:
        ccol = _concept_col(expansion)
        exp_n = expansion.groupby("passage_id")[ccol].nunique()

    all_ids = sorted(
        set(meta["passage_id"].astype(str))
        | set(lex_n.index.astype(str))
        | set(bq_n.index.astype(str))
        | set(exp_n.index.astype(str))
    )
    out = pd.DataFrame({"passage_id": all_ids})
    if not meta.empty:
        out = out.merge(meta, on="passage_id", how="left")
    out["in_lexical"] = out["passage_id"].isin(set(lex_n.index.astype(str)))
    out["in_qualified_bm25"] = out["passage_id"].isin(set(bq_n.index.astype(str)))
    out["in_bm25_expansion"] = out["passage_id"].isin(set(exp_n.index.astype(str)))
    out["n_lexically_retrieved_concepts"] = out["passage_id"].map(lex_n).fillna(0).astype(int)
    out["n_bm25_retrieved_concepts"] = (
        out["passage_id"].map(bq_n).fillna(0).astype(int)
        + out["passage_id"].map(exp_n).fillna(0).astype(int)
    )
    # Prefer qualified best rank; leave NaN if only expansion ranks exist.
    out["best_bm25_rank"] = out["passage_id"].map(best_rank)
    out["has_new_concept_assignment"] = out["in_lexical"] & out["in_bm25_expansion"]
    out["primary_stratum"] = [
        assign_primary_stratum(
            in_lexical=bool(r.in_lexical),
            in_qualified_bm25=bool(r.in_qualified_bm25),
            in_bm25_expansion=bool(r.in_bm25_expansion),
            best_bm25_rank=r.best_bm25_rank,
        )
        for r in out.itertuples(index=False)
    ]
    return out


def build_passage_membership_v05(
    log,
    *,
    held_out_books: set[str] | None = None,
    include_non_retrieved: bool = True,
) -> pd.DataFrame:
    """Build eligible passage-level membership + exclusive primary_stratum.

    Eligible = analysis-set body paragraphs in the candidate passage manifests,
    excluding lexicon-development held-out books. Every eligible passage gets
    exactly one ``primary_stratum``.
    """
    held_out_books = held_out_books if held_out_books is not None else load_held_out_books()
    retrieved_frames = []
    for lang in ("en", "de"):
        lexical = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv"))
        qualified = _ensure_pid(_load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_{lang}.csv"))
        expansion = _ensure_pid(
            _load(DATA_PROCESSED / f"pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv")
        )
        filtered = []
        for df in (lexical, qualified, expansion):
            d = df.copy()
            if not d.empty and held_out_books and "book_id" in d.columns:
                d = d[~d["book_id"].isin(held_out_books)]
            filtered.append(d)
        lexical, qualified, expansion = filtered
        part = _agg_retrieval_membership(lexical, qualified, expansion)
        if not part.empty:
            part["language"] = part["language"].where(part["language"].astype(str).str.len() > 0, lang)
            retrieved_frames.append(part)

    retrieved = (
        pd.concat(retrieved_frames, ignore_index=True)
        if retrieved_frames
        else pd.DataFrame()
    )

    # Eligible universe from passage manifests (analysis snapshot).
    manifest_frames = []
    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv"
        if not path.is_file():
            log.warning(f"missing passage manifest {path}")
            continue
        m = pd.read_csv(path, dtype=str).fillna("")
        if held_out_books:
            m = m[~m["book_id"].isin(held_out_books)]
        m["language"] = lang
        manifest_frames.append(m)
    if not manifest_frames:
        raise SystemExit("No passage manifests found for membership table.")
    manifest = pd.concat(manifest_frames, ignore_index=True)

    # Attach work_id from manifest; merge retrieval indicators.
    base = manifest[
        [
            "passage_id",
            "book_id",
            "work_id",
            "language",
            "primary_group",
            "chapter_idx",
            "para_idx",
            "corpus_role",
        ]
    ].drop_duplicates("passage_id")

    if retrieved.empty:
        merged = base.copy()
        for col, default in (
            ("in_lexical", False),
            ("in_qualified_bm25", False),
            ("in_bm25_expansion", False),
            ("has_new_concept_assignment", False),
            ("best_bm25_rank", np.nan),
            ("n_lexically_retrieved_concepts", 0),
            ("n_bm25_retrieved_concepts", 0),
            ("title", ""),
            ("author", ""),
            ("edition_year", ""),
        ):
            merged[col] = default
    else:
        keep = [
            "passage_id",
            "in_lexical",
            "in_qualified_bm25",
            "in_bm25_expansion",
            "has_new_concept_assignment",
            "best_bm25_rank",
            "n_lexically_retrieved_concepts",
            "n_bm25_retrieved_concepts",
            "title",
            "author",
            "edition_year",
        ]
        r = retrieved[keep].drop_duplicates("passage_id")
        merged = base.merge(r, on="passage_id", how="left")
        for col in (
            "in_lexical",
            "in_qualified_bm25",
            "in_bm25_expansion",
            "has_new_concept_assignment",
        ):
            merged[col] = merged[col].fillna(False).astype(bool)
        for col in ("n_lexically_retrieved_concepts", "n_bm25_retrieved_concepts"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        merged["best_bm25_rank"] = pd.to_numeric(merged["best_bm25_rank"], errors="coerce")
        for col in ("title", "author", "edition_year"):
            if col not in merged.columns:
                merged[col] = ""
            else:
                merged[col] = merged[col].fillna("")

    # Passages present in retrieval but missing from manifest (should be rare).
    if not retrieved.empty:
        missing = retrieved[~retrieved["passage_id"].isin(set(merged["passage_id"]))]
        if len(missing):
            log.warning(
                f"{len(missing)} retrieved passages absent from passage manifests; appending"
            )
            extra = missing.copy()
            if "work_id" not in extra.columns:
                extra["work_id"] = ""
            merged = pd.concat([merged, extra[merged.columns]], ignore_index=True)

    if not include_non_retrieved:
        merged = merged[
            merged["in_lexical"]
            | merged["in_qualified_bm25"]
            | merged["in_bm25_expansion"]
        ].copy()

    merged["primary_group"] = merged["primary_group"].replace("", "unknown").fillna("unknown")
    merged["primary_stratum"] = [
        assign_primary_stratum(
            in_lexical=bool(r.in_lexical),
            in_qualified_bm25=bool(r.in_qualified_bm25),
            in_bm25_expansion=bool(r.in_bm25_expansion),
            best_bm25_rank=r.best_bm25_rank,
        )
        for r in merged.itertuples(index=False)
    ]

    # Integrity: exclusive populations + full coverage of eligible set.
    stratum_ids = {
        s: set(merged.loc[merged["primary_stratum"] == s, "passage_id"])
        for s in PRIMARY_STRATA_V05
    }
    assert_strata_pairwise_exclusive(stratum_ids)
    union = set().union(*stratum_ids.values()) if stratum_ids else set()
    eligible = set(merged["passage_id"])
    if union != eligible:
        raise AssertionError(
            f"Eligible population not fully partitioned: "
            f"|eligible|={len(eligible)} |union|={len(union)} "
            f"missing={len(eligible - union)} extra={len(union - eligible)}"
        )

    cols = [
        "passage_id",
        "book_id",
        "work_id",
        "language",
        "primary_group",
        "in_lexical",
        "in_qualified_bm25",
        "in_bm25_expansion",
        "has_new_concept_assignment",
        "best_bm25_rank",
        "n_lexically_retrieved_concepts",
        "n_bm25_retrieved_concepts",
        "primary_stratum",
        "title",
        "author",
        "edition_year",
        "corpus_role",
        "chapter_idx",
        "para_idx",
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols]


def build_sample_prob_v05(
    log,
    *,
    seed: int = SEED_V05,
    held_out_books: set[str] | None = None,
    with_passages: bool = True,
    membership: pd.DataFrame | None = None,
) -> Path:
    """Probability sample from mutually exclusive stratum populations (v0.5)."""
    held_out_books = held_out_books if held_out_books is not None else load_held_out_books()
    if membership is None:
        membership = build_passage_membership_v05(log, held_out_books=held_out_books)

    # Persist membership table.
    OUT.mkdir(parents=True, exist_ok=True)
    mem_path = OUT / "eval_passage_membership_v05.csv"
    membership.to_csv(mem_path, index=False)
    mem_meta = base_run_meta(
        kind="eval_passage_membership_v05",
        n_passages=len(membership),
        strata=membership["primary_stratum"].value_counts().to_dict(),
        languages=membership["language"].value_counts().to_dict(),
        held_out_books=sorted(held_out_books),
        exclusive_priority=list(PRIMARY_STRATA_V05),
        note=(
            "Every eligible passage (analysis-set manifests minus lexicon-dev held-out) "
            "belongs to exactly one primary_stratum. Membership indicators may still be "
            "true for multiple retrieval methods."
        ),
        work_id_limitation=(
            "work_id is missing for most books; do not assume each book_id is a unique work. "
            "Edition leakage across held-out / eval splits cannot be ruled out automatically."
        ),
    )
    write_json(OUT / "eval_passage_membership_v05.meta.json", mem_meta)
    log.info(f"wrote {mem_path} ({len(membership)} passages)")

    selected_rows: list[dict] = []
    stratum_meta: dict = {"master_seed": seed, "languages": {}, "quotas": {}}

    for lang in ("en", "de"):
        lang_df = membership[membership["language"] == lang]
        # Population sizes for retrieved strata + non_retrieved total.
        pop_sizes = {
            s: int((lang_df["primary_stratum"] == s).sum()) for s in PRIMARY_STRATA_V05
        }
        quotas = redistribute_quotas_v05(pop_sizes, PROB_QUOTAS_V05)
        stratum_meta["quotas"][lang] = {
            "targets": dict(PROB_QUOTAS_V05),
            "population_sizes": pop_sizes,
            "after_redistribution": quotas,
        }
        lang_meta: dict = {}

        for stratum in PRIMARY_STRATA_V05:
            if stratum == "non_retrieved":
                continue
            pop = lang_df[lang_df["primary_stratum"] == stratum].copy()
            s_seed = derive_stratum_seed(seed, f"{lang}::{stratum}")
            sampled, sm = _srs_without_replacement(pop, quotas.get(stratum, 0), s_seed)
            sm["target_sample_size"] = PROB_QUOTAS_V05[stratum]
            sm["actual_sample_size"] = sm["n_h"]
            sm["population_size"] = sm["N_h"]
            sm["sampling_seed"] = s_seed
            sm["stratum_id"] = f"{lang}::{stratum}"
            sm["population_definition"] = (
                f"Exclusive primary stratum={stratum} within independent eval population"
            )
            lang_meta[stratum] = sm
            for _, row in sampled.iterrows():
                d = row.to_dict()
                d["sample_reason"] = stratum
                d["sample_stratum"] = stratum
                d["inclusion_probability"] = sm["inclusion_probability"]
                d["sampling_weight"] = sm["sampling_weight"]
                d["stratum_population_N"] = sm["N_h"]
                d["stratum_sample_n"] = sm["n_h"]
                d["concept_id"] = ""
                d["retrieval_method"] = stratum
                d["bm25_score"] = ""
                d["rank"] = d.get("best_bm25_rank", "")
                d["matched_terms"] = ""
                d["matched_regex"] = ""
                d["candidate_status"] = stratum
                d["evidence_kind"] = "membership_table"
                selected_rows.append(d)

        # Non-retrieved: further stratify by primary_group.
        nr_quota = quotas.get("non_retrieved", 0)
        nr_pop = lang_df[lang_df["primary_stratum"] == "non_retrieved"].copy()
        groups = sorted(nr_pop["primary_group"].fillna("unknown").unique().tolist())
        if not groups:
            groups = ["unknown"]
        # Equal share; leftover to fiction_manual if present else first group.
        per_group = max(1, nr_quota // max(len(groups), 1)) if nr_quota else 0
        leftovers = nr_quota - per_group * len(groups) if nr_quota else 0
        group_ks = {g: per_group for g in groups}
        prefer = "fiction_manual" if "fiction_manual" in group_ks else groups[0]
        group_ks[prefer] = group_ks.get(prefer, 0) + leftovers
        # Redistribute within non-retrieved groups if some are small.
        group_sizes = {
            g: int((nr_pop["primary_group"] == g).sum()) for g in groups
        }
        group_ks = redistribute_quotas_v05(group_sizes, group_ks)

        nr_meta: dict = {}
        for grp in groups:
            pop = nr_pop[nr_pop["primary_group"] == grp]
            s_seed = derive_stratum_seed(seed, f"{lang}::non_retrieved::{grp}")
            sampled, sm = _srs_without_replacement(pop, group_ks.get(grp, 0), s_seed)
            sm["target_sample_size"] = group_ks.get(grp, 0)
            sm["actual_sample_size"] = sm["n_h"]
            sm["population_size"] = sm["N_h"]
            sm["sampling_seed"] = s_seed
            sm["stratum_id"] = f"{lang}::non_retrieved::{grp}"
            sm["population_definition"] = (
                "Exclusive non_retrieved within independent eval population; "
                f"primary_group={grp}"
            )
            nr_meta[grp] = sm
            for _, row in sampled.iterrows():
                d = row.to_dict()
                d["sample_reason"] = "non_retrieved"
                d["sample_stratum"] = "non_retrieved"
                d["inclusion_probability"] = sm["inclusion_probability"]
                d["sampling_weight"] = sm["sampling_weight"]
                d["stratum_population_N"] = sm["N_h"]
                d["stratum_sample_n"] = sm["n_h"]
                d["concept_id"] = ""
                d["retrieval_method"] = "random_non_retrieved"
                d["bm25_score"] = ""
                d["rank"] = ""
                d["matched_terms"] = ""
                d["matched_regex"] = ""
                d["candidate_status"] = "non_retrieved"
                d["evidence_kind"] = "none"
                selected_rows.append(d)
        lang_meta["non_retrieved"] = nr_meta
        stratum_meta["languages"][lang] = lang_meta

    out_df = pd.DataFrame(selected_rows)
    # Final uniqueness check.
    if out_df["passage_id"].duplicated().any():
        raise AssertionError("Duplicate passage_id in v05 sample")

    meta = base_run_meta(
        kind="manual_eval_sample_prob_v05",
        sampling_design="A_probability_based_exclusive_populations",
        seed=seed,
        master_seed=seed,
        n_unique_passages=len(out_df),
        strata=out_df["sample_stratum"].value_counts().to_dict() if len(out_df) else {},
        languages=out_df["language"].value_counts().to_dict() if len(out_df) else {},
        review_fields=REVIEW_FIELDS,
        stratum_meta=stratum_meta,
        held_out_books=sorted(held_out_books),
        population_scope=(
            "independent_evaluation_population "
            "(candidate passage manifests minus lexicon_dev held-out books); "
            "mutually exclusive primary strata fixed before sampling"
        ),
        estimate_refers_to=(
            "Stratum populations within the independent evaluation population. "
            "Weights are not valid for the complete unfiltered corpus or for "
            "prevalence among development books."
        ),
        inclusion_probability_formula="pi_h = n_h / N_h",
        sampling_weight_formula="w_h = N_h / n_h",
        quota_redistribution=list(QUOTA_REDISTRIBUTE_ORDER_V05),
        note=(
            "Design A v0.5: exclusive populations before SRS. "
            "If N_h <= target, census with pi=1. "
            "Do not transfer annotations by chapter/paragraph coordinates alone."
        ),
    )
    return _write_sample_bundle(
        out_df,
        stem="manual_eval_sample_prob_v05",
        sample_id_prefix="v05p",
        meta=meta,
        log=log,
        with_passages=with_passages,
    )


def write_reconciliation_v04_v05(log) -> Path:
    """Reconcile v0.4 probability sample with v0.5; content-hash safe transfer flag."""
    path_v04 = OUT / "manual_eval_sample_prob_v04.csv"
    path_v05 = OUT / "manual_eval_sample_prob_v05.csv"
    v04 = (
        pd.read_csv(path_v04, dtype=str).fillna("")
        if path_v04.is_file()
        else pd.DataFrame(columns=["passage_id"])
    )
    v05 = (
        pd.read_csv(path_v05, dtype=str).fillna("")
        if path_v05.is_file()
        else pd.DataFrame(columns=["passage_id"])
    )
    set04 = set(v04["passage_id"]) if "passage_id" in v04.columns else set()
    set05 = set(v05["passage_id"]) if "passage_id" in v05.columns else set()

    # Content hashes from passage manifests.
    hash_map: dict[str, str] = {}
    for lang in ("en", "de"):
        mp = DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv"
        if not mp.is_file():
            continue
        m = pd.read_csv(
            mp,
            dtype=str,
            usecols=["passage_id", "raw_content_hash", "normalized_content_hash"],
        ).fillna("")
        for _, row in m.iterrows():
            hash_map[str(row["passage_id"])] = (
                f"{row['raw_content_hash']}|{row['normalized_content_hash']}"
            )

    all_ids = sorted(set04 | set05)
    rows = []
    for pid in all_ids:
        in04 = pid in set04
        in05 = pid in set05
        h = hash_map.get(pid, "")
        content_hash_agreement = bool(h) and in04 and in05
        # Safe transfer only when same passage_id AND content hash known+present.
        annotation_transfer_safe = bool(in04 and in05 and h)
        if in04 and in05:
            status = "identical"
        elif in05 and not in04:
            status = "newly_selected"
        else:
            status = "removed_from_sample"
        rows.append(
            {
                "passage_id": pid,
                "in_prob_v04": int(in04),
                "in_prob_v05": int(in05),
                "status": status,
                "content_hash": h,
                "content_hash_agreement": int(content_hash_agreement),
                "annotation_transfer_safe": int(annotation_transfer_safe),
                "transfer_rule": (
                    "passage_id+content_hash match required; "
                    "never transfer by chapter/para coordinates alone"
                ),
            }
        )
    out = OUT / "manual_eval_sample_v04_v05_reconciliation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    summary = base_run_meta(
        kind="manual_eval_sample_v04_v05_reconciliation",
        n_prob_v04=len(set04),
        n_prob_v05=len(set05),
        n_identical=len(set04 & set05),
        n_newly_selected=len(set05 - set04),
        n_removed=len(set04 - set05),
        note=(
            "v0.4 samples preserved. Annotation transfer requires passage_id and "
            "content hash agreement; coordinate-only matches are unsafe."
        ),
    )
    write_json(OUT / "manual_eval_sample_v04_v05_reconciliation.meta.json", summary)
    log.info(f"wrote {out} ({len(rows)} rows)")
    return out


def write_reconciliation_v03_v04(log) -> Path:
    """Map passage overlap across frozen v0.3 and new v0.4 samples."""
    paths = {
        "v03": OUT / "manual_eval_sample_v03.csv",
        "prob_v04": OUT / "manual_eval_sample_prob_v04.csv",
        "diversity_v04": OUT / "manual_eval_sample_diversity_v04.csv",
    }
    sets = {}
    for name, path in paths.items():
        if path.is_file():
            df = pd.read_csv(path, dtype=str).fillna("")
            sets[name] = set(df["passage_id"]) if "passage_id" in df.columns else set()
        else:
            sets[name] = set()
    all_ids = sorted(sets["v03"] | sets["prob_v04"] | sets["diversity_v04"])
    rows = []
    for pid in all_ids:
        rows.append(
            {
                "passage_id": pid,
                "in_v03": int(pid in sets["v03"]),
                "in_prob_v04": int(pid in sets["prob_v04"]),
                "in_diversity_v04": int(pid in sets["diversity_v04"]),
            }
        )
    out = OUT / "manual_eval_sample_v03_v04_reconciliation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    summary = base_run_meta(
        kind="manual_eval_sample_reconciliation",
        n_v03=len(sets["v03"]),
        n_prob_v04=len(sets["prob_v04"]),
        n_diversity_v04=len(sets["diversity_v04"]),
        overlap_v03_prob=len(sets["v03"] & sets["prob_v04"]),
        overlap_v03_diversity=len(sets["v03"] & sets["diversity_v04"]),
        overlap_prob_diversity=len(sets["prob_v04"] & sets["diversity_v04"]),
        note="v0.3 sample preserved; v0.4 designs are additive.",
    )
    write_json(OUT / "manual_eval_sample_v03_v04_reconciliation.meta.json", summary)
    log.info(f"wrote {out} ({len(rows)} rows)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        choices=[
            "v02",
            "v03",
            "v04",
            "v04_prob",
            "v04_diversity",
            "v04_reconcile",
            "v05",
            "v05_prob",
            "v05_reconcile",
            "v05_membership",
        ],
        default="v05",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument(
        "--held-out-books",
        type=str,
        default="",
        help="Comma-separated book_ids, or path to a text file (one id per line).",
    )
    parser.add_argument(
        "--held-out-file",
        type=str,
        default="",
        help="Path to held-out book list (default: lexicon_dev_held_out_books_v03.txt).",
    )
    parser.add_argument("--skip-passages", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)

    held: set[str] = set()
    if args.held_out_file:
        held |= load_held_out_books(Path(args.held_out_file))
    elif args.version.startswith("v04") or args.version.startswith("v05"):
        held |= load_held_out_books()
    if args.held_out_books:
        raw = args.held_out_books
        p = Path(raw)
        if p.is_file():
            held |= load_held_out_books(p)
        else:
            held |= {b.strip() for b in raw.split(",") if b.strip()}

    with_passages = not args.skip_passages
    if args.version == "v03":
        build_sample_v03(
            log,
            seed=args.seed if args.seed is not None else SEED_V03,
            target=args.target,
            held_out_books=held,
        )
    elif args.version == "v02":
        build_sample(log, seed=args.seed if args.seed is not None else SEED, target=args.target)
    elif args.version in ("v04", "v04_prob"):
        build_sample_prob_v04(
            log,
            seed=args.seed if args.seed is not None else SEED_V04,
            held_out_books=held,
            with_passages=with_passages,
        )
        if args.version == "v04":
            build_sample_diversity_v04(
                log,
                seed=(args.seed + 1) if args.seed is not None else SEED_V04 + 1,
                held_out_books=held,
                with_passages=with_passages,
            )
            write_reconciliation_v03_v04(log)
    elif args.version == "v04_diversity":
        build_sample_diversity_v04(
            log,
            seed=args.seed if args.seed is not None else SEED_V04 + 1,
            held_out_books=held,
            with_passages=with_passages,
        )
    elif args.version == "v04_reconcile":
        write_reconciliation_v03_v04(log)
    elif args.version == "v05_membership":
        mem = build_passage_membership_v05(log, held_out_books=held)
        OUT.mkdir(parents=True, exist_ok=True)
        mem.to_csv(OUT / "eval_passage_membership_v05.csv", index=False)
        log.info(f"wrote membership ({len(mem)} passages)")
    elif args.version in ("v05", "v05_prob"):
        build_sample_prob_v05(
            log,
            seed=args.seed if args.seed is not None else SEED_V05,
            held_out_books=held,
            with_passages=with_passages,
        )
        if args.version == "v05":
            write_reconciliation_v04_v05(log)
    elif args.version == "v05_reconcile":
        write_reconciliation_v04_v05(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
