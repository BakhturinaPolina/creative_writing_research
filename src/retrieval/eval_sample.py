"""Stratified bilingual manual-evaluation sample (no gold labels assigned).

v0.2 builder preserved as ``build_sample`` (legacy stems).
v0.3 builder uses reservoir sampling and versioned outputs.
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
from src.retrieval.sampling import StratifiedReservoirs

OUT = DATA_INTERIM / "retrieval"
SEED = 20261121
SEED_V03 = 20261122
TARGET = 260  # within 200–300

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v02", "v03"], default="v03")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument(
        "--held-out-books",
        type=str,
        default="",
        help="Comma-separated book_ids excluded from the eval sample (dev-sample holdout).",
    )
    args = parser.parse_args()
    log = setup_logger(__file__)
    held = {b.strip() for b in args.held_out_books.split(",") if b.strip()}
    if args.version == "v03":
        build_sample_v03(
            log,
            seed=args.seed if args.seed is not None else SEED_V03,
            target=args.target,
            held_out_books=held,
        )
    else:
        build_sample(log, seed=args.seed if args.seed is not None else SEED, target=args.target)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
