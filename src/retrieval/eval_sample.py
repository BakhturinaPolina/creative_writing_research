"""Stratified bilingual manual-evaluation sample (no gold labels assigned)."""
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

OUT = DATA_INTERIM / "retrieval"
SEED = 20261121
TARGET = 260  # within 200–300

REVIEW_FIELDS = [
    "relevant_to_narrative_perspective",
    "concept_relevance",
    "explicit_or_implicit_discussion",
    "instruction_or_fictional_example",
    "correct_concept_ids",
    "false_positive_reason",
    "reviewer_notes",
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
    # Evidence-type tags
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

    # High / mid / low BM25 ranks
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

    # Subgroup coverage
    for lang in ("en", "de"):
        for grp in ("fiction_manual", "pedagogy_genre_craft", "contrast"):
            sub = pool[(pool["language"] == lang) & (pool["primary_group"] == grp)]
            take(sub, 8, f"subgroup_{grp}")

    # Potential FPs: weak selection rules / perspective_general only
    weak = pool[pool.get("selection_rule", pd.Series(dtype=str)).isin(
        ["weak_pov_plus_supporting", "two_independent_supporting"]
    )] if "selection_rule" in pool.columns else pd.DataFrame()
    take(weak, 20, "potential_false_positive")

    # Random non-retrieved passages
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

    # Trim / pad toward target unique passages
    # Deduplicate by passage_id keeping first reason
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
        out_df[f] = ""

    # Pointers-only public-ish table (still interim; may contain titles)
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

    # Passage text sidecar (gitignored pattern *passages*)
    # Fetch text only for sampled IDs
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()
    log = setup_logger(__file__)
    build_sample(log, seed=args.seed, target=args.target)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
