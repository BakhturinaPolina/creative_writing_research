"""Baseline C — streaming BM25 passage retrieval with PsyPer lexicon v0.2.

Two corpus passes (never holds all paragraphs in RAM):
  1) DF / avgdl for the query-token vocabulary
  2) score each paragraph; keep a top-k heap per concept

Validated candidates require concept-specific literal evidence.
Optional BM25-only exploratory pool is labelled ``bm25_only_unvalidated``.

Tracked CSVs store pointers/scores only — never book text.
Optional ``--with-passages`` writes a gitignored sidecar.

Usage:
  python -m src.retrieval.bm25_passages --lang de
  python -m src.retrieval.bm25_passages --lang en --top-k 500
  python -m src.retrieval.bm25_passages --lang all --with-passages
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.bm25_lib import bm25_score_doc, compute_idf
from src.retrieval.evidence import candidate_status, collect_concept_evidence
from src.retrieval.lexicon_v02 import load_lexicon
from src.retrieval.paragraphs import CorpusScanStats, iter_body_paragraphs
from src.retrieval.queries import QueryConfig, build_all_queries
from src.retrieval.regex_map import load_regex_map
from src.retrieval.run_meta import base_run_meta, write_json
from src.retrieval.textnorm import normalize_text

# Documented Okapi defaults (shared with bm25_lib)
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

TIERS_PATH = DATA_INTERIM / "retrieval" / "lexicon_term_tiers.yml"
PASSAGE_SIDECAR = DATA_INTERIM / "retrieval"
DEFAULT_TOP_K = 500
DEFAULT_SCORE_FLOOR = 2.0
DEFAULT_BM25_ONLY_K = 100

META_COLS = [
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
    "psyper_tags",
    "retrieval_method",
    "bm25_score",
    "rank",
    "matched_terms",
    "matched_regex",
    "matched_spans",
    "evidence_tiers",
    "candidate_status",
    "selection_rule",
    "n_tokens",
]


def _load_demoted(lang: str) -> set[str]:
    demoted: set[str] = set()
    if TIERS_PATH.is_file():
        tiers = yaml.safe_load(TIERS_PATH.read_text(encoding="utf-8")) or {}
        for item in (tiers.get("languages") or {}).get(lang, {}).get("demoted") or []:
            demoted.add(str(item.get("term", "")).casefold())
    return demoted


def retrieve_lang(
    lang: str,
    top_k: int,
    score_floor: float,
    log,
    *,
    with_passages: bool = False,
    bm25_only_k: int = DEFAULT_BM25_ONLY_K,
    query_cfg: QueryConfig | None = None,
) -> pd.DataFrame:
    lex = load_lexicon()
    rmap = load_regex_map()
    demoted = _load_demoted(lang)
    queries = build_all_queries(lex, lang, demoted_terms=demoted, cfg=query_cfg or QueryConfig())
    query_vocab = {t for q in queries.values() for t in q.tokens}
    # Cheap literal needles per concept — skip expensive evidence when absent.
    concept_needles: dict[str, list[str]] = {}
    for cid, ct in lex.concepts[lang].items():
        ns = []
        for term in list(ct.academic) + list(ct.craft) + list(ct.paraphrases):
            n = normalize_text(term).replace("-", " ")
            if len(n) >= 3:
                ns.append(n)
        concept_needles[cid] = ns
    regex_concepts: dict[str, list[str]] = {}
    for cid in lex.concepts[lang]:
        for name in rmap.names_for_concept(lang, cid):
            regex_concepts.setdefault(name, []).append(cid)
    log.info(f"{lang}: query vocab {len(query_vocab)} tokens, {len(queries)} concepts")

    # Pass 1: DF for query vocab + avgdl
    log.info(f"{lang}: pass 1 — document frequencies")
    scan1 = CorpusScanStats()
    df_q: Counter = Counter()
    n_docs = 0
    sum_len = 0
    for i, para in enumerate(iter_body_paragraphs(lang, scan_stats=scan1)):
        toks = para.tokens()
        n_docs += 1
        sum_len += len(toks)
        for t in set(toks) & query_vocab:
            df_q[t] += 1
        if i and i % 50000 == 0:
            log.info(f"  pass1 {i} paragraphs")
        del toks
    avgdl = (sum_len / n_docs) if n_docs else 0.0
    idf = compute_idf(n_docs, df_q, epsilon=BM25_EPSILON)
    log.info(
        f"{lang}: {n_docs} paragraphs, avgdl={avgdl:.1f}, "
        f"books {scan1.processed_books}/{scan1.expected_books}"
    )
    if scan1.skipped_missing:
        log.warning(f"  missing JSON ({len(scan1.skipped_missing)}): {scan1.skipped_missing[:10]}")
    if scan1.skipped_unreadable:
        log.warning(f"  unreadable: {scan1.skipped_unreadable}")

    # Pass 2: validated top-k + optional BM25-only exploratory heaps
    log.info(f"{lang}: pass 2 — BM25 top-{top_k} (+ bm25_only top-{bm25_only_k})")
    heaps: dict[str, list] = {cid: [] for cid in queries}
    only_heaps: dict[str, list] = {cid: [] for cid in queries}
    tie = 0
    scan2 = CorpusScanStats()

    for i, para in enumerate(
        iter_body_paragraphs(lang, with_neighbors=with_passages, scan_stats=scan2)
    ):
        toks = para.tokens()
        tf = Counter(toks)
        doc_len = len(toks)
        del toks
        tokset = set(tf)
        hay = normalize_text(para.text).replace("-", " ")
        # Concepts with any literal/regex hint in this paragraph
        literal_cands: set[str] = set()
        for cid, ns in concept_needles.items():
            if any(n in hay for n in ns):
                literal_cands.add(cid)
        for name, cids in regex_concepts.items():
            pat = lex.example_regex.get(lang, {}).get(name)
            if pat is not None and pat.search(para.text):
                literal_cands.update(cids)

        for concept_id, cq in queries.items():
            q_tokens = cq.tokens
            if not (tokset & set(q_tokens)):
                continue
            score = bm25_score_doc(
                tf,
                doc_len,
                q_tokens,
                idf,
                avgdl,
                k1=BM25_K1,
                b=BM25_B,
                token_weights=cq.token_weights,
            )
            if score < score_floor:
                continue

            # Exploratory BM25-only pool (no literal gate)
            oh = only_heaps[concept_id]
            if len(oh) < bm25_only_k or score > oh[0][0]:
                tie += 1
                only_row = (
                    score,
                    tie,
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
                        "concept_id": concept_id,
                        "psyper_tags": concept_id,
                        "retrieval_method": "baseline_c_bm25_only",
                        "bm25_score": round(score, 4),
                        "matched_terms": "",
                        "matched_regex": "",
                        "matched_spans": "[]",
                        "evidence_tiers": "{}",
                        "candidate_status": "bm25_only_unvalidated",
                        "selection_rule": "bm25_score_only",
                        "n_tokens": doc_len,
                        "_passage": para.text if with_passages else "",
                        "_prev": para.prev_text if with_passages else "",
                        "_next": para.next_text if with_passages else "",
                    },
                )
                if len(oh) < bm25_only_k:
                    heapq.heappush(oh, only_row)
                else:
                    heapq.heapreplace(oh, only_row)

            # Validated path: requires literal/regex hint + evidence rule
            if concept_id not in literal_cands:
                continue
            heap = heaps[concept_id]
            if len(heap) >= top_k and score <= heap[0][0]:
                continue
            ct = lex.concepts[lang][concept_id]
            ev = collect_concept_evidence(para.text, concept_id, ct, lang, lex, rmap)
            ok, rule = candidate_status(ev, lang)
            if not ok:
                continue
            tie += 1
            row = (
                score,
                tie,
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
                    "concept_id": concept_id,
                    "psyper_tags": concept_id,
                    "retrieval_method": "baseline_c_v02_bm25",
                    "bm25_score": round(score, 4),
                    "matched_terms": "; ".join(ev.matched_terms),
                    "matched_regex": "; ".join(ev.matched_regex),
                    "matched_spans": json.dumps(ev.matched_spans, ensure_ascii=False),
                    "evidence_tiers": json.dumps(ev.evidence_tiers_summary()),
                    "candidate_status": "bm25_and_literal",
                    "selection_rule": rule,
                    "n_tokens": doc_len,
                    "_passage": para.text if with_passages else "",
                    "_prev": para.prev_text if with_passages else "",
                    "_next": para.next_text if with_passages else "",
                },
            )
            if len(heap) < top_k:
                heapq.heappush(heap, row)
            else:
                heapq.heapreplace(heap, row)

        del tf
        if i and i % 20000 == 0:
            log.info(f"  pass2 {i}/{n_docs} paragraphs")

    def _flush(heaps_map: dict[str, list], status_filter: str | None = None) -> list[dict]:
        rows = []
        for concept_id, heap in heaps_map.items():
            ranked = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (_score, _, payload) in enumerate(ranked, 1):
                payload = dict(payload)
                payload["rank"] = rank
                if status_filter and payload.get("candidate_status") != status_filter:
                    continue
                rows.append(payload)
            log.info(f"  {concept_id}: kept {len(ranked)}")
        return rows

    validated = _flush(heaps)
    exploratory = _flush(only_heaps)

    # Drop BM25-only rows that already appear as validated for same passage+concept
    validated_keys = {(r["passage_id"], r["concept_id"]) for r in validated}
    exploratory = [
        r for r in exploratory if (r["passage_id"], r["concept_id"]) not in validated_keys
    ]

    df_val = pd.DataFrame(validated)
    df_only = pd.DataFrame(exploratory)

    def _write_df(df: pd.DataFrame, stem: str) -> None:
        if df.empty:
            df = pd.DataFrame(columns=META_COLS)
        passage_cols = {}
        for c in ("_passage", "_prev", "_next"):
            if c in df.columns:
                passage_cols[c] = df.pop(c)
        out_cols = [c for c in META_COLS if c in df.columns]
        out = DATA_PROCESSED / f"{stem}.csv"
        df[out_cols].to_csv(out, index=False)
        if with_passages and passage_cols:
            PASSAGE_SIDECAR.mkdir(parents=True, exist_ok=True)
            side = PASSAGE_SIDECAR / f"{stem}_passages.csv"
            side_df = df[out_cols].copy()
            side_df["passage"] = passage_cols.get("_passage", pd.Series(dtype=str)).values
            side_df["prev_passage"] = passage_cols.get("_prev", pd.Series(dtype=str)).values
            side_df["next_passage"] = passage_cols.get("_next", pd.Series(dtype=str)).values
            side_df.to_csv(side, index=False)
            log.info(f"wrote gitignored sidecar {side}")
        log.info(f"wrote {out} ({len(df)} rows)")

    _write_df(df_val, f"pov_baseline_c_bm25_{lang}")
    _write_df(df_only, f"pov_baseline_c_bm25_only_unvalidated_{lang}")

    # Compact top-50 view for rapid review
    if not df_val.empty:
        top50 = (
            df_val.sort_values(["concept_id", "rank"])
            .groupby("concept_id", as_index=False)
            .head(50)
        )
        top50_cols = [c for c in META_COLS if c in top50.columns]
        top50_path = DATA_PROCESSED / f"pov_baseline_c_bm25_top50_{lang}.csv"
        top50[top50_cols].to_csv(top50_path, index=False)
        log.info(f"wrote {top50_path} ({len(top50)} rows)")

    # Legacy alias for compatibility
    legacy = DATA_PROCESSED / f"pov_passage_candidates_{lang}.csv"
    if not df_val.empty:
        df_val[[c for c in META_COLS if c in df_val.columns]].to_csv(legacy, index=False)
    else:
        pd.DataFrame(columns=META_COLS).to_csv(legacy, index=False)

    meta = base_run_meta(
        baseline="C",
        lang=lang,
        lexicon_version=lex.version,
        lexicon_sha256=lex.sha256,
        regex_map_version=rmap.version,
        n_paragraphs_indexed=n_docs,
        n_candidate_rows=len(df_val),
        n_bm25_only_rows=len(df_only),
        n_unique_passages=int(df_val.groupby("passage_id").ngroups) if len(df_val) else 0,
        n_concepts_with_hits=int(df_val["concept_id"].nunique()) if len(df_val) else 0,
        top_k=top_k,
        bm25_only_k=bm25_only_k,
        score_floor=score_floor,
        bm25_k1=BM25_K1,
        bm25_b=BM25_B,
        bm25_epsilon=BM25_EPSILON,
        bm25_backend="streaming_okapi",
        peak_strategy="two_pass_query_vocab_df_topk_heap",
        expected_books=scan1.expected_books,
        processed_books=scan1.processed_books,
        skipped_missing=scan1.skipped_missing,
        skipped_unreadable=scan1.skipped_unreadable,
        stores_book_text=False,
        passage_sidecar=with_passages,
        query_construction={
            "dedupe_expressions": True,
            "dedupe_tokens": True,
            "drop_stopwords": True,
            "allow_repeated_token_weighting": False,
        },
    )
    write_json(DATA_PROCESSED / f"pov_baseline_c_bm25_{lang}.meta.json", meta)
    write_json(DATA_PROCESSED / f"pov_passage_candidates_{lang}.meta.json", meta)
    return df_val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--bm25-only-k", type=int, default=DEFAULT_BM25_ONLY_K)
    parser.add_argument("--score-floor", type=float, default=DEFAULT_SCORE_FLOOR)
    parser.add_argument(
        "--with-passages",
        action="store_true",
        help="also write gitignored sidecar CSVs that contain passage text",
    )
    args = parser.parse_args()
    log = setup_logger(__file__)
    langs = ["en", "de"] if args.lang == "all" else [args.lang]
    for lang in langs:
        retrieve_lang(
            lang,
            args.top_k,
            args.score_floor,
            log,
            with_passages=args.with_passages,
            bm25_only_k=args.bm25_only_k,
        )
    log.info("Log finished.")


if __name__ == "__main__":
    main()
