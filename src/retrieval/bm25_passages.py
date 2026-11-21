"""Streaming BM25 passage retrieval with PsyPer lexicon v0.2.

Two corpus passes only (never holds all paragraphs in RAM):
  1) DF / avgdl for the query-token vocabulary
  2) score each paragraph; keep a top-k heap per concept

Tracked output (``data/processed/pov_passage_candidates_{lang}.csv``) stores
pointers and scores only — never book text. Optional ``--with-passages`` writes
a gitignored sidecar under ``data/interim/retrieval/``.

Usage:
  python -m src.retrieval.bm25_passages --lang de
  python -m src.retrieval.bm25_passages --lang en
  python -m src.retrieval.bm25_passages --lang all
  python -m src.retrieval.bm25_passages --lang en --with-passages
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
from collections import Counter
from datetime import datetime

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED, ROOT
from src.retrieval.lexicon_v02 import classify_tier, load_lexicon
from src.retrieval.paragraphs import Paragraph, iter_body_paragraphs
from src.retrieval.textnorm import phrase_in_text, tokenize

TIERS_PATH = DATA_INTERIM / "retrieval" / "lexicon_term_tiers.yml"
PASSAGE_SIDECAR = DATA_INTERIM / "retrieval"  # gitignored *passages*.csv
DEFAULT_TOP_K = 50
DEFAULT_SCORE_FLOOR = 2.0
BM25_K1 = 1.5
BM25_B = 0.75
META_COLS = [
    "lang",
    "concept_id",
    "bm25_score",
    "rank",
    "book_id",
    "corpus_role",
    "primary_group",
    "title",
    "chapter_idx",
    "para_idx",
    "high_hits",
    "supporting_hits",
    "n_tokens",
]

_WEAK_HIGH = {
    "point of view",
    "pov",
    "povs",
    "viewpoint",
    "viewpoints",
    "regex:general_pov",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _surface(hit: str) -> str:
    return hit.casefold().split(":")[-1]


def _passes_candidate_rule(high: list[str], supporting: list[str]) -> bool:
    strong = [
        h for h in high
        if _surface(h) not in _WEAK_HIGH and not h.casefold().startswith("regex:general_pov")
    ]
    if strong:
        return True
    real_supp = [s for s in supporting if _surface(s) not in _WEAK_HIGH]
    weak = [h for h in high if h not in strong]
    if weak and real_supp:
        return True
    return len(real_supp) >= 2


def _literal_evidence(para: Paragraph, concept_id: str, lex, lang: str) -> tuple[list[str], list[str]]:
    ct = lex.concepts[lang].get(concept_id)
    high, supporting = [], []
    if not ct:
        return high, supporting
    norm = para.text_norm()
    for term in ct.academic:
        if phrase_in_text(term, norm):
            high.append(term)
    for term in ct.craft:
        if not phrase_in_text(term, norm):
            continue
        tier = classify_tier(lang, "craft", term)
        if tier == "high":
            high.append(term)
        else:
            supporting.append(term)
    for name, pat in lex.example_regex.get(lang, {}).items():
        if pat.search(para.text):
            high.append(f"regex:{name}")
    return high, supporting


def _bm25_score(
    tf: Counter,
    doc_len: int,
    query_tokens: list[str],
    idf: dict[str, float],
    avgdl: float,
) -> float:
    if not doc_len or not avgdl:
        return 0.0
    score = 0.0
    for q in query_tokens:
        f = tf.get(q, 0)
        if not f:
            continue
        idf_q = idf.get(q)
        if idf_q is None:
            continue
        denom = f + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl)
        score += idf_q * (f * (BM25_K1 + 1) / denom)
    return score


def _build_queries(lex, lang: str, demoted_terms: set[str]) -> dict[str, list[str]]:
    queries: dict[str, list[str]] = {}
    for concept_id, ct in lex.concepts[lang].items():
        q_tokens: list[str] = []
        for phrase in ct.bm25_query_terms():
            if phrase.casefold() in demoted_terms and " " not in phrase and "-" not in phrase:
                continue
            q_tokens.extend(tokenize(phrase))
        if q_tokens:
            queries[concept_id] = q_tokens
    return queries


def retrieve_lang(
    lang: str,
    top_k: int,
    score_floor: float,
    log,
    *,
    with_passages: bool = False,
) -> pd.DataFrame:
    lex = load_lexicon()
    demoted_terms: set[str] = set()
    if TIERS_PATH.is_file():
        tiers = yaml.safe_load(TIERS_PATH.read_text(encoding="utf-8")) or {}
        for item in (tiers.get("languages") or {}).get(lang, {}).get("demoted") or []:
            demoted_terms.add(str(item.get("term", "")).casefold())

    queries = _build_queries(lex, lang, demoted_terms)
    query_vocab = {t for toks in queries.values() for t in toks}
    log.info(f"{lang}: query vocab {len(query_vocab)} tokens, {len(queries)} concepts")

    # Pass 1: DF for query vocab + avgdl (discard tokens after each para).
    log.info(f"{lang}: pass 1 — document frequencies")
    df_q: Counter = Counter()
    n_docs = 0
    sum_len = 0
    for i, para in enumerate(iter_body_paragraphs(lang)):
        toks = para.tokens()
        n_docs += 1
        sum_len += len(toks)
        seen = set(toks) & query_vocab
        for t in seen:
            df_q[t] += 1
        if i and i % 50000 == 0:
            log.info(f"  pass1 {i} paragraphs")
        del toks
    avgdl = (sum_len / n_docs) if n_docs else 0.0
    idf = {}
    for t, freq in df_q.items():
        idf[t] = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
    log.info(f"{lang}: {n_docs} paragraphs, avgdl={avgdl:.1f}")

    # Pass 2: score; keep min-heaps of (score, tie, rowdict) size top_k per concept.
    log.info(f"{lang}: pass 2 — BM25 top-{top_k}")
    heaps: dict[str, list] = {cid: [] for cid in queries}
    tie = 0
    for i, para in enumerate(iter_body_paragraphs(lang)):
        toks = para.tokens()
        tf = Counter(toks)
        doc_len = len(toks)
        del toks
        tokset = set(tf)
        for concept_id, q_tokens in queries.items():
            # Skip concepts with no query-token overlap (avoids useless scoring).
            if not (tokset & set(q_tokens)):
                continue
            score = _bm25_score(tf, doc_len, q_tokens, idf, avgdl)
            if score < score_floor:
                continue
            heap = heaps[concept_id]
            # Literal evidence only if this score can enter the top-k heap.
            if len(heap) >= top_k and score <= heap[0][0]:
                continue
            high, supporting = _literal_evidence(para, concept_id, lex, lang)
            if not _passes_candidate_rule(high, supporting):
                continue
            tie += 1
            row = (
                score,
                tie,
                {
                    "lang": lang,
                    "concept_id": concept_id,
                    "bm25_score": round(score, 4),
                    "book_id": para.book_id,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "title": para.title,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "high_hits": "; ".join(high),
                    "supporting_hits": "; ".join(supporting),
                    "n_tokens": doc_len,
                    # Kept in-memory only; written to a gitignored sidecar if requested.
                    "_passage": para.text[:800] if with_passages else "",
                },
            )
            if len(heap) < top_k:
                heapq.heappush(heap, row)
            else:
                heapq.heapreplace(heap, row)
        del tf
        if i and i % 50000 == 0:
            log.info(f"  pass2 {i}/{n_docs} paragraphs")

    rows = []
    for concept_id, heap in heaps.items():
        ranked = sorted(heap, key=lambda x: x[0], reverse=True)
        for rank, (score, _, payload) in enumerate(ranked, 1):
            payload = dict(payload)
            payload["rank"] = rank
            rows.append(payload)
        log.info(f"  {concept_id}: kept {len(ranked)}")

    df = pd.DataFrame(rows)
    passage_col = df.pop("_passage") if "_passage" in df.columns else None
    # Tracked CSV: metadata pointers only (no book text).
    out_cols = [c for c in META_COLS if c in df.columns]
    out = DATA_PROCESSED / f"pov_passage_candidates_{lang}.csv"
    df[out_cols].to_csv(out, index=False)
    if with_passages and passage_col is not None:
        PASSAGE_SIDECAR.mkdir(parents=True, exist_ok=True)
        side = PASSAGE_SIDECAR / f"pov_passage_candidates_{lang}_passages.csv"
        side_df = df[out_cols].copy()
        side_df["passage"] = passage_col.values
        side_df.to_csv(side, index=False)
        log.info(f"wrote gitignored sidecar {side}")
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lang": lang,
        "lexicon_version": lex.version,
        "git_commit": _git_commit(),
        "n_paragraphs_indexed": n_docs,
        "n_candidate_rows": len(df),
        "n_unique_passages": int(df.groupby(["book_id", "chapter_idx", "para_idx"]).ngroups) if len(df) else 0,
        "n_concepts_with_hits": int(df["concept_id"].nunique()) if len(df) else 0,
        "top_k": top_k,
        "score_floor": score_floor,
        "bm25_backend": "streaming_okapi",
        "peak_strategy": "two_pass_query_vocab_df_topk_heap",
        "stores_book_text": False,
        "passage_sidecar": with_passages,
    }
    (DATA_PROCESSED / f"pov_passage_candidates_{lang}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    log.info(f"wrote {out} ({len(df)} rows, no book text)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
            lang, args.top_k, args.score_floor, log, with_passages=args.with_passages
        )
    log.info("Log finished.")


if __name__ == "__main__":
    main()
