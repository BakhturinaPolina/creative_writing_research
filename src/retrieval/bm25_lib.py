"""Minimal Okapi BM25 (same defaults as rank_bm25.BM25Okapi).

Documented conventions (shared by streaming path in ``bm25_passages``):

* Tokenization: ``textnorm.tokenize`` — NFKC, casefold, Unicode word tokens
  with internal apostrophes/hyphens.
* IDF: ``log((N - df + 0.5) / (df + 0.5))``. Negative IDF (very common terms)
  is floored to ``epsilon * average_idf`` (default epsilon=0.25), matching
  ``rank_bm25.BM25Okapi``.
* ``k1=1.5``, ``b=0.75``.
* Query weighting: optional per-token multipliers from ``queries.QueryConfig``;
  default construction deduplicates tokens (no accidental repeat weighting).
* Do **not** compare raw BM25 scores across languages or across concepts.

Used when the optional ``rank_bm25`` package is not installed.
"""
from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self.doc_freqs: list[Counter] = []
        df: Counter = Counter()
        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for term in frequencies:
                df[term] += 1
        self.idf = {}
        idf_sum = 0.0
        negative_idfs = []
        for term, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term)
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for term in negative_idfs:
            self.idf[term] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not self.corpus_size:
            return scores
        for q in query:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, frequencies in enumerate(self.doc_freqs):
                f = frequencies.get(q, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


def compute_idf(
    n_docs: int,
    df: dict[str, int],
    *,
    epsilon: float = 0.25,
) -> dict[str, float]:
    """Streaming-friendly IDF with negative-IDF floor (rank_bm25 convention)."""
    idf: dict[str, float] = {}
    if not n_docs:
        return idf
    raw = {}
    idf_sum = 0.0
    negative = []
    for t, freq in df.items():
        val = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
        raw[t] = val
        idf_sum += val
        if val < 0:
            negative.append(t)
    avg = idf_sum / len(raw) if raw else 0.0
    eps = epsilon * avg
    for t, val in raw.items():
        idf[t] = eps if t in negative else val
    return idf


def bm25_score_doc(
    tf: Counter,
    doc_len: int,
    query_tokens: list[str],
    idf: dict[str, float],
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    token_weights: dict[str, float] | None = None,
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
        w = 1.0 if not token_weights else float(token_weights.get(q, 1.0))
        denom = f + k1 * (1 - b + b * doc_len / avgdl)
        score += w * idf_q * (f * (k1 + 1) / denom)
    return score


def get_bm25(corpus: list[list[str]]) -> BM25Okapi:
    try:
        from rank_bm25 import BM25Okapi as ExtBM25  # type: ignore

        return ExtBM25(corpus)
    except ImportError:
        return BM25Okapi(corpus)
