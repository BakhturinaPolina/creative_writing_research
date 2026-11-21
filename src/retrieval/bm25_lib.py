"""Minimal Okapi BM25 (same defaults as rank_bm25.BM25Okapi).

Used when the optional ``rank_bm25`` package is not installed (e.g. read-only
venv). Prefer ``from rank_bm25 import BM25Okapi`` when available.
"""
from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
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


def get_bm25(corpus: list[list[str]]) -> BM25Okapi:
    try:
        from rank_bm25 import BM25Okapi as ExtBM25  # type: ignore

        return ExtBM25(corpus)
    except ImportError:
        return BM25Okapi(corpus)
