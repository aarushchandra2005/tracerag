"""Okapi BM25 over the same inverted index (single "all" field = title + body).

    score(q, d) = sum_{t in q} qtf_t * idf_t * tf_td * (k1 + 1) / (tf_td + k1 * (1 - b + b * |d| / avgdl))
    idf_t       = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))      (Lucene's non-negative variant)

Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond".
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

import config
from src.index.inverted_index import InvertedIndex
from src.retrieve.base import Retriever


def bm25_idf(df: int, n_docs: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = config.BM25_K1, b: float = config.BM25_B,
                 zone: str = "all"):
        super().__init__(index)
        self.k1, self.b, self.zone = float(k1), float(b), zone

    def __repr__(self) -> str:
        return f"BM25Retriever(k1={self.k1}, b={self.b}, zone={self.zone!r}, chunking={self.index.chunking!r})"

    def _term_part(self, tfs: np.ndarray, lengths: np.ndarray, avg_len: float) -> np.ndarray:
        tfs = tfs.astype(np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * lengths / avg_len)
        return tfs * (self.k1 + 1.0) / (tfs + norm)

    def score_all(self, query: str) -> np.ndarray:
        z = self.index.zones[self.zone]
        scores = np.zeros(z.n_docs)
        for term, qtf in Counter(self.index.tokenizer.tokenize(query)).items():
            docs, tfs = z.postings(term)
            if len(docs) == 0:
                continue
            idf = bm25_idf(len(docs), z.n_docs)
            scores[docs] += qtf * idf * self._term_part(tfs, z.length[docs], z.avg_len)
        return scores

    def explain(self, query: str, chunk_id: str) -> list[dict]:
        c = self.index.chunk_pos[chunk_id]
        z = self.index.zones[self.zone]
        rows = []
        for term, qtf in Counter(self.index.tokenizer.tokenize(query)).items():
            docs, tfs = z.postings(term)
            hit = np.searchsorted(docs, c)
            if hit < len(docs) and docs[hit] == c:
                idf = bm25_idf(len(docs), z.n_docs)
                part = float(self._term_part(tfs[hit:hit + 1], z.length[c:c + 1], z.avg_len)[0])
                rows.append({"term": term, "tf": int(tfs[hit]), "df": len(docs), "idf": idf,
                             "contribution": qtf * idf * part})
        rows.sort(key=lambda r: -r["contribution"])
        return rows
