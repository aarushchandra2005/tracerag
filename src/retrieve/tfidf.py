"""tf-idf retrieval with SMART lnc.ltc weighting and weighted zone scoring.

Document side, lnc:  w(t,d) = (1 + log10 tf_td) / ||d||          (log tf, no idf, cosine)
Query side,    ltc:  w(t,q) = (1 + log10 tf_tq) * log10(N/df_t) / ||q||   (log tf, idf, cosine)
Zone score:          cos_z(q, d) = sum_t w(t,q) * w(t,d)          over the zone's vectors
Final score:         sum_z W_z * cos_z(q, d) / sum_z W_z          (W = zone weights)

Scores are accumulated term-at-a-time into one array (one pass over each
query term's postings list) and the top K are taken with a heap.
Reference: Manning, Raghavan & Schütze, Introduction to IR, ch. 6.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

import config
from src.index.inverted_index import InvertedIndex
from src.retrieve.base import Retriever


# ----------------------------------------------------------------- the weighting formulas
def ltc_weights(query_tf: dict[str, int], df: dict[str, int], n_docs: int) -> dict[str, float]:
    """Cosine-normalised query weights. Terms with df == 0 are dropped."""
    raw = {}
    for t, tf in query_tf.items():
        d = df.get(t, 0)
        if tf > 0 and d > 0:
            w = (1.0 + math.log10(tf)) * math.log10(n_docs / d)
            if w > 0:
                raw[t] = w
    norm = math.sqrt(sum(w * w for w in raw.values()))
    return {t: w / norm for t, w in raw.items()} if norm > 0 else {}


def lnc_weights(doc_tf: dict[str, int]) -> dict[str, float]:
    """Cosine-normalised document weights (no idf)."""
    raw = {t: 1.0 + math.log10(tf) for t, tf in doc_tf.items() if tf > 0}
    norm = math.sqrt(sum(w * w for w in raw.values()))
    return {t: w / norm for t, w in raw.items()} if norm > 0 else {}


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, index: InvertedIndex, zone_weights=config.DEFAULT_ZONE_WEIGHTS,
                 zones: tuple[str, ...] = ("title", "body")):
        super().__init__(index)
        if len(zone_weights) != len(zones):
            raise ValueError("one weight per zone")
        self.zone_weights = {z: float(w) for z, w in zip(zones, zone_weights) if w > 0}
        if not self.zone_weights:
            raise ValueError("at least one zone needs a positive weight")
        self._weight_sum = sum(self.zone_weights.values())

    def __repr__(self) -> str:
        return f"TfidfRetriever(zone_weights={self.zone_weights}, chunking={self.index.chunking!r})"

    # ------------------------------------------------------------- query side
    def query_weights(self, query: str, zone: str) -> dict[str, float]:
        z = self.index.zones[zone]
        tf = Counter(self.index.tokenizer.tokenize(query))
        df = {t: z.doc_freq(t) for t in tf}
        return ltc_weights(tf, df, z.n_docs)

    # ------------------------------------------------------------- scoring
    def zone_scores(self, query: str, zone: str) -> np.ndarray:
        """cos_lnc.ltc(q, d_zone) for every chunk, term-at-a-time."""
        z = self.index.zones[zone]
        scores = np.zeros(z.n_docs)
        for term, wq in self.query_weights(query, zone).items():
            docs, tfs = z.postings(term)
            wd = (1.0 + np.log10(tfs)) / z.lnc_norm[docs]
            scores[docs] += wq * wd          # docs are unique within one postings list
        return scores

    def score_all(self, query: str) -> np.ndarray:
        total = np.zeros(self.index.n_chunks)
        for zone, w in self.zone_weights.items():
            total += w * self.zone_scores(query, zone)
        return total / self._weight_sum

    # ------------------------------------------------------------- explanation
    def explain(self, query: str, chunk_id: str) -> list[dict]:
        """Per (term, zone) contribution to one chunk's score; contributions sum to the score."""
        c = self.index.chunk_pos[chunk_id]
        rows = []
        for zone, w in self.zone_weights.items():
            z = self.index.zones[zone]
            for term, wq in self.query_weights(query, zone).items():
                docs, tfs = z.postings(term)
                hit = np.searchsorted(docs, c)
                if hit < len(docs) and docs[hit] == c:
                    tf = int(tfs[hit])
                    wd = (1.0 + math.log10(tf)) / z.lnc_norm[c]
                    rows.append({"term": term, "zone": zone, "tf": tf, "df": z.doc_freq(term),
                                 "w_query": wq, "w_doc": float(wd),
                                 "contribution": w * wq * float(wd) / self._weight_sum})
        rows.sort(key=lambda r: -r["contribution"])
        return rows
