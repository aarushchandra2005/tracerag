"""One place to build retrievers by name, sharing indexes between them.

Names:
  tfidf             lnc.ltc with title/body zones (zone weights tuned on train)
  tfidf_single      lnc.ltc on one merged title+body field (ablation)
  bm25              BM25 on the merged field
  dense             MiniLM cosine
  hybrid_rrf        RRF(BM25, dense)
  hybrid_weighted   alpha * BM25 + (1 - alpha) * dense after min-max (alpha tuned on train)
  hybrid_rrf_tfidf  RRF(tf-idf, dense)
"""
from __future__ import annotations

import config
from src.index.dense_index import DenseIndex
from src.index.inverted_index import InvertedIndex, get_index
from src.retrieve.bm25 import BM25Retriever
from src.retrieve.dense import DenseRetriever
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.tfidf import TfidfRetriever
from src.utils import tuned

NAMES = ("tfidf", "tfidf_single", "bm25", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf_tfidf")
NEEDS_DENSE = {"dense", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf_tfidf"}


def zone_weights_for(chunking: str) -> tuple[float, float]:
    return tuple(tuned(f"zone_weights_{chunking}", config.DEFAULT_ZONE_WEIGHTS))


def alpha_for(chunking: str) -> float:
    return float(tuned(f"hybrid_alpha_{chunking}", config.DEFAULT_HYBRID_ALPHA))


class RetrieverFactory:
    def __init__(self, dense_model: str = config.DENSE_MODEL):
        self.dense_model = dense_model
        self._indexes: dict[tuple, InvertedIndex] = {}
        self._dense: dict[str, DenseRetriever] = {}

    def index(self, chunking: str = "whole", stem: bool = config.USE_STEMMING,
              stopwords: bool = config.REMOVE_STOPWORDS) -> InvertedIndex:
        key = (chunking, stem, stopwords)
        if key not in self._indexes:
            self._indexes[key] = get_index(chunking, stem, stopwords)
        return self._indexes[key]

    def dense(self, chunking: str = "whole") -> DenseRetriever:
        if chunking not in self._dense:
            index = self.index(chunking)
            self._dense[chunking] = DenseRetriever(index, DenseIndex(index.chunks, self.dense_model))
        return self._dense[chunking]

    def get(self, name: str, chunking: str = "whole", *, stem: bool = config.USE_STEMMING,
            stopwords: bool = config.REMOVE_STOPWORDS, zone_weights=None, alpha=None,
            k1: float = config.BM25_K1, b: float = config.BM25_B):
        if name == "tfidf":
            weights = zone_weights if zone_weights is not None else zone_weights_for(chunking)
            return TfidfRetriever(self.index(chunking, stem, stopwords), weights)
        if name == "tfidf_single":
            return TfidfRetriever(self.index(chunking, stem, stopwords), (1.0,), zones=("all",))
        if name == "bm25":
            return BM25Retriever(self.index(chunking, stem, stopwords), k1, b)
        if name == "dense":
            return self.dense(chunking)
        if name == "hybrid_rrf":
            return HybridRetriever(self.get("bm25", chunking), self.dense(chunking), "rrf")
        if name == "hybrid_weighted":
            a = alpha if alpha is not None else alpha_for(chunking)
            return HybridRetriever(self.get("bm25", chunking), self.dense(chunking), "weighted", alpha=a)
        if name == "hybrid_rrf_tfidf":
            return HybridRetriever(self.get("tfidf", chunking), self.dense(chunking), "rrf")
        raise ValueError(f"unknown retriever {name!r}; choose from {NAMES}")


_DEFAULT: RetrieverFactory | None = None


def get_retriever(name: str, chunking: str = "whole", **kwargs):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RetrieverFactory()
    return _DEFAULT.get(name, chunking, **kwargs)
