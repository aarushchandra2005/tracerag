"""Hybrid retrieval: fuse a sparse and a dense ranking.

  * RRF (Cormack, Clarke & Büttcher, SIGIR 2009):
        score(d) = sum_lists 1 / (k + rank_list(d)),  k = 60, ranks start at 1
    Uses ranks only, so the two models' score scales do not matter.
  * Weighted: min-max normalise each list's scores to [0, 1], then
        score(d) = alpha * sparse(d) + (1 - alpha) * dense(d)
    An item missing from a list gets 0 from that list.

Ties are broken by first appearance (sparse list first), so output is deterministic.
"""
from __future__ import annotations

from typing import Sequence

import config
from src.retrieve.base import Retriever


def rrf(rank_lists: Sequence[Sequence[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])  # stable sort keeps first-seen order on ties


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {item: 1.0 for item in scores}
    return {item: (s - lo) / (hi - lo) for item, s in scores.items()}


def weighted(score_lists: Sequence[dict[str, float]], alpha: float | Sequence[float] = 0.5) -> list[tuple[str, float]]:
    """alpha is the weight on the first list (or one weight per list)."""
    if isinstance(alpha, (int, float)):
        if len(score_lists) != 2:
            raise ValueError("a single alpha needs exactly two lists")
        weights = [float(alpha), 1.0 - float(alpha)]
    else:
        weights = [float(a) for a in alpha]
    fused: dict[str, float] = {}
    for w, scores in zip(weights, score_lists):
        for item, s in minmax(scores).items():
            fused[item] = fused.get(item, 0.0) + w * s
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    """Fuses two retrievers that share one chunk catalogue (same InvertedIndex)."""

    def __init__(self, sparse: Retriever, dense: Retriever, method: str = "rrf",
                 alpha: float = config.DEFAULT_HYBRID_ALPHA, rrf_k: int = config.RRF_K,
                 depth: int = config.FUSION_DEPTH):
        if method not in ("rrf", "weighted"):
            raise ValueError("method must be 'rrf' or 'weighted'")
        if sparse.index is not dense.index and sparse.index.chunk_ids != dense.index.chunk_ids:
            raise ValueError("both retrievers must score the same chunks")
        self.sparse, self.dense = sparse, dense
        self.index = sparse.index
        self.method, self.alpha, self.rrf_k, self.depth = method, alpha, rrf_k, depth
        self.name = f"hybrid_{method}"

    def __repr__(self) -> str:
        extra = f"alpha={self.alpha}" if self.method == "weighted" else f"k={self.rrf_k}"
        return f"HybridRetriever({self.sparse.name}+{self.dense.name}, {self.method}, {extra})"

    def prepare(self, queries: list[str]) -> None:
        self.sparse.prepare(queries)
        self.dense.prepare(queries)

    def _fuse(self, lists: list[list[tuple[str, float]]], k: int) -> list[tuple[str, float]]:
        if self.method == "rrf":
            fused = rrf([[item for item, _ in lst] for lst in lists], self.rrf_k)
        else:
            fused = weighted([dict(lst) for lst in lists], self.alpha)
        return fused[:k]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k chunks."""
        lists = [self.sparse.search(query, self.depth), self.dense.search(query, self.depth)]
        return self._fuse(lists, k)

    def rank_docs(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Top-k papers: each model is first reduced to papers (MaxP), then fused."""
        lists = [self.sparse.rank_docs(query, self.depth), self.dense.rank_docs(query, self.depth)]
        return self._fuse(lists, k)
