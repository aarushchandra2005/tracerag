"""Shared retriever plumbing: heap top-K and chunk -> paper aggregation.

Every retriever scores *chunks*. Relevance judgments in SciFact are per
*paper*, so for evaluation a paper's score is the score of its best chunk
("MaxP"). Without this step, sentence windows from one relevant paper could
fill several of the top-10 slots and inflate P@10.
"""
from __future__ import annotations

import heapq

import numpy as np

from src.index.inverted_index import InvertedIndex


def heap_top_k(scores: np.ndarray, k: int, positive_only: bool = True) -> list[int]:
    """Indices of the k largest scores, best first, using a heap (O(n log k)).

    Ties keep index order (heapq.nlargest is stable), so runs are deterministic.
    positive_only drops zero scores: a chunk sharing no term with the query is
    not retrieved by a sparse model at all.
    """
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    return heapq.nlargest(k, candidates.tolist(), key=scores.__getitem__)


class Retriever:
    """Base class. Subclasses implement score_all(query) -> one score per chunk."""

    name = "base"
    positive_only = True  # sparse models: only chunks with a matching term are hits

    def __init__(self, index: InvertedIndex):
        self.index = index

    # ------------------------------------------------------------- to implement
    def score_all(self, query: str) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------- shared
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k chunks as [(chunk_id, score)]."""
        scores = self.score_all(query)
        return [(self.index.chunk_ids[i], float(scores[i]))
                for i in heap_top_k(scores, k, self.positive_only)]

    def paper_scores(self, chunk_scores: np.ndarray) -> np.ndarray:
        """MaxP: a paper's score is the max over its chunks."""
        idx = self.index
        if idx.n_chunks == len(idx.doc_ids):  # whole-abstract chunks: already one per paper
            return chunk_scores
        out = np.full(len(idx.doc_ids), -np.inf)
        np.maximum.at(out, idx.chunk_doc, chunk_scores)
        return out

    def rank_docs(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Top-k *papers* as [(doc_id, score)], for evaluation against qrels."""
        scores = self.paper_scores(self.score_all(query))
        return [(self.index.doc_ids[i], float(scores[i]))
                for i in heap_top_k(scores, k, self.positive_only)]

    def prepare(self, queries: list[str]) -> None:
        """Hook for batch work before many queries (dense models encode them at once)."""
