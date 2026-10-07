"""Ranking metrics, written out by hand and checked against trec_eval in tests/.

ranked:   list of doc ids, best first (duplicates are ignored after the first)
qrels_q:  {doc_id: grade} for one query; grade > 0 means relevant

P@k     = |relevant in top k| / k
R@k     = |relevant in top k| / |relevant|
MRR@k   = 1 / rank of the first relevant doc within the top k (0 if none)
nDCG@k  = DCG@k / IDCG@k,  DCG@k = sum_{i=1..k} grade_i / log2(i + 1)
          (linear gain, as in trec_eval; identical to 2^g - 1 for SciFact's binary grades)
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _dedupe(ranked: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ranked))


def _relevant(qrels_q: dict[str, int] | Iterable[str]) -> set[str]:
    if isinstance(qrels_q, dict):
        return {d for d, g in qrels_q.items() if g > 0}
    return set(qrels_q)


def precision_at_k(ranked: Sequence[str], qrels_q, k: int) -> float:
    rel = _relevant(qrels_q)
    return sum(1 for d in _dedupe(ranked)[:k] if d in rel) / k


def recall_at_k(ranked: Sequence[str], qrels_q, k: int) -> float:
    rel = _relevant(qrels_q)
    if not rel:
        return 0.0
    return sum(1 for d in _dedupe(ranked)[:k] if d in rel) / len(rel)


def mrr(ranked: Sequence[str], qrels_q, k: int | None = None) -> float:
    rel = _relevant(qrels_q)
    top = _dedupe(ranked)
    if k is not None:
        top = top[:k]
    for i, d in enumerate(top, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], qrels_q: dict[str, int], k: int) -> float:
    if not isinstance(qrels_q, dict):
        qrels_q = {d: 1 for d in qrels_q}
    dcg = sum(qrels_q.get(d, 0) / math.log2(i + 1) for i, d in enumerate(_dedupe(ranked)[:k], start=1)
              if qrels_q.get(d, 0) > 0)
    ideal = sorted((g for g in qrels_q.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def first_relevant_rank(ranked: Sequence[str], qrels_q, cutoff: int | None = None) -> int | None:
    rel = _relevant(qrels_q)
    for i, d in enumerate(_dedupe(ranked), start=1):
        if cutoff is not None and i > cutoff:
            break
        if d in rel:
            return i
    return None


METRICS = ("P@5", "P@10", "R@10", "MRR@10", "nDCG@10")


def per_query(ranked: Sequence[str], qrels_q: dict[str, int]) -> dict[str, float]:
    return {
        "P@5": precision_at_k(ranked, qrels_q, 5),
        "P@10": precision_at_k(ranked, qrels_q, 10),
        "R@10": recall_at_k(ranked, qrels_q, 10),
        "MRR@10": mrr(ranked, qrels_q, 10),
        "nDCG@10": ndcg_at_k(ranked, qrels_q, 10),
    }


def evaluate(run: dict[str, Sequence[str]], qrels: dict[str, dict[str, int]]) -> tuple[dict[str, float], dict[str, dict]]:
    """Mean of each metric over every judged query (a query missing from the run scores 0)."""
    per = {q: per_query(run.get(q, []), qrels[q]) for q in qrels}
    means = {m: float(np.mean([per[q][m] for q in per])) if per else 0.0 for m in METRICS}
    return means, per


# ----------------------------------------------------------------- significance
def paired_randomization_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 42) -> float:
    """Two-sided p-value for mean(a) != mean(b) on paired per-query scores
    (Fisher randomization test; Smucker, Allan & Carterette, CIKM 2007)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(diff.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(diff)))
    perm_means = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(perm_means >= observed - 1e-12) + 1) / (n_perm + 1))


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean over queries."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
