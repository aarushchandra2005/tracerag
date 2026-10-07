"""Why did a chunk score high? Per-term contributions and dense evidence.

top_terms(query, chunk_id) ranks the query terms by their share of the
lnc.ltc score (contributions sum exactly to the score; tested). For BM25 the
same view uses BM25 term scores. For the dense model there are no terms, so
explain_dense() shows the chunk sentence closest to the query instead.
"""
from __future__ import annotations

from src.retrieve.bm25 import BM25Retriever
from src.retrieve.tfidf import TfidfRetriever


def top_terms(query: str, chunk_id: str, retriever: TfidfRetriever | BM25Retriever, n: int = 5) -> list[dict]:
    rows = retriever.explain(query, chunk_id)
    total = sum(r["contribution"] for r in rows)
    for r in rows:
        r["share"] = r["contribution"] / total if total > 0 else 0.0
    return rows[:n]


def format_terms(rows: list[dict]) -> str:
    if not rows:
        return "(no query term occurs in this chunk)"
    parts = []
    for r in rows:
        zone = f"/{r['zone']}" if "zone" in r else ""
        parts.append(f"{r['term']}{zone} {r['share']:.0%}")
    return ", ".join(parts)


def explain_dense(query: str, chunk_id: str, dense_retriever, n: int = 1) -> list[dict]:
    return dense_retriever.explain(query, chunk_id)[:n]


def query_overlap(query: str, text: str, tokenizer) -> tuple[float, list[str], list[str]]:
    """Share of the query's index terms that also occur in `text`, with the matched and missing terms."""
    q = list(dict.fromkeys(tokenizer.tokenize(query)))
    d = set(tokenizer.tokenize(text))
    hit = [t for t in q if t in d]
    miss = [t for t in q if t not in d]
    return (len(hit) / len(q) if q else 0.0), hit, miss
