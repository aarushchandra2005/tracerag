import math
import random
from collections import Counter

import numpy as np
import pytest

from src.retrieve.base import heap_top_k
from src.retrieve.bm25 import BM25Retriever
from src.retrieve.hybrid import minmax, rrf, weighted
from src.retrieve.tfidf import TfidfRetriever, lnc_weights, ltc_weights
from tests.conftest import needs_scifact


# ----------------------------------------------------------------- textbook check
def test_iir_example_6_4_lnc_ltc():
    """Manning et al., IIR Example 6.4: query 'best car insurance' vs document
    'car insurance auto insurance' with N = 1,000,000 scores 0.8."""
    df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
    q = ltc_weights({"best": 1, "car": 1, "insurance": 1}, df, 1_000_000)
    d = lnc_weights({"car": 1, "insurance": 2, "auto": 1})
    score = sum(w * d.get(t, 0.0) for t, w in q.items())
    assert round(q["insurance"], 2) == 0.78 and round(d["insurance"], 2) == 0.68
    assert score == pytest.approx(0.80, abs=0.005)


# ----------------------------------------------------------------- brute-force reference
def brute_force_tfidf(index, query, zone_weights, zones=("title", "body")):
    """Dense numpy reference: build every lnc and ltc vector explicitly."""
    total = np.zeros(index.n_chunks)
    tok = index.tokenizer.tokenize
    for zone, w in zip(zones, zone_weights):
        if w == 0:
            continue
        texts = [{"title": c.title, "body": c.text, "all": f"{c.title} {c.text}"}[zone] for c in index.chunks]
        doc_tf = [Counter(tok(t)) for t in texts]
        n = len(doc_tf)
        df = Counter(t for c in doc_tf for t in c)
        qw = ltc_weights(Counter(tok(query)), df, n)
        for i, c in enumerate(doc_tf):
            dw = lnc_weights(c)
            total[i] += w * sum(v * dw.get(t, 0.0) for t, v in qw.items())
    return total / sum(w for w in zone_weights if w > 0)


@pytest.mark.parametrize("weights", [(1.0, 1.0), (0.1, 1.0), (0.0, 1.0), (3.0, 1.0)])
def test_tfidf_matches_brute_force_on_toy(toy_index, weights):
    r = TfidfRetriever(toy_index, weights)
    for q in ("car insurance fraud", "tumor growth training", "best year", "unknownword"):
        assert np.allclose(r.score_all(q), brute_force_tfidf(toy_index, q, weights), atol=1e-12)


def test_single_field_matches_brute_force(toy_sent_index):
    r = TfidfRetriever(toy_sent_index, (1.0,), zones=("all",))
    q = "insurance law drivers"
    assert np.allclose(r.score_all(q), brute_force_tfidf(toy_sent_index, q, (1.0,), ("all",)), atol=1e-12)


def test_explain_contributions_sum_to_score(toy_index):
    r = TfidfRetriever(toy_index, (0.5, 1.0))
    q = "car insurance basics"
    scores = r.score_all(q)
    for cid in toy_index.chunk_ids:
        total = sum(row["contribution"] for row in r.explain(q, cid))
        assert total == pytest.approx(scores[toy_index.chunk_pos[cid]], abs=1e-12)


@needs_scifact
def test_tfidf_matches_brute_force_on_scifact_sample():
    from src.index.inverted_index import InvertedIndex
    from src.index.tokenizer import Tokenizer
    from src.ingest.chunking import build_chunks
    from src.ingest.load_scifact import load_corpus, load_split

    corpus = load_corpus()
    ids = sorted(corpus)[:300]
    index = InvertedIndex.build(build_chunks({i: corpus[i] for i in ids}, "whole"), Tokenizer(), "whole")
    queries, _ = load_split("test")
    rng = random.Random(0)
    for q in rng.sample(list(queries.values()), 5):
        assert np.allclose(TfidfRetriever(index, (0.1, 1.0)).score_all(q),
                           brute_force_tfidf(index, q, (0.1, 1.0)), atol=1e-12)


# ----------------------------------------------------------------- BM25
def test_bm25_matches_hand_formula(toy_index):
    k1, b = 1.2, 0.75
    r = BM25Retriever(toy_index, k1, b)
    q = "insurance fraud insurance"
    tok = toy_index.tokenizer.tokenize
    docs = [Counter(tok(f"{c.title} {c.text}")) for c in toy_index.chunks]
    n, avgdl = len(docs), sum(sum(d.values()) for d in docs) / len(docs)
    expected = []
    for d in docs:
        dl, s = sum(d.values()), 0.0
        for term, qtf in Counter(tok(q)).items():
            df = sum(1 for x in docs if term in x)
            if term in d:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += qtf * idf * d[term] * (k1 + 1) / (d[term] + k1 * (1 - b + b * dl / avgdl))
        expected.append(s)
    assert np.allclose(r.score_all(q), expected, atol=1e-12)
    top = r.search(q, 2)
    assert top[0][0] == "d4"   # 'insurance fraud' paper
    assert sum(row["contribution"] for row in r.explain(q, "d4")) == pytest.approx(top[0][1])


# ----------------------------------------------------------------- top-k and MaxP
def test_heap_top_k_order_ties_and_zeros():
    scores = np.array([0.0, 0.5, 0.9, 0.5, 0.0, 0.7])
    assert heap_top_k(scores, 3) == [2, 5, 1]           # tie 1 vs 3 keeps index order
    assert heap_top_k(scores, 10) == [2, 5, 1, 3]       # zero scores are not hits
    assert heap_top_k(scores, 10, positive_only=False)[-2:] == [0, 4]


def test_maxp_ranks_papers_by_best_window(toy_corpus):
    from src.index.inverted_index import InvertedIndex
    from src.index.tokenizer import Tokenizer
    from src.ingest.chunking import chunk_sentences

    chunks = [c for d, v in toy_corpus.items() for c in chunk_sentences({"doc_id": d, **v}, window=1)]
    index = InvertedIndex.build(chunks, Tokenizer(), "sent")
    assert index.n_chunks > len(index.doc_ids)          # several windows per paper
    r = BM25Retriever(index)
    q = "tumor training mice insurance"
    papers = r.rank_docs(q, 10)
    chunk_scores = r.score_all(q)
    best = {}
    for i, c in enumerate(index.chunks):
        best[c.doc_id] = max(best.get(c.doc_id, 0.0), chunk_scores[i])
    assert papers[0][0] == "d3"
    assert [s for _, s in papers] == sorted((s for _, s in papers), reverse=True)
    assert all(s == pytest.approx(best[d]) for d, s in papers)
    assert len({d for d, _ in papers}) == len(papers)    # one entry per paper


# ----------------------------------------------------------------- fusion
def test_rrf_by_hand():
    fused = dict(rrf([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert [x for x, _ in rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)] == ["b", "c", "a", "d"]


def test_weighted_fusion_min_max():
    assert minmax({"x": 2.0, "y": 4.0, "z": 3.0}) == {"x": 0.0, "y": 1.0, "z": 0.5}
    fused = dict(weighted([{"a": 10, "b": 0}, {"b": 0.9, "c": 0.1}], alpha=0.25))
    assert fused == pytest.approx({"a": 0.25, "b": 0.75, "c": 0.0})
