import math
import random

import pytest

from src.eval.metrics import (bootstrap_ci, evaluate, first_relevant_rank, mrr, ndcg_at_k,
                              paired_randomization_test, precision_at_k, recall_at_k)

RANKED = ["d3", "d1", "d7", "d2", "d9"]
QRELS = {"d1": 1, "d2": 1, "d5": 1}


def test_hand_example():
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 3)
    assert mrr(RANKED, QRELS) == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, QRELS, 5) == pytest.approx(dcg / idcg)
    assert first_relevant_rank(RANKED, QRELS) == 2


def test_edge_cases():
    assert precision_at_k([], QRELS, 10) == 0.0
    assert mrr(["x", "y"], QRELS) == 0.0
    assert mrr(RANKED, QRELS, k=1) == 0.0
    assert ndcg_at_k(["d1", "d1", "d2"], {"d1": 1, "d2": 1}, 2) == pytest.approx(1.0)  # duplicates ignored
    assert recall_at_k(RANKED, {}, 10) == 0.0


def test_evaluate_counts_missing_queries_as_zero():
    means, per = evaluate({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}})
    assert means["nDCG@10"] == pytest.approx(0.5) and per["q2"]["MRR@10"] == 0.0


def test_matches_trec_eval_on_random_runs():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(7)
    docs = [f"d{i}" for i in range(60)]
    qrels, run = {}, {}
    for q in range(40):
        qid = f"q{q}"
        qrels[qid] = {d: rng.choice([1, 1, 2]) for d in rng.sample(docs, rng.randint(1, 6))}
        run[qid] = rng.sample(docs, 30)
    # trec_eval sorts by score, so give strictly decreasing scores; truncate to 10 for MRR@10
    trec_run = {q: {d: float(100 - i) for i, d in enumerate(r)} for q, r in run.items()}
    trec_run10 = {q: {d: float(100 - i) for i, d in enumerate(r[:10])} for q, r in run.items()}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"P_5", "P_10", "recall_10", "ndcg_cut_10"})
    ev10 = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    ref, ref10 = ev.evaluate(trec_run), ev10.evaluate(trec_run10)
    for q in qrels:
        assert precision_at_k(run[q], qrels[q], 5) == pytest.approx(ref[q]["P_5"])
        assert precision_at_k(run[q], qrels[q], 10) == pytest.approx(ref[q]["P_10"])
        assert recall_at_k(run[q], qrels[q], 10) == pytest.approx(ref[q]["recall_10"])
        assert ndcg_at_k(run[q], qrels[q], 10) == pytest.approx(ref[q]["ndcg_cut_10"])
        assert mrr(run[q], qrels[q], 10) == pytest.approx(ref10[q]["recip_rank"])


def test_randomization_test_and_bootstrap():
    a = [0.9, 0.8, 0.85, 0.95, 0.7] * 20
    assert paired_randomization_test(a, a) == 1.0
    assert paired_randomization_test(a, [x - 0.2 for x in a]) < 0.001
    lo, hi = bootstrap_ci(a)
    assert lo < sum(a) / len(a) < hi


def test_saved_runs_score_the_same_under_trec_eval():
    """The main table, recomputed from results/runs/*.trec with trec_eval."""
    import csv

    import config
    from tests.conftest import scifact_available

    pytrec_eval = pytest.importorskip("pytrec_eval")
    main_csv = config.RESULTS_DIR / "retrieval_main.csv"
    if not (scifact_available() and main_csv.exists()):
        pytest.skip("needs SciFact and results from run_retrieval_eval")
    from src.ingest.load_scifact import load_qrels

    qrels = load_qrels("test")
    main = {r["Retriever"]: r for r in csv.DictReader(open(main_csv, encoding="utf-8"))}
    labels = {"tfidf": "tf-idf lnc.ltc (zones)", "bm25": "BM25", "dense": "Dense (MiniLM)",
              "hybrid_rrf": "Hybrid RRF (BM25 + dense)", "hybrid_weighted": "Hybrid weighted (BM25 + dense)"}
    for name, label in labels.items():
        path = config.RESULTS_DIR / "runs" / f"{name}_whole.trec"
        if label not in main or not path.exists():
            continue
        run = {}
        for line in open(path, encoding="utf-8"):
            q, _, d, rank, _, _ = line.split()
            run.setdefault(q, {})[d] = -int(rank)
        res = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"}).evaluate(run)
        ndcg = sum(res[q]["ndcg_cut_10"] for q in qrels if q in res) / len(qrels)
        assert ndcg == pytest.approx(float(main[label]["nDCG@10"]), abs=1e-6)
