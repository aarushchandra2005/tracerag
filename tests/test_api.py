"""HTTP API tests. They use the real SciFact files and indexes (skipped if not downloaded).

Model-backed endpoints (answers, verification) run only when the dense embedding
caches exist, i.e. after `python scripts/build_index.py --dense`.
"""
import math

import numpy as np
import pytest

import config
from tests.conftest import needs_scifact

pytest.importorskip("fastapi")
pytest.importorskip("httpx")           # FastAPI's TestClient needs it (requirements-dev.txt)
from fastapi.testclient import TestClient  # noqa: E402

from src.api.engine import AppError, Engine  # noqa: E402
from src.api.server import _clean, create_app  # noqa: E402

CLAIM_53 = "ALDH1 expression is associated with poorer prognosis in breast cancer."


def dense_caches_exist() -> bool:
    try:
        from src.index.dense_index import cache_path_for
        from src.index.inverted_index import get_index

        return all(cache_path_for(get_index(ch).chunks, config.DENSE_MODEL).exists() for ch in config.CHUNKINGS)
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def sparse_client():
    engine = Engine(nli_enabled=False)
    engine.prepare(log=lambda *a: None)          # no background loading: dense stays unavailable
    return TestClient(create_app(engine))


@pytest.fixture(scope="module")
def full_client():
    if not dense_caches_exist():
        pytest.skip("dense caches not built (python scripts/build_index.py --dense)")
    engine = Engine(nli_enabled=False)
    engine.prepare(log=lambda *a: None)
    engine.start_background()
    engine.wait(600)
    if not engine.components["dense_sent"].ready:
        pytest.skip(f"dense index unavailable: {engine.components['dense_sent'].message}")
    return TestClient(create_app(engine))


def test_clean_handles_numpy_and_nan():
    out = _clean({"a": np.float32(0.5), "b": np.int64(3), "c": float("nan"), "d": np.bool_(True), "e": [np.inf]})
    assert out == {"a": 0.5, "b": 3, "c": None, "d": True, "e": [None]}


# ------------------------------------------------------------------ status and data
@needs_scifact
def test_status_reports_components(sparse_client):
    s = sparse_client.get("/api/status").json()
    assert s["components"]["sparse"]["state"] == "ready"
    assert s["components"]["nli"]["state"] == "disabled"
    assert s["data"]["docs"] == 5183 and s["data"]["test_claims"] == 300
    assert s["retrievers"]["available"]["whole"]["bm25"] is True
    assert s["retrievers"]["available"]["whole"]["dense"] is False
    assert s["verifier"]["available"] == ["cosine"]


@needs_scifact
def test_claims_endpoints(sparse_client):
    r = sparse_client.get("/api/claims", params={"q": "onchocerciasis"}).json()
    assert r["total"] == 1 and r["items"][0]["qid"] == "660" and r["items"][0]["relevant"] == ["1215116"]
    c = sparse_client.get("/api/claims/53").json()
    assert c["label"] == "SUPPORT" and c["papers"][0] == {
        "doc_id": "45638119", "label": "SUPPORT",
        "title": "ALDH1 is a marker of normal and malignant human mammary stem cells and a predictor of poor clinical outcome."}
    assert sparse_client.get("/api/claims/random", params={"label": "CONTRADICT"}).json()["label"] == "CONTRADICT"
    bad = sparse_client.get("/api/claims/99999")
    assert bad.status_code == 404 and "99999" in bad.json()["detail"]


@needs_scifact
def test_paper_endpoint(sparse_client):
    p = sparse_client.get("/api/papers/45638119").json()
    assert p["title"].startswith("ALDH1 is a marker")
    assert p["windows"][0] == {"chunk_id": "45638119_0", "start": 0, "end": 3}
    assert sparse_client.get("/api/papers/1").status_code == 404


# ------------------------------------------------------------------ retrieval
@needs_scifact
def test_search_marks_judged_papers(sparse_client):
    r = sparse_client.post("/api/search", json={"query": CLAIM_53, "retriever": "bm25", "chunking": "whole", "k": 3}).json()
    assert r["claim"]["qid"] == "53"                       # recognised from the exact text
    top = r["results"][0]
    assert top["chunk_id"] == "45638119" and top["relevant"] is True
    assert abs(sum(t["share"] for t in top["terms"]) - 1) < 0.5   # top-4 shares of the tf-idf score


@needs_scifact
def test_dense_unavailable_is_a_clear_409(sparse_client):
    r = sparse_client.post("/api/search", json={"query": CLAIM_53, "retriever": "hybrid_rrf", "chunking": "sent"})
    assert r.status_code == 409 and "BM25" in r.json()["detail"]


@needs_scifact
def test_compare_reports_per_claim_metrics(sparse_client):
    r = sparse_client.post("/api/compare", json={"query": "Ivermectin is used to treat onchocerciasis."}).json()
    bm25 = r["retrievers"]["bm25"]
    assert bm25["first_relevant"] == 1 and bm25["metrics"]["nDCG@10"] == 1.0
    assert r["retrievers"]["dense"]["available"] is False
    assert r["query_terms"] == ["ivermectin", "us", "treat", "onchocerciasi"]


@needs_scifact
def test_validation_errors(sparse_client):
    assert sparse_client.post("/api/search", json={"query": "x", "retriever": "nope"}).status_code == 422
    assert sparse_client.post("/api/search", json={"query": "", "retriever": "bm25"}).status_code == 422


# ------------------------------------------------------------------ inside the index
@needs_scifact
def test_term_and_tokenizer(sparse_client):
    t = sparse_client.get("/api/index/term", params={"word": "Tumors", "limit": 2}).json()
    assert t["term"] == "tumor" and t["zones"]["all"]["df"] == 542 and t["total"] == 542
    assert t["postings"][0]["tf"] >= t["postings"][1]["tf"]
    assert sparse_client.get("/api/index/term", params={"word": "the"}).json()["removed"] is True
    steps = sparse_client.post("/api/index/tokenize", json={"text": "The tumors were growing"}).json()
    assert steps["terms"] == ["tumor", "grow"] and steps["tokens"][0] == {"token": "the", "stopword": True, "stem": "the", "term": None}
    stats = sparse_client.get("/api/index/stats").json()
    assert stats["sent"]["chunks"] == 34718


@needs_scifact
@pytest.mark.parametrize("retriever,chunk", [("tfidf", "45638119"), ("tfidf", "45638119_2"), ("bm25", "45638119")])
def test_explain_matches_library_score(sparse_client, retriever, chunk):
    r = sparse_client.post("/api/explain", json={"query": CLAIM_53, "chunk_id": chunk, "retriever": retriever}).json()
    assert r["matches_library"] is True and math.isclose(r["score"], r["library_score"], abs_tol=1e-9)


# ------------------------------------------------------------------ answers and verification
@needs_scifact
def test_answer_needs_the_encoder(sparse_client):
    r = sparse_client.post("/api/ask", json={"query": CLAIM_53, "retriever": "bm25", "chunking": "whole"})
    assert r.status_code == 409 and "encoder" in r.json()["detail"]


@needs_scifact
def test_llm_settings_are_validated(sparse_client, monkeypatch):
    engine = sparse_client.app.state.engine
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AppError, match="model name"):
        engine._make_llm("groq", "", None, None)
    with pytest.raises(AppError, match="API key"):
        engine._make_llm("openrouter", "some-model", None, None)
    assert engine._make_llm("openrouter", "some-model", "key-for-this-request", None).model == "some-model"


@needs_scifact
def test_ask_verify_and_corrupt(full_client):
    r = full_client.post("/api/ask", json={"query": CLAIM_53, "verifier": "cascade"})
    assert r.status_code == 200
    res = r.json()
    assert res["verification"]["method_used"] == "cosine" and res["verification"]["note"]   # NLI is off here
    assert all(s["cited_ids"] for s in res["verification"]["sentences"])
    assert any(x["cited"] for x in res["retrieved"])
    ids = [x["chunk_id"] for x in res["retrieved"]]

    bad = full_client.post("/api/verify", json={"answer": res["answer"] + " It cures cancer [999_9].", "chunk_ids": ids,
                                                "verifier": "cosine"}).json()
    assert bad["sentences"][-1]["verdict"]["label"] == "unsupported" and bad["sentences"][-1]["invalid_ids"] == ["999_9"]

    for kind in ("negate", "swap", "fabricate"):
        c = full_client.post("/api/corrupt", json={"answer": res["answer"], "chunk_ids": ids, "kind": kind,
                                                   "query": CLAIM_53, "verifier": "cosine", "seed": 3})
        assert c.status_code == 200, c.json()
        body = c.json()
        assert body["verification"]["sentences"][body["index"]]["sentence"] == body["corrupted"]
    fab = full_client.post("/api/corrupt", json={"answer": res["answer"], "chunk_ids": ids, "kind": "fabricate",
                                                 "query": CLAIM_53, "verifier": "cosine", "seed": 3}).json()
    assert fab["verification"]["sentences"][fab["index"]]["verdict"]["label"] == "unsupported"


@needs_scifact
def test_web_app_and_docs_are_served(sparse_client):
    page = sparse_client.get("/")
    assert page.status_code == 200 and "TraceRAG" in page.text
    assert sparse_client.get("/js/main.js").status_code == 200
    assert sparse_client.get("/openapi.json").json()["info"]["title"] == "TraceRAG API"
    assert sparse_client.get("/api/evaluation").status_code == 200
