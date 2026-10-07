"""HTTP API and static web app.

    python run_app.py            # starts this server and opens the browser

Interactive API documentation is served at /docs once the server runs.
All routes return JSON; errors come back as {"detail": "<what went wrong and how to fix it>"}.
"""
from __future__ import annotations

import json
import math
import mimetypes
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.api.engine import AppError, Engine
from src.generate.llm import LLMError

WEB_DIR = config.ROOT / "web"

# Some Windows machines map .js to text/plain in the registry, and browsers refuse to run
# ES modules served that way. Register the right types explicitly.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

RetrieverName = Literal["tfidf", "tfidf_single", "bm25", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf_tfidf"]
Chunking = Literal["whole", "sent"]
Verifier = Literal["cosine", "nli", "cascade"]
Backend = Literal["extractive", "anthropic", "openai", "groq", "gemini", "openrouter", "ollama"]


# ------------------------------------------------------------------ JSON that never fails on numpy or NaN
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------------ request bodies
class LLMSettings(BaseModel):
    backend: Backend = "extractive"
    model: str = Field("", max_length=200, description="model id; required for OpenAI-compatible providers")
    api_key: str | None = Field(None, max_length=500, description="used for this request only, never stored")
    base_url: str | None = Field(None, max_length=500, description="custom OpenAI-compatible endpoint")


class AskBody(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    qid: str | None = None
    retriever: RetrieverName = "hybrid_rrf"
    chunking: Chunking = "sent"
    k: int = Field(5, ge=1, le=20)
    llm: LLMSettings = LLMSettings()
    verifier: Verifier = "cascade"


class SearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    qid: str | None = None
    retriever: RetrieverName = "bm25"
    chunking: Chunking = "whole"
    k: int = Field(10, ge=1, le=100)


class CompareBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    qid: str | None = None
    chunking: Chunking = "whole"
    k: int = Field(10, ge=1, le=50)
    retrievers: list[RetrieverName] | None = None


class VerifyBody(BaseModel):
    answer: str = Field(..., min_length=1, max_length=20000)
    chunk_ids: list[str] = Field(..., max_length=50, description="the chunks the answer may cite")
    verifier: Verifier = "cascade"


class CorruptBody(VerifyBody):
    kind: Literal["negate", "swap", "fabricate"]
    query: str = Field(..., min_length=1, max_length=2000)
    qid: str | None = None
    seed: int | None = None


class ExplainBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chunk_id: str
    retriever: Literal["tfidf", "bm25", "dense"] = "tfidf"


class TokenizeBody(BaseModel):
    text: str = Field(..., max_length=5000)
    stem: bool = True
    stopwords: bool = True


# ------------------------------------------------------------------ app
def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="TraceRAG API", version="1.0",
                  description="Retrieval, cited answers and claim-level citation checking on BEIR SciFact.")
    app.state.engine = engine

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError):
        return SafeJSON({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError):
        return SafeJSON({"detail": f"The language model call failed: {exc}"}, status_code=502)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        return SafeJSON({"detail": f"Unexpected server error ({type(exc).__name__}: {exc}). "
                                   "The terminal running the app shows the full traceback."}, status_code=500)

    # -------------------------------------------------------------- status and data
    @app.get("/api/status", tags=["status"], summary="Readiness of data, indexes and models")
    def status():
        return SafeJSON(engine.status())

    @app.get("/api/claims", tags=["data"], summary="SciFact claims, filterable")
    def claims(split: Literal["test", "train", "all"] = "test", label: str | None = None, q: str | None = None,
               limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return SafeJSON(engine.list_claims(split, label or None, q, limit, offset))

    @app.get("/api/claims/random", tags=["data"], summary="A random SciFact claim")
    def random_claim(split: Literal["test", "train", "all"] = "test", label: str | None = None):
        return SafeJSON(engine.random_claim(split, label or None))

    @app.get("/api/claims/{qid}", tags=["data"], summary="One claim with its judged papers")
    def claim(qid: str):
        return SafeJSON(engine.claim_info(engine.resolve_qid("", qid)))

    @app.get("/api/papers/{doc_id}", tags=["data"], summary="One abstract, its sentences and windows")
    def paper(doc_id: str):
        return SafeJSON(engine.paper(doc_id))

    # -------------------------------------------------------------- retrieval, answers, verification
    @app.post("/api/search", tags=["retrieval"], summary="Top-k chunks from one retriever")
    def search(body: SearchBody):
        return SafeJSON(engine.search(body.query, body.retriever, body.chunking, body.k, body.qid))

    @app.post("/api/compare", tags=["retrieval"], summary="Paper rankings from several retrievers side by side")
    def compare(body: CompareBody):
        return SafeJSON(engine.compare(body.query, body.chunking, body.k, body.qid, body.retrievers))

    @app.post("/api/ask", tags=["answers"], summary="Retrieve, write a cited answer, check every sentence")
    def ask(body: AskBody):
        return SafeJSON(engine.ask(body.query, body.qid, body.retriever, body.chunking, body.k, body.llm.backend,
                                   body.llm.model, body.llm.api_key, body.llm.base_url, body.verifier))

    @app.post("/api/verify", tags=["answers"], summary="Check an answer (for example, one you edited)")
    def verify(body: VerifyBody):
        return SafeJSON(engine.verify_answer(body.answer, body.chunk_ids, body.verifier))

    @app.post("/api/corrupt", tags=["answers"], summary="Corrupt one sentence on purpose and check again")
    def corrupt(body: CorruptBody):
        return SafeJSON(engine.corrupt(body.answer, body.chunk_ids, body.kind, body.query, body.qid,
                                       body.verifier, body.seed))

    # -------------------------------------------------------------- inside the index
    @app.get("/api/index/stats", tags=["index"], summary="Dictionary and postings sizes")
    def index_stats():
        return SafeJSON(engine.index_stats())

    @app.get("/api/index/term", tags=["index"], summary="Dictionary entry and postings list of a word")
    def index_term(word: str = Query(..., min_length=1, max_length=100), chunking: Chunking = "whole",
                   zone: Literal["title", "body", "all"] = "all", sort: Literal["tf", "chunk"] = "tf",
                   limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
        return SafeJSON(engine.term(word, chunking, zone, sort, limit, offset))

    @app.post("/api/index/tokenize", tags=["index"], summary="Every tokenizer step for a piece of text")
    def tokenize(body: TokenizeBody):
        return SafeJSON(engine.tokenize_steps(body.text, body.stem, body.stopwords))

    @app.post("/api/explain", tags=["index"], summary="Full score breakdown of one chunk for one query")
    def explain(body: ExplainBody):
        return SafeJSON(engine.explain(body.query, body.chunk_id, body.retriever))

    # -------------------------------------------------------------- evaluation
    @app.get("/api/evaluation", tags=["evaluation"], summary="Tables produced by the evaluation scripts")
    def evaluation():
        return SafeJSON(engine.evaluation())

    # -------------------------------------------------------------- files and the web app
    app.mount("/files/results", StaticFiles(directory=config.RESULTS_DIR, check_dir=False), name="results")
    app.mount("/files/report", StaticFiles(directory=config.REPORT_DIR, check_dir=False), name="report")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
