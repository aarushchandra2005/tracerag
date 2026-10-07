"""Everything the web app does, behind one object.

The Engine owns the indexes, the models and their readiness. Sparse indexes
are built (once) before the server starts; the dense indexes and the NLI model
load in background threads, so the app is usable within seconds while those
finish. Every operation the API exposes is a method here, so it can be tested
without HTTP.
"""
from __future__ import annotations

import csv
import math
import os
import random
import threading
import time
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.eval.metrics import first_relevant_rank, per_query
from src.generate.llm import PRESETS, AnthropicLLM, LLMError, OpenAICompatibleLLM
from src.generate.rag import NOT_ENOUGH, SYSTEM_PROMPT, ExtractiveGenerator, build_user_prompt
from src.index.inverted_index import ZONES, get_index, index_path
from src.index.tokenizer import _TOKEN_RE, STOPWORDS, Tokenizer, porter
from src.ingest import load_scifact
from src.ingest.chunking import Chunk, build_chunks, chunks_path, save_chunks, split_sentences
from src.retrieve.bm25 import bm25_idf
from src.retrieve.factory import NEEDS_DENSE, RetrieverFactory, alpha_for, zone_weights_for
from src.utils import load_tuned
from src.verify import inject
from src.verify.checker import CitationChecker, thresholds

RETRIEVER_LABELS = {
    "tfidf": "tf-idf (lnc.ltc, zones)",
    "tfidf_single": "tf-idf (one field)",
    "bm25": "BM25",
    "dense": "Dense (MiniLM)",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_weighted": "Hybrid weighted",
    "hybrid_rrf_tfidf": "Hybrid RRF (tf-idf + dense)",
}
COMPARE_DEFAULT = ("tfidf", "bm25", "dense", "hybrid_rrf", "hybrid_weighted")
CHUNKING_LABELS = {"whole": "Whole abstracts", "sent": "3-sentence windows"}
VERIFIER_LABELS = {"cosine": "Cosine similarity", "nli": "NLI model", "cascade": "Cosine gate, then NLI"}


class AppError(Exception):
    """An error with an HTTP status and a message meant for the user."""

    status = 400

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status


class NotReady(AppError):
    status = 409


@dataclass
class Component:
    name: str
    label: str
    state: str = "waiting"            # waiting | loading | building | ready | error | disabled
    message: str = ""
    started: float | None = None
    finished: float | None = None
    detail: dict = field(default_factory=dict)

    def set(self, state: str, message: str = "") -> None:
        if state in ("loading", "building") and self.started is None:
            self.started = time.time()
        if state in ("ready", "error", "disabled"):
            self.finished = time.time()
        self.state, self.message = state, message

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict:
        elapsed = None
        if self.started is not None:
            elapsed = round((self.finished or time.time()) - self.started, 1)
        return {"name": self.name, "label": self.label, "state": self.state, "message": self.message,
                "elapsed": elapsed, **self.detail}


def _num(value: str):
    """CSV cell -> int/float when it looks like a number."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return value
    if not math.isfinite(f):
        return None
    return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f


def _quiet_hf() -> None:
    """Hide model-loading progress bars in the app's console (status shows on the page instead)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Library deprecation notices (e.g. torch.jit.script in newer PyTorch) are for developers, not app users.
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def _read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


class Engine:
    def __init__(self, nli_enabled: bool = True):
        self.factory = RetrieverFactory()
        self.checker = CitationChecker("cascade")
        self.nli_enabled = nli_enabled
        self.components = {
            "data": Component("data", "SciFact data"),
            "sparse": Component("sparse", "Inverted indexes"),
            "encoder": Component("encoder", "MiniLM encoder"),
            "dense_whole": Component("dense_whole", "Dense index, whole abstracts"),
            "dense_sent": Component("dense_sent", "Dense index, sentence windows"),
            "nli": Component("nli", "NLI model", detail={"model": config.NLI_MODEL}),
        }
        if not nli_enabled:
            self.components["nli"].set("disabled", "started with --no-nli")
        self._dense_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self.queries: dict[str, dict] = {}
        self.qrels: dict[str, dict] = {}
        self.text_to_qid: dict[str, str] = {}
        self.stats: dict = {}
        self._eval_cache: tuple[float, dict] | None = None

    # ================================================================== start-up
    def prepare(self, log=print) -> None:
        """Download/verify SciFact and build the sparse indexes. Blocking; seconds once built."""
        c = self.components["data"]
        c.set("loading", "checking SciFact files")
        load_scifact.download()
        self.stats = load_scifact.verify(verbose=False)
        if not (config.PROCESSED_DIR / "qrels.json").exists():
            load_scifact.prepare()
        self.queries = load_scifact.load_queries()
        self.qrels = {s: load_scifact.load_qrels(s) for s in ("train", "test")}
        self.text_to_qid = {self._norm(v["text"]): q for q, v in self.queries.items()}
        c.set("ready", f"{self.stats['corpus']:,} abstracts, {self.stats['queries']:,} claims")
        log(f"  SciFact: {c.message}")

        s = self.components["sparse"]
        s.set("building", "building inverted indexes")
        corpus = None
        tok = Tokenizer(config.USE_STEMMING, config.REMOVE_STOPWORDS)
        for chunking in config.CHUNKINGS:
            if not chunks_path(chunking).exists():
                corpus = corpus or load_scifact.load_corpus()
                log(f"  chunking abstracts ({CHUNKING_LABELS[chunking].lower()})")
                save_chunks(build_chunks(corpus, chunking), chunking)
            if not index_path(chunking, tok).exists():
                log(f"  building the inverted index ({CHUNKING_LABELS[chunking].lower()})")
            self.factory._indexes[(chunking, tok.stem, tok.stopwords)] = get_index(chunking)
        idx = self.index("whole")
        s.detail = {"terms": len(idx.zones["all"].terms), "chunks": {ch: self.index(ch).n_chunks
                                                                      for ch in config.CHUNKINGS}}
        s.set("ready", f"{len(idx.zones['all'].terms):,} terms; "
                       f"{self.index('sent').n_chunks:,} sentence windows")
        log(f"  Indexes: {s.message}")

    def start_background(self) -> None:
        """Load the encoder and dense indexes, and the NLI model, without blocking requests."""
        t = threading.Thread(target=self._load_dense, name="dense-loader", daemon=True)
        t.start()
        self._threads.append(t)
        if self.nli_enabled:
            t = threading.Thread(target=self._load_nli, name="nli-loader", daemon=True)
            t.start()
            self._threads.append(t)

    def wait(self, timeout: float | None = None) -> None:
        for t in self._threads:
            t.join(timeout)

    def _load_dense(self) -> None:
        _quiet_hf()
        from src.index.dense_index import DenseIndex, cache_path_for, get_encoder, load_encoder
        from src.retrieve.dense import DenseRetriever

        enc = self.components["encoder"]
        enc.set("loading", f"loading {config.DENSE_MODEL} (downloads about 90 MB the first time)")
        try:
            get_encoder(config.DENSE_MODEL)
        except Exception as e:  # noqa: BLE001 - offline, blocked download, bad model id
            enc.set("error", f"could not load {config.DENSE_MODEL}: {type(e).__name__}: {e}")
            for ch in config.CHUNKINGS:
                self.components[f"dense_{ch}"].set("error", "needs the MiniLM encoder")
            return
        enc.set("ready", config.DENSE_MODEL)
        builder = None
        for ch in ("whole", "sent"):
            comp = self.components[f"dense_{ch}"]
            index = self.index(ch)
            try:
                if cache_path_for(index.chunks, config.DENSE_MODEL).exists():
                    comp.set("loading", "reading cached embeddings")
                    dense_index = DenseIndex(index.chunks, config.DENSE_MODEL, show_progress=False)
                else:
                    def progress(done: int, total: int, comp=comp) -> None:
                        comp.detail = {"done": done, "total": total}
                        comp.set("building", f"encoding {total:,} chunks once ({done:,} done); "
                                             "they are cached afterwards")

                    progress(0, index.n_chunks)
                    builder = builder or load_encoder(config.DENSE_MODEL)   # private copy: requests keep theirs
                    dense_index = DenseIndex(index.chunks, config.DENSE_MODEL, show_progress=False, encoder=builder,
                                             on_progress=progress)
                    comp.detail = {}
                with self._dense_lock:
                    self.factory._dense[ch] = DenseRetriever(index, dense_index)
                comp.set("ready", f"{index.n_chunks:,} vectors")
            except Exception as e:  # noqa: BLE001
                comp.set("error", f"{type(e).__name__}: {e}")

    def _load_nli(self) -> None:
        _quiet_hf()
        c = self.components["nli"]
        c.set("loading", f"loading {config.NLI_MODEL} (downloads about 570 MB the first time)")
        try:
            _ = self.checker.nli
            c.set("ready", config.NLI_MODEL)
        except Exception as e:  # noqa: BLE001
            c.set("error", f"could not load {config.NLI_MODEL}: {type(e).__name__}: {e}")

    # ================================================================== helpers
    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())

    def index(self, chunking: str):
        if chunking not in config.CHUNKINGS:
            raise AppError(f"unknown chunking {chunking!r}")
        return self.factory.index(chunking)

    def dense_ready(self, chunking: str) -> bool:
        return self.components[f"dense_{chunking}"].ready

    def retriever(self, name: str, chunking: str):
        if name not in RETRIEVER_LABELS:
            raise AppError(f"unknown retriever {name!r}")
        if name in NEEDS_DENSE and not self.dense_ready(chunking):
            comp = self.components[f"dense_{chunking}"]
            if comp.state == "error":
                raise NotReady(f"{RETRIEVER_LABELS[name]} is unavailable: {comp.message}", 503)
            raise NotReady(f"{RETRIEVER_LABELS[name]} needs the dense index for "
                           f"{CHUNKING_LABELS[chunking].lower()}, which is still {comp.state}. "
                           "Use tf-idf or BM25 meanwhile.")
        with self._dense_lock:
            return self.factory.get(name, chunking)

    def get_chunk(self, chunk_id: str) -> Chunk:
        chunking = "sent" if "_" in chunk_id else "whole"
        idx = self.index(chunking)
        pos = idx.chunk_pos.get(chunk_id)
        if pos is None:
            raise AppError(f"no chunk with id {chunk_id!r}", 404)
        return idx.chunks[pos]

    def resolve_qid(self, query: str, qid: str | None) -> str | None:
        if qid:
            if qid not in self.queries:
                raise AppError(f"no SciFact claim with id {qid!r}", 404)
            return qid
        return self.text_to_qid.get(self._norm(query))

    def relevant_docs(self, qid: str | None) -> set[str] | None:
        if not qid:
            return None
        split = self.queries[qid]["split"]
        return set(self.qrels.get(split, {}).get(qid, {}))

    def claim_info(self, qid: str | None) -> dict | None:
        if not qid:
            return None
        q = self.queries[qid]
        rel = self.relevant_docs(qid) or set()
        papers = []
        for doc_id in sorted(rel):
            entries = q["evidence"].get(doc_id)
            label = entries[0]["label"] if entries else "NO_EVIDENCE"
            papers.append({"doc_id": doc_id, "label": label, "title": self.index("whole").chunks[
                self.index("whole").chunk_pos[doc_id]].title})
        return {"qid": qid, "text": q["text"], "label": q["label"], "split": q["split"], "papers": papers}

    # ================================================================== status
    def status(self) -> dict:
        comps = {k: c.to_dict() for k, c in self.components.items()}
        avail = {}
        for ch in config.CHUNKINGS:
            avail[ch] = {name: (name not in NEEDS_DENSE or self.dense_ready(ch)) for name in RETRIEVER_LABELS}
        llm = {"extractive": {"label": "Offline extractive (no key)", "needs_key": False, "key_in_env": True,
                              "needs_model": False}}
        llm["anthropic"] = {"label": "Claude (Anthropic API)", "needs_key": True, "needs_model": False,
                            "key_in_env": bool(os.environ.get("ANTHROPIC_API_KEY")),
                            "default_model": config.ANTHROPIC_DEFAULT_MODEL, "key_env": "ANTHROPIC_API_KEY"}
        names = {"openai": "OpenAI or any compatible API", "groq": "Groq", "gemini": "Google Gemini",
                 "openrouter": "OpenRouter", "ollama": "Ollama (local)"}
        for b, (url, key_env) in PRESETS.items():
            llm[b] = {"label": names[b], "needs_key": key_env is not None, "needs_model": True,
                      "key_in_env": bool(key_env and os.environ.get(key_env)), "key_env": key_env,
                      "base_url": url}
        t = thresholds()
        methods = ["cosine"] + (["nli", "cascade"] if self.components["nli"].ready else [])
        return {
            "components": comps,
            "data": {"docs": self.stats.get("corpus"), "claims": self.stats.get("queries"),
                     "test_claims": self.stats.get("test_queries"), "train_claims": self.stats.get("train_queries"),
                     "fingerprint": self.stats.get("fingerprint")},
            "retrievers": {"labels": RETRIEVER_LABELS, "available": avail},
            "chunkings": CHUNKING_LABELS,
            "llm": llm,
            "verifier": {"labels": VERIFIER_LABELS, "available": methods, "thresholds": t,
                         "window": config.PREMISE_WINDOW},
            "defaults": {"retriever": config.GEN_RETRIEVER, "chunking": config.GEN_CHUNKING,
                         "k": config.TOP_K_GEN, "verifier": "cascade", "llm": config.LLM_BACKEND},
            "tuned": {"zone_weights": {ch: list(zone_weights_for(ch)) for ch in config.CHUNKINGS},
                      "hybrid_alpha": {ch: alpha_for(ch) for ch in config.CHUNKINGS},
                      "bm25": {"k1": config.BM25_K1, "b": config.BM25_B}, "rrf_k": config.RRF_K},
            "busy": any(c.state in ("waiting", "loading", "building") for c in self.components.values()
                        if c.name != "nli" or self.nli_enabled),
        }

    # ================================================================== claims and papers
    def list_claims(self, split: str = "test", label: str | None = None, q: str | None = None,
                    limit: int = 50, offset: int = 0) -> dict:
        items = []
        needle = self._norm(q) if q else None
        for qid, v in self.queries.items():
            if split != "all" and v["split"] != split:
                continue
            if label and v["label"] != label:
                continue
            if needle and needle not in self._norm(v["text"]) and needle != qid:
                continue
            items.append({"qid": qid, "text": v["text"], "label": v["label"], "split": v["split"],
                          "relevant": sorted(self.relevant_docs(qid) or [])})
        items.sort(key=lambda x: int(x["qid"]) if x["qid"].isdigit() else 0)
        return {"total": len(items), "items": items[offset:offset + limit]}

    def random_claim(self, split: str = "test", label: str | None = None) -> dict:
        pool = [q for q, v in self.queries.items() if (split == "all" or v["split"] == split)
                and (not label or v["label"] == label)]
        if not pool:
            raise AppError("no claim matches that filter", 404)
        return self.claim_info(random.choice(pool))

    def paper(self, doc_id: str) -> dict:
        whole = self.index("whole")
        pos = whole.chunk_pos.get(doc_id)
        if pos is None:
            raise AppError(f"no paper with id {doc_id!r}", 404)
        c = whole.chunks[pos]
        windows = [{"chunk_id": w.chunk_id, "start": w.start, "end": w.end}
                   for w in self.index("sent").chunks if w.doc_id == doc_id]
        return {"doc_id": doc_id, "title": c.title, "text": c.text, "sentences": split_sentences(c.text),
                "windows": windows}

    # ================================================================== retrieval
    def _chunk_row(self, chunk: Chunk, score: float, rank: int, rel: set | None, tfidf=None, query=None) -> dict:
        row = {"rank": rank, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "title": chunk.title,
               "text": chunk.text, "score": float(score),
               "relevant": (chunk.doc_id in rel) if rel is not None else None}
        if tfidf is not None:
            row["terms"] = self._term_shares(tfidf, query, chunk.chunk_id, 4)
        return row

    @staticmethod
    def _term_shares(retriever, query: str, chunk_id: str, n: int) -> list[dict]:
        """Each query term's share of the score, summed over zones (title + body)."""
        rows = retriever.explain(query, chunk_id)
        total = sum(r["contribution"] for r in rows)
        merged: dict[str, dict] = {}
        for r in rows:
            m = merged.setdefault(r["term"], {"term": r["term"], "zones": [], "contribution": 0.0})
            m["contribution"] += r["contribution"]
            m["zones"].append(r.get("zone", "title + body"))
        out = sorted(merged.values(), key=lambda m: -m["contribution"])[:n]
        return [{"term": m["term"], "zones": m["zones"], "share": m["contribution"] / total if total else 0.0} for m in out]

    def search(self, query: str, retriever: str, chunking: str, k: int, qid: str | None = None) -> dict:
        qid = self.resolve_qid(query, qid)
        rel = self.relevant_docs(qid)
        r = self.retriever(retriever, chunking)
        t0 = time.perf_counter()
        hits = r.search(query, k)
        ms = (time.perf_counter() - t0) * 1000
        tfidf = self.factory.get("tfidf", chunking)
        idx = self.index(chunking)
        rows = [self._chunk_row(idx.chunks[idx.chunk_pos[cid]], s, i, rel, tfidf, query)
                for i, (cid, s) in enumerate(hits, 1)]
        return {"query": query, "claim": self.claim_info(qid), "retriever": retriever, "chunking": chunking,
                "results": rows, "ms": round(ms, 1), "query_terms": idx.tokenizer.tokenize(query)}

    def compare(self, query: str, chunking: str = "whole", k: int = 10, qid: str | None = None,
                retrievers: list[str] | None = None) -> dict:
        qid = self.resolve_qid(query, qid)
        rel = self.relevant_docs(qid)
        idx = self.index(chunking)
        whole = self.index("whole")
        out = {}
        for name in retrievers or COMPARE_DEFAULT:
            col = {"name": name, "label": RETRIEVER_LABELS.get(name, name)}
            try:
                r = self.retriever(name, chunking)
            except NotReady as e:
                out[name] = col | {"available": False, "message": e.message}
                continue
            t0 = time.perf_counter()
            ranked = r.rank_docs(query, 100)
            col["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            explainer = r if name in ("tfidf", "bm25") else None
            papers = []
            for rank, (doc_id, score) in enumerate(ranked[:k], 1):
                p = {"rank": rank, "doc_id": doc_id, "score": float(score),
                     "title": whole.chunks[whole.chunk_pos[doc_id]].title,
                     "relevant": (doc_id in rel) if rel is not None else None}
                if explainer is not None and chunking == "whole":
                    p["terms"] = self._term_shares(explainer, query, doc_id, 3)
                papers.append(p)
            col.update(available=True, papers=papers)
            if rel:
                ids = [d for d, _ in ranked]
                col["first_relevant"] = first_relevant_rank(ids, {d: 1 for d in rel})
                col["metrics"] = per_query(ids, {d: 1 for d in rel})
            out[name] = col
        return {"query": query, "claim": self.claim_info(qid), "chunking": chunking, "k": k,
                "query_terms": idx.tokenizer.tokenize(query), "retrievers": out}

    # ================================================================== generation + verification
    def _make_llm(self, backend: str, model: str, api_key: str | None, base_url: str | None):
        if backend not in ("anthropic",) + tuple(PRESETS):
            raise AppError(f"unknown answer backend {backend!r}")
        key_env = "ANTHROPIC_API_KEY" if backend == "anthropic" else PRESETS[backend][1]
        if backend in PRESETS and not (model or "").strip():
            raise AppError("Enter a model name for this provider (copy it from the provider's model list).")
        if key_env and not (api_key or os.environ.get(key_env)):
            raise AppError(f"Enter an API key, or set {key_env} before starting the app.")
        try:
            if backend == "anthropic":
                return AnthropicLLM(model or "", api_key=api_key or None)
            return OpenAICompatibleLLM(backend, model.strip(), api_key=api_key or None, base_url=base_url or None)
        except LLMError as e:
            raise AppError(str(e)) from e

    def _method(self, requested: str) -> tuple[str, str | None]:
        if requested not in VERIFIER_LABELS:
            raise AppError(f"unknown verifier {requested!r}")
        if not self.components["encoder"].ready:
            enc = self.components["encoder"]
            raise NotReady(f"The verifier needs the MiniLM encoder, which is {enc.state}"
                           + (f": {enc.message}" if enc.state == "error" else ". Try again in a moment."))
        if requested != "cosine" and not self.components["nli"].ready:
            nli = self.components["nli"]
            reason = {"disabled": "the app was started with --no-nli",
                      "error": nli.message}.get(nli.state, f"the NLI model is still {nli.state}")
            return "cosine", f"Checked with cosine similarity because {reason}."
        return requested, None

    def verify(self, answer: str, chunks: list[Chunk], method: str) -> dict:
        used, note = self._method(method)
        self.checker.t = thresholds()               # picks up thresholds re-tuned while the app runs
        t0 = time.perf_counter()
        rows = self.checker.check_answer(answer, chunks, used)
        ms = (time.perf_counter() - t0) * 1000
        sentences = []
        for i, r in enumerate(rows):
            v = r["verdict"].to_dict()
            sentences.append({"index": i, "sentence": r["sentence"], "text": r["text"], "cited_ids": r["cited_ids"],
                              "invalid_ids": r["invalid_ids"], "abstain": r["abstain"], "verdict": v})
        counts = Counter(s["verdict"]["label"] for s in sentences)
        return {"sentences": sentences, "method_requested": method, "method_used": used, "note": note,
                "thresholds": self.checker.t, "ms": round(ms, 1),
                "summary": {"supported": counts.get("supported", 0), "unsupported": counts.get("unsupported", 0),
                            "abstain": counts.get("abstain", 0), "total": len(sentences)}}

    def ask(self, query: str, qid: str | None, retriever: str, chunking: str, k: int, backend: str,
            model: str, api_key: str | None, base_url: str | None, verifier: str) -> dict:
        qid = self.resolve_qid(query, qid)
        rel = self.relevant_docs(qid)
        r = self.retriever(retriever, chunking)
        if backend == "extractive" and not self.components["encoder"].ready:
            raise NotReady("The offline answer writer needs the MiniLM encoder, which is "
                           f"{self.components['encoder'].state}. Try again in a moment.")
        self._method(verifier)                       # fail before spending an API call
        llm = None if backend == "extractive" else self._make_llm(backend, model, api_key, base_url)

        t0 = time.perf_counter()
        hits = r.search(query, k)
        t1 = time.perf_counter()
        idx = self.index(chunking)
        chunks = [idx.chunks[idx.chunk_pos[cid]] for cid, _ in hits]
        if llm is None:
            text = ExtractiveGenerator().generate_from_chunks(query, chunks) if chunks else NOT_ENOUGH
            model_used = config.DENSE_MODEL
        else:
            text = llm.generate(SYSTEM_PROMPT, build_user_prompt(query, chunks))
            model_used = llm.model
        t2 = time.perf_counter()
        check = self.verify(text, chunks, verifier)
        tfidf = self.factory.get("tfidf", chunking)
        retrieved = [self._chunk_row(c, s, i, rel, tfidf, query) for i, (c, (_, s)) in enumerate(zip(chunks, hits), 1)]
        cited = {cid for s in check["sentences"] for cid in s["cited_ids"]}
        for row in retrieved:
            row["cited"] = row["chunk_id"] in cited
        return {
            "query": query, "claim": self.claim_info(qid),
            "settings": {"retriever": retriever, "chunking": chunking, "k": k, "backend": backend,
                         "model": model_used, "verifier": verifier},
            "retrieved": retrieved,
            "answer": text,
            "verification": check,
            "timings": {"retrieve_ms": round((t1 - t0) * 1000, 1), "generate_ms": round((t2 - t1) * 1000, 1),
                        "verify_ms": check["ms"]},
        }

    def verify_answer(self, answer: str, chunk_ids: list[str], method: str) -> dict:
        chunks = [self.get_chunk(cid) for cid in dict.fromkeys(chunk_ids)]
        return self.verify(answer, chunks, method)

    def corrupt(self, answer: str, chunk_ids: list[str], kind: str, query: str, qid: str | None,
                method: str, seed: int | None = None) -> dict:
        from src.eval.run_verifier_eval import _on_topic_other_paper

        chunks = [self.get_chunk(cid) for cid in dict.fromkeys(chunk_ids)]
        valid = [c.chunk_id for c in chunks]
        rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        qid = self.resolve_qid(query, qid)
        exclude = (self.relevant_docs(qid) or set()) | {c.doc_id for c in chunks}
        whole = self.index("whole")
        bm25 = self.factory.get("bm25", "whole")
        corpus_chunk = {c.doc_id: c for c in whole.chunks}
        if kind == "negate":
            corr = inject.negate(answer, rng, valid)
        elif kind == "swap":
            _, other = _on_topic_other_paper(query, exclude, bm25, corpus_chunk, rng)
            corr = inject.swap_citation(answer, rng, chunks, valid, [other] if other else [])
        elif kind == "fabricate":
            fab, _ = _on_topic_other_paper(query, exclude, bm25, corpus_chunk, rng)
            corr = inject.add_fabricated(answer, rng, fab, valid, valid) if fab else None
        else:
            raise AppError(f"unknown corruption {kind!r}; use negate, swap or fabricate")
        if corr is None:
            raise AppError("Nothing to corrupt: the answer needs at least one sentence that cites a chunk.")
        pool = list(chunks)
        extra = None
        if corr.extra_chunk is not None:
            pool.append(corr.extra_chunk)
            extra = {"chunk_id": corr.extra_chunk.chunk_id, "doc_id": corr.extra_chunk.doc_id,
                     "title": corr.extra_chunk.title, "text": corr.extra_chunk.text}
        check = self.verify(corr.answer, pool, method)
        return {"kind": kind, "answer": corr.answer, "index": corr.index, "original": corr.original,
                "corrupted": corr.corrupted, "extra_chunk": extra, "verification": check}

    # ================================================================== inside the index
    def tokenize_steps(self, text: str, stem: bool = True, stopwords: bool = True) -> dict:
        folded = unicodedata.normalize("NFKC", text or "").casefold()
        rows = []
        for tok in _TOKEN_RE.findall(folded):
            is_stop = tok in STOPWORDS
            stemmed = porter(tok)
            kept = None if (stopwords and is_stop) else (stemmed if stem else tok)
            rows.append({"token": tok, "stopword": is_stop, "stem": stemmed, "term": kept})
        return {"text": text, "stem": stem, "stopwords": stopwords, "tokens": rows,
                "terms": Tokenizer(stem, stopwords).tokenize(text)}

    def term(self, word: str, chunking: str = "whole", zone: str = "all", sort: str = "tf",
             limit: int = 25, offset: int = 0) -> dict:
        if zone not in ZONES:
            raise AppError(f"zone must be one of {ZONES}")
        idx = self.index(chunking)
        toks = idx.tokenizer.tokenize(word)
        base = {"word": word, "chunking": chunking, "zone": zone, "tokenizer": idx.tokenizer.settings,
                "n_chunks": idx.n_chunks}
        if not toks:
            folded = unicodedata.normalize("NFKC", word or "").casefold().strip()
            reason = "it is a stopword" if folded in STOPWORDS else "it has no letters or digits"
            return base | {"term": None, "removed": True, "reason": reason}
        term = toks[0]
        zones = {}
        for zname in ZONES:
            z = idx.zones[zname]
            df = z.doc_freq(term)
            zones[zname] = {"df": df, "idf": z.idf(term), "bm25_idf": bm25_idf(df, z.n_docs) if df else 0.0,
                            "avg_len": z.avg_len}
        z = idx.zones[zone]
        docs, tfs = z.postings(term)
        order = (np.argsort(-tfs, kind="stable") if sort == "tf" else np.arange(len(docs)))
        page = order[offset:offset + limit]
        postings = []
        for j in page:
            c = idx.chunks[int(docs[j])]
            postings.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "tf": int(tfs[j]),
                             "length": int(z.length[docs[j]]), "title": c.title,
                             "w_lnc": (1 + math.log10(int(tfs[j]))) / float(z.lnc_norm[docs[j]])})
        return base | {"term": term, "extra_terms": toks[1:], "removed": False, "zones": zones,
                       "total": int(len(docs)), "offset": offset, "postings": postings,
                       "in_vocab": term in z.vocab, "sort": sort}

    def index_stats(self) -> dict:
        out = {}
        for ch in config.CHUNKINGS:
            idx = self.index(ch)
            out[ch] = {"chunks": idx.n_chunks, "papers": len(idx.doc_ids),
                       "zones": {z: {"terms": len(idx.zones[z].terms), "postings": int(idx.zones[z].offsets[-1]),
                                     "avg_len": idx.zones[z].avg_len} for z in ZONES},
                       "tokenizer": idx.tokenizer.settings}
        return out

    def explain(self, query: str, chunk_id: str, retriever: str = "tfidf") -> dict:
        chunk = self.get_chunk(chunk_id)
        chunking = "sent" if "_" in chunk_id else "whole"
        idx = self.index(chunking)
        c = idx.chunk_pos[chunk_id]
        base = {"query": query, "chunk": {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                                          "title": chunk.title, "text": chunk.text},
                "chunking": chunking, "retriever": retriever}
        tok = idx.tokenizer
        if retriever == "tfidf":
            r = self.factory.get("tfidf", chunking)
            q_tf = Counter(tok.tokenize(query))
            zones_out = []
            for zone, w in r.zone_weights.items():
                z = idx.zones[zone]
                qw = r.query_weights(query, zone)
                zone_text = chunk.title if zone == "title" else chunk.text
                doc_tf = Counter(tok.tokenize(zone_text))
                rows, raw_sq = [], 0.0
                for term, tf in q_tf.items():
                    df = z.doc_freq(term)
                    idf = math.log10(z.n_docs / df) if df else 0.0
                    raw = (1 + math.log10(tf)) * idf if df else 0.0
                    raw_sq += raw * raw
                    tfd = doc_tf.get(term, 0)
                    wd = (1 + math.log10(tfd)) / float(z.lnc_norm[c]) if tfd and z.lnc_norm[c] > 0 else 0.0
                    rows.append({"term": term, "tf_q": tf, "df": df, "idf": idf, "w_q_raw": raw,
                                 "w_q": qw.get(term, 0.0), "tf_d": tfd,
                                 "w_d_raw": (1 + math.log10(tfd)) if tfd else 0.0, "w_d": wd,
                                 "product": qw.get(term, 0.0) * wd})
                cos = sum(x["product"] for x in rows)
                zones_out.append({"zone": zone, "weight": w, "n_docs": z.n_docs, "query_norm": math.sqrt(raw_sq),
                                  "doc_norm": float(z.lnc_norm[c]), "doc_terms": len(doc_tf), "cosine": cos,
                                  "rows": rows})
            wsum = sum(x["weight"] for x in zones_out)
            final = sum(x["weight"] * x["cosine"] for x in zones_out) / wsum
            check = float(r.score_all(query)[c])
            return base | {"zones": zones_out, "weight_sum": wsum, "score": final, "library_score": check,
                           "matches_library": abs(final - check) < 1e-9}
        if retriever == "bm25":
            r = self.factory.get("bm25", chunking)
            z = idx.zones[r.zone]
            rows = []
            for term, qtf in Counter(tok.tokenize(query)).items():
                df = z.doc_freq(term)
                docs, tfs = z.postings(term)
                hit = np.searchsorted(docs, c) if len(docs) else 0
                tfd = int(tfs[hit]) if len(docs) and hit < len(docs) and docs[hit] == c else 0
                idf = bm25_idf(df, z.n_docs) if df else 0.0
                part = tfd * (r.k1 + 1) / (tfd + r.k1 * (1 - r.b + r.b * z.length[c] / z.avg_len)) if tfd else 0.0
                rows.append({"term": term, "qtf": qtf, "df": df, "idf": idf, "tf_d": tfd, "tf_part": part,
                             "score": qtf * idf * part})
            total = sum(x["score"] for x in rows)
            check = float(r.score_all(query)[c])
            return base | {"rows": rows, "score": total, "k1": r.k1, "b": r.b, "doc_len": int(z.length[c]),
                           "avg_len": z.avg_len, "n_docs": z.n_docs, "library_score": check,
                           "matches_library": abs(total - check) < 1e-9}
        if retriever == "dense":
            d = self.retriever("dense", chunking)
            score = float(d.score_all(query)[c])
            return base | {"score": score, "sentences": d.explain(query, chunk_id), "model": config.DENSE_MODEL}
        raise AppError("retriever must be tfidf, bm25 or dense")

    # ================================================================== evaluation results
    EVAL_FILES = {
        "main": "retrieval_main.csv", "by_claim_type": "retrieval_by_claim_type.csv",
        "significance": "significance.csv", "ablations": "retrieval_ablations.csv",
        "tuning": "tuning_train.csv", "winners": "winner_analysis.csv",
        "verifier_gold": "verifier_gold.csv", "verifier_by_kind": "verifier_gold_by_kind.csv",
        "verifier_injected": "verifier_injected.csv", "verifier_injected_by_kind": "verifier_injected_by_kind.csv",
        "verifier_windows": "verifier_window_ablation.csv",
    }

    def evaluation(self) -> dict:
        paths = [config.RESULTS_DIR / f for f in self.EVAL_FILES.values()] + [config.TUNED_PARAMS_PATH]
        stamp = max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)
        if self._eval_cache and self._eval_cache[0] == stamp:
            return self._eval_cache[1]
        data = {k: _read_csv(config.RESULTS_DIR / f) for k, f in self.EVAL_FILES.items()}
        data["tuned"] = load_tuned()
        winners = data.get("winners")
        if winners:
            from src.eval.winner_analysis import _perm_test_unpaired

            groups = {w: [r for r in winners if r["winner"] == w] for w in ("sparse", "tie", "dense")}
            data["winner_summary"] = [{"winner": w, "claims": len(g),
                                       "mean_overlap": float(np.mean([r["overlap"] for r in g])) if g else None}
                                      for w, g in groups.items()]
            a = np.array([r["overlap"] for r in groups["sparse"]], dtype=float)
            b = np.array([r["overlap"] for r in groups["dense"]], dtype=float)
            data["winner_p"] = _perm_test_unpaired(a, b) if len(a) and len(b) else None
        data["available"] = data["main"] is not None
        data["verifier_available"] = data["verifier_gold"] is not None
        data["figures"] = sorted(p.name for p in config.RESULTS_DIR.glob("*.png")) if config.RESULTS_DIR.exists() else []
        self._eval_cache = (stamp, data)
        return data
