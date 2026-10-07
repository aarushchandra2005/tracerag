"""TraceRAG live demo: retrieve -> answer with citations -> verify every sentence.

    python scripts/demo.py --qid 2                           # a SciFact test claim by id
    python scripts/demo.py "Vitamin D reduces fracture risk in older adults."
    python scripts/demo.py --qid 2 --inject negate           # corrupt the answer, watch the verifier
    python scripts/demo.py --postings tumor                  # dictionary entry + postings list
    python scripts/demo.py --interactive

Defaults: hybrid RRF retrieval over sentence windows, k = 5, extractive answers
(no API key needed; pass --llm anthropic/groq/... for a real LLM), cascade
verifier (falls back to cosine if the NLI model cannot be loaded).
"""
from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.explain.contrib import format_terms, top_terms  # noqa: E402
from src.generate.llm import API_BACKENDS, make_llm  # noqa: E402
from src.generate.rag import answer  # noqa: E402
from src.ingest.chunking import Chunk  # noqa: E402
from src.ingest.load_scifact import load_queries, load_qrels  # noqa: E402
from src.retrieve.factory import NAMES, RetrieverFactory  # noqa: E402
from src.utils import setup_console  # noqa: E402
from src.verify import inject  # noqa: E402
from src.verify.checker import CitationChecker  # noqa: E402

W = 100


def header(title: str) -> None:
    print("\n" + f" {title} ".center(W, "="))


def wrap(text: str, indent: int = 4) -> str:
    return textwrap.fill(text, W, initial_indent=" " * indent, subsequent_indent=" " * indent)


def make_checker(method: str) -> CitationChecker:
    checker = CitationChecker(method)
    if method != "cosine":
        try:
            _ = checker.nli
        except Exception as e:  # noqa: BLE001 - offline or model missing: degrade gracefully
            print(f"(NLI model unavailable: {type(e).__name__}; using the cosine checker)")
            checker.method = "cosine"
    return checker


def show_verdicts(rows: list[dict], marked: int | None = None) -> None:
    for i, r in enumerate(rows):
        v = r["verdict"]
        tag = {"supported": "SUPPORTED  ", "unsupported": "UNSUPPORTED", "abstain": "ABSTAIN    "}[v.label]
        star = "  <- injected" if i == marked else ""
        print(f" [{tag}] {r['sentence']}{star}")
        extra = ""
        if v.entail is not None:   # NLI ran: show both signals
            extra = f"   [cosine {v.cosine:.2f}; P(entail) {v.entail:.2f}; P(contradict) {v.contradict:.2f}]"
        print(wrap(v.reason + extra, 15))
        if v.evidence:
            print(wrap(f"evidence from {v.best_chunk}: \"{v.evidence}\"", 15))


def run_claim(claim: str, args, factory: RetrieverFactory, checker: CitationChecker, llm, qid: str | None) -> None:
    rel = set(load_qrels("test").get(qid, {})) if qid else set()
    if qid and not rel:
        rel = set(load_qrels("train").get(qid, {}))
    header("CLAIM")
    print(wrap(claim, 1))
    if qid:
        meta = load_queries()[qid]
        print(f" SciFact claim {qid}: label {meta['label']}, relevant papers {sorted(rel)}")

    retriever = factory.get(args.retriever, args.chunking)
    tfidf = factory.get("tfidf", args.chunking)
    result = answer(claim, retriever, args.k, llm)
    header(f"RETRIEVAL  {args.retriever}, {args.chunking} chunks, top {args.k}")
    for rank, r in enumerate(result["retrieved"], 1):
        mark = "  (relevant paper)" if r["doc_id"] in rel else ""
        print(f" {rank}. [{r['chunk_id']}] score {r['score']:.4f}{mark}")
        print(wrap(r["title"]))
        print(wrap(r["text"][:300] + ("..." if len(r["text"]) > 300 else "")))
        print(wrap("tf-idf term shares: " + format_terms(top_terms(claim, r["chunk_id"], tfidf, 4))))

    header(f"ANSWER  ({result['backend']}{' / ' + result['model'] if llm else ''})")
    print(wrap(result["answer"], 1))

    retrieved = [Chunk(r["chunk_id"], r["doc_id"], r["title"], r["text"]) for r in result["retrieved"]]
    header(f"VERIFIER  ({checker.method})")
    show_verdicts(checker.check_answer(result["answer"], retrieved))

    if args.inject:
        rng = random.Random(config.SEED)
        valid = [c.chunk_id for c in retrieved]
        if args.inject == "negate":
            corr = inject.negate(result["answer"], rng, valid)
        elif args.inject == "swap":
            corr = inject.swap_citation(result["answer"], rng, retrieved, valid)
        else:
            corr = inject.add_fabricated(
                result["answer"], rng, "This treatment doubled overall survival in a large randomized trial.",
                valid, valid)
        header(f"VERIFIER ON A CORRUPTED ANSWER  ({args.inject})")
        if corr is None:
            print(" nothing to corrupt (the answer has no cited sentence)")
        else:
            show_verdicts(checker.check_answer(corr.answer, retrieved), marked=corr.index)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("claim", nargs="?", help="a claim or question")
    ap.add_argument("--qid", help="use a SciFact claim by id (shows its relevant papers)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--postings", metavar="WORD", help="print a word's dictionary entry and postings")
    ap.add_argument("--retriever", default=config.GEN_RETRIEVER, choices=NAMES)
    ap.add_argument("--chunking", default=config.GEN_CHUNKING, choices=config.CHUNKINGS)
    ap.add_argument("--k", type=int, default=config.TOP_K_GEN)
    ap.add_argument("--llm", default=config.LLM_BACKEND, choices=("extractive",) + API_BACKENDS)
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--verifier", default="cascade", choices=("cosine", "nli", "cascade"))
    ap.add_argument("--inject", choices=("negate", "swap", "fabricate"), help="also verify a corrupted answer")
    args = ap.parse_args(argv)
    setup_console()

    factory = RetrieverFactory()
    if args.postings:
        factory.index(args.chunking).print_postings(args.postings)
        if not (args.claim or args.qid or args.interactive):
            return
    llm = make_llm(args.llm, args.model)
    checker = make_checker(args.verifier)
    if args.interactive:
        while True:
            try:
                claim = input("\nclaim (empty line to quit)> ").strip()
            except EOFError:
                break
            if not claim:
                break
            run_claim(claim, args, factory, checker, llm, None)
        return
    if args.qid:
        meta = load_queries()
        if args.qid not in meta:
            sys.exit(f"unknown claim id {args.qid}")
        run_claim(meta[args.qid]["text"], args, factory, checker, llm, args.qid)
    elif args.claim:
        run_claim(args.claim, args, factory, checker, llm, None)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
