"""Generate cited answers for a fixed, seeded sample of test claims.

    python -m src.generate.run_generation                                   # offline (extractive)
    python -m src.generate.run_generation --llm anthropic                   # Claude, default model
    python -m src.generate.run_generation --llm groq --model <model-name>   # any OpenAI-compatible API
    python -m src.generate.run_generation --llm ollama --model <local-model>

Writes results/generation/answers_<tag>.jsonl (+ a readable .md next to it).
"""
from __future__ import annotations

import argparse
import random
import re
import time

import config
from src.generate.llm import API_BACKENDS, make_llm
from src.generate.rag import answer
from src.ingest.load_scifact import load_queries, load_split
from src.retrieve.factory import get_retriever
from src.utils import setup_console, write_jsonl

OUT = config.RESULTS_DIR / "generation"


def pick_queries(n: int, seed: int = config.SEED) -> list[str]:
    queries, _ = load_split("test")
    qids = list(queries)
    if n >= len(qids):
        return qids
    picked = random.Random(seed).sample(qids, n)
    return sorted(picked, key=lambda q: int(q) if q.isdigit() else q)


def tag_for(backend: str, model: str) -> str:
    if backend == "extractive" or not model:
        return backend
    return f"{backend}_{re.sub(r'[^A-Za-z0-9.]+', '-', model).strip('-')}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default=config.LLM_BACKEND, choices=("extractive",) + API_BACKENDS)
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--n", type=int, default=config.N_GEN_QUERIES, help="number of test claims (300 = all)")
    ap.add_argument("--retriever", default=config.GEN_RETRIEVER)
    ap.add_argument("--chunking", default=config.GEN_CHUNKING, choices=config.CHUNKINGS)
    ap.add_argument("--k", type=int, default=config.TOP_K_GEN)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between API calls (free-tier rate limits)")
    args = ap.parse_args(argv)
    setup_console()

    llm = make_llm(args.llm, args.model)
    retriever = get_retriever(args.retriever, args.chunking)
    meta = load_queries()
    _, qrels = load_split("test")
    rows = []
    qids = pick_queries(args.n)
    for i, qid in enumerate(qids, 1):
        out = answer(meta[qid]["text"], retriever, args.k, llm)
        out.update(qid=qid, claim_label=meta[qid]["label"], relevant_docs=sorted(qrels[qid]),
                   retriever=args.retriever, chunking=args.chunking, k=args.k)
        rows.append(out)
        print(f"[{i}/{len(qids)}] claim {qid}: {len(out['cited_chunks'])} chunks cited")
        if llm is not None and args.sleep:
            time.sleep(args.sleep)

    tag = tag_for(args.llm, llm.model if llm else "")
    path = OUT / f"answers_{tag}.jsonl"
    write_jsonl(path, rows)
    md = [f"# Generated answers ({tag})", "",
          f"{len(rows)} test claims, retriever `{args.retriever}`, {args.chunking} chunks, k = {args.k}.",
          "Relevant (judged) papers are marked with *.", ""]
    for r in rows:
        rel = set(r["relevant_docs"])
        md += [f"## Claim {r['qid']} ({r['claim_label']})", "", f"> {r['claim']}", "",
               "Retrieved: " + ", ".join(f"{x['chunk_id']}{'*' if x['doc_id'] in rel else ''}"
                                          for x in r["retrieved"]), "", r["answer"], ""]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
