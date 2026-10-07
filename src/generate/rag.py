"""Retrieval-augmented answering with forced chunk citations.

answer(claim, retriever, k=5, llm=...) retrieves k chunks, asks the LLM to
judge the claim using only those chunks and to cite a chunk id after every
sentence, then parses the citations.

With llm=None the ExtractiveGenerator is used instead: no API, deterministic.
It copies the chunk sentences closest to the claim (MiniLM cosine) and cites
their chunk. Every sentence it writes is supported by construction, which makes
it a clean source of "original" sentences for the corruption tests.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import config
from src.index.dense_index import encode
from src.ingest.chunking import split_sentences
from src.verify import splitter

SYSTEM_PROMPT = """You check scientific claims against evidence chunks taken from paper abstracts.
Rules:
1. Use only the evidence chunks below. Do not add outside knowledge.
2. End every sentence with the id of the chunk that supports it, in square brackets, exactly as shown (for example [4983_2]). If two chunks support a sentence, write [4983_2][1234_0].
3. First say in one sentence whether the evidence supports the claim, contradicts it, or does not settle it. Then give at most three sentences with the key findings.
4. If no chunk is relevant to the claim, reply with exactly: Not enough evidence."""

NOT_ENOUGH = "Not enough evidence."


def build_user_prompt(claim: str, chunks: Sequence) -> str:
    lines = [f"Claim: {claim}", "", "Evidence chunks:"]
    for c in chunks:
        lines += [f"[{c.chunk_id}] {c.title}", c.text, ""]
    lines.append("Answer (cite chunk ids):")
    return "\n".join(lines)


class ExtractiveGenerator:
    """Offline stand-in for an LLM: the top sentences by cosine to the claim, each cited."""

    name = "extractive"

    def __init__(self, model_name: str = config.DENSE_MODEL, max_sentences: int = 3, min_similarity: float = 0.35):
        self.model = model_name
        self.max_sentences, self.min_similarity = max_sentences, min_similarity

    def generate_from_chunks(self, claim: str, chunks: Sequence) -> str:
        cands = [(c.chunk_id, s) for c in chunks for s in split_sentences(c.text)]
        if not cands:
            return NOT_ENOUGH
        sims = encode([s for _, s in cands], self.model) @ encode([claim], self.model)[0]
        order = np.argsort(-sims, kind="stable")
        if sims[order[0]] < self.min_similarity:
            return NOT_ENOUGH
        picked, seen = [], set()
        for i in order:
            if sims[i] < self.min_similarity or len(picked) == self.max_sentences:
                break
            cid, sent = cands[i]
            if sent in seen:            # overlapping windows repeat sentences
                continue
            seen.add(sent)
            picked.append(splitter.with_citations(sent, [cid]))
        return " ".join(picked)


def answer(claim: str, retriever, k: int = config.TOP_K_GEN, llm=None) -> dict:
    """Retrieve k chunks, generate a cited answer, and parse its citations."""
    index = retriever.index
    hits = retriever.search(claim, k)
    chunks = [index.chunks[index.chunk_pos[cid]] for cid, _ in hits]
    if llm is None:
        gen = ExtractiveGenerator()
        text = gen.generate_from_chunks(claim, chunks)
        backend, model = gen.name, gen.model
    else:
        text = llm.generate(SYSTEM_PROMPT, build_user_prompt(claim, chunks))
        backend, model = llm.name, llm.model
    parsed = splitter.split(text, [c.chunk_id for c in chunks])
    cited = list(dict.fromkeys(i for p in parsed for i in p["cited_ids"]))
    return {
        "claim": claim,
        "answer": text,
        "backend": backend,
        "model": model,
        "retrieved": [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": s, "title": c.title, "text": c.text}
                      for c, (_, s) in zip(chunks, hits)],
        "cited_chunks": cited,
        "sentences": parsed,
    }
