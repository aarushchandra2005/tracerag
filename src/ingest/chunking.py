"""Split abstracts into retrieval units ("chunks").

Two strategies, compared in the ablation table:
  * whole: one chunk per abstract (chunk_id == doc_id)
  * sent:  overlapping windows of `window` sentences moved by `stride`
           (chunk_id == f"{doc_id}_{first_sentence_index}")

Every chunk keeps its paper title so the title zone can be scored separately.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import config
from src.utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int = 0      # index of the first sentence of the abstract in this chunk
    end: int = 0        # one past the last sentence

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- sentence splitting
# Abbreviations whose final period must not end a sentence (case-sensitive).
_ABBREVIATIONS = (
    "e.g.", "i.e.", "et al.", "vs.", "cf.", "viz.", "approx.", "ca.", "Fig.", "Figs.",
    "Eq.", "Eqs.", "Ref.", "Refs.", "No.", "Nos.", "Vol.", "vol.", "pp.", "Dr.", "Drs.",
    "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr.", "Sr.", "Inc.", "Ltd.", "Co.", "Corp.",
    "U.S.", "U.K.", "E.U.", "spp.", "sp.", "subsp.", "var.", "resp.", "Suppl.",
)
_ABBR_RE = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")"
)
_PLACEHOLDER = "․"  # "one dot leader", restored to "." after splitting

# A sentence ends at . ! or ? (plus closing quotes/brackets) when the next word
# starts like a sentence: a capital, a digit, a Greek letter, an opening
# bracket/quote, or a biomedical token such as mRNA, p53 or siRNA.
_END_RE = re.compile(
    r"[.!?]+[\"”’)\]]*"
    r"(?=\s+(?:[A-Z0-9Ͱ-Ͽ(\[“\"']|[a-z]+[A-Z0-9]))"
)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tuned for scientific abstracts."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    protected = _ABBR_RE.sub(lambda m: m.group(1).replace(".", _PLACEHOLDER), text)
    sentences, start = [], 0
    for m in _END_RE.finditer(protected):
        piece = protected[start:m.end()].strip()
        if piece:
            sentences.append(piece)
        start = m.end()
    tail = protected[start:].strip()
    if tail:
        sentences.append(tail)
    return [s.replace(_PLACEHOLDER, ".") for s in sentences]


# ---------------------------------------------------------------- chunkers
def chunk_whole(doc: dict) -> list[Chunk]:
    """doc = {"doc_id", "title", "text"} -> a single chunk holding the full abstract."""
    n = len(split_sentences(doc["text"]))
    return [Chunk(doc["doc_id"], doc["doc_id"], doc["title"], doc["text"], 0, n)]


def chunk_sentences(doc: dict, window: int = config.SENT_WINDOW, stride: int = config.SENT_STRIDE) -> list[Chunk]:
    """Overlapping sentence windows; every sentence lands in at least one window."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    sents = split_sentences(doc["text"])
    n = len(sents)
    if n == 0:
        return [Chunk(f"{doc['doc_id']}_0", doc["doc_id"], doc["title"], "", 0, 0)]
    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:          # stride skipped the tail: add a final window
        starts.append(n - window)
    return [
        Chunk(f"{doc['doc_id']}_{s}", doc["doc_id"], doc["title"],
              " ".join(sents[s:s + window]), s, min(s + window, n))
        for s in starts
    ]


def build_chunks(corpus: dict[str, dict], mode: str) -> list[Chunk]:
    """corpus: doc_id -> {"title", "text"}; mode: "whole" or "sent"."""
    if mode not in config.CHUNKINGS:
        raise ValueError(f"unknown chunking {mode!r}; use one of {config.CHUNKINGS}")
    fn = chunk_whole if mode == "whole" else chunk_sentences
    chunks: list[Chunk] = []
    for doc_id, d in corpus.items():
        chunks.extend(fn({"doc_id": doc_id, "title": d["title"], "text": d["text"]}))
    return chunks


def chunks_path(mode: str) -> Path:
    return config.PROCESSED_DIR / f"chunks_{mode}.jsonl"


def save_chunks(chunks: Iterable[Chunk], mode: str) -> Path:
    path = chunks_path(mode)
    write_jsonl(path, (c.to_dict() for c in chunks))
    return path


def load_chunks(mode: str) -> list[Chunk]:
    """Load chunks written by scripts/build_index.py (builds them on first use)."""
    path = chunks_path(mode)
    if not path.exists():
        from src.ingest.load_scifact import load_corpus

        save_chunks(build_chunks(load_corpus(), mode), mode)
    return [Chunk(**r) for r in read_jsonl(path)]
