"""Hand-built inverted index with zones.

Layout (per zone, the classic "dictionary + postings file" design):

    dictionary   terms[i]            sorted vocabulary
                 df[i]               document frequency of terms[i]
                 offsets[i]          where terms[i]'s postings start
    postings     post_doc[o:o+df]    chunk indices, ascending
                 post_tf[o:o+df]     term frequency in that chunk
    per chunk    length[c]           number of tokens in the zone
                 lnc_norm[c]         sqrt(sum_t (1 + log10 tf)^2), the lnc cosine norm

Zones: "title", "body" and "all" (title + body as one field, used by BM25 and
by the single-field ablation). Postings are built with plain Python lists and
frozen into numpy arrays so that scoring stays fast on 40k sentence windows.

Indexes are saved with pickle; only load files you built yourself.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import config
from src.index.tokenizer import Tokenizer

ZONES = ("title", "body", "all")


class ZoneIndex:
    """Dictionary and postings for one zone."""

    def __init__(self, terms: list[str], df: np.ndarray, offsets: np.ndarray,
                 post_doc: np.ndarray, post_tf: np.ndarray, length: np.ndarray):
        self.terms = terms
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.df = df
        self.offsets = offsets
        self.post_doc = post_doc
        self.post_tf = post_tf
        self.length = length
        self.n_docs = int(len(length))
        self.avg_len = float(length.mean()) if len(length) else 0.0
        # lnc document norms: one pass over all postings
        log_tf = 1.0 + np.log10(post_tf.astype(np.float64))
        self.lnc_norm = np.sqrt(np.bincount(post_doc, weights=log_tf ** 2, minlength=self.n_docs))

    @classmethod
    def from_term_counts(cls, counts: Sequence[Counter]) -> "ZoneIndex":
        """counts[c] = Counter(term -> tf) for chunk c."""
        raw: dict[str, tuple[list[int], list[int]]] = {}
        length = np.zeros(len(counts), dtype=np.int32)
        for c, counter in enumerate(counts):
            length[c] = sum(counter.values())
            for term, tf in counter.items():
                docs, tfs = raw.setdefault(term, ([], []))
                docs.append(c)          # chunks are visited in order -> postings stay sorted
                tfs.append(tf)
        terms = sorted(raw)
        df = np.fromiter((len(raw[t][0]) for t in terms), dtype=np.int32, count=len(terms))
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(df, out=offsets[1:])
        post_doc = np.empty(int(offsets[-1]), dtype=np.int32)
        post_tf = np.empty(int(offsets[-1]), dtype=np.int32)
        for i, t in enumerate(terms):
            o, e = offsets[i], offsets[i + 1]
            post_doc[o:e] = raw[t][0]
            post_tf[o:e] = raw[t][1]
        return cls(terms, df, offsets, post_doc, post_tf, length)

    # ------------------------------------------------------------- lookups
    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """(chunk indices, term frequencies) for term; empty arrays if unseen."""
        i = self.vocab.get(term)
        if i is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        o, e = self.offsets[i], self.offsets[i + 1]
        return self.post_doc[o:e], self.post_tf[o:e]

    def doc_freq(self, term: str) -> int:
        i = self.vocab.get(term)
        return 0 if i is None else int(self.df[i])

    def idf(self, term: str) -> float:
        """log10(N / df), the "t" in ltc. Zero for unseen terms."""
        df = self.doc_freq(term)
        return math.log10(self.n_docs / df) if df else 0.0

    def to_state(self) -> dict:
        return {"terms": self.terms, "df": self.df, "offsets": self.offsets,
                "post_doc": self.post_doc, "post_tf": self.post_tf, "length": self.length}

    @classmethod
    def from_state(cls, s: dict) -> "ZoneIndex":
        return cls(s["terms"], s["df"], s["offsets"], s["post_doc"], s["post_tf"], s["length"])


class InvertedIndex:
    VERSION = 2

    def __init__(self, chunks: Sequence, zones: dict[str, ZoneIndex], tokenizer: Tokenizer, chunking: str = ""):
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        self.chunk_pos = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.zones = zones
        self.tokenizer = tokenizer
        self.chunking = chunking
        # chunk -> paper mapping, used to turn chunk rankings into paper rankings
        self.doc_ids: list[str] = []
        doc_pos: dict[str, int] = {}
        chunk_doc = np.empty(len(self.chunks), dtype=np.int32)
        for i, c in enumerate(self.chunks):
            if c.doc_id not in doc_pos:
                doc_pos[c.doc_id] = len(self.doc_ids)
                self.doc_ids.append(c.doc_id)
            chunk_doc[i] = doc_pos[c.doc_id]
        self.doc_pos = doc_pos
        self.chunk_doc = chunk_doc

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, chunks: Iterable, tokenizer: Tokenizer | None = None, chunking: str = "") -> "InvertedIndex":
        chunks = list(chunks)
        tokenizer = tokenizer or Tokenizer(config.USE_STEMMING, config.REMOVE_STOPWORDS)
        title_cache: dict[str, Counter] = {}
        per_zone: dict[str, list[Counter]] = {z: [] for z in ZONES}
        for c in chunks:
            if c.doc_id not in title_cache:  # windows of one paper share its title
                title_cache[c.doc_id] = Counter(tokenizer.tokenize(c.title))
            title = title_cache[c.doc_id]
            body = Counter(tokenizer.tokenize(c.text))
            per_zone["title"].append(title)
            per_zone["body"].append(body)
            per_zone["all"].append(title + body)
        zones = {z: ZoneIndex.from_term_counts(per_zone[z]) for z in ZONES}
        return cls(chunks, zones, tokenizer, chunking)

    # ------------------------------------------------------------- views used in the roadmap
    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def dictionary(self, zone: str = "all") -> dict[str, int]:
        """{term: df} for one zone."""
        z = self.zones[zone]
        return dict(zip(z.terms, z.df.tolist()))

    def get_postings(self, term: str, zone: str = "all", analyze: bool = False) -> list[tuple[str, int]]:
        """[(chunk_id, tf), ...]; analyze=True runs the raw word through the tokenizer first."""
        if analyze:
            toks = self.tokenizer.tokenize(term)
            term = toks[0] if toks else ""
        docs, tfs = self.zones[zone].postings(term)
        return [(self.chunk_ids[d], int(t)) for d, t in zip(docs, tfs)]

    def doc_len(self, zone: str = "all") -> np.ndarray:
        return self.zones[zone].length

    def print_postings(self, word: str, zone: str = "all", limit: int = 10) -> str:
        """Pretty-print a word's dictionary entry and postings (used in the demo video)."""
        toks = self.tokenizer.tokenize(word)
        lines = []
        if not toks:
            lines.append(f"{word!r} is removed by the tokenizer ({self.tokenizer}).")
        else:
            term = toks[0]
            z = self.zones[zone]
            df = z.doc_freq(term)
            lines.append(f"word {word!r} -> index term {term!r}   zone={zone}   tokenizer={self.tokenizer}")
            if df == 0:
                lines.append("  not in the dictionary")
            else:
                lines.append(f"  df = {df} of N = {z.n_docs} chunks   idf = log10(N/df) = {z.idf(term):.4f}")
                docs, tfs = z.postings(term)
                order = np.argsort(-tfs, kind="stable")[:limit]
                lines.append(f"  postings (top {min(limit, df)} by tf; stored sorted by chunk):")
                for j in order:
                    c = self.chunks[int(docs[j])]
                    lines.append(f"    {c.chunk_id:>14}  tf={int(tfs[j]):<3} len={int(z.length[docs[j]]):<4} {c.title[:60]}")
        out = "\n".join(lines)
        print(out)
        return out

    def stats(self) -> dict:
        return {z: {"terms": len(self.zones[z].terms), "postings": int(self.zones[z].offsets[-1]),
                    "avg_len": round(self.zones[z].avg_len, 2)} for z in ZONES} | {
            "chunks": self.n_chunks, "papers": len(self.doc_ids), "tokenizer": self.tokenizer.settings}

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.VERSION,
            "chunking": self.chunking,
            "tokenizer": self.tokenizer.settings,
            "chunks": [c.to_dict() for c in self.chunks],
            "zones": {z: zi.to_state() for z, zi in self.zones.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        from src.ingest.chunking import Chunk

        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("version") != cls.VERSION:
            raise ValueError(f"{path} was built by an older version; rebuild it with scripts/build_index.py")
        chunks = [Chunk(**c) for c in state["chunks"]]
        zones = {z: ZoneIndex.from_state(s) for z, s in state["zones"].items()}
        return cls(chunks, zones, Tokenizer(**state["tokenizer"]), state["chunking"])


# ----------------------------------------------------------------- cached access
def index_path(chunking: str, tokenizer: Tokenizer) -> Path:
    return config.INDEX_DIR / f"index_{chunking}_{tokenizer.tag}.pkl"


def get_index(chunking: str = "whole", stem: bool = config.USE_STEMMING,
              stopwords: bool = config.REMOVE_STOPWORDS, rebuild: bool = False) -> InvertedIndex:
    """Load the index for these settings, building and saving it on first use."""
    from src.ingest.chunking import load_chunks

    tokenizer = Tokenizer(stem, stopwords)
    path = index_path(chunking, tokenizer)
    if path.exists() and not rebuild:
        return InvertedIndex.load(path)
    index = InvertedIndex.build(load_chunks(chunking), tokenizer, chunking)
    index.save(path)
    return index
