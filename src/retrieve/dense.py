"""Dense retrieval: cosine similarity between MiniLM embeddings (dot product of unit vectors)."""
from __future__ import annotations

import numpy as np

import config
from src.index.dense_index import DenseIndex, encode
from src.index.inverted_index import InvertedIndex
from src.ingest.chunking import split_sentences
from src.retrieve.base import Retriever


class DenseRetriever(Retriever):
    name = "dense"
    positive_only = False  # every chunk gets a similarity, possibly negative

    def __init__(self, index: InvertedIndex, dense_index: DenseIndex | None = None,
                 model_name: str = config.DENSE_MODEL):
        super().__init__(index)
        self.dense = dense_index or DenseIndex(index.chunks, model_name)
        if self.dense.embeddings.shape[0] != index.n_chunks:
            raise ValueError("dense index and inverted index hold different chunks")
        self.model_name = self.dense.model_name
        self._query_cache: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        return f"DenseRetriever(model={self.model_name!r}, chunking={self.index.chunking!r})"

    def prepare(self, queries: list[str]) -> None:
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        if missing:
            for q, v in zip(missing, encode(missing, self.model_name)):
                self._query_cache[q] = v

    def query_vector(self, query: str) -> np.ndarray:
        if query not in self._query_cache:
            self.prepare([query])
        return self._query_cache[query]

    def score_all(self, query: str) -> np.ndarray:
        return (self.dense.embeddings @ self.query_vector(query)).astype(np.float64)

    def explain(self, query: str, chunk_id: str) -> list[dict]:
        """Which sentence of the chunk sits closest to the query in embedding space."""
        chunk = self.index.chunks[self.index.chunk_pos[chunk_id]]
        sents = split_sentences(chunk.text) or [chunk.text]
        sims = encode(sents, self.model_name) @ self.query_vector(query)
        order = np.argsort(-sims, kind="stable")
        return [{"sentence": sents[i], "cosine": float(sims[i])} for i in order]
