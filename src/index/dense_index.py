"""Dense chunk embeddings (sentence-transformers) with an on-disk cache.

Each chunk is embedded as "title + text", L2-normalised, so cosine similarity
is a plain dot product. The cache file name contains a hash of the model name
and every chunk's id and text, so changing the chunking or the model can never
reuse stale vectors.

Note: all-MiniLM-L6-v2 truncates input at 256 word pieces. Whole SciFact
abstracts are often longer, which is one reason sentence windows help it.
"""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

import config

_ENCODERS: dict = {}
_LOCKS: dict = {}
_LOAD_LOCK = threading.Lock()


def load_encoder(model_name: str = config.DENSE_MODEL, device: str | None = config.DEVICE):
    """A new SentenceTransformer instance (not shared)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def get_encoder(model_name: str = config.DENSE_MODEL, device: str | None = config.DEVICE):
    """Load a SentenceTransformer once per process."""
    key = (model_name, device)
    with _LOAD_LOCK:
        if key not in _ENCODERS:
            _ENCODERS[key] = load_encoder(model_name, device)
            _LOCKS[key] = threading.Lock()
    return _ENCODERS[key]


def encode(texts: Sequence[str], model_name: str = config.DENSE_MODEL,
           batch_size: int = config.DENSE_BATCH_SIZE, show_progress: bool = False,
           encoder=None, on_progress=None) -> np.ndarray:
    """Unit-length float32 embeddings, one row per text.

    The shared encoder is used under a lock: Hugging Face fast tokenizers raise
    "Already borrowed" if two threads use one instance at once (the web app
    serves requests from a thread pool). Pass `encoder` to use a private
    instance instead, as the web app's background index builder does.

    `on_progress(done, total)`, if given, is called after each block of texts.
    Texts are encoded longest first in blocks, as sentence-transformers itself
    batches them, and returned in the original order.
    """
    texts = list(texts)
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    kwargs = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=show_progress and on_progress is None)
    if encoder is not None:
        model, lock = encoder, None
    else:
        model, lock = get_encoder(model_name), _LOCKS[(model_name, config.DEVICE)]

    def run(part: list[str]) -> np.ndarray:
        if lock is None:
            return np.asarray(model.encode(part, **kwargs), dtype=np.float32)
        with lock:
            return np.asarray(model.encode(part, **kwargs), dtype=np.float32)

    if on_progress is None:
        return run(texts)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    block = max(int(batch_size), 1) * 16
    out = None
    for start in range(0, len(texts), block):
        idx = order[start:start + block]
        part = run([texts[i] for i in idx])
        if out is None:
            out = np.empty((len(texts), part.shape[1]), dtype=np.float32)
        out[idx] = part
        on_progress(min(start + block, len(texts)), len(texts))
    return out


def chunk_text(chunk) -> str:
    return f"{chunk.title} {chunk.text}".strip()


def model_key(name: str) -> str:
    """Canonical model name for cache keys: the last path component, so
    'sentence-transformers/all-MiniLM-L6-v2' and a local copy of the same model
    in '.../all-MiniLM-L6-v2' share one cache."""
    return name.replace("\\", "/").rstrip("/").split("/")[-1]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_key(name)).strip("-")


def cache_path_for(chunks: Sequence, model_name: str = config.DENSE_MODEL,
                   cache_dir: Path = config.CACHE_DIR) -> Path:
    h = hashlib.sha1(model_key(model_name).encode("utf-8"))
    for c in chunks:
        h.update(f"{c.chunk_id}\t{chunk_text(c)}\n".encode("utf-8"))
    return Path(cache_dir) / f"emb_{_slug(model_name)}_{len(chunks)}_{h.hexdigest()[:12]}.npy"


class DenseIndex:
    def __init__(self, chunks: Sequence, model_name: str = config.DENSE_MODEL,
                 cache_dir: Path = config.CACHE_DIR, batch_size: int = config.DENSE_BATCH_SIZE,
                 show_progress: bool = True, encoder=None, on_progress=None):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.embeddings = self._load_or_build(show_progress, encoder, on_progress)

    def cache_path(self) -> Path:
        return cache_path_for(self.chunks, self.model_name, self.cache_dir)

    def _load_or_build(self, show_progress: bool, encoder=None, on_progress=None) -> np.ndarray:
        path = self.cache_path()
        if path.exists():
            emb = np.load(path)
            if emb.shape[0] == len(self.chunks):
                return emb
        print(f"Encoding {len(self.chunks)} chunks with {self.model_name} (cached afterwards)")
        emb = encode([chunk_text(c) for c in self.chunks], self.model_name, self.batch_size, show_progress,
                     encoder=encoder, on_progress=on_progress)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")      # write then rename: a crash never leaves half a cache
        np.save(tmp, emb)
        tmp.replace(path)
        return emb
