"""Dense encoding helpers, with a fake encoder (no model download)."""
import zlib
from types import SimpleNamespace

import numpy as np

from src.index.dense_index import DenseIndex, encode, model_key


class FakeEncoder:
    """Deterministic 'embedding' of each text; records the batches it was given."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, zlib.crc32(t.encode()) % 8] = 1.0
            out[i, len(t) % 8] += 0.5
        return out / np.linalg.norm(out, axis=1, keepdims=True)


TEXTS = [("x" * (i % 13 + 1)) + f" text {i}" for i in range(70)]


def test_progress_blocks_keep_the_original_order():
    plain = encode(TEXTS, encoder=FakeEncoder())
    seen = []
    enc = FakeEncoder()
    blocked = encode(TEXTS, encoder=enc, batch_size=2, on_progress=lambda d, t: seen.append((d, t)))
    assert np.array_equal(plain, blocked)
    assert seen == [(32, 70), (64, 70), (70, 70)]                    # blocks of 16 batches
    assert len(enc.calls[0][0]) >= len(enc.calls[-1][-1])            # longest texts first


def test_dense_index_builds_once_then_reads_the_cache(tmp_path):
    chunks = [SimpleNamespace(chunk_id=f"c{i}", title="T", text=t) for i, t in enumerate(TEXTS[:10])]
    enc = FakeEncoder()
    ticks = []
    first = DenseIndex(chunks, "some/all-MiniLM-L6-v2", cache_dir=tmp_path, show_progress=False, encoder=enc,
                       on_progress=lambda d, t: ticks.append(d))
    assert ticks[-1] == 10 and first.cache_path().exists()
    unused = FakeEncoder()
    again = DenseIndex(chunks, "elsewhere/all-MiniLM-L6-v2", cache_dir=tmp_path, encoder=unused)
    assert np.array_equal(first.embeddings, again.embeddings) and unused.calls == []   # read from the cache
    assert model_key("C:\\models\\all-MiniLM-L6-v2\\") == "all-MiniLM-L6-v2"
