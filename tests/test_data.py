import io
import json
import zipfile

import pytest

from src.ingest import load_scifact
from tests.conftest import needs_scifact

CORPUS = [{"_id": "10", "title": "T", "text": "A. B.", "metadata": {}}]
QUERIES = [{"_id": "1", "text": "claim one", "metadata": {"10": [{"sentences": [0], "label": "SUPPORT"}]}},
           {"_id": "2", "text": "claim two", "metadata": {}}]
QRELS = "query-id\tcorpus-id\tscore\n1\t10\t1\n"


def _beir_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("scifact/corpus.jsonl", "\n".join(json.dumps(r) for r in CORPUS) + "\n")
        zf.writestr("scifact/queries.jsonl", "\n".join(json.dumps(r) for r in QUERIES) + "\n")
        zf.writestr("scifact/qrels/test.tsv", QRELS)
        zf.writestr("scifact/qrels/train.tsv", "query-id\tcorpus-id\tscore\n2\t10\t1\n")
    return buf.getvalue()


def test_extract_and_load_beir_zip(tmp_path):
    load_scifact._extract_beir_zip(_beir_zip(), tmp_path)
    assert load_scifact.load_corpus(tmp_path) == {"10": {"title": "T", "text": "A. B."}}
    assert load_scifact.load_qrels("test", tmp_path) == {"1": {"10": 1}}
    q = load_scifact.load_queries(tmp_path)
    assert q["1"]["label"] == "SUPPORT" and q["1"]["split"] == "test"
    assert q["2"]["label"] == "NEI" and q["2"]["split"] == "train"


def test_manual_zip_is_used_without_network(tmp_path):
    (tmp_path / "scifact.zip").write_bytes(_beir_zip())
    raw = tmp_path / "scifact"
    load_scifact.download(raw)
    assert (raw / "qrels" / "train.tsv").exists()


def test_parquet_fallback_parser(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    buf = io.BytesIO()
    pd.DataFrame([{"_id": "10", "title": "T", "text": "A."}]).to_parquet(buf)
    load_scifact._parquet_to_jsonl(buf.getvalue(), tmp_path / "corpus.jsonl")
    rows = [json.loads(line) for line in open(tmp_path / "corpus.jsonl", encoding="utf-8")]
    assert rows == [{"_id": "10", "title": "T", "text": "A.", "metadata": {}}]


@needs_scifact
def test_real_scifact_matches_published_counts():
    stats = load_scifact.verify(verbose=False)
    assert stats["test_labels"] == {"SUPPORT": 124, "CONTRADICT": 64, "NEI": 112}
    assert stats["corpus"] == 5183 and stats["test_pairs"] == 339
