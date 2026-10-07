"""Download, verify and load BEIR SciFact.

BEIR SciFact = 5,183 paper abstracts (corpus), 1,109 expert-written scientific
claims (queries) and doc-level relevance judgments (qrels): 300 test claims and
809 train claims. The BEIR copy of queries.jsonl also carries SciFact's
sentence-level evidence annotations in a "metadata" field, e.g.
    {"_id": "2", "text": "1 in 5 million in UK have abnormal PrP positivity.",
     "metadata": {"13734012": [{"sentences": [4], "label": "CONTRADICT"}]}}
We keep that field because the citation verifier is evaluated against it.

Sources, tried in order:
  1. data/raw/scifact.zip if you downloaded it by hand
  2. the official BEIR zip (config.BEIR_SCIFACT_URL)
  3. the Hugging Face copy (BeIR/scifact parquet + BeIR/scifact-qrels); this
     copy drops the evidence field, so evidence is then taken from the original
     SciFact release (claims_*.jsonl).
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import config
from src.utils import read_jsonl, write_json, write_jsonl

FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "qrels/train.tsv")

EXPECTED = {
    "corpus": 5183,
    "queries": 1109,
    "test_queries": 300,
    "train_queries": 809,
    "test_pairs": 339,
    "train_pairs": 919,
    # SciFact claim labels (Wadden et al., 2020): the BEIR test split is SciFact dev
    "test_labels": {"SUPPORT": 124, "CONTRADICT": 64, "NEI": 112},
    "train_labels": {"SUPPORT": 332, "CONTRADICT": 173, "NEI": 304},
}

HF_CORPUS = "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet"
HF_QUERIES = "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet"
HF_QRELS = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/{split}.tsv"
SCIFACT_ORIGINAL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


# ====================================================================== download
def _fetch(url: str, timeout: int = 120) -> bytes:
    import requests  # local import: only needed when downloading

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _extract_beir_zip(data: bytes, raw_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        for rel in FILES:
            member = next((n for n in names if n.replace("\\", "/").endswith("scifact/" + rel)), None)
            if member is None:
                raise FileNotFoundError(f"{rel} not found inside the SciFact zip")
            target = raw_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _parquet_to_jsonl(data: bytes, target: Path) -> None:
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pandas + pyarrow are needed for the Hugging Face fallback") from e
    df = pd.read_parquet(io.BytesIO(data))
    rows = []
    for rec in df.to_dict(orient="records"):
        rows.append({"_id": str(rec["_id"]), "title": rec.get("title", "") or "",
                     "text": rec.get("text", "") or "", "metadata": {}})
    write_jsonl(target, rows)


def _merge_original_evidence(raw_dir: Path) -> None:
    """Fill the evidence field from the original SciFact release (claims_*.jsonl)."""
    data = _fetch(SCIFACT_ORIGINAL, timeout=300)
    evidence: dict[str, dict] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            base = member.name.rsplit("/", 1)[-1]
            if base in ("claims_train.jsonl", "claims_dev.jsonl"):
                fh = tf.extractfile(member)
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if line.strip():
                        c = json.loads(line)
                        evidence[str(c["id"])] = c.get("evidence", {}) or {}
    path = raw_dir / "queries.jsonl"
    rows = read_jsonl(path)
    for r in rows:
        r["metadata"] = evidence.get(r["_id"], {})
    write_jsonl(path, rows)


def download(raw_dir: Path = config.RAW_DIR, force: bool = False) -> Path:
    """Make sure the four BEIR files exist in raw_dir, downloading if needed."""
    raw_dir = Path(raw_dir)
    if not force and all((raw_dir / f).exists() for f in FILES):
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    manual_zip = raw_dir.parent / "scifact.zip"
    errors = []
    if manual_zip.exists():
        print(f"Extracting {manual_zip}")
        _extract_beir_zip(manual_zip.read_bytes(), raw_dir)
        return raw_dir
    try:
        print(f"Downloading {config.BEIR_SCIFACT_URL}")
        blob = _fetch(config.BEIR_SCIFACT_URL, timeout=300)
        manual_zip.write_bytes(blob)
        _extract_beir_zip(blob, raw_dir)
        return raw_dir
    except Exception as e:  # noqa: BLE001 - try the next source
        errors.append(f"BEIR: {e}")
    try:
        print("BEIR server failed, trying the Hugging Face copy")
        _parquet_to_jsonl(_fetch(HF_CORPUS), raw_dir / "corpus.jsonl")
        _parquet_to_jsonl(_fetch(HF_QUERIES), raw_dir / "queries.jsonl")
        for split in ("test", "train"):
            target = raw_dir / "qrels" / f"{split}.tsv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_fetch(HF_QRELS.format(split=split)))
        print("Adding SciFact evidence labels from the original release")
        _merge_original_evidence(raw_dir)
        return raw_dir
    except Exception as e:  # noqa: BLE001
        errors.append(f"Hugging Face: {e}")
    raise RuntimeError(
        "Could not download SciFact.\n  " + "\n  ".join(errors) +
        f"\nDownload {config.BEIR_SCIFACT_URL} in a browser, save it as {manual_zip}, and rerun."
    )


# ====================================================================== loading
def load_corpus(raw_dir: Path = config.RAW_DIR) -> dict[str, dict]:
    """doc_id -> {"title", "text"} in file order."""
    corpus = {}
    for rec in read_jsonl(Path(raw_dir) / "corpus.jsonl"):
        corpus[str(rec["_id"])] = {"title": (rec.get("title") or "").strip(),
                                    "text": (rec.get("text") or "").strip()}
    return corpus


def load_qrels(split: str, raw_dir: Path = config.RAW_DIR) -> dict[str, dict[str, int]]:
    """qid -> {doc_id: relevance} for split in {"train", "test"}."""
    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    with open(Path(raw_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].strip().lstrip("-").isdigit():
                continue  # header row: query-id corpus-id score
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0].strip(), parts[1].strip(), int(parts[2])
            qrels[qid][doc_id] = score
    return dict(qrels)


def claim_label(evidence: dict) -> str:
    """SUPPORT / CONTRADICT / NEI (no evidence) / MIXED (both, never happens in SciFact)."""
    labels = {e["label"] for evs in evidence.values() for e in evs}
    if not labels:
        return "NEI"
    return labels.pop() if len(labels) == 1 else "MIXED"


def load_queries(raw_dir: Path = config.RAW_DIR) -> dict[str, dict]:
    """qid -> {"text", "evidence", "label", "split"} (split is filled from qrels)."""
    splits = {}
    for split in ("train", "test"):
        for qid in load_qrels(split, raw_dir):
            splits[qid] = split
    queries = {}
    for rec in read_jsonl(Path(raw_dir) / "queries.jsonl"):
        qid = str(rec["_id"])
        evidence = rec.get("metadata") or {}
        queries[qid] = {"text": rec["text"].strip(), "evidence": evidence,
                        "label": claim_label(evidence), "split": splits.get(qid)}
    return queries


def fingerprint(corpus: dict[str, dict]) -> str:
    """Order- and format-independent hash of the corpus content."""
    h = hashlib.sha256()
    for doc_id in sorted(corpus):
        d = corpus[doc_id]
        h.update(f"{doc_id}\t{d['title']}\t{d['text']}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def verify(raw_dir: Path = config.RAW_DIR, verbose: bool = True) -> dict:
    """Check the files against the published SciFact/BEIR statistics."""
    corpus = load_corpus(raw_dir)
    queries = load_queries(raw_dir)
    test, train = load_qrels("test", raw_dir), load_qrels("train", raw_dir)
    stats = {
        "corpus": len(corpus),
        "queries": len(queries),
        "test_queries": len(test),
        "train_queries": len(train),
        "test_pairs": sum(len(v) for v in test.values()),
        "train_pairs": sum(len(v) for v in train.values()),
        "test_labels": dict(collections.Counter(queries[q]["label"] for q in test)),
        "train_labels": dict(collections.Counter(queries[q]["label"] for q in train)),
    }
    problems = [f"{k}: expected {v}, got {stats[k]}" for k, v in EXPECTED.items() if stats[k] != v]
    missing = [d for qr in (test, train) for rel in qr.values() for d in rel if d not in corpus]
    if missing:
        problems.append(f"{len(missing)} judged docs missing from the corpus")
    stats["fingerprint"] = fingerprint(corpus)
    if problems:
        raise ValueError("SciFact files look wrong:\n  " + "\n  ".join(problems))
    if verbose:
        print(f"SciFact OK: {stats['corpus']} docs, {stats['test_queries']} test / "
              f"{stats['train_queries']} train claims, fingerprint {stats['fingerprint']}")
    return stats


# ====================================================================== processed copies
def prepare(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR) -> None:
    """Write data/processed/{corpus,queries}.jsonl and qrels.json."""
    corpus = load_corpus(raw_dir)
    queries = load_queries(raw_dir)
    write_jsonl(Path(out_dir) / "corpus.jsonl",
                ({"doc_id": k, **v} for k, v in corpus.items()))
    write_jsonl(Path(out_dir) / "queries.jsonl",
                ({"qid": k, **v} for k, v in queries.items()))
    write_json(Path(out_dir) / "qrels.json",
               {"train": load_qrels("train", raw_dir), "test": load_qrels("test", raw_dir)})


def load_split(split: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Convenience: ({qid: claim text}, qrels) for one split, from the raw files."""
    queries = load_queries()
    qrels = load_qrels(split)
    return {q: queries[q]["text"] for q in sorted(qrels, key=_qid_key)}, qrels


def _qid_key(qid: str):
    return (0, int(qid)) if qid.isdigit() else (1, qid)
