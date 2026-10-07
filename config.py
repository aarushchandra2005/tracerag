"""Central settings for TraceRAG.

Every module and script reads its paths, model names and hyperparameters from
here, so one edit changes the whole pipeline. Values that are *tuned* (zone
weights, fusion alpha, verifier thresholds) are chosen on the SciFact TRAIN
split by the evaluation scripts and written to results/tuned_params.json; the
defaults below are only used when that file does not exist yet.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "scifact"          # BEIR files: corpus.jsonl, queries.jsonl, qrels/*.tsv
PROCESSED_DIR = DATA_DIR / "processed"          # chunks_*.jsonl, queries.jsonl, qrels.json
INDEX_DIR = DATA_DIR / "index"                  # pickled inverted indexes
CACHE_DIR = DATA_DIR / "cache"                  # dense embeddings, LLM answers
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
TUNED_PARAMS_PATH = RESULTS_DIR / "tuned_params.json"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- data
# Official BEIR mirror first, Hugging Face second (see src/ingest/load_scifact.py).
BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

# ---------------------------------------------------------------- chunking
CHUNKINGS = ("whole", "sent")
SENT_WINDOW = 3          # sentences per window
SENT_STRIDE = 1          # step between window starts

# ---------------------------------------------------------------- tokenizer
USE_STEMMING = True
REMOVE_STOPWORDS = True

# ---------------------------------------------------------------- sparse retrieval
# tf-idf: score = sum_z w_z * cos_lnc.ltc(q, d_z) / sum_z w_z over zones (title, body)
DEFAULT_ZONE_WEIGHTS = (0.1, 1.0)                 # (w_title, w_body); tuned on train
ZONE_WEIGHT_GRID = tuple((w, 1.0) for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0))
BM25_K1 = 1.2
BM25_B = 0.75

# ---------------------------------------------------------------- dense retrieval
DENSE_MODEL = os.environ.get("TRACERAG_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_BATCH_SIZE = 64
DEVICE = os.environ.get("TRACERAG_DEVICE") or None   # None lets torch pick (cpu/cuda/mps)

# ---------------------------------------------------------------- fusion
RRF_K = 60
FUSION_DEPTH = 1000                               # items per ranked list fed to fusion
DEFAULT_HYBRID_ALPHA = 0.5                        # weight on the sparse list; tuned on train
ALPHA_GRID = tuple(round(a / 10, 1) for a in range(11))

# ---------------------------------------------------------------- evaluation
METRIC_KS = (5, 10)
EVAL_DEPTH = 100                                  # docs kept per query in saved runs

# ---------------------------------------------------------------- generation
TOP_K_GEN = 5
N_GEN_QUERIES = 30
GEN_RETRIEVER = "hybrid_rrf"
GEN_CHUNKING = "sent"
LLM_BACKEND = os.environ.get("TRACERAG_LLM", "extractive")   # extractive | openai | anthropic
LLM_MODEL = os.environ.get("TRACERAG_LLM_MODEL", "")         # required for openai-compatible APIs
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 400
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------- verifier
NLI_MODEL = os.environ.get("TRACERAG_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
NLI_BATCH_SIZE = 16
NLI_MAX_LENGTH = 512
PREMISE_WINDOW = 2       # sentences per evidence window; best cosine AUROC on train (1, 2, 3 tried)
DEFAULT_COSINE_THRESHOLD = 0.70
DEFAULT_NLI_THRESHOLD = 0.5
DEFAULT_CASCADE_GATE = 0.30   # cosine below this is rejected without calling NLI


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, CACHE_DIR, RESULTS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
