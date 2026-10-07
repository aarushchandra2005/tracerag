import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.index.inverted_index import InvertedIndex  # noqa: E402
from src.index.tokenizer import Tokenizer  # noqa: E402
from src.ingest.chunking import build_chunks  # noqa: E402

TOY_CORPUS = {
    "d1": {"title": "Car insurance basics",
           "text": "Car insurance protects drivers. Auto insurance is required by law in most states."},
    "d2": {"title": "Best cars of the year",
           "text": "The best car of the year is fast. Cars are reviewed by experts every year."},
    "d3": {"title": "Tumor growth in mice",
           "text": "Swim training suppresses tumor growth in mice. Tumors grew slower after training. "
                   "The effect was large."},
    "d4": {"title": "Insurance fraud",
           "text": "Insurance fraud costs billions. Fraud detection uses machine learning."},
}


@pytest.fixture
def toy_corpus():
    return TOY_CORPUS


@pytest.fixture
def toy_index():
    return InvertedIndex.build(build_chunks(TOY_CORPUS, "whole"), Tokenizer(), "whole")


@pytest.fixture
def toy_sent_index():
    return InvertedIndex.build(build_chunks(TOY_CORPUS, "sent"), Tokenizer(), "sent")


def scifact_available() -> bool:
    return (config.RAW_DIR / "corpus.jsonl").exists() and (config.RAW_DIR / "qrels" / "test.tsv").exists()


needs_scifact = pytest.mark.skipif(not scifact_available(), reason="SciFact not downloaded")
