"""One command: download SciFact, verify it, chunk it, build every index.

    python scripts/build_index.py              # sparse indexes for both chunkings
    python scripts/build_index.py --dense      # also cache MiniLM embeddings
    python scripts/build_index.py --ablations  # also the no-stem / no-stopword indexes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.index.inverted_index import get_index  # noqa: E402
from src.ingest import load_scifact  # noqa: E402
from src.ingest.chunking import build_chunks, save_chunks  # noqa: E402
from src.utils import setup_console, timer  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunking", nargs="+", default=list(config.CHUNKINGS), choices=config.CHUNKINGS)
    ap.add_argument("--dense", action="store_true", help="also encode chunks with the dense model")
    ap.add_argument("--ablations", action="store_true", help="also build stemming/stopword ablation indexes")
    ap.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = ap.parse_args(argv)
    setup_console()
    config.ensure_dirs()

    load_scifact.download()
    load_scifact.verify()
    load_scifact.prepare()
    corpus = load_scifact.load_corpus()

    for chunking in args.chunking:
        with timer(f"chunk {chunking}"):
            chunks = build_chunks(corpus, chunking)
            save_chunks(chunks, chunking)
        print(f"{chunking}: {len(chunks)} chunks")
        settings = [(config.USE_STEMMING, config.REMOVE_STOPWORDS)]
        if args.ablations:
            settings += [(False, True), (True, False), (False, False)]
        for stem, stop in settings:
            with timer(f"index {chunking} stem={stem} stopwords={stop}"):
                index = get_index(chunking, stem, stop, rebuild=args.force)
            s = index.stats()
            print(f"  {s['chunks']} chunks, {s['papers']} papers, "
                  f"{s['all']['terms']} terms, {s['all']['postings']} postings (title+body field)")
        if args.dense:
            from src.index.dense_index import DenseIndex

            with timer(f"dense {chunking}"):
                DenseIndex(get_index(chunking).chunks, config.DENSE_MODEL)

    sample = get_index("whole")
    sample.print_postings("tumor")


if __name__ == "__main__":
    main()
