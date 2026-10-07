"""Retrieval evaluation on BEIR SciFact.

    python -m src.eval.run_retrieval_eval            # everything
    python -m src.eval.run_retrieval_eval --no-dense # sparse models only (no model download)

1. Tune on TRAIN (809 claims): tf-idf title weight, weighted-fusion alpha.
   The test split is never used to pick a setting.
2. Main table on TEST (300 claims), whole-abstract chunks (comparable to BEIR).
3. Ablations (train and test): tokenizer, title weight, chunking, fusion, BM25 k1/b.
4. Per-query scores, TREC run files, randomization tests, bootstrap CIs, plots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import config
from src.eval import plots
from src.eval.metrics import (METRICS, bootstrap_ci, evaluate, first_relevant_rank,
                              paired_randomization_test)
from src.ingest.load_scifact import load_queries, load_split
from src.retrieve.factory import NEEDS_DENSE, RetrieverFactory
from src.retrieve.tfidf import TfidfRetriever
from src.utils import markdown_table, save_tuned, setup_console, timer, write_json

LABELS = {
    "tfidf": "tf-idf lnc.ltc (zones)",
    "tfidf_single": "tf-idf lnc.ltc (one field)",
    "bm25": "BM25",
    "dense": "Dense (MiniLM)",
    "hybrid_rrf": "Hybrid RRF (BM25 + dense)",
    "hybrid_weighted": "Hybrid weighted (BM25 + dense)",
    "hybrid_rrf_tfidf": "Hybrid RRF (tf-idf + dense)",
}
MAIN_SYSTEMS = ("tfidf", "bm25", "dense", "hybrid_rrf", "hybrid_weighted")
OUT = config.RESULTS_DIR


# ====================================================================== running
def run_retriever(retriever, queries: dict[str, str], depth: int = config.EVAL_DEPTH) -> dict[str, list]:
    retriever.prepare(list(queries.values()))
    return {q: retriever.rank_docs(text, depth) for q, text in queries.items()}


def score(run: dict[str, list], qrels) -> tuple[dict, dict]:
    return evaluate({q: [d for d, _ in lst] for q, lst in run.items()}, qrels)


def write_trec(run: dict[str, list], path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q, lst in run.items():
            for rank, (d, s) in enumerate(lst, start=1):
                f.write(f"{q} Q0 {d} {rank} {s:.6f} {tag}\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def best_by_ndcg(rows: list[dict], key: str):
    best = max(rows, key=lambda r: r["nDCG@10"])  # max() keeps the first of equal values
    return best[key]


# ====================================================================== tuning on train
def tune_zone_weights(factory, chunking: str, queries, qrels) -> tuple[tuple, list[dict]]:
    index = factory.index(chunking)
    rows = []
    for w in config.ZONE_WEIGHT_GRID:
        means, _ = score(run_retriever(TfidfRetriever(index, w), queries), qrels)
        rows.append({"param": "title_weight", "chunking": chunking, "value": w[0], **means})
    best = best_by_ndcg(rows, "value")
    return (float(best), 1.0), rows


def tune_alpha(factory, chunking: str, queries, qrels) -> tuple[float, list[dict]]:
    rows = []
    for a in config.ALPHA_GRID:
        means, _ = score(run_retriever(factory.get("hybrid_weighted", chunking, alpha=a), queries), qrels)
        rows.append({"param": "hybrid_alpha", "chunking": chunking, "value": a, **means})
    return float(best_by_ndcg(rows, "value")), rows


# ====================================================================== main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dense", action="store_true", help="skip dense and hybrid models")
    args = ap.parse_args(argv)
    setup_console()
    config.ensure_dirs()
    use_dense = not args.no_dense

    factory = RetrieverFactory()
    q_test, qrels_test = load_split("test")
    q_train, qrels_train = load_split("train")
    claim_label = {q: v["label"] for q, v in load_queries().items()}
    systems = [s for s in MAIN_SYSTEMS if use_dense or s not in NEEDS_DENSE]

    # ---------------------------------------------------------- 1. tuning on train
    tuning_rows, tuned = [], {}
    with timer("tune on train"):
        for chunking in config.CHUNKINGS:
            w, rows = tune_zone_weights(factory, chunking, q_train, qrels_train)
            tuned[f"zone_weights_{chunking}"] = list(w)
            tuning_rows += rows
            if use_dense:
                a, rows = tune_alpha(factory, chunking, q_train, qrels_train)
                tuned[f"hybrid_alpha_{chunking}"] = a
                tuning_rows += rows
        for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
            for b in (0.25, 0.5, 0.75, 1.0):
                means, _ = score(run_retriever(factory.get("bm25", "whole", k1=k1, b=b), q_train), qrels_train)
                tuning_rows.append({"param": "bm25_k1_b", "chunking": "whole", "value": f"{k1}/{b}", **means})
    save_tuned(tuned)
    write_csv(OUT / "tuning_train.csv", tuning_rows)
    print("tuned on train:", tuned)

    # ---------------------------------------------------------- 2. main table (test) + per-query
    results: dict[tuple, dict] = {}       # (system, chunking, split) -> means
    per_query: dict[tuple, dict] = {}     # (system, chunking) -> {qid: metrics} on test
    runs_test: dict[tuple, dict] = {}
    with timer("evaluate systems"):
        for chunking in config.CHUNKINGS:
            names = systems + (["tfidf_single", "hybrid_rrf_tfidf"] if use_dense else ["tfidf_single"])
            for name in names:
                retriever = factory.get(name, chunking)
                for split, queries, qrels in (("test", q_test, qrels_test), ("train", q_train, qrels_train)):
                    run = run_retriever(retriever, queries)
                    means, per = score(run, qrels)
                    results[(name, chunking, split)] = means
                    if split == "test":
                        per_query[(name, chunking)] = per
                        runs_test[(name, chunking)] = run
                        write_trec(run, OUT / "runs" / f"{name}_{chunking}.trec", f"{name}_{chunking}")

    main_rows = []
    for name in systems:
        m = results[(name, "whole", "test")]
        lo, hi = bootstrap_ci([per_query[(name, "whole")][q]["nDCG@10"] for q in qrels_test])
        main_rows.append({"Retriever": LABELS[name], **{k: m[k] for k in METRICS},
                          "nDCG@10 95% CI": f"{lo:.3f}\u2013{hi:.3f}"})
    write_csv(OUT / "retrieval_main.csv", main_rows)

    # per-query file (test) with the rank of the first relevant paper
    pq_rows = []
    for (name, chunking), per in per_query.items():
        for q in qrels_test:
            ranked = [d for d, _ in runs_test[(name, chunking)][q]]
            pq_rows.append({"qid": q, "claim_label": claim_label[q], "system": name, "chunking": chunking,
                            **per[q], "first_rel_rank": first_relevant_rank(ranked, qrels_test[q]) or ""})
    write_csv(OUT / "per_query_test.csv", pq_rows)

    # nDCG@10 by claim type (SUPPORT / CONTRADICT / NEI)
    by_label = []
    for name in systems:
        row = {"Retriever": LABELS[name]}
        for lab in ("SUPPORT", "CONTRADICT", "NEI"):
            qs = [q for q in qrels_test if claim_label[q] == lab]
            row[f"{lab} (n={len(qs)})"] = float(np.mean([per_query[(name, "whole")][q]["nDCG@10"] for q in qs]))
        by_label.append(row)
    write_csv(OUT / "retrieval_by_claim_type.csv", by_label)

    # ---------------------------------------------------------- 3. significance (test, whole)
    pairs = [("bm25", "tfidf"), ("tfidf", "tfidf_single")]
    if use_dense:
        pairs += [("dense", "bm25"), ("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense"),
                  ("hybrid_weighted", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rrf_tfidf")]
    sig_rows = []
    for a, b in pairs:
        for chunking in config.CHUNKINGS:
            xa = [per_query[(a, chunking)][q]["nDCG@10"] for q in qrels_test]
            xb = [per_query[(b, chunking)][q]["nDCG@10"] for q in qrels_test]
            sig_rows.append({"A": LABELS[a], "B": LABELS[b], "chunking": chunking,
                             "nDCG@10 A": float(np.mean(xa)), "nDCG@10 B": float(np.mean(xb)),
                             "diff": float(np.mean(xa) - np.mean(xb)),
                             "p (randomization, 10k)": paired_randomization_test(xa, xb)})
    write_csv(OUT / "significance.csv", sig_rows)

    # ---------------------------------------------------------- 4. ablations
    abl = []

    def add(section: str, setting: str, name: str, chunking: str, train: dict, test: dict) -> None:
        abl.append({"ablation": section, "setting": setting, "retriever": LABELS[name], "chunking": chunking,
                    "nDCG@10 train": train["nDCG@10"], "nDCG@10 test": test["nDCG@10"],
                    "MRR@10 test": test["MRR@10"], "R@10 test": test["R@10"], "P@5 test": test["P@5"]})

    with timer("ablations"):
        # tokenizer
        for stem, stop in ((True, True), (False, True), (True, False), (False, False)):
            setting = f"stemming {'on' if stem else 'off'}, stopwords {'removed' if stop else 'kept'}"
            for name in ("tfidf", "bm25"):
                r = factory.get(name, "whole", stem=stem, stopwords=stop)
                tr, _ = score(run_retriever(r, q_train), qrels_train)
                te, _ = score(run_retriever(r, q_test), qrels_test)
                add("tokenizer", setting, name, "whole", tr, te)
        # title zone weight (train rows reused from tuning)
        for chunking in config.CHUNKINGS:
            train_by_w = {r["value"]: r for r in tuning_rows
                          if r["param"] == "title_weight" and r["chunking"] == chunking}
            for w in config.ZONE_WEIGHT_GRID:
                te, _ = score(run_retriever(TfidfRetriever(factory.index(chunking), w), q_test), qrels_test)
                add("title weight", f"w_title={w[0]:g} x w_body", "tfidf", chunking, train_by_w[w[0]], te)
            add("title weight", "one merged field", "tfidf_single", chunking,
                results[("tfidf_single", chunking, "train")], results[("tfidf_single", chunking, "test")])
        # chunking
        for name in systems:
            for chunking in config.CHUNKINGS:
                add("chunking", f"{chunking} ({'whole abstract' if chunking == 'whole' else 'sentence windows'})",
                    name, chunking, results[(name, chunking, "train")], results[(name, chunking, "test")])
        # fusion inputs
        if use_dense:
            for name in ("hybrid_rrf", "hybrid_rrf_tfidf", "hybrid_weighted"):
                add("fusion", LABELS[name], name, "whole",
                    results[(name, "whole", "train")], results[(name, "whole", "test")])
        # BM25 parameters
        for k1, b in ((0.9, 0.4), (1.2, 0.75), (1.5, 0.75), (2.0, 0.75), (1.2, 1.0), (1.2, 0.5)):
            r = factory.get("bm25", "whole", k1=k1, b=b)
            tr, _ = score(run_retriever(r, q_train), qrels_train)
            te, _ = score(run_retriever(r, q_test), qrels_test)
            add("BM25 k1/b", f"k1={k1}, b={b}", "bm25", "whole", tr, te)
    write_csv(OUT / "retrieval_ablations.csv", abl)

    # ---------------------------------------------------------- 5. markdown + plots
    md = ["# Retrieval results (BEIR SciFact)", "",
          f"Test split: {len(qrels_test)} claims. Settings tuned on the {len(qrels_train)} train claims: "
          f"`{tuned}`.", "", "## Main table (whole-abstract chunks, test)", "",
          markdown_table(main_rows), "",
          "## nDCG@10 by claim type (test)", "", markdown_table(by_label), "",
          "## Paired randomization tests on per-query nDCG@10 (test)", "",
          markdown_table([{**r, "p (randomization, 10k)": fmt_p(r["p (randomization, 10k)"])} for r in sig_rows]), "",
          "## Ablations", "", markdown_table(abl)]
    (OUT / "retrieval_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cats = [LABELS[s].replace(" (", "\n(") for s in systems]
    plots.grouped_bars(
        cats, {"nDCG@10": [results[(s, "whole", "test")]["nDCG@10"] for s in systems],
               "MRR@10": [results[(s, "whole", "test")]["MRR@10"] for s in systems],
               "R@10": [results[(s, "whole", "test")]["R@10"] for s in systems]},
        OUT / "retrieval_main", "Retrieval quality on SciFact test claims",
        f"{len(qrels_test)} claims, whole-abstract chunks; values shown for nDCG@10",
        ylim=(0, 1.0), label_series=["nDCG@10"])
    for chunking in config.CHUNKINGS:
        xs = [w[0] for w in config.ZONE_WEIGHT_GRID]
        tr = [r["nDCG@10 train"] for r in abl if r["ablation"] == "title weight" and r["chunking"] == chunking
              and r["retriever"] == LABELS["tfidf"]]
        te = [r["nDCG@10 test"] for r in abl if r["ablation"] == "title weight" and r["chunking"] == chunking
              and r["retriever"] == LABELS["tfidf"]]
        plots.lines(xs, {"train": tr, "test": te}, OUT / f"ablation_title_weight_{chunking}",
                    f"tf-idf: title zone weight ({chunking} chunks)",
                    "score = (w_title * cos_title + cos_body) / (w_title + 1)",
                    xlabel="w_title (w_body = 1)", ylabel="nDCG@10",
                    mark_x=tuned[f"zone_weights_{chunking}"][0])
    plots.grouped_bars(
        [LABELS[s].replace(" (", "\n(") for s in systems],
        {("whole abstracts" if c == "whole" else "3-sentence windows"): [results[(s, c, "test")]["nDCG@10"]
                                                                       for s in systems] for c in config.CHUNKINGS},
        OUT / "ablation_chunking", "Whole abstracts vs sentence windows",
        "nDCG@10 on test; papers ranked by their best chunk (MaxP)", ylim=(0, 1.0))
    write_json(OUT / "retrieval_summary.json",
               {f"{n}|{c}|{s}": m for (n, c, s), m in results.items()})
    print(markdown_table(main_rows))
    print(f"\nWrote {OUT / 'retrieval_results.md'}")


if __name__ == "__main__":
    main()
