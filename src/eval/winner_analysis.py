"""Sparse vs dense, claim by claim: who ranked the first relevant paper higher, and why?

For each test claim we take the rank of the first relevant paper under BM25
(sparse) and MiniLM (dense), whole-abstract chunks. Lower rank wins; equal
ranks are a tie; a paper outside the top 100 counts as rank 101.

"Why" is measured with lexical overlap: the share of the claim's index terms
(stemmed, stopwords removed) that occur in its best-matching relevant paper.
If sparse wins on exact word matches and dense wins on paraphrase, sparse
wins should have higher overlap than dense wins.

    python -m src.eval.winner_analysis
"""
from __future__ import annotations

import statistics

import numpy as np

import config
from src.eval import plots
from src.eval.metrics import first_relevant_rank
from src.eval.run_retrieval_eval import write_csv
from src.explain.contrib import format_terms, query_overlap, top_terms
from src.ingest.load_scifact import load_queries, load_split
from src.retrieve.factory import RetrieverFactory
from src.utils import markdown_table, setup_console

OUT = config.RESULTS_DIR
DEPTH = 100


def main(argv=None) -> None:
    setup_console()
    factory = RetrieverFactory()
    bm25, dense, tfidf = (factory.get(n, "whole") for n in ("bm25", "dense", "tfidf"))
    index = bm25.index
    tok = index.tokenizer
    queries, qrels = load_split("test")
    meta = load_queries()
    dense.prepare(list(queries.values()))

    rows = []
    for qid, claim in queries.items():
        ranks = {}
        for name, r in (("bm25", bm25), ("dense", dense), ("tfidf", tfidf)):
            ranked = [d for d, _ in r.rank_docs(claim, DEPTH)]
            ranks[name] = first_relevant_rank(ranked, qrels[qid]) or DEPTH + 1
        best_doc, best_ov, hit, miss = None, -1.0, [], []
        for doc_id in qrels[qid]:
            c = index.chunks[index.chunk_pos[doc_id]]
            ov, h, m = query_overlap(claim, f"{c.title} {c.text}", tok)
            if ov > best_ov:
                best_doc, best_ov, hit, miss = doc_id, ov, h, m
        if ranks["bm25"] < ranks["dense"]:
            winner = "sparse"
        elif ranks["dense"] < ranks["bm25"]:
            winner = "dense"
        else:
            winner = "tie"
        rows.append({"qid": qid, "claim": claim, "claim_label": meta[qid]["label"], "winner": winner,
                     "rank_bm25": ranks["bm25"], "rank_dense": ranks["dense"], "rank_tfidf": ranks["tfidf"],
                     "margin": ranks["dense"] - ranks["bm25"], "relevant_paper": best_doc,
                     "overlap": best_ov, "matched_terms": " ".join(hit), "missing_terms": " ".join(miss)})
    write_csv(OUT / "winner_analysis.csv", rows)

    groups = {w: [r for r in rows if r["winner"] == w] for w in ("sparse", "tie", "dense")}
    summary = []
    for w, g in groups.items():
        ov = [r["overlap"] for r in g]
        summary.append({"winner": w, "claims": len(g), "share": len(g) / len(rows),
                        "mean overlap": statistics.mean(ov) if ov else float("nan"),
                        "median overlap": statistics.median(ov) if ov else float("nan"),
                        "mean claim terms missing": statistics.mean(
                            [len(r["missing_terms"].split()) for r in g]) if g else float("nan")})
    # is the overlap gap between sparse and dense wins larger than chance? (unpaired permutation test)
    a = np.array([r["overlap"] for r in groups["sparse"]])
    b = np.array([r["overlap"] for r in groups["dense"]])
    p_value = _perm_test_unpaired(a, b) if len(a) and len(b) else float("nan")

    def rk(x: int) -> str:
        return f">{DEPTH}" if x > DEPTH else str(x)

    def describe(r, kind: str) -> list[str]:
        out = [f"**Claim {r['qid']}** ({r['claim_label']}): {r['claim']}",
               f"- first relevant paper {r['relevant_paper']}: BM25 rank {rk(r['rank_bm25'])}, "
               f"dense rank {rk(r['rank_dense'])}, tf-idf rank {rk(r['rank_tfidf'])}",
               f"- lexical overlap {r['overlap']:.0%}; matched: `{r['matched_terms']}`; "
               f"missing: `{r['missing_terms'] or '-'}`"]
        if kind == "sparse":
            out.append(f"- BM25 term shares for the paper: {format_terms(top_terms(r['claim'], r['relevant_paper'], bm25))}")
        else:
            best = dense.explain(r["claim"], r["relevant_paper"])[0]
            out.append(f"- closest sentence in embedding space (cosine {best['cosine']:.2f}): \"{best['sentence']}\"")
        return out + [""]

    md = ["# Sparse vs dense: winner analysis (test, whole abstracts)", "",
          "Rank of the first relevant paper under BM25 and MiniLM (a paper outside the top 100 counts as 101 "
          "when deciding the winner); overlap = share of claim terms present in the relevant paper.", "", markdown_table(summary), "",
          f"Permutation test, mean overlap of sparse wins vs dense wins: p = {p_value:.4f}", ""]
    for kind in ("sparse", "dense"):
        g = sorted(groups[kind], key=lambda r: (-abs(r["margin"]), int(r["qid"])))[:3]
        md += [f"## Three clear {kind} wins", ""]
        for r in g:
            md += describe(r, kind)
    (OUT / "winner_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plots.grouped_bars(
        [f"{'ties' if w == 'tie' else w + ' wins'}\n(n={len(g)})" for w, g in groups.items()],
        {"mean lexical overlap": [s["mean overlap"] for s in summary]},
        OUT / "winner_overlap", "Claims sparse retrieval wins share more words with the paper",
        "share of claim terms found in the relevant paper, by which model ranked it higher", ylim=(0, 1), fmt="{:.0%}")
    print(markdown_table(summary))
    print(f"p = {p_value:.4f}\nWrote {OUT / 'winner_analysis.md'}")


def _perm_test_unpaired(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = config.SEED) -> float:
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


if __name__ == "__main__":
    main()
