"""Evaluate the citation verifier two ways.

A. Gold benchmark from SciFact's expert annotations (no hand labelling needed).
   Each item is (claim, cited paper):
     support            paper annotated as SUPPORTING the claim        -> supported
     contradict         paper annotated as CONTRADICTING the claim     -> unsupported
     cited_no_evidence  paper cited for the claim but holding no
                        evidence for it (SciFact "NEI" citations)      -> unsupported
     wrong_paper        the top BM25 paper that is not judged relevant
                        (a swapped citation), for SUPPORT claims       -> unsupported
   Thresholds are tuned on TRAIN claims, results are reported on TEST claims.

B. Injected corruptions on generated answers (src/verify/inject.py): one
   sentence per corrupted copy is known to be unsupported. Original sentences
   count as supported unless hand labels exist (scripts/label_answers.py);
   extractive answers are supported by construction.

    python -m src.eval.run_verifier_eval             # cosine, NLI and cascade
    python -m src.eval.run_verifier_eval --no-nli    # cosine only (no NLI model)
    python -m src.eval.run_verifier_eval --answers results/generation/answers_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import config
from src.eval.run_retrieval_eval import write_csv
from src.ingest.chunking import Chunk, split_sentences
from src.ingest.load_scifact import load_queries, load_qrels
from src.retrieve.factory import RetrieverFactory
from src.utils import markdown_table, read_json, read_jsonl, save_tuned, setup_console, timer, write_jsonl
from src.verify import inject, splitter
from src.verify.checker import CitationChecker

OUT = config.RESULTS_DIR
VDIR = OUT / "verifier"
COS_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
NLI_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
GATE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


# ====================================================================== metrics
def flag_metrics(gold_unsup, pred_unsup) -> dict:
    g, p = np.asarray(gold_unsup, bool), np.asarray(pred_unsup, bool)
    tp, fp = int(np.sum(g & p)), int(np.sum(~g & p))
    fn, tn = int(np.sum(g & ~p)), int(np.sum(~g & ~p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # the same for the "supported" class, so a checker that flags everything cannot look good
    prec_s = tn / (tn + fn) if tn + fn else 0.0
    rec_s = tn / (tn + fp) if tn + fp else 0.0
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if prec_s + rec_s else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "macro_f1": (f1 + f1_s) / 2,
            "balanced_acc": (rec + rec_s) / 2, "accuracy": (tp + tn) / max(1, len(g)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(support_scores, gold_unsup) -> float:
    """P(a supported item scores higher than an unsupported one); ties count 1/2."""
    s = np.asarray(support_scores, float)
    g = np.asarray(gold_unsup, bool)
    pos, neg = s[~g], s[g]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                        # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


SELECT_BY = "macro_f1"   # F1 of the flag alone is maximised by flagging everything when most items are unsupported


def best_threshold(scores, gold_unsup, grid) -> tuple[float, dict]:
    best_t, best_m = None, None
    for t in grid:
        m = flag_metrics(gold_unsup, np.asarray(scores) < t)
        if best_m is None or m[SELECT_BY] > best_m[SELECT_BY] + 1e-12:
            best_t, best_m = float(t), m
    return best_t, best_m


def cascade_predict(cos, ent, gate, t):
    cos, ent = np.asarray(cos), np.asarray(ent)
    return (cos < gate) | (ent < t)


# ====================================================================== A. gold benchmark
def build_gold_pairs(split: str, meta: dict, qrels: dict, bm25) -> list[dict]:
    pairs = []
    for qid in sorted(qrels, key=lambda q: int(q)):
        claim, ev = meta[qid]["text"], meta[qid]["evidence"]
        for doc_id, entries in ev.items():
            lab = entries[0]["label"]
            pairs.append({"split": split, "qid": qid, "claim": claim, "doc_id": doc_id,
                          "kind": "support" if lab == "SUPPORT" else "contradict",
                          "gold": "supported" if lab == "SUPPORT" else "unsupported"})
        for doc_id in qrels[qid]:
            if doc_id not in ev:
                pairs.append({"split": split, "qid": qid, "claim": claim, "doc_id": doc_id,
                              "kind": "cited_no_evidence", "gold": "unsupported"})
        if any(e[0]["label"] == "SUPPORT" for e in ev.values()):
            exclude = set(qrels[qid]) | set(ev)
            for doc_id, _ in bm25.rank_docs(claim, 50):
                if doc_id not in exclude:
                    pairs.append({"split": split, "qid": qid, "claim": claim, "doc_id": doc_id,
                                  "kind": "wrong_paper", "gold": "unsupported"})
                    break
    return pairs


def score_pairs(checker: CitationChecker, pairs: list[dict], chunk_of: dict, use_nli: bool) -> None:
    checker.warm([p["claim"] for p in pairs], [chunk_of[p["doc_id"]] for p in pairs])
    for i, p in enumerate(pairs):
        s = checker.pair_scores(p["claim"], [chunk_of[p["doc_id"]]], use_nli=use_nli)
        p["cosine"] = s["cosine"]
        p["cos_evidence"] = s["cos_evidence"]
        if use_nli:
            p.update(entail=s["entail"], contradict=s["contradict"], nli_evidence=s["nli_evidence"])
        if use_nli and (i + 1) % 200 == 0:
            print(f"  NLI scored {i + 1}/{len(pairs)} pairs", flush=True)


def gold_benchmark(checker, use_nli: bool, factory: RetrieverFactory) -> dict:
    meta = load_queries()
    index = factory.index("whole")
    chunk_of = {c.doc_id: c for c in index.chunks}
    bm25 = factory.get("bm25", "whole")
    pairs = {s: build_gold_pairs(s, meta, load_qrels(s), bm25) for s in ("train", "test")}
    for s in ("train", "test"):
        with timer(f"score gold pairs ({s}, {len(pairs[s])} pairs)"):
            score_pairs(checker, pairs[s], chunk_of, use_nli)

    def gold(s):
        return [p["gold"] == "unsupported" for p in pairs[s]]

    tuned, rows, preds = {}, [], {}
    rows.append({"checker": "Baseline: flag every sentence", "threshold (train)": "-",
                 **flag_metrics(gold("test"), np.ones(len(pairs["test"]), bool)), "AUROC": 0.5, "NLI calls": "0%"})
    # cosine
    t_cos, _ = best_threshold([p["cosine"] for p in pairs["train"]], gold("train"), COS_GRID)
    tuned["verifier_cosine_threshold"] = t_cos
    pred = np.array([p["cosine"] for p in pairs["test"]]) < t_cos
    preds["cosine"] = pred
    rows.append({"checker": "Cosine (MiniLM)", "threshold (train)": f"cos < {t_cos:.2f}",
                 **flag_metrics(gold("test"), pred),
                 "AUROC": auroc([p["cosine"] for p in pairs["test"]], gold("test")), "NLI calls": "0%"})
    if use_nli:
        t_nli, _ = best_threshold([p["entail"] for p in pairs["train"]], gold("train"), NLI_GRID)
        tuned["verifier_nli_threshold"] = t_nli
        pred = np.array([p["entail"] for p in pairs["test"]]) < t_nli
        preds["nli"] = pred
        rows.append({"checker": f"NLI ({Path(checker.nli_model_name).name})",
                     "threshold (train)": f"P(entail) < {t_nli:.2f}",
                     **flag_metrics(gold("test"), pred),
                     "AUROC": auroc([p["entail"] for p in pairs["test"]], gold("test")), "NLI calls": "100%"})
        cos_tr = [p["cosine"] for p in pairs["train"]]
        ent_tr = [p["entail"] for p in pairs["train"]]
        best = None
        for gate in GATE_GRID:
            for t in NLI_GRID:
                m = flag_metrics(gold("train"), cascade_predict(cos_tr, ent_tr, gate, t))
                key = (round(m[SELECT_BY], 10), gate)     # ties: prefer the higher gate (fewer NLI calls)
                if best is None or key > best[0]:
                    best = (key, gate, float(t))
        _, gate, t_c = best
        tuned.update(verifier_cascade_gate=gate, verifier_cascade_nli_threshold=t_c)
        cos_te = np.array([p["cosine"] for p in pairs["test"]])
        pred = cascade_predict(cos_te, [p["entail"] for p in pairs["test"]], gate, t_c)
        preds["cascade"] = pred
        calls = float(np.mean(cos_te >= gate))
        rows.append({"checker": "Cosine gate + NLI (cascade)",
                     "threshold (train)": f"cos < {gate:.2f} or P(entail) < {t_c:.2f}",
                     **flag_metrics(gold("test"), pred), "AUROC": None, "NLI calls": f"{calls:.0%}"})
    save_tuned(tuned)

    # flag rate per item kind (test)
    kinds = ("support", "contradict", "cited_no_evidence", "wrong_paper")
    by_kind = []
    for method, pred in preds.items():
        row = {"checker": method}
        for k in kinds:
            mask = np.array([p["kind"] == k for p in pairs["test"]])
            row[f"{k} (n={int(mask.sum())})"] = float(pred[mask].mean()) if mask.any() else float("nan")
        by_kind.append(row)
    for s in ("train", "test"):
        for i, p in enumerate(pairs[s]):
            if s == "test":
                for method, pred in preds.items():
                    p[f"pred_{method}"] = "unsupported" if pred[i] else "supported"
        write_csv(VDIR / f"gold_pairs_{s}.csv", pairs[s])
    composition = {s: dict(Counter(p["kind"] for p in pairs[s])) for s in pairs}

    # evidence window size for the cosine checker (AUROC is threshold-free; the choice is made on train)
    window_rows = []
    for w in (1, 2, 3):
        if w == checker.window:
            sc = {s: [p["cosine"] for p in pairs[s]] for s in pairs}
        else:
            alt = CitationChecker("cosine", embed_model=checker.embed_model, nli_model=None, window=w)
            sc = {}
            for s in pairs:
                alt.warm([p["claim"] for p in pairs[s]], [chunk_of[p["doc_id"]] for p in pairs[s]])
                sc[s] = [alt.score_cosine(p["claim"], chunk_of[p["doc_id"]])[0] for p in pairs[s]]
        window_rows.append({"evidence window": f"title + {w}-sentence windows",
                            "AUROC train": auroc(sc["train"], gold("train")),
                            "AUROC test": auroc(sc["test"], gold("test")),
                            "used": "yes" if w == checker.window else ""})
    return {"rows": rows, "by_kind": by_kind, "tuned": tuned, "pairs": pairs, "composition": composition,
            "windows": window_rows}


# ====================================================================== B. injected corruptions
def _on_topic_other_paper(claim: str, exclude_docs: set, bm25, corpus_chunk: dict, rng: random.Random):
    """(a sentence, a whole-abstract chunk) from the best BM25 paper that was neither
    retrieved nor judged relevant: plausible, on-topic, and not evidence for the claim."""
    for doc_id, _ in bm25.rank_docs(claim, 30):
        if doc_id in exclude_docs:
            continue
        sents = [s for s in split_sentences(corpus_chunk[doc_id].text)
                 if len(s.split()) >= 8 and s[0].isupper() and s[-1] in ".!?"]
        if sents:
            return rng.choice(sents), corpus_chunk[doc_id]
    return None, None


def injected_eval(checker, answers_path: Path, methods: list[str], factory: RetrieverFactory) -> dict:
    answers = read_jsonl(answers_path)
    tag = answers_path.stem.replace("answers_", "")
    labels_path = answers_path.parent / f"labels_{tag}.json"
    hand = read_json(labels_path) if labels_path.exists() else {}
    bm25 = factory.get("bm25", "whole")
    corpus_chunk = {c.doc_id: c for c in factory.index("whole").chunks}
    qrels = load_qrels("test")
    cases = []
    for a in answers:
        rng = random.Random(f"{config.SEED}-{a['qid']}")
        retrieved = [Chunk(r["chunk_id"], r["doc_id"], r["title"], r["text"]) for r in a["retrieved"]]
        valid = [c.chunk_id for c in retrieved]
        by_id = {c.chunk_id: c for c in retrieved}
        mine = []
        for i, p in enumerate(splitter.split(a["answer"], valid)):
            if p["abstain"]:
                continue
            hand_label = hand.get(a["qid"], {}).get(str(i))
            if hand_label:
                gold, source = hand_label, "hand label"
            elif not p["cited_ids"] or p["invalid_ids"]:
                gold, source = "unsupported", "missing or invalid citation"
            else:
                gold, source = "supported", "presumed"
            mine.append({"qid": a["qid"], "kind": "original", "sentence": p["text"], "cited": p["cited_ids"],
                         "invalid": p["invalid_ids"], "gold": gold, "gold_source": source})
        exclude = set(qrels.get(a["qid"], {})) | {c.doc_id for c in retrieved}
        fab, other = _on_topic_other_paper(a["claim"], exclude, bm25, corpus_chunk, rng)
        for corr in (inject.negate(a["answer"], rng, valid),
                     inject.swap_citation(a["answer"], rng, retrieved, valid, [other] if other else []),
                     inject.add_fabricated(a["answer"], rng, fab, valid, valid) if fab else None):
            if corr is None:
                continue
            ids = valid + ([corr.extra_chunk.chunk_id] if corr.extra_chunk else [])
            target = splitter.split(corr.answer, ids)[corr.index]
            mine.append({"qid": a["qid"], "kind": corr.kind, "sentence": target["text"],
                         "cited": target["cited_ids"], "invalid": target["invalid_ids"],
                         "gold": "unsupported", "gold_source": "injected", "original": corr.original,
                         "_extra": corr.extra_chunk})
        for c in mine:
            pool = dict(by_id)
            if c.get("_extra") is not None:
                pool[c["_extra"].chunk_id] = c["_extra"]
            c["_chunks"] = [pool[i] for i in c["cited"] if i in pool]
            c.pop("_extra", None)
        cases += mine
    checker.warm([c["sentence"] for c in cases], [ch for c in cases for ch in c["_chunks"]])
    for c in cases:
        for m in methods:
            v = checker.verdict(c["sentence"], c["_chunks"], m, c["invalid"])
            c[f"pred_{m}"] = v.label
            c[f"reason_{m}"] = v.reason
            c["cosine"] = v.cosine if v.cosine is not None else c.get("cosine")
            if v.entail is not None:
                c["entail"], c["contradict"] = v.entail, v.contradict
    rows, by_kind = [], []
    kinds = ("original", "negate", "swap_citation", "add_fabricated")
    for m in methods:
        g = [c["gold"] == "unsupported" for c in cases]
        p = [c[f"pred_{m}"] == "unsupported" for c in cases]
        rows.append({"checker": m, "answers": tag, **flag_metrics(g, p)})
        row = {"checker": m}
        for k in kinds:
            sel = [c for c in cases if c["kind"] == k]
            row[f"{k} flagged (n={len(sel)})"] = (sum(c[f"pred_{m}"] == "unsupported" for c in sel) / len(sel)
                                                 if sel else float("nan"))
        by_kind.append(row)
    for c in cases:
        c.pop("_chunks", None)
    write_jsonl(VDIR / f"injected_cases_{tag}.jsonl", cases)
    n_presumed = sum(c["gold_source"] == "presumed" for c in cases)
    return {"rows": rows, "by_kind": by_kind, "tag": tag, "n_presumed": n_presumed, "cases": cases}


# ====================================================================== failure cases
def failure_cases(gold: dict, methods: list[str], k: int = 3) -> list[str]:
    md = []
    test = gold["pairs"]["test"]
    for m in methods:
        score_key = "cosine" if m == "cosine" else "entail"
        missed = [p for p in test if p["gold"] == "unsupported" and p.get(f"pred_{m}") == "supported"]
        false_flags = [p for p in test if p["gold"] == "supported" and p.get(f"pred_{m}") == "unsupported"]
        missed.sort(key=lambda p: -p[score_key])
        false_flags.sort(key=lambda p: p[score_key])
        md += [f"### {m}: unsupported claims it let through (most confident first)", ""]
        for p in missed[:k]:
            ev = p.get("nli_evidence") if m != "cosine" else p.get("cos_evidence")
            md += [f"- claim {p['qid']} vs paper {p['doc_id']} ({p['kind']}), {score_key} = {p[score_key]:.2f}",
                   f"  - claim: {p['claim']}", f"  - closest evidence: {ev}"]
        md += ["", f"### {m}: supported claims it flagged", ""]
        for p in false_flags[:k]:
            ev = p.get("nli_evidence") if m != "cosine" else p.get("cos_evidence")
            md += [f"- claim {p['qid']} vs paper {p['doc_id']}, {score_key} = {p[score_key]:.2f}",
                   f"  - claim: {p['claim']}", f"  - closest evidence: {ev}"]
        md.append("")
    return md


# ====================================================================== main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-nli", action="store_true", help="cosine checker only")
    ap.add_argument("--nli-model", default=config.NLI_MODEL)
    ap.add_argument("--answers", type=Path, default=None,
                    help="answers_*.jsonl for the corruption test (default: every file in results/generation)")
    args = ap.parse_args(argv)
    setup_console()
    config.ensure_dirs()
    use_nli = not args.no_nli
    factory = RetrieverFactory()
    checker = CitationChecker("cosine", nli_model=args.nli_model)
    nli_note = "NLI rows not run (--no-nli)."
    if use_nli:
        try:
            _ = checker.nli
        except Exception as e:  # noqa: BLE001 - offline, blocked download, wrong model id
            use_nli = False
            nli_note = f"NLI rows not run: could not load {args.nli_model} ({type(e).__name__}: {e})."
            print(f"WARNING: {nli_note} Continuing with the cosine checker only.")
    methods = ["cosine"] + (["nli", "cascade"] if use_nli else [])

    gold = gold_benchmark(checker, use_nli, factory)
    checker.t.update({"cosine": gold["tuned"]["verifier_cosine_threshold"]})
    if use_nli:
        checker.t.update(nli=gold["tuned"]["verifier_nli_threshold"], gate=gold["tuned"]["verifier_cascade_gate"],
                         cascade_nli=gold["tuned"]["verifier_cascade_nli_threshold"])
    gold_cols = ["checker", "threshold (train)", "precision", "recall", "f1", "macro_f1", "balanced_acc", "AUROC",
                 "NLI calls", "tp", "fp", "fn", "tn"]
    write_csv(OUT / "verifier_gold.csv", [{c: r.get(c) for c in gold_cols} for r in gold["rows"]])
    write_csv(OUT / "verifier_gold_by_kind.csv", gold["by_kind"])
    write_csv(OUT / "verifier_window_ablation.csv", gold["windows"])

    files = [args.answers] if args.answers else sorted((OUT / "generation").glob("answers_*.jsonl"))
    injected = [injected_eval(checker, f, methods, factory) for f in files if f.exists()]
    inj_rows = [r for res in injected for r in res["rows"]]
    inj_kind = [{"answers": res["tag"], **r} for res in injected for r in res["by_kind"]]
    write_csv(OUT / "verifier_injected.csv", inj_rows)
    write_csv(OUT / "verifier_injected_by_kind.csv", inj_kind)

    md = ["# Citation verifier results", "",
          "## A. SciFact expert labels (claim vs cited paper)", "",
          f"Pairs: {json.dumps(gold['composition'])}. Thresholds tuned on train (macro-F1), scores on test. "
          "precision/recall/f1 are for the 'unsupported' flag; macro_f1 averages it with the F1 of the "
          "'supported' class, so flagging everything (first row) does not score well.", "",
          markdown_table([{c: r.get(c) for c in gold_cols[:9]} for r in gold["rows"]]), "",
          "Share of test pairs flagged as unsupported, by kind (support = false-alarm rate):", "",
          markdown_table(gold["by_kind"]), "",
          "Cosine checker: how much evidence to compare against (config.PREMISE_WINDOW, picked on train):", "",
          markdown_table(gold["windows"]), ""]
    if not use_nli:
        md += [nli_note + " Run `python -m src.eval.run_verifier_eval` with the model available to add them.", ""]
    md += ["## B. Injected corruptions on generated answers", ""]
    if injected:
        for res in injected:
            md += [f"Answers: `{res['tag']}`. Original sentences without a hand label "
                   f"({res['n_presumed']} items) are presumed supported.", ""]
        md += [markdown_table(inj_rows), "", markdown_table(inj_kind), ""]
    else:
        md += ["No generated answers found; run `python -m src.generate.run_generation` first.", ""]
    md += ["## Failure cases (gold benchmark, test)", ""] + failure_cases(gold, methods)
    (OUT / "verifier_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(markdown_table([{c: r.get(c) for c in gold_cols[:9]} for r in gold["rows"]]))
    if inj_rows:
        print(markdown_table(inj_rows))
    print(f"\nWrote {OUT / 'verifier_results.md'}")


if __name__ == "__main__":
    main()
