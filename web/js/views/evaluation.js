// Evaluation: the tables and figures produced by the evaluation scripts, made interactive.

import { api } from "../api.js";
import { groupedBars, lineChart, rankScatter } from "../charts.js";
import { append, clear, fmt, fmtP, h, icon, pct, table } from "../dom.js";
import { navigate } from "../state.js";

const S = { root: null, data: null, loaded: false };

const SHORT = {
  "tf-idf lnc.ltc (zones)": "tf-idf (zones)",
  "tf-idf lnc.ltc (one field)": "tf-idf (one field)",
  "BM25": "BM25",
  "Dense (MiniLM)": "Dense (MiniLM)",
  "Hybrid RRF (BM25 + dense)": "Hybrid RRF",
  "Hybrid weighted (BM25 + dense)": "Hybrid weighted",
  "Hybrid RRF (tf-idf + dense)": "Hybrid RRF (tf-idf)",
};
const short = (name) => SHORT[name] || name;
const CORRUPTION_LABEL = { original: "Untouched", negate: "Negated", swap_citation: "Swapped citation", add_fabricated: "Fabricated" };

export function mount(root) { S.root = root; }

export function enter() { if (!S.loaded) load(); }

async function load() {
  const root = clear(S.root);
  root.append(h("div", { class: "view-head" }, h("h1", { id: "h-evaluation", text: "Evaluation" })), h("p", { class: "muted" }, icon("spinner"), " Loading results"));
  try {
    S.data = await api.evaluation();
    S.loaded = true;
    render(S.data);
  } catch (err) {
    clear(root).append(h("h1", { id: "h-evaluation", text: "Evaluation" }), h("p", { class: "notice notice-bad", text: err.message }));
  }
}

function section(title, lede, ...children) {
  return h("section", { class: "section" }, h("h2", { text: title }), lede ? h("p", { class: "lede", text: lede }) : null, ...children);
}

function card(title, sub, body) {
  return h("div", { class: "panel chart-card" }, h("h3", { text: title }), sub ? h("p", { class: "sub", text: sub }) : null, body);
}

function render(d) {
  const root = clear(S.root);
  const head = h("div", { class: "view-head" }, h("h1", { id: "h-evaluation", text: "Evaluation" }),
    h("button", { class: "btn btn-small", type: "button", onclick: () => { S.loaded = false; load(); } }, "Reload results"));
  root.append(head);
  if (!d.available) {
    root.append(h("div", { class: "panel empty" },
      h("h2", { text: "No results yet" }),
      h("p", { text: "Run the evaluation once; this page then shows its tables and charts." }),
      h("code", { class: "cmd", style: { marginTop: "14px", textAlign: "left", display: "inline-block" }, text: "python scripts/run_all.py" })));
    return;
  }
  const main = d.main;
  const best = main.reduce((a, b) => (b["nDCG@10"] > a["nDCG@10"] ? b : a));
  const bm25 = main.find((r) => r.Retriever === "BM25");
  const dense = main.find((r) => r.Retriever.startsWith("Dense"));
  root.append(h("p", { class: "eval-lede" },
    "On the 300 SciFact test claims, ", h("strong", { text: short(best.Retriever).toLowerCase() }), " retrieval reaches nDCG@10 ",
    h("strong", { text: fmt(best["nDCG@10"]) }),
    bm25 ? `, against ${fmt(bm25["nDCG@10"])} for BM25` : "", dense ? ` and ${fmt(dense["nDCG@10"])} for dense retrieval alone` : "",
    ". Every setting was chosen on the 809 train claims; the test claims were only used to report."));
  root.append(statStrip(d, best, bm25));

  // ---------------------------------------------------------------- retrieval
  const chart = h("div");
  root.append(section("Retrieval", "Papers ranked by each retriever over whole abstracts, compared with SciFact's relevance judgments.",
    card("Ranking quality by retriever", "Values shown for nDCG@10; hover a bar for the others.", chart),
    h("div", { style: { marginTop: "14px" } }, table([
      { key: "Retriever", label: "Retriever" },
      { key: "P@5", label: "P@5", num: true, format: (v) => fmt(v) },
      { key: "P@10", label: "P@10", num: true, format: (v) => fmt(v) },
      { key: "R@10", label: "R@10", num: true, format: (v) => fmt(v) },
      { key: "MRR@10", label: "MRR@10", num: true, format: (v) => fmt(v) },
      { key: "nDCG@10", label: "nDCG@10", num: true, format: (v) => fmt(v) },
      { key: "nDCG@10 95% CI", label: "95% CI (bootstrap)", num: true },
    ], main, { rowClass: (r) => (r === best ? "best" : "") }))));
  groupedBars(chart, {
    categories: main.map((r) => short(r.Retriever)),
    series: [
      { name: "nDCG@10", values: main.map((r) => r["nDCG@10"]), color: "--series-1" },
      { name: "MRR@10", values: main.map((r) => r["MRR@10"]), color: "--series-2" },
      { name: "R@10", values: main.map((r) => r["R@10"]), color: "--series-3" },
    ],
    yMax: 1, labelSeries: [0], ariaLabel: "nDCG@10, MRR@10 and R@10 by retriever",
  });

  // ---------------------------------------------------------------- significance
  if (d.significance) {
    root.append(section("Are the differences real?",
      "Paired randomization test on per-claim nDCG@10 (10,000 permutations). A small p means the gap is unlikely to come from which claims happened to be in the test set.",
      table([
        { key: "A", label: "Comparison", format: (v, r) => `${short(r.A)} vs ${short(r.B)}` },
        { key: "chunking", label: "Chunks", format: (v) => (v === "whole" ? "whole abstracts" : "3-sentence windows") },
        { key: "nDCG@10 A", label: "A", num: true, format: (v) => fmt(v) },
        { key: "nDCG@10 B", label: "B", num: true, format: (v) => fmt(v) },
        { key: "diff", label: "Difference", num: true, format: (v) => `${v > 0 ? "+" : ""}${fmt(v)}` },
        { key: "p (randomization, 10k)", label: "p", num: true, format: (v) => fmtP(v) },
      ], d.significance, { rowClass: (r) => (r["p (randomization, 10k)"] < 0.05 ? "" : "dim"),
        caption: "Grey rows: p ≥ 0.05, no reliable difference." })));
  }

  // ---------------------------------------------------------------- tuning and ablations
  const abl = d.ablations || [];
  const tune = d.tuning || [];
  const tw = abl.filter((r) => r.ablation === "title weight" && r.chunking === "whole" && r.retriever.startsWith("tf-idf lnc.ltc (zones)"));
  const titleChart = h("div");
  const alphaChart = h("div");
  const chunkChart = h("div");
  const grid = h("div", { class: "grid-2" },
    card("tf-idf title weight", "nDCG@10 as the title zone's weight grows (body weight 1, whole abstracts)", titleChart),
    card("Weighted hybrid: share of BM25", "alpha × BM25 + (1 − alpha) × dense, after min-max scaling; train claims", alphaChart));
  const tok = abl.filter((r) => r.ablation === "tokenizer");
  const k1b = abl.filter((r) => r.ablation === "BM25 k1/b");
  root.append(section("Tuning and ablations",
    "What each design choice is worth. Weights were picked where the train curve peaks; the test curve is shown to check that the choice transfers.",
    grid,
    h("div", { style: { marginTop: "16px" } }, card("Whole abstracts vs 3-sentence windows", "Test nDCG@10; with windows, a paper's score is its best window", chunkChart)),
    h("div", { class: "grid-2", style: { marginTop: "16px" } },
      h("div", {}, h("h3", { style: { marginBottom: "8px" }, text: "Tokenizer" }), table([
        { key: "setting", label: "Setting" }, { key: "retriever", label: "Retriever", format: (v) => short(v) },
        { key: "nDCG@10 train", label: "Train", num: true, format: (v) => fmt(v) },
        { key: "nDCG@10 test", label: "Test", num: true, format: (v) => fmt(v) },
      ], tok, { caption: "nDCG@10, whole abstracts" })),
      h("div", {}, h("h3", { style: { marginBottom: "8px" }, text: "BM25 parameters" }), table([
        { key: "setting", label: "Setting" },
        { key: "nDCG@10 train", label: "Train", num: true, format: (v) => fmt(v) },
        { key: "nDCG@10 test", label: "Test", num: true, format: (v) => fmt(v) },
      ], k1b, { caption: "The defaults k1 = 1.2, b = 0.75 are used in the main table" })))));
  if (tw.length) {
    const xs = tw.map((r) => Number(String(r.setting).match(/w_title=([\d.]+)/)[1]));
    lineChart(titleChart, {
      xs, xName: "title weight", xLabel: "title weight (body = 1)", yLabel: "nDCG@10",
      series: [{ name: "train", values: tw.map((r) => r["nDCG@10 train"]), color: "--series-1" },
        { name: "test", values: tw.map((r) => r["nDCG@10 test"]), color: "--series-2" }],
      markX: d.tuned.zone_weights_whole ? d.tuned.zone_weights_whole[0] : null, markLabel: "chosen on train",
      xTicks: [0, 0.5, 1, 1.5, 2, 2.5, 3], ariaLabel: "nDCG@10 against title weight",
    });
  }
  const alphaRows = (ch) => tune.filter((r) => r.param === "hybrid_alpha" && r.chunking === ch);
  if (alphaRows("whole").length) {
    const xs = alphaRows("whole").map((r) => r.value);
    lineChart(alphaChart, {
      xs, xName: "alpha", xLabel: "alpha (0 = dense only, 1 = BM25 only)", yLabel: "nDCG@10",
      series: [{ name: "whole abstracts", values: alphaRows("whole").map((r) => r["nDCG@10"]), color: "--series-1" },
        { name: "3-sentence windows", values: alphaRows("sent").map((r) => r["nDCG@10"]), color: "--series-3" }],
      markX: d.tuned.hybrid_alpha_whole, markLabel: "chosen (whole)", xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], ariaLabel: "nDCG@10 against alpha",
    });
  } else {
    alphaChart.append(h("p", { class: "muted", text: "Run the retrieval evaluation with dense retrieval to fill this chart." }));
  }
  const ch = abl.filter((r) => r.ablation === "chunking");
  const retr = [...new Set(ch.map((r) => r.retriever))];
  if (retr.length) {
    groupedBars(chunkChart, {
      categories: retr.map(short),
      series: [
        { name: "whole abstracts", values: retr.map((n) => (ch.find((r) => r.retriever === n && r.chunking === "whole") || {})["nDCG@10 test"] ?? null), color: "--series-1" },
        { name: "3-sentence windows", values: retr.map((n) => (ch.find((r) => r.retriever === n && r.chunking === "sent") || {})["nDCG@10 test"] ?? null), color: "--series-2" },
      ],
      yMax: 1, labelSeries: [0, 1], height: 260, ariaLabel: "nDCG@10 by chunking",
    });
  }

  // ---------------------------------------------------------------- claim types
  if (d.by_claim_type) {
    const cols = Object.keys(d.by_claim_type[0]).filter((k) => k !== "Retriever");
    const names = { SUPPORT: "Supported claims", CONTRADICT: "Contradicted claims", NEI: "Paper holds no evidence" };
    root.append(section("Which claims are hard to retrieve for",
      "Test nDCG@10 by SciFact claim type. When the cited paper holds no evidence for the claim (NEI), it also tends to share fewer words with it.",
      table([{ key: "Retriever", label: "Retriever", format: (v) => short(v) },
        ...cols.map((c) => ({ key: c, label: `${names[c.split(" ")[0]] || c} ${c.match(/\(n=\d+\)/) ? c.match(/\(n=\d+\)/)[0] : ""}`, num: true,
          format: (v) => h("span", { class: "cell-bar", style: { "--w": `${Math.round(v * 100)}%`, display: "block" } }, h("span", { text: fmt(v) })) }))],
      d.by_claim_type)));
  }

  // ---------------------------------------------------------------- winners
  if (d.winners && d.winners.length) root.append(winnerSection(d));

  // ---------------------------------------------------------------- verifier
  root.append(verifierSection(d));

  // ---------------------------------------------------------------- figures
  if (d.figures && d.figures.length) {
    root.append(section("Figures for the report", "The same results as PNG files (SVG versions sit next to them in results/).",
      h("div", { class: "figure-links" }, d.figures.map((f) => h("a", { class: "btn btn-small", href: `/files/results/${f}`, target: "_blank", rel: "noopener" }, icon("external"), f)))));
  }
}

function statStrip(d, best, bm25) {
  const tiles = [];
  const tile = (label, value, note) => h("div", { class: "stat" }, h("div", { class: "label", text: label }), h("div", { class: "value", text: value }), h("div", { class: "note", text: note }));
  tiles.push(tile("Best retrieval, nDCG@10", fmt(best["nDCG@10"]), `${short(best.Retriever)}; BM25 scores ${bm25 ? fmt(bm25["nDCG@10"]) : "–"}`));
  const sig = (d.significance || []).find((r) => r.chunking === "whole" && r.A.startsWith("Hybrid RRF (BM25") && r.B === "BM25");
  if (sig) tiles.push(tile("Hybrid RRF over BM25", `${sig.diff > 0 ? "+" : ""}${fmt(sig.diff)}`, `nDCG@10 gain, randomization test p ${sig["p (randomization, 10k)"] < 0.001 ? "< 0.001" : `= ${fmtP(sig["p (randomization, 10k)"])}`}`));
  const gold = (d.verifier_gold || []).filter((r) => !/Baseline/.test(r.checker));
  const base = (d.verifier_gold || []).find((r) => /Baseline/.test(r.checker));
  if (gold.length) {
    const bestV = gold.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
    tiles.push(tile("Citation checker, macro-F1", fmt(bestV.macro_f1), `${bestV.checker}; flagging everything scores ${base ? fmt(base.macro_f1) : "–"}`));
  }
  const inj = (d.verifier_injected_by_kind || [])[0];
  if (inj) {
    const key = (prefix) => Object.keys(inj).find((k) => k.startsWith(prefix));
    tiles.push(tile("Swapped citations caught", pct(inj[key("swap_citation")]),
      `${inj.checker} checker; fabricated ${pct(inj[key("add_fabricated")])}, negated ${pct(inj[key("negate")])}`));
  }
  return h("div", { class: "stats" }, tiles);
}

function winnerSection(d) {
  const rows = d.winners;
  const groups = [
    { key: "sparse", name: "BM25 ranked it higher", color: "--series-1" },
    { key: "dense", name: "dense ranked it higher", color: "--series-2" },
    { key: "tie", name: "same rank", color: "--series-3" },
  ];
  const buckets = new Map();
  for (const r of rows) {
    const key = `${r.rank_bm25}|${r.rank_dense}`;
    if (!buckets.has(key)) buckets.set(key, { x: r.rank_bm25, y: r.rank_dense, n: 0, group: r.winner, items: [] });
    const b = buckets.get(key);
    b.n += 1;
    b.items.push(r);
  }
  const chart = h("div");
  const picked = h("div", { class: "picked-claims", "aria-live": "polite" },
    h("p", { class: "hint", text: "Click a dot to list its claims here, then open any of them in the retriever comparison." }));
  const summary = d.winner_summary || [];
  const label = { sparse: "BM25 ranked it higher", tie: "Same rank", dense: "Dense ranked it higher" };
  const sec = section("Sparse vs dense, claim by claim",
    "Each dot places a claim by the rank of its first relevant paper under BM25 (across) and MiniLM (up), on a log scale; higher means a worse rank, and bigger dots hold more claims. Dots above the diagonal are claims where BM25 found the paper earlier.",
    h("div", { class: "grid-2" },
      card("First relevant rank, BM25 vs dense", `${rows.length} test claims, whole abstracts`, chart),
      h("div", {},
        table([
          { key: "winner", label: "Outcome", format: (v) => label[v] },
          { key: "claims", label: "Claims", num: true },
          { key: "mean_overlap", label: "Claim words found in the paper", num: true, format: (v) => pct(v) },
        ], summary, { caption: d.winner_p !== null && d.winner_p !== undefined
          ? `Claims BM25 wins share more words with the relevant paper than claims dense wins (permutation test, p ${d.winner_p < 0.001 ? "< 0.001" : `= ${fmtP(d.winner_p)}`}).` : "" }),
        picked)));
  rankScatter(chart, {
    points: [...buckets.values()], groups, xLabel: "BM25: rank of the first relevant paper", yLabel: "Dense: rank of the first relevant paper",
    ariaLabel: "Scatter of first relevant rank under BM25 and dense retrieval",
    onSelect: (p) => {
      append(clear(picked), [h("h3", { style: { margin: "14px 0 6px" }, text: `${p.n} ${p.n === 1 ? "claim" : "claims"}: BM25 rank ${p.x > 100 ? ">100" : p.x}, dense rank ${p.y > 100 ? ">100" : p.y}` }),
        h("ul", {}, p.items.slice(0, 12).map((r) => h("li", {},
          h("a", { class: "claim-link", href: "#", onclick: (e) => { e.preventDefault(); navigate("compare", { q: r.claim, qid: String(r.qid) }); } }, r.claim),
          h("span", { class: "hint", text: ` (claim ${r.qid}, ${pct(r.overlap)} of its words in the paper)` })))),
        p.items.length > 12 ? h("p", { class: "hint", text: `and ${p.items.length - 12} more` }) : null]);
    },
  });
  return sec;
}

function verifierSection(d) {
  const sec = section("Citation verifier",
    "Each item pairs a claim with a paper: one SciFact experts marked as supporting it, one marked as contradicting it, one cited without evidence, or a wrong paper (the top BM25 paper not judged relevant). Only the first kind should pass. Thresholds come from train claims; scores are on test claims.");
  if (!d.verifier_available) {
    sec.append(h("div", { class: "notice" }, icon("info"), h("span", { text: "No verifier results yet. Run python -m src.eval.run_verifier_eval." })));
    return sec;
  }
  const gold = d.verifier_gold;
  const hasNli = gold.some((r) => /NLI/.test(r.checker));
  if (!hasNli) {
    sec.append(h("div", { class: "notice notice-warn" }, icon("info"),
      h("span", { text: "Only the cosine checker has results so far. Run python -m src.eval.run_verifier_eval on a machine that can download the NLI model to add the NLI and cascade rows." })));
  }
  sec.append(table([
    { key: "checker", label: "Checker" },
    { key: "threshold (train)", label: "Flags a sentence when" },
    { key: "precision", label: "Precision", num: true, format: (v) => fmt(v) },
    { key: "recall", label: "Recall", num: true, format: (v) => fmt(v) },
    { key: "f1", label: "F1", num: true, format: (v) => fmt(v) },
    { key: "macro_f1", label: "Macro-F1", num: true, format: (v) => fmt(v) },
    { key: "AUROC", label: "AUROC", num: true, format: (v) => (v === null || v === undefined ? "–" : fmt(v)) },
    { key: "NLI calls", label: "NLI calls", num: true },
  ], gold, { rowClass: (r) => (/Baseline/.test(r.checker) ? "dim" : ""),
    caption: "Precision, recall and F1 are for the unsupported flag. Most items are unsupported, so flagging everything (grey row) already scores a high F1; macro-F1 averages both classes and is what the thresholds were tuned on." }));

  const kinds = d.verifier_by_kind || [];
  if (kinds.length) {
    const cols = Object.keys(kinds[0]).filter((k) => k !== "checker");
    const nice = { support: "Supporting paper", contradict: "Contradicting paper", cited_no_evidence: "Cited, no evidence", wrong_paper: "Wrong paper" };
    const chart = h("div");
    sec.append(h("div", { style: { marginTop: "16px" } }, card("How often each kind of item is flagged",
      "Share of test pairs flagged unsupported. For supporting papers this is the false-alarm rate; for the other three, higher is better.", chart)));
    const colors = ["--series-1", "--series-2", "--series-3", "--series-4"];
    groupedBars(chart, {
      categories: cols.map((c) => `${nice[c.split(" ")[0]] || c} ${(c.match(/\(n=\d+\)/) || [""])[0]}`),
      series: kinds.map((r, i) => ({ name: r.checker, values: cols.map((c) => r[c]), color: colors[i % 4] })),
      yMax: 1, labelSeries: kinds.map((_, i) => i), format: (v) => pct(v), tickFormat: (t) => pct(t), height: 260,
      ariaLabel: "Flag rate by item kind",
    });
  }
  const extra = h("div", { class: "grid-2", style: { marginTop: "16px" } });
  if (d.verifier_injected_by_kind && d.verifier_injected_by_kind.length) {
    const r0 = d.verifier_injected_by_kind[0];
    const cols = Object.keys(r0).filter((k) => !["answers", "checker"].includes(k));
    extra.append(h("div", {}, h("h3", { style: { marginBottom: "8px" }, text: "Corrupted answers" }),
      table([{ key: "checker", label: "Checker" }, ...cols.map((c) => ({ key: c, label: CORRUPTION_LABEL[c.split(" ")[0]] || c,
        title: c, num: true, format: (v) => pct(v) }))],
        d.verifier_injected_by_kind, { caption: `Share flagged (${cols.map((c) => `${CORRUPTION_LABEL[c.split(" ")[0]] || c}: ${(c.match(/n=(\d+)/) || [, "?"])[1]}`).join(", ")} sentences). Untouched sentences should pass; every corrupted one should be flagged.` })));
  }
  if (d.verifier_windows && d.verifier_windows.length) {
    extra.append(h("div", {}, h("h3", { style: { marginBottom: "8px" }, text: "How much evidence to compare against" }),
      table([
        { key: "evidence window", label: "Evidence unit" },
        { key: "AUROC train", label: "AUROC train", num: true, format: (v) => fmt(v) },
        { key: "AUROC test", label: "AUROC test", num: true, format: (v) => fmt(v) },
        { key: "used", label: "Used", format: (v) => (v ? "yes" : "") },
      ], d.verifier_windows, { caption: "Cosine checker; the unit with the best train AUROC is used" })));
  }
  if (extra.childElementCount) sec.append(extra);
  return sec;
}
