// Compare retrievers: one claim, five rankings side by side.

import { api } from "../api.js";
import { $$, clear, fmt, fmtMs, h, icon, labelBadge, normSpace, toast } from "../dom.js";
import { navigate, store } from "../state.js";
import { openPaper, pickClaim } from "../shared.js";

const S = { els: {}, pickedQid: null, pickedText: null, busy: false, last: null };

const SUGGESTIONS = [
  { qid: "660", why: "BM25 finds the paper on one rare word; dense retrieval misses it" },
  { qid: "1", why: "No word is shared with the paper; only dense retrieval finds it" },
  { qid: "53", why: "Every retriever agrees" },
];

export function mount(root) {
  const e = S.els;
  e.query = h("textarea", { class: "input compare-input", rows: 2, "aria-label": "Claim to compare retrievers on",
    placeholder: "Type a claim, or pick a SciFact claim to see which papers experts judged relevant",
    onkeydown: (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey || !ev.shiftKey)) { ev.preventDefault(); run(); } } });
  e.chunking = h("select", { class: "input", "aria-label": "Chunks", onchange: () => store.set("compare-chunking", e.chunking.value) },
    h("option", { value: "whole", text: "Whole abstracts" }), h("option", { value: "sent", text: "3-sentence windows (best window per paper)" }));
  e.chunking.value = store.get("compare-chunking", "whole");
  e.k = h("select", { class: "input", "aria-label": "Papers per retriever" },
    ...[5, 10, 20].map((k) => h("option", { value: k, text: `Top ${k}` })));
  e.k.value = "10";
  e.run = h("button", { class: "btn btn-primary", type: "button", onclick: run }, "Compare");
  e.out = h("div", { "aria-live": "polite" });
  root.append(
    h("div", { class: "view-head" },
      h("h1", { id: "h-compare", text: "Compare retrievers" }),
      h("p", { text: "The same claim through tf-idf, BM25, dense retrieval and both hybrids. For a SciFact claim, the papers experts judged relevant are marked. Point at a paper to find it in the other columns; click it to see why it scored as it did." })),
    h("div", { class: "panel compare-box" },
      e.query,
      h("div", { class: "row", style: { marginTop: "10px" } },
        h("button", { class: "btn", type: "button", onclick: randomClaim }, icon("dice"), "Random SciFact claim"),
        h("button", { class: "btn", type: "button", onclick: browse }, icon("list"), "Browse claims"),
        h("div", { style: { minWidth: "220px" } }, e.chunking),
        h("div", { style: { width: "110px" } }, e.k),
        h("div", { class: "row-end" }, e.run))),
    e.out);
  renderEmpty();
}

export function enter(params) {
  if (params.q) {
    S.els.query.value = params.q;
    S.pickedText = params.q;
    S.pickedQid = params.qid || null;
    if (params.chunking) S.els.chunking.value = params.chunking;
    history.replaceState(null, "", "#/compare");
    run();
  }
}

async function randomClaim() {
  try { const c = await api.randomClaim({ split: "test" }); set(c.qid, c.text); } catch (err) { toast(err.message); }
}
async function browse() { const c = await pickClaim(); if (c) set(c.qid, c.text); }
function set(qid, text) { S.els.query.value = text; S.pickedQid = qid; S.pickedText = text; S.els.query.focus(); }

function renderEmpty() {
  const list = h("div", { class: "examples" });
  clear(S.els.out).append(h("div", { class: "panel empty", style: { marginTop: "22px" } },
    h("h2", { text: "Pick a claim to compare" }),
    h("p", { text: "Three SciFact claims that show where the retrievers differ:" }), list));
  Promise.all(SUGGESTIONS.map((s) => api.claim(s.qid).then((c) => ({ ...s, c })).catch(() => null))).then((rows) => {
    for (const r of rows.filter(Boolean)) {
      list.append(h("button", { class: "example", type: "button", onclick: () => { set(r.c.qid, r.c.text); run(); } },
        h("span", { text: r.c.text }), h("span", { class: "hint", style: { display: "block", fontFamily: "var(--sans)" }, text: r.why })));
    }
  });
}

async function run() {
  if (S.busy) return;
  const query = S.els.query.value.trim();
  if (!query) { toast("Type a claim first."); return; }
  const qid = S.pickedText && normSpace(query) === normSpace(S.pickedText) ? S.pickedQid : null;
  S.busy = true;
  S.els.run.disabled = true;
  clear(S.els.run).append(icon("spinner"), "Comparing");
  try {
    const res = await api.compare({ query, qid, chunking: S.els.chunking.value, k: parseInt(S.els.k.value, 10) });
    S.last = res;
    render(res);
  } catch (err) {
    clear(S.els.out).append(h("div", { class: "notice notice-bad", style: { marginTop: "18px" } }, icon("alert"), h("span", { text: err.message })));
  } finally {
    S.busy = false;
    S.els.run.disabled = false;
    clear(S.els.run).append("Compare");
  }
}

function render(res) {
  const out = clear(S.els.out);
  const meta = h("div", { class: "terms-line" });
  if (res.claim) {
    meta.append(h("span", { text: `SciFact claim ${res.claim.qid}` }), labelBadge(res.claim.label),
      h("span", { class: "muted", text: `${res.claim.papers.length} judged ${res.claim.papers.length === 1 ? "paper" : "papers"}.` }));
  } else {
    meta.append(h("span", { text: "Free-text claim: no judged papers to mark." }));
  }
  meta.append(h("span", { class: "muted", style: { marginLeft: "8px" }, text: "Index terms after the tokenizer:" }),
    ...(res.query_terms.length ? res.query_terms.map((t) => h("span", { class: "term-chip", text: t })) : [h("span", { class: "muted", text: "none" })]));
  out.append(meta);

  const cols = Object.values(res.retrievers);
  const ranks = cols.filter((c) => c.available && c.first_relevant).map((c) => c.first_relevant);
  const best = ranks.length ? Math.min(...ranks) : null;
  const grid = h("div", { class: "cols", style: { "--n": cols.length } });
  for (const col of cols) grid.append(column(col, res, best));
  out.append(h("div", { class: "cols-wrap" }, grid));
  if (res.claim) {
    out.append(h("p", { class: "hint", style: { marginTop: "10px" },
      text: "Metrics are for this one claim: nDCG@10 and the rank of the first judged-relevant paper among the top 100." }));
  }
  out.append(h("div", { class: "row", style: { marginTop: "14px" } },
    h("button", { class: "btn btn-small", type: "button", onclick: () => navigate("check", { q: res.query, qid: res.claim ? res.claim.qid : "", run: "1" }) },
      "Check this claim with a cited answer")));
}

function column(col, res, best) {
  const panel = h("section", { class: "panel col", "aria-label": col.label });
  const head = h("div", { class: "col-head" }, h("div", { class: "row", style: { gap: "6px", justifyContent: "space-between" } },
    h("h3", { text: col.label }),
    col.available && best !== null && col.first_relevant === best ? h("span", { class: "badge badge-best", title: "Ranked a judged-relevant paper highest", text: "best" }) : null));
  panel.append(head);
  if (!col.available) {
    panel.append(h("p", { class: "col-off", text: col.message }));
    return panel;
  }
  const stats = h("div", { class: "col-stats" });
  if (res.claim) {
    stats.append(h("span", {}, "First relevant paper: ", h("strong", { text: col.first_relevant ? `rank ${col.first_relevant}` : "not in top 100" })));
    stats.append(h("span", {}, "nDCG@10 ", h("strong", { text: fmt(col.metrics["nDCG@10"]) })));
  }
  stats.append(h("span", { class: "muted", text: `ranked in ${fmtMs(col.ms)}` }));
  head.append(stats);
  const list = h("ol", { class: "col-list" });
  for (const p of col.papers) {
    const row = h("button", {
      class: `paper-row${p.relevant ? " rel" : ""}`, type: "button", "data-doc": p.doc_id,
      onmouseenter: () => link(p.doc_id, true), onmouseleave: () => link(p.doc_id, false),
      onfocus: () => link(p.doc_id, true), onblur: () => link(p.doc_id, false),
      onclick: () => openPaper(p.doc_id, { query: res.query, retriever: col.name, chunkId: p.doc_id, relevant: p.relevant,
        note: res.chunking === "sent" ? "The breakdown below scores the whole abstract; the ranking itself used the paper's best 3-sentence window." : null }),
    },
    h("span", { class: "paper-rank", text: p.rank }),
    h("span", {},
      h("span", { class: "paper-title", text: p.title }),
      h("span", { class: "paper-meta" },
        p.relevant ? h("span", { class: "badge badge-rel", text: "judged relevant" }) : null,
        h("span", { class: "num", text: `score ${fmt(p.score, col.name.startsWith("hybrid") ? 4 : 3)}` }),
        p.terms && p.terms.length ? h("span", { text: p.terms.map((t) => `${t.term} ${Math.round(t.share * 100)}%`).join(", ") }) : null)));
    list.append(h("li", {}, row));
  }
  panel.append(list);
  return panel;
}

function link(docId, on) {
  $$(`.paper-row[data-doc="${CSS.escape(docId)}"]`).forEach((el) => el.classList.toggle("linked", on));
}
