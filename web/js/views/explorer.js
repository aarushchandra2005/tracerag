// Inside the index: tokenizer steps, dictionary + postings, and full score arithmetic.

import { api } from "../api.js";
import { clear, fmt, h, icon, table, toast } from "../dom.js";
import { openPaper } from "../shared.js";

const S = { els: {}, termOffset: 0, tokTimer: null };
const PAGE = 25;

const TABS = [["tok", "1. Tokenizer"], ["post", "2. Dictionary and postings"], ["score", "3. Score breakdown"]];

export function mount(root) {
  S.tools = { tok: tokenizerTool(), post: postingsTool(), score: scoreTool() };
  S.tabBar = h("div", { class: "seg", role: "tablist", "aria-label": "Index tools", style: { marginBottom: "16px" } },
    TABS.map(([key, text]) => h("button", { type: "button", role: "tab", id: `tab-${key}`, "aria-controls": `panel-${key}`,
      "aria-selected": "false", onclick: () => showTab(key) }, text)));
  for (const [key] of TABS) {
    S.tools[key].id = `panel-${key}`;
    S.tools[key].setAttribute("role", "tabpanel");
    S.tools[key].setAttribute("aria-labelledby", `tab-${key}`);
  }
  root.append(
    h("div", { class: "view-head" },
      h("h1", { id: "h-index", text: "Inside the index" }),
      h("p", { text: "The sparse retrievers run on an inverted index built for this project. Follow a piece of text into index terms, look up a term's dictionary entry and postings list, and recompute a paper's score by hand." })),
    S.tabBar,
    h("div", { class: "explorer-grid" }, S.tools.tok, S.tools.post, S.tools.score));
  showTab("tok");
  runTokenize();
  lookUp(0);
  api.indexStats().then(renderStats).catch(() => {});
}

function showTab(key) {
  for (const [k] of TABS) {
    S.tools[k].hidden = k !== key;
    document.getElementById(`tab-${k}`).setAttribute("aria-selected", String(k === key));
  }
}

export function enter(params) {
  if (params.query && params.chunk) {
    S.els.sq.value = params.query;
    S.els.sc.value = params.chunk;
    S.els.sr.value = params.retriever || "tfidf";
    showTab("score");
    explain();
    history.replaceState(null, "", "#/index");
  } else if (params.term) {
    S.els.word.value = params.term;
    showTab("post");
    lookUp(0);
    history.replaceState(null, "", "#/index");
  }
}

// ================================================================== 1. tokenizer
function tokenizerTool() {
  const e = S.els;
  e.tokText = h("textarea", { class: "input", rows: 2, style: { fontFamily: "var(--serif)", fontSize: "1.02rem" }, "aria-label": "Text to tokenize",
    oninput: () => { clearTimeout(S.tokTimer); S.tokTimer = setTimeout(runTokenize, 250); } });
  e.tokText.value = "Tumors were growing more slowly in ALDH1-positive mice, weren't they?";
  e.stem = h("input", { type: "checkbox", checked: true, onchange: runTokenize });
  e.stop = h("input", { type: "checkbox", checked: true, onchange: runTokenize });
  e.tokOut = h("div", { class: "tool-out" });
  return h("section", { class: "panel tool", "aria-labelledby": "h-tok" },
    h("h2", { id: "h-tok", text: "How text becomes index terms" }),
    h("p", { class: "lede", text: "Text is Unicode-normalised and case-folded, split into runs of letters and digits, stripped of NLTK's English stopwords, and reduced to Porter stems. Documents and queries go through exactly the same steps; switching a step off here is what the tokenizer ablation measures." }),
    e.tokText,
    h("div", { class: "row", style: { marginTop: "10px", gap: "18px" } },
      h("label", { class: "check" }, e.stem, "Porter stemming"),
      h("label", { class: "check" }, e.stop, "Remove stopwords")),
    e.tokOut);
}

async function runTokenize() {
  const e = S.els;
  try {
    const r = await api.tokenize({ text: e.tokText.value, stem: e.stem.checked, stopwords: e.stop.checked });
    const out = clear(e.tokOut);
    out.append(h("p", { class: "result-line" }, h("span", { class: "muted", text: "Index terms: " }),
      ...(r.terms.length ? r.terms.flatMap((t) => [h("span", { class: "term-chip", text: t }), " "]) : [h("span", { class: "muted", text: "none" })])));
    out.append(h("div", { style: { marginTop: "10px" } }, table([
      { key: "token", label: "Token (case-folded)", format: (v) => h("code", { text: v }) },
      { key: "stopword", label: "Stopword", format: (v) => (v ? "yes" : "no") },
      { key: "stem", label: "Porter stem", format: (v) => h("code", { text: v }) },
      { key: "term", label: "Index term", format: (v) => (v ? h("code", { text: v }) : h("span", { class: "muted", text: "removed" })) },
    ], r.tokens, { rowClass: (row) => (row.term ? "" : "dim") })));
  } catch (err) { clear(e.tokOut).append(h("p", { class: "notice", text: err.message })); }
}

// ================================================================== 2. dictionary and postings
function postingsTool() {
  const e = S.els;
  e.word = h("input", { class: "input", value: "tumor", "aria-label": "Word to look up",
    onkeydown: (ev) => { if (ev.key === "Enter") lookUp(0); } });
  e.zone = h("select", { class: "input", onchange: () => lookUp(0) },
    h("option", { value: "all", text: "Title + body (merged field)" }), h("option", { value: "body", text: "Body zone" }), h("option", { value: "title", text: "Title zone" }));
  e.tchunk = h("select", { class: "input", onchange: () => lookUp(0) },
    h("option", { value: "whole", text: "Whole abstracts" }), h("option", { value: "sent", text: "3-sentence windows" }));
  e.sort = h("select", { class: "input", onchange: () => lookUp(0) },
    h("option", { value: "tf", text: "Highest tf first" }), h("option", { value: "chunk", text: "Stored order" }));
  e.termOut = h("div", { class: "tool-out" });
  e.stats = h("div", { class: "tool-out" });
  return h("section", { class: "panel tool", "aria-labelledby": "h-post" },
    h("h2", { id: "h-post", text: "Dictionary and postings" }),
    h("p", { class: "lede", text: "Each zone has a sorted dictionary (term, document frequency, offset) and a postings list of (chunk, term frequency) pairs, stored in chunk order. Zones let tf-idf weigh titles and bodies separately; BM25 uses the merged field." }),
    h("div", { class: "tool-form" },
      h("label", { class: "field grow" }, h("span", { text: "Word" }), e.word),
      h("label", { class: "field" }, h("span", { text: "Zone" }), e.zone),
      h("label", { class: "field" }, h("span", { text: "Chunks" }), e.tchunk),
      h("label", { class: "field" }, h("span", { text: "Order" }), e.sort),
      h("button", { class: "btn btn-primary", type: "button", onclick: () => lookUp(0) }, "Look up")),
    e.termOut, e.stats);
}

async function lookUp(offset) {
  const e = S.els;
  const word = e.word.value.trim();
  if (!word) return;
  try {
    const r = await api.term({ word, chunking: e.tchunk.value, zone: e.zone.value, sort: e.sort.value, limit: PAGE, offset });
    S.termOffset = offset;
    renderTerm(r);
  } catch (err) { clear(e.termOut).append(h("p", { class: "notice notice-bad", text: err.message })); }
}

function renderTerm(r) {
  const out = clear(S.els.termOut);
  if (r.removed) {
    out.append(h("p", { class: "notice" }, icon("info"), h("span", { text: `“${r.word}” never reaches the index: ${r.reason}.` })));
    return;
  }
  out.append(h("p", { class: "result-line" }, h("span", { class: "muted", text: `“${r.word}” is stored as ` }), h("span", { class: "big-term", text: r.term }),
    r.extra_terms.length ? h("span", { class: "muted", text: ` (only the first term is looked up; the input also produced ${r.extra_terms.join(", ")})` }) : null));
  const zoneName = { title: "Title zone", body: "Body zone", all: "Title + body" };
  out.append(h("div", { style: { marginTop: "10px" } }, table([
    { key: "zone", label: "Zone", format: (v) => (v === r.zone ? h("strong", { text: zoneName[v] }) : zoneName[v]) },
    { key: "df", label: "df", num: true, title: "chunks containing the term" },
    { key: "idf", label: "log10(N / df)", num: true, format: (v, row) => (row.df ? fmt(v, 4) : "–") },
    { key: "bm25_idf", label: "BM25 idf", num: true, format: (v, row) => (row.df ? fmt(v, 4) : "–") },
    { key: "avg_len", label: "Avg. zone length", num: true, format: (v) => fmt(v, 1) },
  ], Object.entries(r.zones).map(([zone, z]) => ({ zone, ...z })), { caption: `N = ${r.n_chunks.toLocaleString()} chunks (${r.chunking === "whole" ? "whole abstracts" : "3-sentence windows"}).` })));
  if (!r.total) {
    out.append(h("p", { class: "notice", style: { marginTop: "12px" }, text: `“${r.term}” does not occur in this zone.` }));
    return;
  }
  out.append(h("h3", { style: { margin: "18px 0 8px" }, text: `Postings list: ${r.total.toLocaleString()} ${r.total === 1 ? "entry" : "entries"}` }));
  out.append(table([
    { key: "chunk_id", label: "Chunk", format: (v, row) => h("button", { class: "cite", type: "button", title: "Open the paper", text: v, onclick: () => openPaper(row.doc_id) }) },
    { key: "tf", label: "tf", num: true },
    { key: "length", label: "Zone length", num: true },
    { key: "w_lnc", label: "lnc weight", num: true, format: (v) => fmt(v, 4), title: "(1 + log10 tf) / ‖d‖" },
    { key: "title", label: "Paper title" },
  ], r.postings));
  const start = r.offset + 1;
  const end = r.offset + r.postings.length;
  out.append(h("div", { class: "pager" },
    h("button", { class: "btn btn-small", type: "button", disabled: r.offset === 0, onclick: () => lookUp(Math.max(0, r.offset - PAGE)) }, "Previous"),
    h("span", { text: `${start}–${end} of ${r.total.toLocaleString()}` }),
    h("button", { class: "btn btn-small", type: "button", disabled: end >= r.total, onclick: () => lookUp(r.offset + PAGE) }, "Next")));
}

function renderStats(stats) {
  const rows = [];
  for (const [chunking, s] of Object.entries(stats)) {
    for (const [zone, z] of Object.entries(s.zones)) {
      rows.push({ chunking: chunking === "whole" ? "Whole abstracts" : "3-sentence windows", chunks: s.chunks, zone: { title: "Title", body: "Body", all: "Title + body" }[zone], ...z });
    }
  }
  clear(S.els.stats).append(
    h("h3", { style: { margin: "6px 0 8px" }, text: "Index size" }),
    table([
      { key: "chunking", label: "Chunks" }, { key: "chunks", label: "Count", num: true, format: (v) => v.toLocaleString() },
      { key: "zone", label: "Zone" }, { key: "terms", label: "Dictionary terms", num: true, format: (v) => v.toLocaleString() },
      { key: "postings", label: "Postings", num: true, format: (v) => v.toLocaleString() },
      { key: "avg_len", label: "Avg. length", num: true, format: (v) => fmt(v, 1) },
    ], rows));
}

// ================================================================== 3. score breakdown
function scoreTool() {
  const e = S.els;
  e.sq = h("textarea", { class: "input", rows: 2, style: { fontFamily: "var(--serif)", fontSize: "1.02rem" }, "aria-label": "Query" });
  e.sq.value = "ALDH1 expression is associated with poorer prognosis in breast cancer.";
  e.sc = h("input", { class: "input mono", value: "45638119", "aria-label": "Chunk id", onkeydown: (ev) => { if (ev.key === "Enter") explain(); } });
  e.sr = h("select", { class: "input" },
    h("option", { value: "tfidf", text: "tf-idf (lnc.ltc, zones)" }), h("option", { value: "bm25", text: "BM25" }), h("option", { value: "dense", text: "Dense (MiniLM)" }));
  e.scoreOut = h("div", { class: "tool-out" });
  e.scoreTool = h("section", { class: "panel tool", "aria-labelledby": "h-score" },
    h("h2", { id: "h-score", text: "Score breakdown" }),
    h("p", { class: "lede", text: "Every number behind one chunk's score for one query, computed from the dictionary and postings, then checked against the retriever's own score. Chunk ids look like 45638119 (whole abstract) or 45638119_2 (window starting at sentence 2)." }),
    h("div", { class: "stack" },
      h("label", { class: "field" }, h("span", { text: "Query" }), e.sq),
      h("div", { class: "tool-form" },
        h("label", { class: "field", style: { width: "200px" } }, h("span", { text: "Chunk id" }), e.sc),
        h("label", { class: "field" }, h("span", { text: "Retriever" }), e.sr),
        h("button", { class: "btn btn-primary", type: "button", onclick: explain }, "Break down the score"))),
    e.scoreOut);
  return e.scoreTool;
}

async function explain() {
  const e = S.els;
  const body = { query: e.sq.value.trim(), chunk_id: e.sc.value.trim(), retriever: e.sr.value };
  if (!body.query || !body.chunk_id) { toast("Enter a query and a chunk id."); return; }
  const out = clear(e.scoreOut);
  out.append(h("p", { class: "muted" }, icon("spinner"), " Computing"));
  try {
    const r = await api.explain(body);
    clear(out);
    out.append(h("p", { class: "small muted", style: { marginBottom: "10px" } }, h("span", { class: "cite", text: r.chunk.chunk_id }), ` ${r.chunk.title}`));
    if (r.retriever === "tfidf") renderTfidf(out, r);
    else if (r.retriever === "bm25") renderBm25(out, r);
    else renderDense(out, r);
  } catch (err) { clear(out).append(h("p", { class: "notice notice-bad", text: err.message })); }
}

function check(r, value) {
  return h("p", { class: "result-line" },
    r.matches_library ? h("span", { class: "ok-text", text: "Matches " }) : h("span", { class: "bad-text", text: "Does not match " }),
    `the retriever's own score (${fmt(r.library_score, 6)}); hand computation ${fmt(value, 6)}.`);
}

function renderTfidf(out, r) {
  out.append(
    h("p", { class: "formula" }, "Query weights (ltc): ", h("var", { text: "w(t,q)" }), " = (1 + log₁₀ tf) × log₁₀(N / df), then divided by the vector length ‖q‖."),
    h("p", { class: "formula" }, "Document weights (lnc): ", h("var", { text: "w(t,d)" }), " = (1 + log₁₀ tf), divided by ‖d‖, which runs over every term of the zone, not only the query's."),
    h("p", { class: "formula" }, "Zone score: cos(q, d) = Σ ", h("var", { text: "w(t,q)" }), " × ", h("var", { text: "w(t,d)" }), " over the query terms."));
  for (const z of r.zones) {
    const name = { title: "Title zone", body: "Body zone", all: "Merged field" }[z.zone];
    out.append(h("div", { class: "zone-block" },
      h("h3", { style: { margin: "14px 0 8px" }, text: `${name} (weight ${z.weight})` }),
      table([
        { key: "term", label: "Term", format: (v) => h("code", { text: v }) },
        { key: "tf_q", label: "tf(q)", num: true }, { key: "df", label: "df", num: true },
        { key: "idf", label: "idf", num: true, format: (v) => fmt(v, 4) },
        { key: "w_q_raw", label: "w raw", num: true, format: (v) => fmt(v, 4), title: "(1 + log10 tf) × idf" },
        { key: "w_q", label: "w(t,q)", num: true, format: (v) => fmt(v, 4) },
        { key: "tf_d", label: "tf(d)", num: true },
        { key: "w_d_raw", label: "1 + log tf(d)", num: true, format: (v) => fmt(v, 4) },
        { key: "w_d", label: "w(t,d)", num: true, format: (v) => fmt(v, 4) },
        { key: "product", label: "Product", num: true, format: (v) => fmt(v, 4) },
      ], z.rows, {
        rowClass: (row) => (row.tf_d ? "" : "dim"),
        caption: `N = ${z.n_docs.toLocaleString()}. ‖q‖ = ${fmt(z.query_norm, 4)}; ‖d‖ = ${fmt(z.doc_norm, 4)} over ${z.doc_terms} distinct terms in this zone. cos = ${fmt(z.cosine, 4)}.`,
      })));
  }
  const parts = r.zones.map((z) => `${z.weight} × ${fmt(z.cosine, 4)}`).join(" + ");
  out.append(h("p", { class: "formula", style: { marginTop: "16px" } }, "Final score = (", parts, `) / ${r.weight_sum} = `, h("strong", { text: fmt(r.score, 4) })),
    check(r, r.score));
}

function renderBm25(out, r) {
  out.append(
    h("p", { class: "formula" }, "score = Σ qtf × idf × tf × (k₁ + 1) / (tf + k₁ × (1 − b + b × |d| / avgdl)), with idf = ln(1 + (N − df + 0.5) / (df + 0.5))."),
    h("p", { class: "small muted", text: `k₁ = ${r.k1}, b = ${r.b}, |d| = ${r.doc_len}, avgdl = ${fmt(r.avg_len, 2)}, N = ${r.n_docs.toLocaleString()} (merged title + body field).` }),
    h("div", { style: { marginTop: "10px" } }, table([
      { key: "term", label: "Term", format: (v) => h("code", { text: v }) },
      { key: "qtf", label: "qtf", num: true }, { key: "df", label: "df", num: true },
      { key: "idf", label: "idf", num: true, format: (v) => fmt(v, 4) },
      { key: "tf_d", label: "tf(d)", num: true },
      { key: "tf_part", label: "tf part", num: true, format: (v) => fmt(v, 4) },
      { key: "score", label: "Contribution", num: true, format: (v) => fmt(v, 4) },
    ], r.rows, { rowClass: (row) => (row.tf_d ? "" : "dim") })),
    h("p", { class: "formula", style: { marginTop: "12px" } }, "Total = ", h("strong", { text: fmt(r.score, 4) })),
    check(r, r.score));
}

function renderDense(out, r) {
  out.append(
    h("p", { class: "formula", text: `Claim and chunk are embedded with ${r.model.split("/").pop()} into unit vectors; the score is their dot product, the cosine: ${fmt(r.score, 4)}.` }),
    h("p", { class: "small muted", style: { margin: "8px 0" }, text: "There are no terms to break down. Instead, each sentence of the chunk is embedded on its own; the closest ones show what the model matched:" }),
    table([
      { key: "cosine", label: "Cosine", num: true, format: (v) => fmt(v, 3) },
      { key: "sentence", label: "Sentence", format: (v) => h("span", { class: "claim-link", text: v }) },
    ], r.sentences));
}
