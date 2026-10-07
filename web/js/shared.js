// Pieces several views use: the claim picker and the paper drawer.

import { api } from "./api.js";
import { append, clear, fmt, h, icon, labelBadge, normSpace, table, toast } from "./dom.js";

// ------------------------------------------------------------------ claim picker
let pickerResolve = null;
let pickerOffset = 0;
let pickerTimer = null;

function setupPicker() {
  const dlg = document.getElementById("claim-picker");
  if (dlg.dataset.ready) return dlg;
  dlg.dataset.ready = "1";
  dlg.querySelector("[data-close]").append(icon("close"));
  dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
  dlg.addEventListener("close", () => { if (pickerResolve) { pickerResolve(null); pickerResolve = null; } });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  const reload = () => { pickerOffset = 0; loadClaims(false); };
  document.getElementById("picker-q").addEventListener("input", () => { clearTimeout(pickerTimer); pickerTimer = setTimeout(reload, 200); });
  document.getElementById("picker-split").addEventListener("change", reload);
  document.getElementById("picker-label").addEventListener("change", reload);
  document.getElementById("picker-more").addEventListener("click", () => loadClaims(true));
  return dlg;
}

async function loadClaims(more) {
  const list = document.getElementById("picker-list");
  const params = {
    q: document.getElementById("picker-q").value.trim(),
    split: document.getElementById("picker-split").value,
    label: document.getElementById("picker-label").value,
    limit: 40,
    offset: more ? pickerOffset : 0,
  };
  try {
    const data = await api.claims(params);
    if (!more) clear(list);
    for (const c of data.items) {
      list.append(h("li", {}, h("button", {
        type: "button",
        onclick: () => {
          const r = pickerResolve; pickerResolve = null;
          document.getElementById("claim-picker").close();
          if (r) r(c);
        },
      }, h("span", { class: "qid", text: c.qid }), h("span", { class: "ctext", text: c.text }), labelBadge(c.label))));
    }
    pickerOffset = params.offset + data.items.length;
    document.getElementById("picker-count").textContent =
      data.total ? `${data.total} claims match. Click one to use it.` : "No claim matches. Try fewer words.";
    document.getElementById("picker-more").hidden = pickerOffset >= data.total;
  } catch (e) { toast(e.message); }
}

export function pickClaim() {
  const dlg = setupPicker();
  pickerOffset = 0;
  loadClaims(false);
  dlg.showModal();
  setTimeout(() => document.getElementById("picker-q").focus(), 30);
  return new Promise((resolve) => { pickerResolve = resolve; });
}

// ------------------------------------------------------------------ paper drawer
function setupDrawer() {
  const dlg = document.getElementById("paper-drawer");
  if (!dlg.dataset.ready) {
    dlg.dataset.ready = "1";
    dlg.querySelector("[data-close]").append(icon("close"));
    dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }
  return dlg;
}

// Opens a paper with the score explanation for one retriever and query.
export async function openPaper(docId, { query, retriever, chunkId, relevant, note } = {}) {
  const dlg = setupDrawer();
  const body = clear(document.getElementById("drawer-body"));
  document.getElementById("drawer-title").textContent = `Paper ${docId}`;
  body.append(h("p", { class: "muted" }, icon("spinner"), " Loading"));
  dlg.showModal();
  try {
    const paper = await api.paper(docId);
    append(clear(body), [
      h("h3", { style: { fontSize: "1.05rem", marginBottom: "8px" }, text: paper.title }),
      relevant === true ? h("p", { style: { marginBottom: "8px" } }, h("span", { class: "badge badge-rel", text: "Judged relevant for this claim" })) : null,
      h("p", { class: "paper-abstract", text: normSpace(paper.text) }),
      note ? h("p", { class: "hint", style: { marginTop: "12px" }, text: note }) : null,
    ]);
    if (query) {
      const kinds = retriever === "dense" ? ["dense"]
        : retriever === "tfidf" ? ["tfidf"]
        : retriever === "bm25" ? ["bm25"]
        : ["bm25", "dense"];                       // hybrids: show both halves
      for (const kind of kinds) {
        const section = h("div", { class: "section", style: { marginTop: "22px" } });
        body.append(section);
        await renderExplanation(section, { query, chunk_id: chunkId || docId, retriever: kind });
      }
    }
  } catch (e) {
    clear(body).append(h("p", { class: "notice notice-bad", text: e.message }));
  }
}

const RETRIEVER_TITLE = { tfidf: "tf-idf score, term by term", bm25: "BM25 score, term by term", dense: "Dense similarity, sentence by sentence" };

export async function renderExplanation(container, body) {
  container.append(h("h3", { text: RETRIEVER_TITLE[body.retriever] }));
  let ex;
  try { ex = await api.explain(body); } catch (e) { container.append(h("p", { class: "notice", text: e.message })); return; }
  if (body.retriever === "dense") {
    container.append(h("p", { class: "small muted", style: { margin: "4px 0 8px" }, text: `Cosine between the claim and this chunk: ${fmt(ex.score)}. The sentence closest to the claim:` }));
    container.append(table([
      { key: "cosine", label: "Cosine", num: true, format: (v) => fmt(v) },
      { key: "sentence", label: "Sentence", format: (v) => h("span", { class: "claim-link", text: v }) },
    ], ex.sentences.slice(0, 5)));
    return;
  }
  if (body.retriever === "bm25") {
    container.append(h("p", { class: "small muted", style: { margin: "4px 0 8px" },
      text: `k1 = ${ex.k1}, b = ${ex.b}, document length ${ex.doc_len} terms, average ${fmt(ex.avg_len, 1)}. Score ${fmt(ex.score)}.` }));
    container.append(table([
      { key: "term", label: "Term", format: (v) => h("code", { text: v }) },
      { key: "df", label: "df", num: true }, { key: "idf", label: "idf", num: true, format: (v) => fmt(v) },
      { key: "tf_d", label: "tf in doc", num: true }, { key: "tf_part", label: "tf part", num: true, format: (v) => fmt(v) },
      { key: "score", label: "Score", num: true, format: (v) => fmt(v) },
    ], ex.rows, { rowClass: (r) => (r.tf_d ? "" : "dim") }));
    return;
  }
  const parts = ex.zones.map((z) => `${z.weight} × ${fmt(z.cosine, 4)} (${z.zone})`).join(" + ");
  container.append(h("p", { class: "small muted", style: { margin: "4px 0 8px" },
    text: `Score = (${parts}) / ${ex.weight_sum} = ${fmt(ex.score, 4)}` }));
  for (const z of ex.zones) {
    const matched = z.rows.filter((r) => r.tf_d > 0);
    container.append(h("p", { class: "small", style: { margin: "10px 0 6px" } },
      h("strong", { text: z.zone === "title" ? "Title zone" : z.zone === "body" ? "Body zone" : "Merged field" }),
      ` cosine ${fmt(z.cosine, 4)}; ${matched.length} of ${z.rows.length} claim terms occur here`));
    container.append(table([
      { key: "term", label: "Term", format: (v) => h("code", { text: v }) },
      { key: "w_q", label: "w(t,q)", num: true, format: (v) => fmt(v, 4) },
      { key: "tf_d", label: "tf(t,d)", num: true },
      { key: "w_d", label: "w(t,d)", num: true, format: (v) => fmt(v, 4) },
      { key: "product", label: "Product", num: true, format: (v) => fmt(v, 4) },
    ], z.rows, { rowClass: (r) => (r.tf_d ? "" : "dim") }));
  }
}
