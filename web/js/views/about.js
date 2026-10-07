// About: what the system does, how it is wired, and the live state of every component.

import { clear, fmt, fmtElapsed, h, icon } from "../dom.js";
import { app } from "../state.js";

const S = { status: null, list: null, data: null, tuned: null };

export function mount(root) {
  S.list = h("ul", { class: "state-list" });
  S.data = h("div");
  S.tuned = h("div");
  root.append(
    h("div", { class: "view-head" },
      h("h1", { id: "h-about", text: "About TraceRAG" }),
      h("p", { text: "A retrieval-augmented system for scientific claims in which every answer sentence must cite the chunk it came from, and a verifier checks whether that chunk really backs it." })),
    h("div", { class: "about-grid" },
      h("div", { class: "stack" },
        h("figure", { class: "diagram", style: { margin: 0 } },
          h("img", { src: "/files/report/pipeline.svg", alt: "Pipeline: SciFact abstracts are chunked, tokenized and indexed; a claim is retrieved against the inverted and dense indexes, an answer is written with citations, split into sentences and verified." })),
        h("div", { class: "panel panel-pad prose" },
          h("h2", { style: { marginBottom: "10px" }, text: "How a claim is checked" }),
          h("p", { text: "Retrieval. Abstracts are split into whole abstracts or overlapping three-sentence windows. A hand-built inverted index serves tf-idf (lnc.ltc with separate title and body zones) and BM25; MiniLM embeddings serve dense retrieval; reciprocal rank fusion or a weighted blend combines the two." }),
          h("p", { text: "Answering. The top chunks go to a writer that must end every sentence with the id of the chunk supporting it, or answer “Not enough evidence.” The writer is an LLM of your choice, or an offline mode that copies the sentences closest to the claim." }),
          h("p", { text: "Verification. Each sentence is compared with the title and every pair of consecutive sentences of the chunk it cites. Cosine similarity catches off-topic citations cheaply; an NLI model (DeBERTa-v3) catches contradictions and negations. A sentence citing nothing, or citing a chunk that was not retrieved, is always flagged." }),
          h("p", { text: "Evaluation. Retrieval is scored against SciFact's relevance judgments on 300 test claims. The verifier is scored on SciFact's expert support and contradiction labels, and on answers corrupted on purpose. All thresholds and weights are chosen on the 809 train claims." }))),
      h("div", { class: "stack" },
        h("div", { class: "panel panel-pad", id: "status" }, h("h2", { style: { marginBottom: "6px" }, text: "Components" }), S.list),
        h("div", { class: "panel panel-pad" }, h("h2", { style: { marginBottom: "10px" }, text: "Data and settings" }), S.data, S.tuned),
        h("div", { class: "panel panel-pad stack" },
          h("h2", { text: "Rebuild the results" }),
          h("p", { class: "small muted", text: "Stop the app first, then run from the project folder:" }),
          h("code", { class: "cmd", text: "python scripts/run_all.py\npython -m src.eval.run_verifier_eval\npython run_app.py" }),
          h("p", { class: "small" }, "The HTTP API behind this page is documented at ", h("a", { href: "/docs", target: "_blank", rel: "noopener", text: "/docs" }), ".")))));
  app.onStatus(render);
}

export function enter() { if (app.status) render(app.status); }

function render(s) {
  const glyph = (st) => (st === "ready" ? icon("check") : st === "error" ? icon("cross") : st === "disabled" ? icon("dash") : icon("spinner"));
  const word = { ready: "Ready", error: "Failed", disabled: "Off", waiting: "Waiting", loading: "Loading", building: "Building" };
  clear(S.list).append(...Object.values(s.components).map((c) => h("li", { "data-state": c.state },
    glyph(c.state),
    h("div", {},
      h("strong", { text: c.label }), " ", h("span", { class: "muted", text: `${word[c.state] || c.state}${c.elapsed && c.state !== "ready" ? ` for ${fmtElapsed(c.elapsed)}` : ""}` }),
      c.message ? h("p", { class: "small muted", style: { overflowWrap: "anywhere" }, text: c.message }) : null))));
  const d = s.data;
  clear(S.data).append(h("p", { class: "small", text: `${(d.docs || 0).toLocaleString()} abstracts and ${(d.claims || 0).toLocaleString()} claims (${d.test_claims} test, ${d.train_claims} train) from BEIR SciFact. Corpus fingerprint ${d.fingerprint}.` }));
  const t = s.tuned;
  const v = s.verifier.thresholds;
  clear(S.tuned).append(h("ul", { class: "small", style: { margin: "10px 0 0", paddingLeft: "18px" } },
    h("li", { text: `tf-idf zone weights (title, body): ${t.zone_weights.whole.join(", ")} for whole abstracts, ${t.zone_weights.sent.join(", ")} for windows` }),
    h("li", { text: `BM25: k1 = ${t.bm25.k1}, b = ${t.bm25.b}. RRF k = ${t.rrf_k}. Weighted hybrid alpha: ${t.hybrid_alpha.whole} (whole), ${t.hybrid_alpha.sent} (windows)` }),
    h("li", { text: `Verifier thresholds: cosine ${fmt(v.cosine, 2)}; NLI P(entail) ${fmt(v.nli, 2)}; cascade gate ${fmt(v.gate, 2)} then P(entail) ${fmt(v.cascade_nli, 2)}` })));
}
