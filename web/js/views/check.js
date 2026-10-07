// Check a claim: retrieve -> cited answer -> per-sentence verdicts, with the evidence traced.

import { api } from "../api.js";
import { clear, fmt, fmtMs, h, icon, labelBadge, normSpace, plural, toast, verdictMark, VERDICT_TEXT } from "../dom.js";
import { app, navigate, retrieverAvailable, store, verifierAvailable } from "../state.js";
import { openPaper, pickClaim } from "../shared.js";

const RETRIEVERS = ["hybrid_rrf", "hybrid_weighted", "bm25", "tfidf", "dense"];
const FALLBACK_RETRIEVER = "bm25";
const EXAMPLE_QIDS = ["53", "274", "660", "1"];
const DESKTOP = matchMedia("(min-width: 961px)");

const SHORT = { hybrid_rrf: "Hybrid RRF", hybrid_weighted: "Hybrid weighted", bm25: "BM25", tfidf: "tf-idf", dense: "Dense" };
const HELP = {
  retriever: {
    hybrid_rrf: "Merges the BM25 and dense rankings by rank (reciprocal rank fusion). A strong, safe default.",
    hybrid_weighted: "Blends BM25 and dense scores after scaling them to 0-1; the weight was tuned on train claims. Best nDCG@10 on the test set.",
    bm25: "Probabilistic ranking on the hand-built inverted index. Fast and strong when the claim shares rare words with the paper.",
    tfidf: "lnc.ltc cosine with separate title and body zones, on the hand-built inverted index.",
    dense: "MiniLM sentence embeddings. Finds paraphrases, but can miss rare exact terms.",
  },
  chunking: {
    sent: "Overlapping 3-sentence windows: short, precise passages to cite.",
    whole: "Whole abstracts: one chunk per paper.",
  },
  verifier: {
    cascade: "Cosine similarity rejects off-topic citations first; the NLI model decides the rest. Recommended.",
    nli: "A DeBERTa-v3 NLI model decides whether the cited passage entails each sentence. Sees negations.",
    cosine: "Embedding similarity only. Fast, but blind to negation and contradiction.",
  },
};

const S = {
  els: {},
  pickedQid: null,
  pickedText: null,
  result: null,
  display: null,       // { kind: original | edited | corrupted, answer, verification, corruption? }
  selected: null,
  editing: false,
  busy: false,
  prefs: {
    retriever: store.get("check-retriever", "hybrid_rrf"),
    chunking: store.get("check-chunking", "sent"),
    k: store.get("check-k", 5),
    backend: store.get("check-backend", "extractive"),
    model: store.get("check-model", ""),
    base_url: store.get("check-base-url", ""),
    verifier: store.get("check-verifier", "cascade"),
  },
  apiKey: "",
};

// ================================================================== form
function segmented(name, options, onPick) {
  const box = h("div", { class: "seg", role: "group", "aria-label": name });
  for (const [value, text] of options) {
    box.append(h("button", { type: "button", "data-value": value, "aria-pressed": "false", onclick: () => onPick(value) }, text));
  }
  return box;
}

function setPressed(box, value, isAvailable) {
  for (const b of box.querySelectorAll("button")) {
    const ok = isAvailable ? isAvailable(b.dataset.value) : true;
    b.disabled = !ok;
    b.setAttribute("aria-pressed", String(b.dataset.value === value));
    const soon = b.querySelector(".soon");
    if (!ok && !soon) b.append(h("span", { class: "soon", text: "loading" }));
    if (ok && soon) soon.remove();
    b.title = ok ? "" : "Still loading; available in a moment";
  }
}

function traceArt() {
  // A real result from this system (SciFact claim 53, cosine verifier), drawn with the app's own pieces.
  return h("div", { class: "hero-art" }, h("figure", { class: "trace-art", style: { margin: 0 }, "aria-label": "Example of a traced sentence" },
    h("p", { class: "art-claim" }, h("strong", { text: "Claim" }), "ALDH1 expression is associated with poorer prognosis in breast cancer."),
    h("div", { class: "art-sent v-supported" }, verdictMark("supported"),
      h("span", {}, h("span", { class: "sent-text" }, "In a series of 577 breast carcinomas, expression of ALDH1 detected by immunostaining correlated with poor prognosis.",
        h("span", { class: "cite", text: "45638119_2" })),
      h("span", { class: "sent-reason", text: "Close to the cited evidence: cosine 0.90, needs 0.70." }))),
    h("div", { class: "art-link", "aria-hidden": "true" }),
    h("div", { class: "art-ev" },
      h("div", { class: "ev-title", text: "ALDH1 is a marker of normal and malignant human mammary stem cells and a predictor of poor clinical outcome." }),
      h("p", { class: "ev-text" }, "…stem/progenitor properties. ",
        h("mark", { class: "trace", text: "In breast carcinomas, high ALDH activity identifies the tumorigenic cell fraction, capable of self-renewal and of generating tumors that recapitulate the heterogeneity of the parental tumor. In a series of 577 breast carcinomas, expression of ALDH1 detected by immunostaining correlated with poor prognosis." }))),
    h("figcaption", { class: "art-meta" }, h("span", { class: "badge badge-rel", text: "judged relevant" }),
      h("span", { text: "A real check from SciFact claim 53" }))));
}

export function mount(root) {
  const e = S.els;
  e.claim = h("textarea", {
    class: "claim-input", rows: 2, id: "claim-text", "aria-label": "Claim to check",
    placeholder: "Type a scientific claim, for example: Vitamin D supplements lower the risk of fractures in older adults.",
    onkeydown: (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); run(); } },
  });
  e.run = h("button", { class: "btn btn-primary", type: "button", onclick: run }, "Check claim");
  e.examples = h("div", { class: "example-row" }, h("span", { class: "hint", text: "Try:" }));

  e.retriever = segmented("Retriever", RETRIEVERS.map((r) => [r, SHORT[r]]), (v) => { S.prefs.retriever = v; save(); sync(); });
  e.chunking = segmented("Chunks", [["sent", "3-sentence windows"], ["whole", "Whole abstracts"]], (v) => { S.prefs.chunking = v; save(); sync(); });
  e.verifier = segmented("Verifier", [["cascade", "Cosine, then NLI"], ["nli", "NLI"], ["cosine", "Cosine"]], (v) => { S.prefs.verifier = v; save(); sync(); });
  e.kOut = h("output", { text: String(S.prefs.k), "aria-live": "polite" });
  const stepK = (d) => { S.prefs.k = Math.max(1, Math.min(10, S.prefs.k + d)); save(); sync(); };
  e.kMinus = h("button", { type: "button", "aria-label": "Fewer chunks", onclick: () => stepK(-1) }, "−");
  e.kPlus = h("button", { type: "button", "aria-label": "More chunks", onclick: () => stepK(1) }, "+");
  e.backend = h("select", { class: "input", id: "set-backend", style: { maxWidth: "340px" }, onchange: () => { S.prefs.backend = e.backend.value; save(); sync(); } });
  e.model = h("input", { class: "input", id: "set-model", autocomplete: "off", spellcheck: "false",
    oninput: () => { S.prefs.model = e.model.value.trim(); save(); } });
  e.key = h("input", { class: "input", id: "set-key", type: "password", autocomplete: "off",
    oninput: () => { S.apiKey = e.key.value.trim(); } });
  e.baseUrl = h("input", { class: "input", id: "set-base-url", autocomplete: "off", placeholder: "https://api.openai.com/v1",
    oninput: () => { S.prefs.base_url = e.baseUrl.value.trim(); save(); } });
  e.baseField = h("label", { class: "field" }, h("span", { text: "Base URL (optional)" }), e.baseUrl);
  e.llmFields = h("div", { class: "settings llm-fields" },
    h("label", { class: "field" }, h("span", { text: "Model" }), e.model),
    h("label", { class: "field" }, h("span", { text: "API key" }), e.key),
    e.baseField);
  e.keyHint = h("p", { class: "hint", style: { marginTop: "8px" } });
  e.help = { retriever: h("p", { class: "seg-help" }), chunking: h("p", { class: "seg-help" }), verifier: h("p", { class: "seg-help" }) };
  e.settingsNote = h("div", { class: "hint", style: { marginTop: "12px" }, "aria-live": "polite" });
  e.summary = h("span", { class: "settings-summary" });
  const group = (label, control, help, wide) => h("div", { class: `setting-group${wide ? " wide" : ""}` }, h("span", { class: "label", text: label }), control, help);
  e.settings = h("details", { class: "panel settings-box", open: store.get("check-settings-open", false) },
    h("summary", {}, e.summary),
    h("div", { class: "settings-grid" },
      group("Retriever", e.retriever, e.help.retriever, true),
      group("Verifier", e.verifier, e.help.verifier),
      group("Chunks", e.chunking, e.help.chunking),
      group("Chunks given to the writer", h("div", { class: "stepper" }, e.kMinus, e.kOut, e.kPlus),
        h("p", { class: "seg-help", text: "How many top chunks the answer may cite (1 to 10)." })),
      group("Answer written by", e.backend, h("p", { class: "seg-help", text: "Offline extraction needs no key. Any listed API works with your own key." }))),
    e.llmFields, e.keyHint, e.settingsNote);
  e.settings.addEventListener("toggle", () => store.set("check-settings-open", e.settings.open));
  e.results = h("div", { id: "check-results", "aria-live": "polite" });
  e.hero = h("div", { class: "hero" },
    h("div", {},
      h("h1", { id: "h-check", text: "Check a scientific claim against the evidence" }),
      h("p", { class: "sub" }, "TraceRAG searches 5,183 SciFact abstracts, writes an answer that cites a passage after every sentence, and checks each sentence against the passage it cites. ",
        h("button", { class: "tour-link", type: "button", onclick: () => window.dispatchEvent(new Event("opentour")) }, "How it works")),
      h("div", { class: "panel claim-box" },
        e.claim,
        h("div", { class: "row claim-actions" },
          h("button", { class: "btn", type: "button", onclick: randomClaim }, icon("dice"), "Random SciFact claim"),
          h("button", { class: "btn", type: "button", onclick: browse }, icon("list"), "Browse claims"),
          h("span", { class: "hint", text: "Ctrl + Enter runs the check" }),
          h("div", { class: "row-end" }, e.run))),
      e.examples),
    traceArt());

  root.append(e.hero, e.settings, e.results);
  fillOptions();
  e.model.value = S.prefs.model;
  e.baseUrl.value = S.prefs.base_url;
  renderEmpty();
  loadExamples();
  app.onStatus(() => { fillOptions(); sync(); });
  sync();
}

export function enter(params) {
  if (params.q) {
    S.els.claim.value = params.q;
    S.pickedText = params.q;
    S.pickedQid = params.qid || null;
    if (params.run === "1") run();
    history.replaceState(null, "", "#/check");
  }
}

function save() {
  store.set("check-retriever", S.prefs.retriever);
  store.set("check-chunking", S.prefs.chunking);
  store.set("check-k", S.prefs.k);
  store.set("check-backend", S.prefs.backend);
  store.set("check-model", S.prefs.model);
  store.set("check-base-url", S.prefs.base_url);
  store.set("check-verifier", S.prefs.verifier);
}

function fillOptions() {
  const s = app.status;
  const e = S.els;
  const current = e.backend.value || S.prefs.backend;
  const llm = s ? s.llm : { extractive: { label: "Offline extractive (no key)" } };
  clear(e.backend).append(...Object.entries(llm).map(([k, v]) => h("option", { value: k, text: v.label })));
  e.backend.value = current in llm ? current : "extractive";
}

function effectiveRetriever() {
  const pref = S.prefs.retriever;
  return retrieverAvailable(pref, S.prefs.chunking) || !app.status ? pref : FALLBACK_RETRIEVER;
}

function effectiveVerifier() {
  const pref = S.prefs.verifier;
  return verifierAvailable(pref) || !app.status ? pref : "cosine";
}

// Keeps the form honest about what can run right now.
function sync() {
  const e = S.els;
  const s = app.status;
  const ch = S.prefs.chunking;
  const r = effectiveRetriever();
  const v = effectiveVerifier();
  setPressed(e.retriever, r, (name) => !s || retrieverAvailable(name, ch));
  setPressed(e.verifier, v, (name) => !s || verifierAvailable(name));
  setPressed(e.chunking, ch);
  e.help.retriever.textContent = HELP.retriever[r];
  e.help.verifier.textContent = HELP.verifier[v];
  e.help.chunking.textContent = HELP.chunking[ch];
  e.kOut.textContent = String(S.prefs.k);
  e.kMinus.disabled = S.prefs.k <= 1;
  e.kPlus.disabled = S.prefs.k >= 10;
  e.backend.value = s && S.prefs.backend in s.llm ? S.prefs.backend : "extractive";

  const notes = [];
  if (r !== S.prefs.retriever) notes.push(`${SHORT[S.prefs.retriever]} needs the dense index, which is still loading; BM25 is used until it is ready.`);
  if (v !== S.prefs.verifier) {
    const nli = s && s.components.nli;
    notes.push(nli && nli.state === "error" ? "The NLI model could not load (see About); the cosine verifier is used."
      : nli && nli.state === "disabled" ? "The NLI model is off (--no-nli); the cosine verifier is used."
      : "The NLI model is still loading; the cosine verifier is used until it is ready.");
  }
  clear(e.settingsNote).append(...notes.map((n) => h("p", { text: n })));

  const backend = e.backend.value;
  const info = s && s.llm[backend];
  e.llmFields.hidden = backend === "extractive";
  e.baseField.hidden = backend !== "openai";
  if (info && backend !== "extractive") {
    e.model.placeholder = info.default_model ? `${info.default_model} (default)` : "model id from your provider";
    e.key.placeholder = info.needs_key ? (info.key_in_env ? `Using ${info.key_env} from your environment` : "Paste a key") : "Not needed";
    e.key.disabled = !info.needs_key;
    e.keyHint.textContent = info.needs_key
      ? "The key goes only to your local TraceRAG server, for each request; it is not saved anywhere."
      : "Runs against a local Ollama server at http://localhost:11434.";
    e.keyHint.hidden = false;
  } else {
    e.keyHint.hidden = true;
  }
  clear(e.summary).append("Settings: ", h("strong", { text: label("retriever", r) }), " over ",
    h("strong", { text: ch === "sent" ? "3-sentence windows" : "whole abstracts" }), `, top ${S.prefs.k}, answers by `,
    h("strong", { text: backend === "extractive" ? "offline extraction" : (info ? info.label : backend) }), ", checked by ",
    h("strong", { text: lowerFirst(label("verifier", v)) }));
}

const lowerFirst = (t) => (t ? t[0].toLowerCase() + t.slice(1) : t);

function label(kind, key) {
  const s = app.status;
  if (!s) return key;
  return kind === "retriever" ? s.retrievers.labels[key] : s.verifier.labels[key];
}

async function loadExamples() {
  const claims = await Promise.all(EXAMPLE_QIDS.map((q) => api.claim(q).catch(() => null)));
  for (const c of claims.filter(Boolean)) {
    const tag = { SUPPORT: ["badge-ok", "supported"], CONTRADICT: ["badge-bad", "contradicted"], NEI: ["badge-neutral", "no evidence"] }[c.label];
    S.els.examples.append(h("button", { class: "example-chip", type: "button", title: c.text,
      onclick: () => { setClaim(c.qid, c.text, false); run(); } },
    h("span", { class: `badge ${tag[0]}`, text: tag[1] }), h("span", { text: c.text })));
  }
}

async function randomClaim() {
  try {
    const c = await api.randomClaim({ split: "test" });
    setClaim(c.qid, c.text);
  } catch (err) { toast(err.message); }
}

async function browse() {
  const c = await pickClaim();
  if (c) setClaim(c.qid, c.text);
}

function setClaim(qid, text, focus = true) {
  S.els.claim.value = text;
  S.pickedQid = qid;
  S.pickedText = text;
  if (focus) S.els.claim.focus();
}

// ================================================================== run
async function run() {
  if (S.busy) return;
  const query = S.els.claim.value.trim();
  if (query.length < 3) { toast("Type a claim of a few words first."); S.els.claim.focus(); return; }
  const qid = S.pickedText && normSpace(query) === normSpace(S.pickedText) ? S.pickedQid : null;
  const backend = S.els.backend.value;
  const body = {
    query, qid, retriever: effectiveRetriever(), chunking: S.prefs.chunking, k: S.prefs.k,
    llm: { backend, model: S.prefs.model || "", api_key: S.apiKey || null, base_url: S.prefs.base_url || null },
    verifier: effectiveVerifier(),
  };
  setBusy(true);
  S.els.hero.classList.add("compact");
  renderLoading(body);
  try {
    const res = await api.ask(body);
    S.result = res;
    S.display = { kind: "original", answer: res.answer, verification: res.verification };
    S.editing = false;
    S.selected = firstChecked(res.verification);
    render();
  } catch (err) {
    clear(S.els.results).append(h("div", { class: "notice notice-bad", style: { marginTop: "22px" } }, icon("alert"), h("span", { text: err.message })));
  } finally {
    setBusy(false);
  }
}

function firstChecked(v) {
  const i = v.sentences.findIndex((s) => !s.abstain);
  return i >= 0 ? i : (v.sentences.length ? 0 : null);
}

function setBusy(busy) {
  S.busy = busy;
  S.els.run.disabled = busy;
  clear(S.els.run).append(...(busy ? [icon("spinner"), "Checking"] : ["Check claim"]));
  S.els.results.setAttribute("aria-busy", busy ? "true" : "false");
}

// ================================================================== render
function renderEmpty() {
  const step = (n, title, text) => h("li", {}, h("span", { class: "step-n", text: String(n) }), h("h3", { text: title }), h("p", { text }));
  clear(S.els.results).append(h("ol", { class: "how-steps", "aria-label": "How a claim is checked" },
    step(1, "Retrieve", "The claim is matched against 5,183 abstracts with tf-idf, BM25, dense embeddings or a hybrid, and the best passages are kept."),
    step(2, "Write with citations", "An answer is written from those passages only, with a passage id after every sentence."),
    step(3, "Verify each sentence", "Each sentence is compared with the passage it cites. Unsupported sentences are flagged, with the reason.")));
}

function renderLoading(body) {
  clear(S.els.results).append(
    h("ol", { class: "steps", style: { marginTop: "22px" } },
      h("li", {}, icon("spinner"), `Retrieving ${body.k} chunks, writing the answer and checking each sentence`)),
    h("div", { class: "results" },
      h("div", { class: "panel answer-panel" }, ...[90, 70, 85].map((w) => h("div", { class: "skeleton", style: { height: "18px", width: `${w}%`, margin: "14px 0" } }))),
      h("div", { class: "panel answer-panel" }, ...[60, 95, 80, 90].map((w) => h("div", { class: "skeleton", style: { height: "14px", width: `${w}%`, margin: "12px 0" } })))));
}

function render(focusSentence = false) {
  const res = S.result;
  const d = S.display;
  if (!res || !d) return;
  const v = d.verification;
  const out = clear(S.els.results);
  out.append(claimMeta(res), steps(res));
  const answerPanel = h("div", { class: "panel answer-panel" });
  const evidence = evidencePanel(res, d);
  out.append(h("div", { class: "results" }, answerPanel, h("div", { class: "evidence-col" }, evidence)));

  const title = d.kind === "corrupted" ? "Corrupted answer" : d.kind === "edited" ? "Your edited answer" : "Answer";
  const by = res.settings.backend === "extractive" ? "written offline from the top chunks" : `written by ${res.settings.backend} / ${res.settings.model}`;
  answerPanel.append(h("div", { class: "panel-head" },
    h("div", {}, h("h2", { text: title }), h("p", { class: "small muted", text: d.kind === "original" ? by : "checked against the same retrieved chunks" })),
    h("div", { class: "row", style: { gap: "4px" } },
      h("button", { class: "btn btn-quiet btn-small", type: "button", title: "Copy the answer with its citations", onclick: () => copyAnswer(d) }, icon("copy"), "Copy"),
      h("button", { class: "btn btn-quiet btn-small", type: "button", title: "Download this check as a Markdown report", onclick: () => downloadReport(res, d) }, icon("download"), "Report"))));
  if (v.sentences.length) answerPanel.append(...verdictSummary(v.summary));
  if (v.note) answerPanel.append(h("div", { class: "notice notice-warn" }, icon("info"), h("span", { text: v.note })));
  if (d.kind === "corrupted") answerPanel.append(corruptionNotice(d.corruption));

  if (S.editing) {
    answerPanel.append(editor(d));
  } else if (!v.sentences.length) {
    answerPanel.append(h("p", { class: "muted", text: "The answer is empty." }));
  } else {
    const list = h("ol", { class: "sentences" });
    v.sentences.forEach((s, i) => list.append(sentenceItem(s, i, d, v)));
    answerPanel.append(list);
  }
  answerPanel.append(actions(d));

  if (focusSentence && S.selected !== null) {
    const btn = answerPanel.querySelector(`.sent[data-index="${S.selected}"]`);
    if (btn) btn.focus({ preventScroll: true });
  }
  scrollEvidence(evidence);
}

function verdictSummary(sum) {
  const total = sum.total || 1;
  const checked = sum.supported + sum.unsupported;
  const text = sum.abstain === sum.total
    ? h("span", { class: "verdict-text", text: "The writer found nothing in the retrieved passages that settles this claim." })
    : h("span", { class: "verdict-text" }, h("strong", { text: `${sum.supported} of ${checked}` }),
      ` ${checked === 1 ? "sentence is" : "sentences are"} backed by the passage ${checked === 1 ? "it cites" : "they cite"}.`,
      sum.unsupported ? ` ${sum.unsupported} flagged.` : "");
  const bar = h("div", { class: "verdict-bar", role: "img", "aria-label": `${sum.supported} supported, ${sum.unsupported} unsupported, ${sum.abstain} abstained` },
    sum.supported ? h("span", { class: "vb-ok", style: { width: `${(100 * sum.supported) / total}%` } }) : null,
    sum.unsupported ? h("span", { class: "vb-bad", style: { width: `${(100 * sum.unsupported) / total}%` } }) : null,
    sum.abstain ? h("span", { class: "vb-abs", style: { width: `${(100 * sum.abstain) / total}%` } }) : null);
  return [h("div", { class: "verdict-line" }, text, tally(sum)), bar];
}

async function copyAnswer(d) {
  try {
    await navigator.clipboard.writeText(d.answer);
    toast("Answer copied with its citations.", "ok", 3000);
  } catch (e) {
    toast("The browser blocked the clipboard; select the text and copy it instead.");
  }
}

function downloadReport(res, d) {
  const v = d.verification;
  const lines = [
    "# TraceRAG check", "",
    `**Claim:** ${res.query}`, "",
    res.claim ? `SciFact claim ${res.claim.qid} (${res.claim.split} split), expert label ${res.claim.label}; judged papers: ${res.claim.papers.map((p) => `${p.doc_id} (${p.label})`).join(", ")}.` : "Free-text claim (not in SciFact).",
    "",
    `Retriever: ${label("retriever", res.settings.retriever)} over ${res.settings.chunking === "sent" ? "3-sentence windows" : "whole abstracts"}, top ${res.settings.k}. ` +
      `Answer: ${res.settings.backend === "extractive" ? "offline extraction" : `${res.settings.backend} / ${res.settings.model}`}. ` +
      `Verifier: ${label("verifier", v.method_used)}.`,
    "", `## ${d.kind === "corrupted" ? "Corrupted answer" : d.kind === "edited" ? "Edited answer" : "Answer"}`, "", d.answer, "",
    `## Verdicts (${v.summary.supported} supported, ${v.summary.unsupported} unsupported, ${v.summary.abstain} abstained)`, "",
  ];
  v.sentences.forEach((s, i) => {
    lines.push(`${i + 1}. **${VERDICT_TEXT[s.verdict.label]}**: ${s.text}`);
    lines.push(`   - Reason: ${reasonText(s, v)}`);
    if (s.verdict.evidence) lines.push(`   - Evidence [${s.verdict.best_chunk}]: "${s.verdict.evidence}"`);
  });
  lines.push("", "## Retrieved passages", "");
  for (const r of res.retrieved) {
    lines.push(`${r.rank}. [${r.chunk_id}] ${r.title}${r.relevant ? " (judged relevant)" : ""}, score ${fmt(r.score, 4)}`, "", `   ${normSpace(r.text)}`, "");
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const a = h("a", { href: URL.createObjectURL(blob), download: `tracerag-check${res.claim ? `-claim-${res.claim.qid}` : ""}.md` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function claimMeta(res) {
  const c = res.claim;
  if (!c) {
    return h("div", { class: "claim-meta" }, h("span", { text: "Free-text claim. It is not in SciFact, so no retrieved paper can be marked as judged relevant." }));
  }
  const papers = c.papers.map((p) => h("span", { class: "chip", title: p.title },
    `paper ${p.doc_id}: ${p.label === "SUPPORT" ? "supports" : p.label === "CONTRADICT" ? "contradicts" : "no evidence"}`));
  return h("div", { class: "claim-meta" },
    h("span", { text: `SciFact claim ${c.qid} (${c.split} split)` }), labelBadge(c.label),
    h("span", { class: "muted", text: "Judged papers:" }), ...papers);
}

function steps(res) {
  const t = res.timings;
  const n = res.retrieved.length;
  const v = res.verification;
  const writer = res.settings.backend === "extractive" ? "Selected the closest sentences" : `Wrote the answer with ${res.settings.backend}`;
  const method = lowerFirst(label("verifier", v.method_used));
  return h("ol", { class: "steps" },
    h("li", {}, h("span", { class: "step-n", text: "1" }), `Retrieved ${plural(n, "chunk")} with ${label("retriever", res.settings.retriever)} in ${fmtMs(t.retrieve_ms)}`),
    h("li", {}, h("span", { class: "step-n", text: "2" }), `${writer} in ${fmtMs(t.generate_ms)}`),
    h("li", {}, h("span", { class: "step-n", text: "3" }), `Checked ${plural(v.summary.total, "sentence")} by ${method} in ${fmtMs(t.verify_ms)}`));
}

function tally(sum) {
  return h("div", { class: "tally" },
    h("span", { class: "t-ok" }, icon("check"), `${sum.supported} supported`),
    h("span", { class: "t-bad" }, icon("cross"), `${sum.unsupported} unsupported`),
    sum.abstain ? h("span", {}, icon("dash"), `${sum.abstain} abstained`) : null);
}

function corruptionNotice(c) {
  const what = { negate: "One sentence was negated.", swap: "One sentence now cites a chunk from a different paper.",
    fabricate: "A sentence taken from an unrelated paper was added, citing a retrieved chunk." }[c.kind];
  return h("div", { class: "notice notice-warn" }, icon("alert"),
    h("span", {}, `${what} It is marked `, h("span", { class: "tag-changed", text: "changed" }), ". A good verifier flags it as unsupported."));
}

// One answer sentence: verdict mark, text with citations, one-line reason, and details when open.
function sentenceItem(s, i, d, v) {
  const open = S.selected === i;
  const changed = d.kind === "corrupted" && d.corruption.index === i;
  const text = h("span", { class: "sent-text" }, s.text || s.sentence);
  for (const cid of s.cited_ids) {
    const bad = s.invalid_ids.includes(cid);
    text.append(h("span", { class: `cite${bad ? " bad" : ""}`, title: bad ? "Not among the retrieved chunks" : `Cites chunk ${cid}`, text: cid }));
  }
  if (changed) text.append(h("span", { class: "tag-changed", text: "changed" }));
  const btn = h("button", {
    class: `sent v-${s.verdict.label}${changed ? " changed" : ""}`, type: "button", "data-index": i, "aria-expanded": open ? "true" : "false",
    onclick: () => { S.selected = open ? null : i; render(true); },
  }, verdictMark(s.verdict.label),
  h("span", {}, h("span", { class: "sr-only", text: `${VERDICT_TEXT[s.verdict.label]}: ` }), text,
    h("span", { class: "sent-reason", text: reasonText(s, v) })));
  const li = h("li", {}, btn);
  if (open) li.append(sentenceDetail(s, v, changed ? d.corruption : null));
  return li;
}

function thresholdsFor(v) {
  const t = v.thresholds;
  if (v.method_used === "cosine") return { cos: t.cosine, cosLabel: "threshold", ent: null };
  if (v.method_used === "nli") return { cos: null, ent: t.nli };
  return { cos: t.gate, cosLabel: "gate", ent: t.cascade_nli };
}

function reasonText(s, v) {
  const vd = s.verdict;
  const r = vd.reason || "";
  const t = thresholdsFor(v);
  if (vd.label === "abstain") return "The writer said the retrieved chunks do not settle the claim.";
  if (r.startsWith("no citation")) return "No citation. Every sentence has to cite a retrieved chunk.";
  if (r.startsWith("cites chunk")) return `Cites ${s.invalid_ids.join(", ")}, which was not among the retrieved chunks.`;
  if (r.startsWith("off-topic")) return `The cited chunk is off-topic (cosine ${fmt(vd.cosine, 2)}, gate ${fmt(t.cos, 2)}), so NLI was skipped.`;
  if (!vd.nli_called) {
    return vd.label === "supported"
      ? `Close to the cited evidence: cosine ${fmt(vd.cosine, 2)}, needs ${fmt(t.cos, 2)}.`
      : `Not close enough to the cited evidence: cosine ${fmt(vd.cosine, 2)}, needs ${fmt(t.cos, 2)}.`;
  }
  if (vd.label === "supported") return `The cited evidence entails it: P(entail) ${fmt(vd.entail, 2)}, needs ${fmt(t.ent, 2)}.`;
  if ((vd.contradict ?? 0) >= 0.5) return `The cited evidence contradicts it: P(contradict) ${fmt(vd.contradict, 2)}.`;
  return `The cited evidence does not entail it: P(entail) ${fmt(vd.entail, 2)}, needs ${fmt(t.ent, 2)}.`;
}

function meter(name, value, tick, kind, tickLabel) {
  const pctv = Math.max(0, Math.min(1, value)) * 100;
  const cls = kind === "contradict" ? (value >= 0.5 ? "bad" : "") : (tick !== null && tick !== undefined ? (value >= tick ? "ok" : "bad") : "");
  return h("div", { class: "meter" },
    h("span", { text: name }),
    h("span", { class: "meter-track", role: "img", "aria-label": `${name} ${fmt(value, 2)}${tick !== null && tick !== undefined ? `, ${tickLabel} ${fmt(tick, 2)}` : ""}` },
      h("span", { class: `meter-fill ${cls}`, style: { width: `${pctv}%` } }),
      tick !== null && tick !== undefined ? h("span", { class: "meter-tick", style: { left: `calc(${tick * 100}% - 1px)` }, title: `${tickLabel} ${fmt(tick, 2)}` }) : null),
    h("span", { class: "meter-value", text: fmt(value, 2) }));
}

function sentenceDetail(s, v, corruption) {
  const vd = s.verdict;
  const t = thresholdsFor(v);
  const box = h("div", { class: "sent-detail" });
  const meters = h("div", { class: "meters" });
  if (vd.cosine !== null && vd.cosine !== undefined) meters.append(meter("Cosine", vd.cosine, t.cos, "cos", t.cosLabel || "threshold"));
  if (vd.entail !== null && vd.entail !== undefined) {
    meters.append(meter("P(entail)", vd.entail, t.ent, "entail", "threshold"));
    meters.append(meter("P(contradict)", vd.contradict, 0.5, "contradict", "contradiction at"));
  }
  if (meters.childElementCount) box.append(h("p", { class: "small muted", style: { margin: "10px 0 8px" }, text: "Scores against the best-matching passage of the cited chunk. The dark tick marks the threshold chosen on SciFact train claims." }), meters);
  if (vd.evidence) {
    box.append(h("blockquote", { class: "quote" }, h("span", { class: "cite", text: vd.best_chunk }), " ", vd.evidence));
    box.append(h("p", { style: { marginTop: "8px" } }, h("button", {
      class: "btn btn-quiet btn-small", type: "button",
      onclick: () => { const el = document.getElementById(`ev-${vd.best_chunk}`); if (el) el.scrollIntoView({ behavior: "smooth", block: "center" }); },
    }, "Show it in the evidence list")));
  }
  if (corruption && corruption.original) {
    box.append(h("div", { class: "diff" }, h("p", { class: "small muted", text: "Before the corruption:" }), h("del", { text: corruption.original })));
  } else if (corruption) {
    box.append(h("p", { class: "small muted", style: { marginTop: "8px" }, text: "This sentence was added; it comes from a paper that was not retrieved." }));
  }
  return box;
}

function editor(d) {
  const area = h("textarea", { class: "input edit-area", "aria-label": "Edit the answer" });
  area.value = d.answer;
  const recheck = h("button", { class: "btn btn-primary", type: "button" }, "Re-check");
  recheck.addEventListener("click", async () => {
    const answer = area.value.trim();
    if (!answer) { toast("The answer is empty."); return; }
    recheck.disabled = true;
    clear(recheck).append(icon("spinner"), "Checking");
    try {
      const ver = await api.verify({ answer, chunk_ids: chunkPool(d).map((c) => c.chunk_id), verifier: effectiveVerifier() });
      S.display = { kind: "edited", answer, verification: ver, extra: d.extra };
      S.editing = false;
      S.selected = firstChecked(ver);
      render();
    } catch (err) {
      toast(err.message);
      recheck.disabled = false;
      clear(recheck).append("Re-check");
    }
  });
  setTimeout(() => area.focus(), 0);
  return h("div", { class: "stack" },
    h("p", { class: "hint", text: "Change any sentence, add one, or cite a different chunk as [chunk_id]. Citing a chunk that was not retrieved is always flagged." }),
    area,
    h("div", { class: "row" }, recheck, h("button", { class: "btn", type: "button", onclick: () => { S.editing = false; render(); } }, "Cancel")));
}

function actions(d) {
  const hasClaims = S.result.verification.sentences.some((s) => s.cited_ids.length && !s.abstain);
  const bar = h("div", { class: "answer-actions" });
  if (S.editing) return bar;
  const first = h("div", { class: "row" },
    h("button", { class: "btn btn-small", type: "button", onclick: () => { S.editing = true; render(); } }, icon("edit"), "Edit and re-check"));
  if (d.kind !== "original") {
    first.append(h("button", { class: "btn btn-small btn-quiet", type: "button", onclick: () => {
      S.display = { kind: "original", answer: S.result.answer, verification: S.result.verification };
      S.selected = firstChecked(S.result.verification);
      render();
    } }, icon("undo"), "Back to the original answer"));
  }
  const second = h("div", { class: "row" }, h("span", { class: "label", text: "Corrupt one sentence of the original answer:" }));
  for (const [kind, text] of [["negate", "Negate it"], ["swap", "Swap its citation"], ["fabricate", "Add a fabricated sentence"]]) {
    second.append(h("button", { class: "btn btn-small", type: "button", disabled: !hasClaims,
      onclick: (ev) => corrupt(kind, ev.currentTarget) }, text));
  }
  bar.append(first, second);
  return bar;
}

async function corrupt(kind, btn) {
  const res = S.result;
  btn.disabled = true;
  try {
    const c = await api.corrupt({
      answer: res.answer, chunk_ids: res.retrieved.map((r) => r.chunk_id), kind,
      query: res.query, qid: res.claim ? res.claim.qid : null, verifier: effectiveVerifier(),
    });
    S.display = { kind: "corrupted", answer: c.answer, verification: c.verification, corruption: c, extra: c.extra_chunk };
    S.selected = c.index;
    S.editing = false;
    render();
  } catch (err) {
    toast(err.message);
    btn.disabled = false;
  }
}

// ================================================================== evidence
function chunkPool(d) {
  const pool = S.result.retrieved.map((r) => ({ ...r }));
  if (d.extra && !pool.some((r) => r.chunk_id === d.extra.chunk_id)) {
    pool.push({ ...d.extra, rank: null, score: null, relevant: null, extra: true });
  }
  return pool;
}

function evidencePanel(res, d) {
  const sel = S.selected !== null ? d.verification.sentences[S.selected] : null;
  const target = sel && sel.verdict.best_chunk;
  const evidence = sel && sel.verdict.evidence;
  const cited = new Set(d.verification.sentences.flatMap((s) => s.cited_ids));
  const selCited = new Set(sel ? sel.cited_ids : []);
  const list = h("div", { class: "evidence-list", style: { position: "relative" } });
  for (const c of chunkPool(d)) list.append(evidenceCard(c, res, { cited: cited.has(c.chunk_id), target: c.chunk_id === target, selCited: selCited.has(c.chunk_id), evidence: c.chunk_id === target ? evidence : null }));
  const chunkWord = res.settings.chunking === "sent" ? "3-sentence windows" : "whole abstracts";
  return h("div", { class: "panel" },
    h("div", { class: "evidence-head" },
      h("h2", { text: "Evidence" }),
      h("p", { class: "small muted", text: `Top ${res.retrieved.length} ${chunkWord} from ${label("retriever", res.settings.retriever)}. Open a sentence of the answer to highlight the passage it was checked against.` })),
    list);
}

function evidenceCard(c, res, o) {
  const titleEl = h("div", { class: "ev-title" });
  const textEl = h("p", { class: "ev-text" });
  const title = normSpace(c.title);
  const text = normSpace(c.text);
  const ev = o.evidence ? normSpace(o.evidence) : null;
  if (ev && ev === title) {
    titleEl.append(h("mark", { class: "trace", text: title }));
    textEl.append(text);
  } else {
    titleEl.append(title);
    const at = ev ? text.indexOf(ev) : -1;
    if (at >= 0) textEl.append(text.slice(0, at), h("mark", { class: "trace", text: ev }), text.slice(at + ev.length));
    else textEl.append(text);
  }
  const badges = [];
  if (c.relevant === true) badges.push(h("span", { class: "badge badge-rel", text: "judged relevant" }));
  if (o.cited) badges.push(h("span", { class: "badge badge-cited", text: "cited" }));
  if (c.extra) badges.push(h("span", { class: "badge badge-bad", text: "added by the corruption" }));
  const terms = c.terms && c.terms.length ? h("div", { class: "ev-terms" },
    h("span", { class: "hint", text: "tf-idf term shares:" }),
    ...c.terms.map((t) => h("span", { class: "chip", title: `found in: ${t.zones.join(", ")}` }, h("code", { text: t.term }), ` ${Math.round(t.share * 100)}%`))) : null;
  return h("article", { class: `ev${o.target || o.selCited ? " is-target" : ""}`, id: `ev-${c.chunk_id}` },
    h("div", { class: "ev-top" },
      c.rank ? h("span", { class: "ev-rank", text: `${c.rank}.` }) : null,
      h("span", { class: "cite", text: c.chunk_id, title: "Chunk id, as the answer cites it" }), ...badges,
      c.score !== null && c.score !== undefined ? h("span", { class: "ev-score", text: `score ${fmt(c.score, 4)}` }) : null),
    titleEl, textEl, terms,
    h("div", { class: "ev-foot" },
      h("button", { class: "btn btn-quiet btn-small", type: "button",
        onclick: () => openPaper(c.doc_id, { query: res.query, retriever: res.settings.retriever, chunkId: c.chunk_id, relevant: c.relevant }) }, "Why this chunk"),
      h("button", { class: "btn btn-quiet btn-small", type: "button",
        onclick: () => navigate("index", { query: res.query, chunk: c.chunk_id }) }, "Full score breakdown")));
}

function scrollEvidence(panel) {
  if (!DESKTOP.matches) return;
  const list = panel.querySelector(".evidence-list");
  const mark = list.querySelector("mark.trace");
  const card = mark ? mark.closest(".ev") : list.querySelector(".ev.is-target");
  if (!card) return;
  // offsets are relative to the list (it is the positioned ancestor)
  const top = mark ? Math.max(card.offsetTop - 8, mark.offsetTop - 140) : card.offsetTop - 8;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  requestAnimationFrame(() => list.scrollTo({ top: Math.max(0, top), behavior: reduce ? "auto" : "smooth" }));
}
