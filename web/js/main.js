// Router, status polling and theme switching.

import { api } from "./api.js";
import { $, $$, clear, fmtElapsed, h, icon } from "./dom.js";
import { app, parseHash } from "./state.js";
import * as check from "./views/check.js";
import * as compare from "./views/compare.js";
import * as explorer from "./views/explorer.js";
import * as evaluation from "./views/evaluation.js";
import * as about from "./views/about.js";

const VIEWS = { check, compare, index: explorer, evaluation, about };
const mounted = new Set();

function route() {
  const { view, params } = parseHash();
  const name = VIEWS[view] ? view : "check";
  for (const [key] of Object.entries(VIEWS)) {
    const section = document.getElementById(`view-${key}`);
    section.hidden = key !== name;
  }
  $$(".tabs a").forEach((a) => {
    if (a.dataset.view === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  const section = document.getElementById(`view-${name}`);
  if (!mounted.has(name)) { VIEWS[name].mount(section); mounted.add(name); }
  VIEWS[name].enter(params);
  const titles = { check: "Check a claim", compare: "Compare retrievers", index: "Inside the index", evaluation: "Evaluation", about: "About" };
  document.title = `${titles[name]} | TraceRAG`;
}

// ------------------------------------------------------------------ status
let pollTimer = null;
let lastError = false;

function pillState(components, names) {
  const states = names.map((n) => components[n]).filter(Boolean);
  if (states.some((c) => c.state === "error")) return states.find((c) => c.state === "error");
  const busy = states.find((c) => ["waiting", "loading", "building"].includes(c.state));
  if (busy) return busy;
  if (states.every((c) => c.state === "disabled")) return states[0];
  return { state: "ready" };
}

const progressPct = (c) => `${Math.floor((100 * (c.done || 0)) / c.total)}%`;

function renderPills(s) {
  const box = clear(document.getElementById("status-pills"));
  const groups = [
    ["Index", ["data", "sparse"]],
    ["Dense", ["encoder", "dense_whole", "dense_sent"]],
    ["NLI", ["nli"]],
  ];
  for (const [label, names] of groups) {
    const st = pillState(s.components, names);
    const word = { ready: "ready", error: "failed", disabled: "off", waiting: "waiting", loading: "loading", building: "building" }[st.state] || st.state;
    const glyph = st.state === "ready" ? icon("check") : st.state === "error" ? icon("cross")
      : st.state === "disabled" ? icon("dash") : icon("spinner");
    // While busy, the spinner says "working", so the pill shows only the progress or the time (keeps the bar from overflowing).
    const busyNow = st.state === "building" || st.state === "loading";
    const extra = st.state === "building" && st.total ? progressPct(st) : busyNow && st.elapsed ? fmtElapsed(st.elapsed) : "";
    const shown = busyNow && extra ? extra : word;
    const full = `${label}: ${word}${extra ? ` ${extra}` : ""}`;
    box.append(h("a", { class: "pill", href: "#/about", "data-state": st.state, "aria-label": full, title: `${full}${st.message ? ` (${st.message})` : ""}` },
      glyph, h("span", {}, label, h("span", { class: "pill-label", text: ` ${shown}` }))));
  }
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const s = await api.status();
    lastError = false;
    app.setStatus(s);
    renderPills(s);
    renderSetup(s);
    pollTimer = setTimeout(poll, s.busy ? 2500 : 20000);
  } catch (e) {
    if (!lastError) {
      lastError = true;
      const box = clear(document.getElementById("status-pills"));
      box.append(h("span", { class: "pill", "data-state": "error", title: e.message }, icon("cross"), h("span", { text: "Server offline" })));
    }
    pollTimer = setTimeout(poll, 4000);
  }
}

// ------------------------------------------------------------------ setup banner
const PART = {
  encoder: "the MiniLM encoder", dense_whole: "the dense index for whole abstracts",
  dense_sent: "the dense index for sentence windows", nli: "the NLI model",
};
let dismissed = "";

function renderSetup(s) {
  const bar = document.getElementById("setup-bar");
  const comps = s.components;
  const failed = Object.keys(PART).filter((k) => comps[k] && comps[k].state === "error");
  const busy = Object.keys(PART).filter((k) => comps[k] && ["waiting", "loading", "building"].includes(comps[k].state));
  let text = "";
  let error = false;
  if (failed.length) {
    error = true;
    text = `Could not load ${failed.map((k) => PART[k]).join(" or ")}. Everything else works; the About page shows the reason.`;
  } else if (busy.length) {
    const parts = busy.map((k) => {
      const c = comps[k];
      const verb = c.state === "building" ? "is being built" : c.state === "loading" ? "is loading" : "is queued";
      const bits = [];
      if (c.state === "building" && c.total) bits.push(`${progressPct(c)} done`);
      if (c.elapsed && c.state !== "waiting") bits.push(fmtElapsed(c.elapsed));
      return `${PART[k]} ${verb}${bits.length ? ` (${bits.join(", ")})` : ""}`;
    });
    text = `Getting ready: ${parts.join("; ")}. tf-idf and BM25 already work; the rest switches on by itself.`;
  }
  // Remember a dismissal by what is pending, not by the text, which changes as progress ticks.
  const key = `${failed.join(",")}|${busy.join(",")}`;
  if (!text || key === dismissed) { bar.hidden = true; return; }
  bar.hidden = false;
  bar.classList.toggle("is-error", error);
  clear(bar).append(h("div", { class: "inner" }, icon(error ? "alert" : "spinner"), h("span", { style: { flex: "1" }, text }),
    error ? h("a", { href: "#/about", text: "Details" }) : null,
    h("button", { class: "btn-quiet", type: "button", "aria-label": "Hide this message", style: { border: "0", background: "none", cursor: "pointer", color: "inherit", fontSize: "1.1rem", lineHeight: "1" },
      onclick: () => { dismissed = key; bar.hidden = true; } }, "×")));
}

// ------------------------------------------------------------------ help tour
function setupTour() {
  const dlg = document.getElementById("tour");
  dlg.querySelector("[data-close]").append(icon("close"));
  const close = () => dlg.close();
  dlg.querySelector("[data-close]").addEventListener("click", close);
  dlg.querySelector("[data-close-btn]").addEventListener("click", () => { close(); location.hash = "#/check"; });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  const open = () => dlg.showModal();
  const btn = document.getElementById("help-open");
  btn.append(icon("help"));
  btn.addEventListener("click", open);
  document.getElementById("help-open-2").addEventListener("click", open);
  window.addEventListener("opentour", open);
}

// ------------------------------------------------------------------ theme
function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function renderThemeButton() {
  const btn = document.getElementById("theme-toggle");
  const dark = currentTheme() === "dark";
  clear(btn).append(icon(dark ? "sun" : "moon"));
  btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  btn.title = btn.getAttribute("aria-label");
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("tracerag-theme", next); } catch (e) { /* ignore */ }
  renderThemeButton();
  window.dispatchEvent(new Event("themechange"));
}

// ------------------------------------------------------------------ start
window.addEventListener("hashchange", route);
document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderThemeButton();
  window.dispatchEvent(new Event("themechange"));
});
renderThemeButton();
setupTour();
route();
poll();
