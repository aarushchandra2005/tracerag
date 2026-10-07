// Small DOM helpers. Text always goes in through textContent, never innerHTML,
// because abstracts and claims are data, not markup.

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const clear = (el) => { el.replaceChildren(); return el; };

// ------------------------------------------------------------------ icons (stroke icons, 24px grid)
const ICONS = {
  check: "M5 12.5l4.5 4.5L19 7.5",
  cross: "M6.5 6.5l11 11M17.5 6.5l-11 11",
  dash: "M6 12h12",
  info: "M12 11v6M12 7.5v.5M12 21a9 9 0 110-18 9 9 0 010 18z",
  alert: "M12 8.5v5M12 16.5v.5M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z",
  sun: "M12 4V2M12 22v-2M4 12H2M22 12h-2M5.6 5.6L4.2 4.2M19.8 19.8l-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M12 17a5 5 0 100-10 5 5 0 000 10z",
  moon: "M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z",
  close: "M6 6l12 12M18 6L6 18",
  dice: "M5 4h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1zM8.5 8.5h.01M15.5 15.5h.01M15.5 8.5h.01M8.5 15.5h.01M12 12h.01",
  list: "M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01",
  edit: "M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4",
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3",
  circle: "M12 20a8 8 0 100-16 8 8 0 000 16z",
  half: "M12 20a8 8 0 100-16 8 8 0 000 16zM12 4v16",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5",
  help: "M9.2 9.2a2.9 2.9 0 015.6 1c0 1.9-2.8 2.6-2.8 4.3M12 17.5h.01M12 21a9 9 0 110-18 9 9 0 010 18z",
  copy: "M9 9h10v10H9zM5 15V5h10",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  arrow: "M5 12h14M13 6l6 6-6 6",
};

export function icon(name, extra = {}) {
  if (name === "spinner") {
    return svg("svg", { viewBox: "0 0 24 24", class: "spinner", "aria-hidden": "true", fill: "none" },
      svg("circle", { cx: 12, cy: 12, r: 9, stroke: "currentColor", "stroke-width": 2.5, opacity: 0.25 }),
      svg("path", { d: "M21 12a9 9 0 00-9-9", stroke: "currentColor", "stroke-width": 2.5, "stroke-linecap": "round" }));
  }
  return svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", ...extra },
  svg("path", { d: ICONS[name] || ICONS.circle }));
}

// ------------------------------------------------------------------ formatting
export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
export const pct = (x, d = 0) => (x === null || x === undefined) ? "–" : `${(100 * x).toFixed(d)}%`;
export const fmtP = (p) => (p === null || p === undefined) ? "–" : (p < 0.001 ? "< 0.001" : Number(p).toFixed(3));
export function fmtMs(ms) {
  if (ms === null || ms === undefined) return "–";
  if (ms < 1) return "under 1 ms";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
export function fmtElapsed(sec) {
  if (sec === null || sec === undefined) return "";
  const s = Math.round(sec);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}
export const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
export const normSpace = (t) => (t || "").replace(/\s+/g, " ").trim();

// ------------------------------------------------------------------ toasts
export function toast(message, kind = "error", timeout = 7000) {
  const box = document.getElementById("toasts");
  const item = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(kind === "error" ? "alert" : "check"),
    h("span", { text: message }),
    h("button", { type: "button", "aria-label": "Dismiss", text: "×", onclick: () => item.remove() }));
  box.append(item);
  if (timeout) setTimeout(() => item.remove(), timeout);
}

// ------------------------------------------------------------------ shared pieces
export function verdictMark(label) {
  const name = label === "supported" ? "check" : label === "unsupported" ? "cross" : "dash";
  return h("span", { class: "verdict-mark" }, icon(name));
}

export const VERDICT_TEXT = { supported: "Supported", unsupported: "Unsupported", abstain: "Abstained" };

export function labelBadge(label) {
  const map = {
    SUPPORT: ["badge-ok", "Supported by a judged paper"],
    CONTRADICT: ["badge-bad", "Contradicted by a judged paper"],
    NEI: ["badge-neutral", "Judged paper has no evidence"],
    NO_EVIDENCE: ["badge-neutral", "cited, no evidence"],
  };
  const [cls, text] = map[label] || ["badge-neutral", label];
  return h("span", { class: `badge ${cls}`, text });
}

export function table(columns, rows, opts = {}) {
  // columns: [{key, label, num, format(value,row), title}]
  const thead = h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "n" : "", scope: "col", title: c.title }, c.label))));
  const tbody = h("tbody", {}, rows.map((r) => {
    const tr = h("tr", { class: opts.rowClass ? opts.rowClass(r) || "" : "" });
    for (const c of columns) {
      const v = c.format ? c.format(r[c.key], r) : r[c.key];
      tr.append(h("td", { class: c.num ? "n" : "" }, v instanceof Node ? v : (v ?? "–")));
    }
    return tr;
  }));
  const t = h("table", { class: "data" }, thead, tbody);
  if (opts.caption) t.append(h("caption", { text: opts.caption }));
  return h("div", { class: "table-wrap" }, t);
}
