// Small SVG charts: grouped bars, lines and a log-log scatter, each with hover tooltips.
// Colours come from CSS variables so light and dark themes both work; charts redraw on resize and theme change.

import { append, clear, h, svg } from "./dom.js";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ------------------------------------------------------------------ tooltip
export function showTip(evt, nodes) {
  const tip = document.getElementById("tooltip");
  clear(tip).append(...nodes);
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const rect = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + rect.width > W - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > H - 8) y = evt.clientY - rect.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
export function hideTip() { document.getElementById("tooltip").hidden = true; }

function tipAtElement(el, nodes) {
  const r = el.getBoundingClientRect();
  showTip({ clientX: r.left + r.width / 2, clientY: r.top }, nodes);
}

// ------------------------------------------------------------------ helpers
function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}

function ticks(min, max, count = 5) {
  const step = niceStep(max - min || 1, count);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function responsive(holder, draw) {
  let last = 0;
  let ro = null;
  const onTheme = () => render(true);
  function render(force) {
    if (last && !holder.isConnected) {          // chart was replaced: stop listening
      window.removeEventListener("themechange", onTheme);
      if (ro) ro.disconnect();
      return;
    }
    const w = Math.floor(holder.clientWidth);
    if (!w || (w === last && !force)) return;
    last = w;
    draw(w);
  }
  ro = new ResizeObserver(() => requestAnimationFrame(() => render(false)));
  ro.observe(holder);
  window.addEventListener("themechange", onTheme);
  requestAnimationFrame(() => render(true));
}

function legend(items, line = false) {
  return h("div", { class: "legend" }, items.map((s) => h("span", {}, h("i", { class: line ? "line" : "", style: { background: css(s.color) } }), s.name)));
}

function wrap(text, maxChars) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > maxChars && cur) { lines.push(cur); cur = w; } else cur = `${cur} ${w}`.trim();
  }
  if (cur) lines.push(cur);
  return lines.slice(0, 3);
}

function tipRow(color, label, value) {
  return h("div", { class: "tt-row" }, color ? h("i", { style: { background: css(color) } }) : null, h("span", { text: label }), h("strong", { style: { marginLeft: "auto", paddingLeft: "10px" }, text: value }));
}

// ------------------------------------------------------------------ grouped bars
// opts: { categories, series: [{ name, values, color }], yMax, format, labelSeries: [indices], height, ariaLabel }
export function groupedBars(el, opts) {
  const holder = h("div", { class: "chart" });
  append(clear(el), [opts.series.length > 1 ? legend(opts.series) : null, holder]);
  const fmtv = opts.format || ((v) => v.toFixed(3));
  responsive(holder, (W) => {
    const H = opts.height || 280;
    const m = { t: 18, r: 6, b: 50, l: 38 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const all = opts.series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
    const yMax = opts.yMax ?? Math.max(...all) * 1.12;
    const y = (v) => m.t + ih - (v / yMax) * ih;
    const band = iw / opts.categories.length;
    const n = opts.series.length;
    const gap = (opts.labelSeries || []).length > 1 ? 10 : 2;    // room for two value labels side by side
    const bw = Math.max(4, Math.min(24, (band * 0.7 - (n - 1) * gap) / n));
    const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.ariaLabel || "bar chart" });
    for (const t of ticks(0, yMax, 5)) {
      root.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: css("--grid"), "stroke-width": 1 }));
      root.append(svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11 }, opts.tickFormat ? opts.tickFormat(t) : String(+t.toFixed(2))));
    }
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: css("--axis"), "stroke-width": 1 }));
    opts.categories.forEach((cat, i) => {
      const cx = m.l + band * (i + 0.5);
      const gw = n * bw + (n - 1) * gap;
      opts.series.forEach((s, j) => {
        const v = s.values[i];
        if (v === null || v === undefined) return;
        const x = cx - gw / 2 + j * (bw + gap);
        const top = y(v);
        const base = y(0);
        const r = Math.min(4, bw / 2, Math.max(0, base - top));
        const d = `M${x},${base} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${base} Z`;
        const tip = [h("strong", { text: cat }), tipRow(s.color, s.name, fmtv(v))];
        const bar = svg("path", { d, fill: css(s.color), tabindex: 0, "aria-label": `${cat}, ${s.name}: ${fmtv(v)}`,
          onmousemove: (e) => showTip(e, tip), onmouseleave: hideTip, onfocus: () => tipAtElement(bar, tip), onblur: hideTip });
        root.append(bar);
        if ((opts.labelSeries || []).includes(j)) {
          root.append(svg("text", { x: x + bw / 2, y: top - 5, "text-anchor": "middle", "font-size": 10 }, fmtv(v)));
        }
      });
      const lines = wrap(cat, Math.max(8, Math.floor(band / 6.6)));
      lines.forEach((line, k) => root.append(svg("text", { x: cx, y: H - m.b + 17 + k * 13, "text-anchor": "middle", "font-size": 11 }, line)));
    });
    clear(holder).append(root);
  });
}

// ------------------------------------------------------------------ lines
// opts: { xs, series: [{ name, values, color }], xLabel, yLabel, markX, markLabel, format, height }
export function lineChart(el, opts) {
  const holder = h("div", { class: "chart" });
  append(clear(el), [opts.series.length > 1 ? legend(opts.series, true) : null, holder]);
  const fmtv = opts.format || ((v) => v.toFixed(3));
  responsive(holder, (W) => {
    const H = opts.height || 250;
    const m = { t: 16, r: 14, b: 42, l: 46 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const xs = opts.xs;
    const all = opts.series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const padY = (hi - lo || 0.1) * 0.12;
    lo -= padY; hi += padY;
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const x = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * iw;
    const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.ariaLabel || "line chart" });
    for (const t of ticks(lo, hi, 4)) {
      root.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: css("--grid") }));
      root.append(svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11 }, t.toFixed(2)));
    }
    for (const t of opts.xTicks || ticks(xMin, xMax, 6)) {
      root.append(svg("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle", "font-size": 11 }, String(t)));
    }
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: css("--axis") }));
    if (opts.xLabel) root.append(svg("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", "font-size": 11 }, opts.xLabel));
    if (opts.yLabel) root.append(svg("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${m.t + ih / 2})` }, opts.yLabel));
    if (opts.markX !== undefined && opts.markX !== null) {
      root.append(svg("line", { x1: x(opts.markX), x2: x(opts.markX), y1: m.t, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1.5 }));
      root.append(svg("text", { x: x(opts.markX) + 6, y: m.t + 10, "font-size": 11 }, opts.markLabel || "chosen"));
    }
    for (const s of opts.series) {
      const pts = xs.map((xv, i) => [xv, s.values[i]]).filter(([, v]) => v !== null && v !== undefined);
      root.append(svg("path", { d: pts.map(([xv, v], i) => `${i ? "L" : "M"}${x(xv)},${y(v)}`).join(" "), fill: "none",
        stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      for (const [xv, v] of pts) root.append(svg("circle", { cx: x(xv), cy: y(v), r: 4, fill: css(s.color), stroke: css("--surface"), "stroke-width": 2 }));
    }
    const cross = svg("line", { y1: m.t, y2: m.t + ih, stroke: css("--line-strong"), "stroke-width": 1, visibility: "hidden" });
    root.append(cross);
    const overlay = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    overlay.addEventListener("mousemove", (e) => {
      const bx = root.getBoundingClientRect().left;
      const mx = e.clientX - bx;
      let best = 0;
      xs.forEach((xv, i) => { if (Math.abs(x(xv) - mx) < Math.abs(x(xs[best]) - mx)) best = i; });
      cross.setAttribute("x1", x(xs[best])); cross.setAttribute("x2", x(xs[best])); cross.setAttribute("visibility", "visible");
      showTip(e, [h("strong", { text: `${opts.xName || "x"} = ${xs[best]}` }), ...opts.series.map((s) => tipRow(s.color, s.name, s.values[best] === null || s.values[best] === undefined ? "–" : fmtv(s.values[best])))]);
    });
    overlay.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    root.append(overlay);
    clear(holder).append(root);
  });
}

// ------------------------------------------------------------------ log-log scatter of first-relevant ranks
// opts: { points: [{ x, y, n, group, label }], groups: [{ key, name, color }], xLabel, yLabel, onSelect }
export function rankScatter(el, opts) {
  const holder = h("div", { class: "chart" });
  clear(el).append(legend(opts.groups), holder);
  responsive(holder, (W) => {
    const size = Math.min(W, 520);
    const H = size;
    const m = { t: 12, r: 12, b: 44, l: 52 };
    const iw = size - m.l - m.r;
    const ih = H - m.t - m.b;
    const LO = 0.6;                                  // pad below rank 1 so big dots clear the axis labels
    const HI = 135;
    const L = (v) => Math.log(v / LO) / Math.log(HI / LO);
    const x = (v) => m.l + L(v) * iw;
    const y = (v) => m.t + ih - L(v) * ih;
    const root = svg("svg", { width: size, height: H, viewBox: `0 0 ${size} ${H}`, role: "img", "aria-label": opts.ariaLabel || "scatter chart" });
    const marks = [1, 2, 5, 10, 20, 50, 101];
    for (const t of marks) {
      const label = t === 101 ? ">100" : String(t);
      root.append(svg("line", { x1: x(t), x2: x(t), y1: m.t, y2: m.t + ih, stroke: css("--grid") }));
      root.append(svg("line", { x1: m.l, x2: m.l + iw, y1: y(t), y2: y(t), stroke: css("--grid") }));
      root.append(svg("text", { x: x(t), y: m.t + ih + 16, "text-anchor": "middle", "font-size": 11 }, label));
      root.append(svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11 }, label));
    }
    root.append(svg("line", { x1: x(LO), y1: y(LO), x2: x(HI), y2: y(HI), stroke: css("--axis"), "stroke-width": 1.5 }));
    root.append(svg("text", { x: x(40), y: y(40) - 8, "font-size": 11, "text-anchor": "end", transform: `rotate(-45 ${x(40)} ${y(40) - 8})` }, "same rank"));
    root.append(svg("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", "font-size": 11 }, opts.xLabel));
    root.append(svg("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${m.t + ih / 2})` }, opts.yLabel));
    const color = Object.fromEntries(opts.groups.map((g) => [g.key, g.color]));
    const name = Object.fromEntries(opts.groups.map((g) => [g.key, g.name]));
    const pts = [...opts.points].sort((a, b) => b.n - a.n);
    for (const p of pts) {
      const r = Math.min(20, 3.5 + 2.4 * Math.sqrt(p.n));
      const tip = [h("strong", { text: `${p.n} ${p.n === 1 ? "claim" : "claims"} (${name[p.group]})` }),
        h("div", { text: `BM25 rank ${p.x > 100 ? ">100" : p.x}, dense rank ${p.y > 100 ? ">100" : p.y}` }),
        h("div", { class: "muted", text: "Click to list the claims" })];
      const c = svg("circle", { cx: x(p.x), cy: y(p.y), r, fill: css(color[p.group]), "fill-opacity": 0.85,
        stroke: css("--surface"), "stroke-width": 1.5, tabindex: 0, role: "button", style: "cursor:pointer",
        "aria-label": `${p.n} claims: BM25 rank ${p.x}, dense rank ${p.y}`,
        onmousemove: (e) => showTip(e, tip), onmouseleave: hideTip,
        onfocus: () => tipAtElement(c, tip), onblur: hideTip,
        onclick: () => opts.onSelect && opts.onSelect(p),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); opts.onSelect && opts.onSelect(p); } } });
      root.append(c);
    }
    clear(holder).append(root);
  });
}
