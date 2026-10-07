// Shared app state: the latest server status, saved preferences, and navigation.

const listeners = new Set();

export const app = {
  status: null,
  onStatus(fn) { listeners.add(fn); if (app.status) fn(app.status); return () => listeners.delete(fn); },
  setStatus(s) { app.status = s; listeners.forEach((fn) => { try { fn(s); } catch (e) { console.error(e); } }); },
};

export const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(`tracerag-${key}`); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`tracerag-${key}`, JSON.stringify(value)); } catch (e) { /* storage blocked: keep going */ }
  },
};

export function navigate(view, params = {}) {
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();
  location.hash = `#/${view}${q ? `?${q}` : ""}`;
}

export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [view, query] = raw.split("?");
  return { view: view || "check", params: Object.fromEntries(new URLSearchParams(query || "")) };
}

export function retrieverAvailable(name, chunking) {
  const s = app.status;
  return !!(s && s.retrievers.available[chunking] && s.retrievers.available[chunking][name]);
}

export function verifierAvailable(method) {
  const s = app.status;
  return !!(s && s.verifier.available.includes(method));
}
