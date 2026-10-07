// Thin wrapper over the TraceRAG HTTP API. Errors surface the server's own message.

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

function detailText(data) {
  if (!data) return "";
  const d = data.detail;
  if (Array.isArray(d)) {
    return d.map((e) => `${(e.loc || []).slice(1).join(".") || "input"}: ${e.msg}`).join("; ");
  }
  return typeof d === "string" ? d : "";
}

async function request(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError("Cannot reach the TraceRAG server. Is `python run_app.py` still running?", 0);
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new ApiError(detailText(data) || `${res.status} ${res.statusText}`, res.status);
  return data;
}

const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();

export const api = {
  status: () => request("GET", "/api/status"),
  claims: (params) => request("GET", `/api/claims?${qs(params)}`),
  randomClaim: (params = {}) => request("GET", `/api/claims/random?${qs(params)}`),
  claim: (qid) => request("GET", `/api/claims/${encodeURIComponent(qid)}`),
  paper: (docId) => request("GET", `/api/papers/${encodeURIComponent(docId)}`),
  search: (body) => request("POST", "/api/search", body),
  compare: (body) => request("POST", "/api/compare", body),
  ask: (body) => request("POST", "/api/ask", body),
  verify: (body) => request("POST", "/api/verify", body),
  corrupt: (body) => request("POST", "/api/corrupt", body),
  indexStats: () => request("GET", "/api/index/stats"),
  term: (params) => request("GET", `/api/index/term?${qs(params)}`),
  tokenize: (body) => request("POST", "/api/index/tokenize", body),
  explain: (body) => request("POST", "/api/explain", body),
  evaluation: () => request("GET", "/api/evaluation"),
};
