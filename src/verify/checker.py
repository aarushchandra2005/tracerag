"""Claim-level citation checking: is each answer sentence backed by the chunk(s) it cites?

Evidence is compared at the level of short windows (the paper title, then
every run of PREMISE_WINDOW consecutive sentences of the cited chunk): a claim
is usually backed by one or two sentences, and comparing against a whole
abstract dilutes both cosine and NLI scores. The best window over all cited
chunks decides.

Three checkers, all with thresholds tuned on SciFact TRAIN claims:
  cosine   supported if max cosine(sentence, window) >= t_cos
  nli      supported if max P(entailment | window, sentence) >= t_nli
  cascade  cheap cosine gate first: below the gate -> unsupported without
           running NLI; otherwise NLI decides as above

A sentence with no citation, or citing a chunk that was not retrieved, is
unsupported whatever the scores. "Not enough evidence." is an abstention and
is not checked.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

import config
from src.ingest.chunking import split_sentences
from src.utils import tuned
from src.verify import splitter

METHODS = ("cosine", "nli", "cascade")


@dataclass
class Verdict:
    label: str                   # supported | unsupported | abstain
    reason: str
    cosine: float | None = None
    entail: float | None = None
    contradict: float | None = None
    neutral: float | None = None
    best_chunk: str | None = None
    evidence: str | None = None
    nli_called: bool = False

    @property
    def supported(self) -> bool:
        return self.label == "supported"

    def to_dict(self) -> dict:
        return asdict(self)


def thresholds() -> dict:
    return {
        "cosine": float(tuned("verifier_cosine_threshold", config.DEFAULT_COSINE_THRESHOLD)),
        "nli": float(tuned("verifier_nli_threshold", config.DEFAULT_NLI_THRESHOLD)),
        "gate": float(tuned("verifier_cascade_gate", config.DEFAULT_CASCADE_GATE)),
        "cascade_nli": float(tuned("verifier_cascade_nli_threshold", config.DEFAULT_NLI_THRESHOLD)),
    }


def evidence_windows(title: str, text: str, window: int = config.PREMISE_WINDOW) -> list[str]:
    sents = split_sentences(text)
    wins = [title.strip()] if title and title.strip() else []
    if len(sents) <= window:
        if sents:
            wins.append(" ".join(sents))
    else:
        wins += [" ".join(sents[i:i + window]) for i in range(len(sents) - window + 1)]
    return wins or [text]


class CitationChecker:
    def __init__(self, method: str = "cascade", embed_model: str = config.DENSE_MODEL,
                 nli_model: str | None = config.NLI_MODEL, window: int = config.PREMISE_WINDOW,
                 thresholds_override: dict | None = None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.embed_model = embed_model
        self.nli_model_name = nli_model
        self.window = window
        self.t = thresholds() | (thresholds_override or {})
        self._nli = None
        self._nli_lock = threading.Lock()
        self._win_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._sent_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- models (lazy)
    @property
    def nli(self):
        if self._nli is None:
            with self._nli_lock:            # two threads must not load the model twice
                if self._nli is None:
                    from src.verify.nli import NLIModel

                    self._nli = NLIModel(self.nli_model_name)
        return self._nli

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        from src.index.dense_index import encode

        return encode(list(texts), self.embed_model)

    def _windows(self, chunk) -> tuple[list[str], np.ndarray]:
        key = f"{chunk.chunk_id}|{self.window}"
        if key not in self._win_cache:
            wins = evidence_windows(chunk.title, chunk.text, self.window)
            self._win_cache[key] = (wins, self._embed(wins))
        return self._win_cache[key]

    def _sentence_vec(self, sentence: str) -> np.ndarray:
        if sentence not in self._sent_cache:
            self._sent_cache[sentence] = self._embed([sentence])[0]
        return self._sent_cache[sentence]

    def warm(self, sentences: Sequence[str], chunks: Sequence) -> None:
        """Batch-encode many sentences and chunk windows at once (used by the evaluation)."""
        new = [s for s in dict.fromkeys(sentences) if s not in self._sent_cache]
        if new:
            for s, v in zip(new, self._embed(new)):
                self._sent_cache[s] = v
        todo = [c for c in {c.chunk_id: c for c in chunks}.values()
                if f"{c.chunk_id}|{self.window}" not in self._win_cache]
        if todo:
            wins = [evidence_windows(c.title, c.text, self.window) for c in todo]
            flat = self._embed([w for ws in wins for w in ws])
            pos = 0
            for c, ws in zip(todo, wins):
                self._win_cache[f"{c.chunk_id}|{self.window}"] = (ws, flat[pos:pos + len(ws)])
                pos += len(ws)

    # ------------------------------------------------------------- raw scores
    def score_cosine(self, sentence: str, chunk) -> tuple[float, str]:
        """Best cosine between the sentence and any window of the chunk, and that window."""
        wins, vecs = self._windows(chunk)
        sims = vecs @ self._sentence_vec(sentence)
        i = int(np.argmax(sims))
        return float(sims[i]), wins[i]

    def score_nli(self, sentence: str, chunk) -> dict:
        """NLI probabilities from the window most likely to entail the sentence,
        plus the strongest contradiction found in any window."""
        wins, _ = self._windows(chunk)
        probs = self.nli.predict([(w, sentence) for w in wins])
        i = int(np.argmax(probs[:, 0]))
        return {"entail": float(probs[i, 0]), "neutral": float(probs[i, 1]),
                "contradict": float(probs[:, 2].max()), "evidence": wins[i]}

    def pair_scores(self, sentence: str, chunks: Sequence, use_nli: bool = True) -> dict:
        """Best scores over all cited chunks (used for threshold tuning)."""
        best = {"cosine": -1.0, "cos_chunk": None, "cos_evidence": None}
        for c in chunks:
            s, ev = self.score_cosine(sentence, c)
            if s > best["cosine"]:
                best.update(cosine=s, cos_chunk=c.chunk_id, cos_evidence=ev)
        if use_nli:
            best.update(entail=-1.0, contradict=0.0, neutral=None, nli_chunk=None, nli_evidence=None)
            for c in chunks:
                r = self.score_nli(sentence, c)
                best["contradict"] = max(best["contradict"], r["contradict"])
                if r["entail"] > best["entail"]:
                    best.update(entail=r["entail"], neutral=r["neutral"], nli_chunk=c.chunk_id,
                                nli_evidence=r["evidence"])
        return best

    # ------------------------------------------------------------- verdicts
    def verdict(self, sentence: str, chunks: Sequence, method: str | None = None,
                invalid_ids: Sequence[str] = (), abstain: bool = False) -> Verdict:
        method = method or self.method
        if abstain:
            return Verdict("abstain", "answer declined: not enough evidence")
        if invalid_ids:
            return Verdict("unsupported", f"cites chunk(s) that were not retrieved: {', '.join(invalid_ids)}")
        if not chunks:
            return Verdict("unsupported", "no citation")
        s = self.pair_scores(sentence, chunks, use_nli=False)
        v = Verdict("unsupported", "", cosine=s["cosine"], best_chunk=s["cos_chunk"], evidence=s["cos_evidence"])
        if method == "cosine":
            ok = s["cosine"] >= self.t["cosine"]
            v.label = "supported" if ok else "unsupported"
            v.reason = f"cosine {s['cosine']:.2f} {'>=' if ok else '<'} {self.t['cosine']:.2f}"
            return v
        if method == "cascade" and s["cosine"] < self.t["gate"]:
            v.reason = f"off-topic: cosine {s['cosine']:.2f} below gate {self.t['gate']:.2f} (NLI skipped)"
            return v
        threshold = self.t["nli"] if method == "nli" else self.t["cascade_nli"]
        n = self.pair_scores(sentence, chunks, use_nli=True)
        v.entail, v.contradict, v.neutral = n["entail"], n["contradict"], n["neutral"]
        v.best_chunk, v.evidence, v.nli_called = n["nli_chunk"], n["nli_evidence"], True
        if n["entail"] >= threshold:
            v.label, v.reason = "supported", f"entailed (p={n['entail']:.2f})"
        elif n["contradict"] >= 0.5:
            v.reason = f"contradicted by the cited evidence (p={n['contradict']:.2f})"
        else:
            v.reason = f"not entailed (p={n['entail']:.2f} < {threshold:.2f})"
        return v

    def check_answer(self, answer: str, retrieved: Sequence, method: str | None = None) -> list[dict]:
        """Split an answer and judge every sentence against the retrieved chunks it cites."""
        by_id = {c.chunk_id: c for c in retrieved}
        rows = []
        for p in splitter.split(answer, by_id):
            cited = [by_id[i] for i in p["cited_ids"] if i in by_id]
            v = self.verdict(p["text"], cited, method, p["invalid_ids"], p["abstain"])
            rows.append({**p, "verdict": v})
        return rows
