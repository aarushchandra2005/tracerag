"""Three-way NLI (entailment / neutral / contradiction) with a cross-encoder.

Default model: cross-encoder/nli-deberta-v3-small, whose config maps
0 -> contradiction, 1 -> entailment, 2 -> neutral. The label order is read
from the model config rather than hard-coded, and loading fails loudly if the
model is not a three-way NLI classifier.

Input pairs are (premise = evidence text, hypothesis = answer sentence).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

import config

ORDER = ("entail", "neutral", "contradict")


class NLIModel:
    def __init__(self, model_name: str = config.NLI_MODEL, device: str | None = config.DEVICE,
                 batch_size: int = config.NLI_BATCH_SIZE, max_length: int = config.NLI_MAX_LENGTH,
                 cache_dir: Path = config.CACHE_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size, self.max_length = batch_size, max_length
        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        self.column = {}
        for key in ORDER:
            hits = [i for i, l in id2label.items() if l.startswith(key)]
            if len(hits) != 1:
                raise ValueError(f"{model_name} is not a 3-way NLI model (labels: {id2label})")
            self.column[key] = hits[0]
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._lock = threading.Lock()          # one tokenizer/model, many web-request threads
        # predictions are also kept on disk, so rerunning the evaluation skips work already done
        slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(model_name).name if Path(model_name).exists() else model_name)
        self._disk_path = Path(cache_dir) / f"nli_{slug.strip('-')}.jsonl"
        self._disk: dict[str, list] = {}
        if self._disk_path.exists():
            with open(self._disk_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._disk[rec["k"]] = rec["p"]

    @staticmethod
    def _key(premise: str, hypothesis: str) -> str:
        return hashlib.sha1(f"{premise}\x00{hypothesis}".encode("utf-8")).hexdigest()

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Probabilities, one row per (premise, hypothesis), columns in ORDER."""
        with self._lock:
            return self._predict(pairs)

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), 3), dtype=np.float64)
        for p in pairs:
            if p not in self._cache:
                hit = self._disk.get(self._key(*p))
                if hit is not None:
                    self._cache[p] = np.asarray(hit, dtype=np.float64)
        first: dict[tuple[str, str], int] = {}
        for i, p in enumerate(pairs):
            if p not in self._cache and p not in first:
                first[p] = i
        todo = list(first.values())
        new_rows = []
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            premises = [pairs[i][0] for i in idx]
            hypotheses = [pairs[i][1] for i in idx]
            enc = self.tokenizer(premises, hypotheses, truncation="longest_first", max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            for i, row in zip(idx, probs):
                vec = np.array([row[self.column[k]] for k in ORDER], dtype=np.float64)
                self._cache[pairs[i]] = vec
                new_rows.append({"k": self._key(*pairs[i]), "p": [round(float(x), 6) for x in vec]})
        if new_rows:
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._disk_path, "a", encoding="utf-8", newline="\n") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
        for i, p in enumerate(pairs):
            out[i] = self._cache[p]
        return out
