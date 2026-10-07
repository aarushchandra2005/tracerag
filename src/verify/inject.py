"""Corrupt a correct answer on purpose, to get labelled "unsupported" sentences for free.

  swap_citation   point one sentence at another retrieved chunk from a different paper
  add_fabricated  append a plausible on-topic sentence (taken from a paper that was not
                  retrieved) and cite one of the retrieved chunks for it
  negate          flip one sentence's claim (not removal, antonym swap, or inserted "not")

Each function returns a Corruption whose `index` is the position of the
corrupted sentence in splitter.split(corruption.answer); that sentence is
labelled unsupported. Functions return None when the answer has nothing to
corrupt (for example, an abstention).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

from src.verify import splitter


@dataclass
class Corruption:
    kind: str
    answer: str
    index: int
    original: str | None
    corrupted: str
    extra_chunk: object = None   # a chunk the corruption cites that the generator never saw


# ----------------------------------------------------------------- negation
_ANTONYMS = [
    ("increases", "decreases"), ("increased", "decreased"), ("increase", "decrease"),
    ("higher", "lower"), ("more", "less"), ("improves", "worsens"), ("improved", "worsened"),
    ("promotes", "inhibits"), ("promoted", "inhibited"), ("enhances", "suppresses"),
    ("enhanced", "suppressed"), ("upregulates", "downregulates"), ("upregulated", "downregulated"),
    ("positive", "negative"), ("positively", "negatively"), ("activates", "inhibits"),
    ("reduces", "increases"), ("reduced", "increased"), ("reduce", "increase"),
    ("prevents", "causes"), ("protective", "harmful"), ("benefit", "harm"),
    ("effective", "ineffective"),
]
_FLIP = {}
for a, b in _ANTONYMS:
    _FLIP.setdefault(a, b)
    _FLIP.setdefault(b, a)
_FLIP_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _FLIP), key=len, reverse=True)) + r")\b",
                      re.IGNORECASE)
_AUX_RE = re.compile(r"\b(is|are|was|were|can|could|does|do|did|has|have|had|will|would|may|might|should|must)\b"
                     r"(?!\s+not\b)", re.IGNORECASE)
_NOT_RE = re.compile(r"\s+not\b", re.IGNORECASE)


def _match_case(src: str, word: str) -> str:
    return word.capitalize() if src[:1].isupper() else word


def negate_sentence(text: str) -> str:
    """Flip the meaning of one sentence with simple, auditable rules (first rule that applies)."""
    if _NOT_RE.search(text):                                    # "did not reduce" -> "did reduce"
        return _NOT_RE.sub("", text, count=1)
    m = _FLIP_RE.search(text)
    if m:                                                       # "increased" -> "decreased"
        word = m.group(1)
        return text[:m.start()] + _match_case(word, _FLIP[word.lower()]) + text[m.end():]
    m = _AUX_RE.search(text)
    if m:                                                       # "is linked" -> "is not linked"
        return text[:m.end()] + " not" + text[m.end():]
    body = text[0].lower() + text[1:] if text[:2] != text[:2].upper() else text
    return "It is not true that " + body                       # last resort


# ----------------------------------------------------------------- helpers
def _claims(parsed: list[dict]) -> list[int]:
    return [i for i, p in enumerate(parsed) if p["cited_ids"] and not p["abstain"]]


def _rebuild(parsed: list[dict], index: int, new_sentence: str) -> str:
    sents = [p["sentence"] for p in parsed]
    sents[index] = new_sentence
    return " ".join(sents)


def _checked(kind: str, answer: str, index: int, original, new: str, valid_ids) -> Corruption | None:
    """Keep the corruption only if re-splitting the answer finds the new sentence at `index`
    (a sentence that starts in lower case, say, could merge with its neighbour)."""
    parsed = splitter.split(answer, valid_ids)
    if index < len(parsed) and parsed[index]["sentence"] == new:
        return Corruption(kind, answer, index, original, new)
    return None


# ----------------------------------------------------------------- corruptions
def negate(answer: str, rng: random.Random, valid_ids=None) -> Corruption | None:
    parsed = splitter.split(answer, valid_ids)
    targets = _claims(parsed)
    if not targets:
        return None
    i = rng.choice(targets)
    p = parsed[i]
    new = splitter.with_citations(negate_sentence(p["text"]), p["cited_ids"])
    return _checked("negate", _rebuild(parsed, i, new), i, p["sentence"], new, valid_ids)


def swap_citation(answer: str, rng: random.Random, retrieved: Sequence, valid_ids=None,
                  fallback: Sequence = ()) -> Corruption | None:
    """Re-point one sentence at a chunk from a different paper.

    retrieved: the Chunk objects the generator was given. When they all come from
    the paper the sentence cites (common with sentence windows), a chunk from
    `fallback` (other on-topic papers) is used and returned as extra_chunk, so the
    evaluation can treat it as retrieved and test the checker rather than the
    "not retrieved" rule.
    """
    parsed = splitter.split(answer, valid_ids)
    doc_of = {c.chunk_id: c.doc_id for c in retrieved}
    targets = _claims(parsed)
    rng.shuffle(targets)
    for i in targets:
        p = parsed[i]
        cited_docs = {doc_of.get(cid) for cid in p["cited_ids"]}
        options = [c for c in retrieved if c.doc_id not in cited_docs]
        extra = None
        if not options:
            options = [c for c in fallback if c.doc_id not in cited_docs]
            if not options:
                continue
            extra = rng.choice(options)
            options = [extra]
        target = rng.choice(options)
        new = splitter.with_citations(p["text"], [target.chunk_id])
        ids = None if valid_ids is None else list(valid_ids) + ([extra.chunk_id] if extra else [])
        corr = _checked("swap_citation", _rebuild(parsed, i, new), i, p["sentence"], new, ids)
        if corr is not None:
            corr.extra_chunk = extra
        return corr
    return None


def add_fabricated(answer: str, rng: random.Random, fabricated: str, cite_ids: Sequence[str],
                   valid_ids=None) -> Corruption | None:
    """Append `fabricated` (a sentence from a paper that was not retrieved) citing one retrieved chunk."""
    parsed = splitter.split(answer, valid_ids)
    if not _claims(parsed) or not cite_ids:
        return None
    new = splitter.with_citations(fabricated, [rng.choice(list(cite_ids))])
    text = " ".join([p["sentence"] for p in parsed] + [new])
    return _checked("add_fabricated", text, len(parsed), None, new, valid_ids)
