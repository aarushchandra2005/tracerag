"""Split a generated answer into sentences and pull out each sentence's citations.

    split("Vitamin D cut fractures [123_4]. No effect on falls. [55]")
    -> [{"sentence": "Vitamin D cut fractures [123_4].", "text": "Vitamin D cut fractures.",
         "cited_ids": ["123_4"], "invalid_ids": [], "abstain": False},
        {"sentence": "No effect on falls. [55]", "text": "No effect on falls.",
         "cited_ids": ["55"], ...}]

A citation written after the full stop ("... falls. [55] Next ...") belongs to
the sentence before it, so leading citation groups are moved back. Bracketed
text that is not an id (e.g. "[Ca2+]i") is left alone.
"""
from __future__ import annotations

import re
from typing import Iterable

from src.ingest.chunking import split_sentences

_GROUP_RE = re.compile(r"\[([^\[\]]{1,200})\]")
_ID_RE = re.compile(r"^\d+(?:_\d+)?$")                     # SciFact chunk ids: 4983 or 4983_2
_PREFIX_RE = re.compile(r"^(?:chunk|doc(?:ument)?|id|source)\s*[:#]?\s*", re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"^\W*(?:there is\s+)?(?:not enough|insufficient|no relevant)\s+evidence\b",
                         re.IGNORECASE)


def _citation_ids(group: str, valid_ids: set[str] | None) -> list[str] | None:
    """Ids inside one [...] group, or None if the group is ordinary bracketed text."""
    parts = [_PREFIX_RE.sub("", p.strip()) for p in re.split(r"[,;]|\band\b", group)]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if all(_ID_RE.match(p) or (valid_ids is not None and p in valid_ids) for p in parts):
        return parts
    return None


def _leading_citations(sentence: str, valid_ids) -> tuple[str, str]:
    """Split '[1][2] Rest' into ('[1][2]', 'Rest') when every leading group is a citation."""
    pos = 0
    while True:
        m = re.match(r"\s*\[([^\[\]]{1,200})\]", sentence[pos:])
        if not m or _citation_ids(m.group(1), valid_ids) is None:
            break
        pos += m.end()
    return sentence[:pos].strip(), sentence[pos:].strip()


def strip_citations(sentence: str, valid_ids: set[str] | None = None) -> str:
    def repl(m):
        return "" if _citation_ids(m.group(1), valid_ids) is not None else m.group(0)

    text = _GROUP_RE.sub(repl, sentence)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split(answer: str, valid_ids: Iterable[str] | None = None) -> list[dict]:
    valid = set(valid_ids) if valid_ids is not None else None
    answer = re.sub(r"([.!?])\[", r"\1 [", answer or "")      # "effect.[12] Next" -> "effect. [12] Next"
    sentences = split_sentences(answer)
    merged: list[str] = []
    for s in sentences:
        lead, rest = _leading_citations(s, valid)
        if lead and merged:                       # "... falls. [55] Next" -> move [55] back
            merged[-1] = f"{merged[-1]} {lead}"
            if rest:
                merged.append(rest)
        else:
            merged.append(s)
    out = []
    for s in merged:
        cited: list[str] = []
        for m in _GROUP_RE.finditer(s):
            ids = _citation_ids(m.group(1), valid)
            if ids:
                cited.extend(i for i in ids if i not in cited)
        text = strip_citations(s, valid)
        out.append({
            "sentence": s,
            "text": text,
            "cited_ids": cited,
            "invalid_ids": [i for i in cited if valid is not None and i not in valid],
            "abstain": not cited and bool(_ABSTAIN_RE.match(text)),
        })
    return [o for o in out if o["text"] or o["cited_ids"]]


def join(parsed: list[dict]) -> str:
    return " ".join(p["sentence"] for p in parsed)


def with_citations(text: str, ids: list[str]) -> str:
    """'Cells died.' + ['12_0'] -> 'Cells died [12_0].'"""
    text = text.strip()
    tag = "".join(f"[{i}]" for i in ids)
    if text and text[-1] in ".!?":
        return f"{text[:-1].rstrip()} {tag}{text[-1]}"
    return f"{text} {tag}."
