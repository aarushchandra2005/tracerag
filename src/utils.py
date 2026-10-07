"""Small helpers shared by every module: JSON/JSONL I/O, tuned parameters,
console setup and markdown tables."""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import config


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- tuned params
def load_tuned() -> dict:
    """Parameters chosen on the train split (empty dict before the first eval run)."""
    if config.TUNED_PARAMS_PATH.exists():
        return read_json(config.TUNED_PARAMS_PATH)
    return {}


def save_tuned(updates: dict) -> dict:
    params = load_tuned()
    params.update(updates)
    write_json(config.TUNED_PARAMS_PATH, params)
    return params


def tuned(key: str, default: Any) -> Any:
    value = load_tuned().get(key, default)
    return tuple(value) if isinstance(default, tuple) and isinstance(value, list) else value


# ---------------------------------------------------------------- console
def setup_console() -> None:
    """Make printing of Greek letters and other symbols safe on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


@contextmanager
def timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.1f}s", flush=True)


# ---------------------------------------------------------------- tables
def markdown_table(rows: list[dict], columns: list[str] | None = None, floatfmt: str = ".3f") -> str:
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |")
    return "\n".join(lines)
