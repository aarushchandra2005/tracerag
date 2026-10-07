"""Hand-label the sentences of generated answers as supported / unsupported.

    python scripts/label_answers.py results/generation/answers_<tag>.jsonl

Shows each answer sentence next to the chunk(s) it cites. Labels are saved
after every answer to results/generation/labels_<tag>.json, so you can quit
and resume. run_verifier_eval uses them instead of "presumed supported".
Tip: split the answers between team members and label independently, then
compare a shared subset to report agreement.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import read_jsonl, setup_console, write_json  # noqa: E402
from src.verify import splitter  # noqa: E402


def main() -> None:
    setup_console()
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    tag = path.stem.replace("answers_", "")
    out = path.parent / f"labels_{tag}.json"
    labels = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for a in read_jsonl(path):
        chunks = {r["chunk_id"]: r for r in a["retrieved"]}
        parsed = splitter.split(a["answer"], list(chunks))
        todo = [i for i, p in enumerate(parsed) if not p["abstain"] and str(i) not in labels.get(a["qid"], {})]
        if not todo:
            continue
        print("\n" + "=" * 100 + f"\nClaim {a['qid']}: {a['claim']}")
        for i in todo:
            p = parsed[i]
            print("-" * 100 + f"\nSentence {i}: {p['text']}\nCites: {p['cited_ids'] or 'nothing'}")
            for cid in p["cited_ids"]:
                c = chunks.get(cid)
                body = f"{c['title']}. {c['text']}" if c else "(not a retrieved chunk)"
                print(textwrap.fill(f"[{cid}] {body}", 100, initial_indent="  ", subsequent_indent="  "))
            while True:
                ans = input("supported? [s]upported / [u]nsupported / [k] skip / [q] quit > ").strip().lower()
                if ans in ("s", "u", "k", "q"):
                    break
            if ans == "q":
                write_json(out, labels)
                print(f"saved {out}")
                return
            if ans != "k":
                labels.setdefault(a["qid"], {})[str(i)] = "supported" if ans == "s" else "unsupported"
        write_json(out, labels)
    print(f"done, saved {out}")


if __name__ == "__main__":
    main()
