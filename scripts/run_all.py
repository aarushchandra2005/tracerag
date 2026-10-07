"""Run the whole pipeline end to end and rebuild every table and figure in results/.

    python scripts/run_all.py                           # everything, offline extractive answers
    python scripts/run_all.py --llm groq --model <m>    # answers from an API LLM as well
    python scripts/run_all.py --no-nli                  # skip the NLI model (cosine verifier only)
    python scripts/run_all.py --no-dense                # sparse retrieval only, no model downloads

Steps: data + indexes -> retrieval eval (+ train tuning, ablations, significance)
-> winner analysis -> generation on 30 test claims -> verifier eval -> RESULTS.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.utils import setup_console, timer  # noqa: E402


def write_results_md() -> Path:
    import numpy

    parts = [f"# TraceRAG results\n\nGenerated {dt.datetime.now():%Y-%m-%d %H:%M} "
             f"on Python {platform.python_version()}, numpy {numpy.__version__}.\n"]
    for name in ("retrieval_results.md", "winner_analysis.md", "verifier_results.md"):
        p = config.RESULTS_DIR / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    out = config.RESULTS_DIR / "RESULTS.md"
    out.write_text("\n\n".join(parts), encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--llm", default="extractive")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--n", type=int, default=config.N_GEN_QUERIES)
    args = ap.parse_args(argv)
    setup_console()
    config.ensure_dirs()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_index
    from src.eval import run_retrieval_eval, run_verifier_eval, winner_analysis
    from src.generate import run_generation

    with timer("build indexes"):
        build_index.main(["--ablations"] + ([] if args.no_dense else ["--dense"]))
    with timer("retrieval evaluation"):
        run_retrieval_eval.main(["--no-dense"] if args.no_dense else [])
    if not args.no_dense:
        with timer("winner analysis"):
            winner_analysis.main([])
        with timer("generation (extractive)"):
            run_generation.main(["--llm", "extractive", "--n", str(args.n)])
        if args.llm != "extractive":
            with timer(f"generation ({args.llm})"):
                run_generation.main(["--llm", args.llm, "--model", args.model, "--n", str(args.n)])
        with timer("verifier evaluation"):
            run_verifier_eval.main(["--no-nli"] if args.no_nli else [])
    print(f"\nWrote {write_results_md()}")


if __name__ == "__main__":
    main()
