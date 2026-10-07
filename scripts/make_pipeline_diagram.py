"""Draw report/pipeline.png and report/pipeline.svg (the system diagram for the report and video)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import config  # noqa: E402

SURFACE, INK, INK_2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
FILL, ACCENT_FILL, ACCENT = "#f0efec", "#cde2fb", "#2a78d6"
EVAL_FILL = "#fde8de"


def box(ax, x, y, w, h, title, body="", fill=FILL, edge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge or fill, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=LINE, linewidth=1.2, shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.2))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.2, 7.0, "TraceRAG pipeline", fontsize=14, fontweight="bold", color=INK, va="top")
    ax.text(0.2, 6.62, "blue = built from scratch for this project; orange = evaluation", fontsize=8.5,
            color=INK_2, va="top")

    # --- offline indexing
    ax.text(0.2, 6.15, "OFFLINE", fontsize=8, color=LINE, fontweight="bold")
    box(ax, 0.2, 4.6, 2.2, 1.4, "SciFact (BEIR)", "5,183 abstracts\n1,109 claims + qrels\nevidence labels")
    box(ax, 2.9, 4.6, 2.3, 1.4, "Chunking", "whole abstract, or\n3-sentence windows\n(stride 1)", ACCENT_FILL)
    box(ax, 5.7, 4.6, 2.3, 1.4, "Tokenizer", "case fold, punctuation,\nstopwords, Porter stem\n(each switchable)", ACCENT_FILL)
    box(ax, 8.5, 4.6, 3.1, 1.4, "Inverted index", "dictionary: term -> df, offset\npostings: (chunk, tf)\ntitle / body / merged zones",
        ACCENT_FILL, ACCENT)
    box(ax, 2.9, 3.6, 2.3, 0.8, "Dense index", "MiniLM-L6 unit vectors")
    arrow(ax, 2.4, 5.3, 2.9, 5.3)
    arrow(ax, 5.2, 5.3, 5.7, 5.3)
    arrow(ax, 8.0, 5.3, 8.5, 5.3)
    arrow(ax, 4.05, 4.6, 4.05, 4.4)

    # --- query time
    ax.text(0.2, 3.35, "QUERY TIME", fontsize=8, color=LINE, fontweight="bold")
    box(ax, 0.2, 1.75, 1.7, 1.4, "Claim", "e.g. '1 in 5 million\nin UK have abnormal\nPrP positivity.'")
    box(ax, 2.3, 1.75, 2.6, 1.4, "Retrievers", "tf-idf lnc.ltc + zones\nBM25 (k1 1.2, b 0.75)\ndense cosine; RRF hybrid", ACCENT_FILL)
    box(ax, 5.3, 1.75, 1.7, 1.4, "Top-5 chunks", "heap top-K;\npapers ranked by\nbest chunk (MaxP)")
    box(ax, 7.4, 1.75, 2.2, 1.4, "Generator", "LLM must cite\n[chunk_id] after every\nsentence (or extractive)")
    box(ax, 10.0, 1.75, 2.1, 1.4, "Claim splitter", "sentences +\ncitations; flags\ninvalid ids", ACCENT_FILL)
    box(ax, 12.5, 1.75, 2.3, 1.4, "Citation verifier", "cosine gate -> NLI\n(DeBERTa-v3)\nsupported / unsupported",
        ACCENT_FILL, ACCENT)
    for x1, x2 in ((1.9, 2.3), (4.9, 5.3), (7.0, 7.4), (9.6, 10.0), (12.1, 12.5)):
        arrow(ax, x1, 2.45, x2, 2.45)
    arrow(ax, 10.05, 4.6, 4.6, 3.15)
    arrow(ax, 4.05, 3.6, 4.05, 3.15)

    # --- evaluation
    box(ax, 2.3, 0.15, 4.7, 1.1, "Retrieval evaluation", "P@5, P@10, R@10, MRR@10, nDCG@10 on 300 test claims;\n"
        "tuned on 809 train claims; randomization tests, ablations", EVAL_FILL)
    box(ax, 9.6, 0.15, 5.2, 1.1, "Verifier evaluation", "SciFact expert labels (support / contradict / no evidence /\n"
        "wrong paper) + injected corruptions; P/R/F1 of the flag", EVAL_FILL)
    arrow(ax, 4.65, 1.75, 4.65, 1.25)
    arrow(ax, 13.65, 1.75, 13.65, 1.25)

    out = config.REPORT_DIR / "pipeline"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
