"""Static charts for the report (matplotlib, PNG + SVG).

Style: one recessive hairline grid, thin bars, text in ink colours (never in
series colours), a legend whenever there are two or more series, and values
written only at bar tips. Palette slots are a colour-blind-checked set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(categories: Sequence[str], series: dict[str, Sequence[float]], path: Path,
                 title: str, subtitle: str = "", ylabel: str = "", ylim: tuple | None = None,
                 fmt: str = "{:.3f}", label_series: Sequence[str] | None = None) -> None:
    """Bars grouped by category, one colour per series; values at the tips of `label_series`."""
    n_cat, n_ser = len(categories), len(series)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.25 * n_cat * max(1, n_ser) ** 0.5 + 2), 4.2))
    width = min(0.8 / n_ser, 0.28)
    label_series = set(series if label_series is None else label_series)
    for j, (name, values) in enumerate(series.items()):
        offset = (j - (n_ser - 1) / 2) * (width + 0.02)
        xs = [i + offset for i in range(n_cat)]
        bars = ax.bar(xs, values, width=width, color=SERIES[j % len(SERIES)], label=name, zorder=2)
        if name in label_series:
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(n_cat))
    ax.set_xticklabels(categories, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    _style(ax, title, subtitle)
    if n_ser > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=min(n_ser, 4),
                  loc="upper left", bbox_to_anchor=(0, -0.12))
    _save(fig, path)


def lines(xs: Sequence, series: dict[str, Sequence[float]], path: Path, title: str,
          subtitle: str = "", xlabel: str = "", ylabel: str = "", mark_x=None, fmt: str = "{:.3f}") -> None:
    """One 2px line per series with end labels; optional vertical marker (e.g. the tuned value)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for j, (name, ys) in enumerate(series.items()):
        color = SERIES[j % len(SERIES)]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5, label=name, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    if mark_x is not None:
        ax.axvline(mark_x, color=AXIS, linewidth=1, zorder=1)
        last = list(series.values())[-1]
        y = last[list(xs).index(mark_x)] if mark_x in list(xs) else None
        label = f"chosen on train: {mark_x:g}" + (f" (test {fmt.format(y)})" if y is not None else "")
        ax.annotate(label, (mark_x, ax.get_ylim()[1]), xytext=(6, -4), textcoords="offset points",
                    color=INK_2, fontsize=8, va="top")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    _style(ax, title, subtitle)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
    ax.margins(x=0.12)
    _save(fig, path)
