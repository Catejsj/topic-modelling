"""Figures for the slides. Every number is read from the data, never typed in.

fig1  coherence against number of topics, both models, with the elbow marked
fig2  median view ratio by LDA topic, as a deviation from the channel norm
fig3  the same question with no model in it: single words

Palette and mark rules follow the project's chart conventions: one hue for
magnitude, blue/red with a neutral midpoint for polarity, recessive grid, direct
value labels, no legend when there is only one series.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROC = ROOT / "data" / "processed"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"
ORANGE = "#eb6834"
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK,
})


def strip(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(GRID)
    ax.tick_params(length=0)


def fig_sweep() -> None:
    d = pd.read_csv(PROC / "k_sweep.csv")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(d["k"], d["lda_cv"], color=BLUE, lw=2, marker="o", ms=5,
            label="LDA", zorder=3)
    ax.fill_between(d["k"], d["lda_cv"] - d["lda_cv_sd"],
                    d["lda_cv"] + d["lda_cv_sd"], color=BLUE, alpha=0.15,
                    lw=0, zorder=2)
    ax.plot(d["k"], d["lsa_cv"], color=ORANGE, lw=2, marker="s", ms=5,
            label="LSA", zorder=3)
    ax.plot(d["k"], d["nmf_cv"], color=RED, lw=1.5, ls=(0, (3, 2)),
            marker="^", ms=4, label="NMF (optional)", zorder=2, alpha=0.85)

    chosen = meta["best_lda"]
    ax.axvline(chosen, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"chosen k = {chosen}", xy=(chosen, ax.get_ylim()[1]),
                xytext=(chosen + 1.5, d["lda_cv"].max() * 0.99),
                color=INK2, fontsize=9)

    ax.set_xlabel("number of topics (k)")
    ax.set_ylabel("c_v coherence   (higher is better)")
    ax.set_title("Coherence keeps rising with k,\nso the peak is not the answer",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "shaded band is +-1 sd across three random seeds",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_coherence_vs_k.png", dpi=200)
    plt.close(fig)


def fig_topic_views() -> None:
    d = pd.read_csv(PROC / "topic_views.csv")
    d = d[d["model"] == "LDA"].sort_values("median_ratio")
    label = [f"{int(t):>2}  {w.split(',')[0].strip()}, "
             f"{w.split(',')[1].strip()}, {w.split(',')[2].strip()}"
             for t, w in zip(d["topic"], d["words"])]

    # deviation from the channel norm: 1.0 is the neutral midpoint
    dev = d["median_ratio"] - 1.0
    colors = [BLUE if v >= 0 else RED for v in dev]

    fig, ax = plt.subplots(figsize=(9, 6.4))
    y = np.arange(len(d))
    ax.barh(y, dev, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=4)

    for yi, v, n in zip(y, dev, d["n"]):
        off = 0.015 if v >= 0 else -0.015
        ax.text(v + off, yi, f"{1 + v:.2f}x", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)

    ax.set_yticks(y, label, fontsize=9)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlabel("median views relative to the same channel's median video")
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4],
                  ["0.4x", "0.6x", "0.8x", "1.0x", "1.2x", "1.4x"])
    ax.set_xlim(dev.min() - 0.14, dev.max() + 0.10)
    ax.set_title("Case-narrative topics beat the channel norm;\n"
                 "instructional topics fall far below it",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, f"LDA, k={len(d)}; each bar is the median of the titles "
                     "assigned to that topic",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_views_by_topic.png", dpi=200)
    plt.close(fig)


def fig_words() -> None:
    docs = pd.read_csv(PROC / "doc_topics.csv")
    docs["tokens"] = docs["tokens"].fillna("").str.split()
    # one title counts once per word, even if it repeats the word
    unique_words = docs["tokens"].map(lambda t: sorted(set(t)))
    ex = docs[["view_ratio"]].join(unique_words.explode().rename("word"))
    ex = ex.dropna(subset=["word"])
    per = ex.groupby("word").agg(n=("view_ratio", "size"),
                                 med=("view_ratio", "median"))
    per = per[per["n"] >= 40].sort_values("med")
    show = pd.concat([per.head(10), per.tail(10)])

    dev = np.log2(show["med"])
    colors = [BLUE if v >= 0 else RED for v in dev]

    fig, ax = plt.subplots(figsize=(8, 6.4))
    y = np.arange(len(show))
    ax.barh(y, dev, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=4)
    for yi, v, m, n in zip(y, dev, show["med"], show["n"]):
        off = 0.08 if v >= 0 else -0.08
        ax.text(v + off, yi, f"{m:.2f}x  (n={int(n)})", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)

    ax.set_yticks(y, [w.replace("_", " ") for w in show.index], fontsize=9)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlabel("median view ratio of titles containing the word (log2 scale)")
    ax.set_xticks([-4, -3, -2, -1, 0, 1, 2],
                  ["0.06x", "0.13x", "0.25x", "0.5x", "1x", "2x", "4x"])
    ax.set_xlim(dev.min() - 1.6, dev.max() + 1.4)
    ax.set_title("The same split shows up\nwith no topic model involved",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "words appearing in at least 40 titles; "
                     "top and bottom 10 shown",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_words_by_views.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_sweep()
    fig_topic_views()
    fig_words()
    for p in sorted(FIG.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
