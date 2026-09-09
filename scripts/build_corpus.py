"""Build the working corpus from the raw Kaggle export.

Reads data/raw/true_crime_channel_stats.csv, removes rows that cannot support
the research question, repairs the text encoding problems in the titles, and
adds the two view measures the analysis needs.

The unit of analysis is one YouTube video title. The research question is
"which kind of title attracts the most views", so the title is the document and
the view count is the outcome variable attached to it.

Output: data/processed/corpus.csv, reports/01_corpus.txt
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "true_crime_channel_stats.csv"
OUT = ROOT / "data" / "processed" / "corpus.csv"
REPORT = ROOT / "reports" / "01_corpus.txt"

# Some titles are written with Cyrillic letters that look identical to Latin
# ones ("тruеcrіmе & makeup"). Left alone they become their own vocabulary
# entries and pollute the topics, so they are folded back to Latin.
CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "А": "A", "Е": "E",
    "О": "O", "Р": "P", "С": "C", "Х": "X", "Т": "T",
    "М": "M", "Н": "H", "К": "K", "В": "B",
}


def fold_confusables(text: str) -> str:
    return "".join(CONFUSABLES.get(ch, ch) for ch in text)


def clean_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = fold_confusables(text)
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} - download the Kaggle dataset first")

    raw = pd.read_csv(RAW)
    n_raw = len(raw)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] SOURCE")
    say(f"    file            {RAW.relative_to(ROOT)}")
    say("    dataset         True Crime Channel Statistics (mhapich, Kaggle)")
    say("    licence         CC0 1.0 Public Domain Dedication")
    say("    collected from  the public YouTube Data API v3")
    say(f"    rows as downloaded {n_raw:,}")
    say()

    # --- filtering -------------------------------------------------------
    df = raw.copy()
    df["Title"] = df["Title"].map(clean_title)

    before = len(df)
    df = df[df["Title"].str.len() > 0]
    n_empty = before - len(df)

    before = len(df)
    df = df.dropna(subset=["Views"])
    n_noviews = before - len(df)

    # A handful of videos have 0 views; they carry no signal for a question
    # about attracting views and would distort the per-channel ratio.
    before = len(df)
    df = df[df["Views"] > 0]
    n_zero = before - len(df)

    # Same title uploaded twice on the same channel is a re-upload, not a
    # second observation.
    before = len(df)
    df = df.drop_duplicates(subset=["Channel", "Title"], keep="first")
    n_dupe = before - len(df)

    df["PublishedDate"] = pd.to_datetime(df["PublishedDate"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["PublishedDate"])
    n_baddate = before - len(df)

    say("[2] FILTERING")
    say(f"    empty title            -{n_empty}")
    say(f"    missing view count     -{n_noviews}")
    say(f"    zero views             -{n_zero}")
    say(f"    duplicate channel+title -{n_dupe}")
    say(f"    unparseable date       -{n_baddate}")
    say(f"    kept                   {len(df):,} of {n_raw:,} "
        f"({len(df) / n_raw:.1%})")
    say()

    # --- view measures ---------------------------------------------------
    # Raw views mostly measure how big the channel is, not how good the title
    # is: the median 48 Hours video outranks almost anything from a small
    # channel. view_ratio divides by the channel's own median so that 1.0
    # means "typical for this channel" and 2.0 means "twice the channel norm".
    df["log_views"] = np.log10(df["Views"])
    channel_median = df.groupby("Channel")["Views"].transform("median")
    df["view_ratio"] = df["Views"] / channel_median
    df["log_view_ratio"] = np.log2(df["view_ratio"])

    say("[3] CORPUS DESCRIPTION")
    say(f"    videos          {len(df):,}")
    say(f"    channels        {df['Channel'].nunique()}")
    say(f"    date range      {df['PublishedDate'].min():%Y-%m-%d} to "
        f"{df['PublishedDate'].max():%Y-%m-%d}")
    wc = df["Title"].str.split().str.len()
    say(f"    title length    mean {wc.mean():.1f} words, median {wc.median():.0f}, "
        f"min {wc.min()}, max {wc.max()}")
    say(f"    views           median {df['Views'].median():,.0f}, "
        f"mean {df['Views'].mean():,.0f}, max {df['Views'].max():,.0f}")
    say()

    say("[4] CHANNEL BALANCE  (the corpus is not balanced - see limitations)")
    say(f"    {'channel':<34} {'videos':>7} {'share':>7} {'median views':>13}")
    counts = df.groupby("Channel").agg(
        n_videos=("Title", "size"), median_views=("Views", "median")
    ).sort_values("n_videos", ascending=False)
    for name, row in counts.iterrows():
        say(f"    {name:<34} {int(row.n_videos):>7,} {row.n_videos / len(df):>6.1%} "
            f"{row.median_views:>13,.0f}")
    say()
    top = counts.iloc[0]
    say(f"    largest channel is {counts.index[0]} with "
        f"{top.n_videos / len(df):.1%} of all videos.")
    say(f"    the most-watched channel's median video gets "
        f"{counts.median_views.max() / counts.median_views.min():,.0f}x the views of")
    say("    the least-watched channel's median video, which is why the analysis")
    say("    uses view_ratio (views divided by that channel's own median).")
    say()

    say("[5] LEGAL AND ETHICAL NOTES")
    say("    - The dataset is published on Kaggle under CC0 1.0, which places it")
    say("      in the public domain and permits reuse including redistribution.")
    say("    - The underlying data came from the official YouTube Data API v3,")
    say("      not from scraping, so the platform terms of service were followed.")
    say("    - Titles, view counts and channel names are public metadata that")
    say("      any visitor to the channel page can read. No private data is used.")
    say("    - Channel names identify creators, who are public figures acting in")
    say("      a professional capacity. They are kept because the analysis has to")
    say("      control for channel size. No comment authors or viewers appear in")
    say("      the data, so there is no personal data to anonymise.")
    say("    - Video descriptions contain creator contact details and sponsor")
    say("      links. They are dropped: the research question is about titles.")
    say("    - Content warning: the corpus is true crime, so titles name real")
    say("      victims and offenders. Topics are reported at the aggregate level")
    say("      and no claim is made about any individual case.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    keep = ["Title", "Channel", "PublishedDate", "Views", "Likes", "Comments",
            "log_views", "view_ratio", "log_view_ratio"]
    df[keep].to_csv(OUT, index=False)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(df):,} rows)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
