"""Preprocess video titles into token lists for topic modelling.

Pipeline, in order, with the reason for each stage:

  1  branding removal    remove the channel name and the recurring series tag
                         ("| Mystery & Makeup | Bailey Sarian"). These repeat in
                         hundreds of titles and would otherwise become a topic
                         about the channel instead of about the content.
  2  normalization       lowercase, expand contractions ("don't" -> "do not"),
                         strip emoji and punctuation, map digits out.
  3  tokenization        NLTK word_tokenize (Treebank). Explained below.
  4  stopword removal    NLTK English list plus a domain list built from words
                         that are frequent but carry no topic ("part", "episode").
  5  lemmatization       WordNet lemmatizer with POS tags. Chosen over stemming;
                         --stem runs the Porter stemmer instead for comparison.
  6  short-token drop    tokens under 3 characters, and tokens that survive as
                         pure digits, carry no topic signal in a 9-word title.
  7  phrase detection    gensim Phrases joins collocations that behave as one
                         concept: serial_killer, death_row, murder_mystery.

Output: data/processed/tokens.csv, reports/02_preprocessing.txt
"""

from __future__ import annotations

import argparse
import collections
import re
import unicodedata
from pathlib import Path

import contractions
import pandas as pd
from gensim.models.phrases import Phraser, Phrases
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
OUT = ROOT / "data" / "processed" / "tokens.csv"
REPORT = ROOT / "reports" / "02_preprocessing.txt"

# Words that are common in the corpus but say nothing about the topic of a
# video. They are format words, not content words.
DOMAIN_STOPWORDS = {
    "part", "episode", "full", "video", "channel", "subscribe", "new",
    "watch", "series", "vlog", "podcast", "ep", "pt", "update", "week",
    "today", "live", "sneak", "peek", "preview", "official", "feat",
    "ft", "special", "edition", "season", "day", "one", "two", "three",
    "get", "got", "make", "made", "go", "going", "let", "like", "know",
    "thing", "things", "way", "time", "year", "years", "story", "stories",
}

# A word only counts as branding boilerplate if it appears in at least this
# share of one channel's titles. 0.30 keeps real content words safe: no topic
# word appears in 30% of a channel's videos, but a series tag does.
BRAND_THRESHOLD = 0.30

WORDNET_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}


def strip_emoji(text: str) -> str:
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"So", "Sk", "Cf", "Cs"}
    )


def learn_branding(df: pd.DataFrame) -> dict[str, set[str]]:
    """Find, per channel, the words that are branding rather than content.

    Two sources: the channel name itself, and any word appearing in at least
    BRAND_THRESHOLD of that channel's titles.
    """
    brand: dict[str, set[str]] = {}
    for channel, group in df.groupby("Channel"):
        words = set(re.findall(r"[a-z]+", channel.lower()))
        seen = collections.Counter()
        for title in group["Title"]:
            seen.update(set(re.findall(r"[a-z']+", title.lower())))
        n = len(group)
        frequent = {w for w, c in seen.items() if c / n >= BRAND_THRESHOLD}
        brand[channel] = words | frequent
    return brand


def normalize(text: str) -> str:
    text = strip_emoji(str(text))
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wordnet_pos(tag: str) -> str:
    return WORDNET_POS.get(tag[0], "n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="store_true",
                    help="Porter stemming instead of WordNet lemmatization")
    ap.add_argument("--no-phrases", action="store_true",
                    help="skip gensim bigram detection")
    ap.add_argument("--min-tokens", type=int, default=2,
                    help="drop titles left with fewer tokens than this")
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit("run scripts/build_corpus.py first")

    df = pd.read_csv(CORPUS)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    stop = set(stopwords.words("english")) | DOMAIN_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    brand = learn_branding(df)

    say("[1] TOKENIZATION")
    say("    Tool: nltk.word_tokenize, the Treebank tokenizer.")
    say("    Why not str.split(): split() leaves punctuation attached, so")
    say("    'killer?' and 'killer' become two different vocabulary entries, and")
    say("    the topic model treats them as unrelated words. The Treebank")
    say("    tokenizer separates punctuation and splits clitics, so 'don't'")
    say("    becomes do + n't and is then expanded to 'do not' by the")
    say("    contraction step that runs before it.")
    say("    Why not a subword tokenizer (BPE/WordPiece): LSA and LDA need whole")
    say("    words to produce readable topics. A topic listed as '##ing, murd,")
    say("    ##er' cannot be interpreted by a human, and interpretation is the")
    say("    point of topic modelling.")
    say()

    say("[2] BRANDING REMOVAL")
    say("    Every title from one channel repeats that channel's series tag.")
    say("    Left in, the strongest 'topic' found is simply the channel.")
    say(f"    A word is treated as branding if it appears in >= {BRAND_THRESHOLD:.0%} of")
    say("    one channel's own titles, or if it is part of the channel name.")
    say("    It is removed only for that channel, so 'crime' stays a content")
    say("    word for channels that do not put it in every title.")
    say()
    for channel in sorted(brand):
        if brand[channel]:
            sample = ", ".join(sorted(brand[channel])[:10])
            say(f"    {channel:<34} {sample}")
    say()

    # --- the pipeline ----------------------------------------------------
    records: list[list[str]] = []
    demo_idx = df.index[df["Title"].str.contains("Serial Killer", case=False,
                                                 na=False)][:1]
    demo: dict[str, str] = {}

    raw_vocab: collections.Counter[str] = collections.Counter()
    for idx, row in df.iterrows():
        title = row["Title"]
        text = normalize(title)
        if idx in demo_idx:
            demo["1 raw"] = title
            demo["2 normalized"] = text

        tokens = word_tokenize(text)
        raw_vocab.update(tokens)
        if idx in demo_idx:
            demo["3 tokenized"] = " | ".join(tokens)

        tokens = [t for t in tokens
                  if t not in stop and t not in brand[row["Channel"]]]
        if idx in demo_idx:
            demo["4 stopwords + branding removed"] = " | ".join(tokens)

        if args.stem:
            tokens = [stemmer.stem(t) for t in tokens]
        else:
            tokens = [lemmatizer.lemmatize(t, wordnet_pos(tag))
                      for t, tag in pos_tag(tokens)]
        if idx in demo_idx:
            key = "5 stemmed" if args.stem else "5 lemmatized"
            demo[key] = " | ".join(tokens)

        tokens = [t for t in tokens if len(t) >= 3 and t.isalpha()]
        if idx in demo_idx:
            demo["6 short tokens dropped"] = " | ".join(tokens)

        records.append(tokens)

    df["tokens"] = records

    if not args.no_phrases:
        phrases = Phrases(records, min_count=20, threshold=10.0)
        bigram = Phraser(phrases)
        df["tokens"] = [bigram[t] for t in records]
        found = sorted({p for toks in df["tokens"] for p in toks if "_" in p})
        demo["7 phrases joined"] = " | ".join(df["tokens"][demo_idx[0]]) \
            if len(demo_idx) else ""
    else:
        found = []

    before = len(df)
    df = df[df["tokens"].map(len) >= args.min_tokens].reset_index(drop=True)
    dropped = before - len(df)

    say("[3] WORKED EXAMPLE  (one title through every stage)")
    for stage, value in demo.items():
        say(f"    {stage:<32} {value}")
    say()

    say("[4] LEMMATIZATION VS STEMMING")
    say("    This run used: " + ("Porter stemming" if args.stem
                                 else "WordNet lemmatization"))
    say("    Lemmatization is the default. Topic modelling output is read by a")
    say("    human, and the Porter stemmer produces non-words:")
    for word in ["murderer", "mysterious", "police", "missing", "families",
                 "disappearance", "confession", "studies"]:
        say(f"      {word:<16} stem -> {stemmer.stem(word):<14} "
            f"lemma -> {lemmatizer.lemmatize(word, 'n')}")
    say("    A topic printed as 'murder, mysteri, polic' cannot be labelled with")
    say("    confidence in a presentation. The cost is speed, which does not")
    say("    matter at 10k short documents.")
    say()

    say("[5] PHRASE DETECTION")
    if found:
        say(f"    gensim Phrases (min_count=20, threshold=10) joined "
            f"{len(found)} bigrams.")
        say("    Examples: " + ", ".join(found[:25]))
        say("    These are pairs that occur together far more often than chance,")
        say("    so treating them as one token stops LDA from splitting a single")
        say("    concept across two topics.")
    else:
        say("    disabled (--no-phrases)")
    say()

    vocab = collections.Counter(t for toks in df["tokens"] for t in toks)
    say("[6] RESULT")
    say(f"    titles in       {before:,}")
    say(f"    titles kept     {len(df):,}  "
        f"(dropped {dropped} left with < {args.min_tokens} tokens)")
    say(f"    vocabulary before preprocessing {len(raw_vocab):,}")
    say(f"    vocabulary after  preprocessing {len(vocab):,}  "
        f"({len(vocab) / len(raw_vocab):.1%} of the original)")
    tok_counts = df["tokens"].map(len)
    say(f"    tokens per title  mean {tok_counts.mean():.1f}, "
        f"median {tok_counts.median():.0f}")
    say(f"    total tokens      {tok_counts.sum():,}")
    say()
    say("    most frequent tokens after cleaning:")
    for word, count in vocab.most_common(25):
        say(f"      {word:<24} {count:,}")

    out = df[["Title", "Channel", "Views", "view_ratio", "log_view_ratio"]].copy()
    out["tokens"] = df["tokens"].map(" ".join)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(out):,} rows)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
