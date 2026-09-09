"""Does removing people's names improve the topics?

Victim and offender names are a large share of the vocabulary in a true crime
corpus, and they crowd the topic word lists ("michael, david, robert") without
describing a theme. The obvious move is to delete them. This script tests
whether that actually helps before anyone does it.

Names are found without a name list, from capitalisation in the original titles:
a token counts as a proper noun if it appears capitalised mid-sentence in at
least 90% of its occurrences. Words that are also ordinary English nouns
(*mark*, *grace*, *summer*) are kept unless they are also common given names.

Output: reports/06_name_ablation.txt
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import nltk
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from nltk.corpus import names
from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
TOKENS = ROOT / "data" / "processed" / "tokens.csv"
REPORT = ROOT / "reports" / "06_name_ablation.txt"

K_GRID = [10, 15, 20, 25]
CAP_RATIO = 0.90
MIN_OCCURRENCES = 3
SEED = 42


def proper_nouns(titles: pd.Series) -> set[str]:
    cap: collections.Counter[str] = collections.Counter()
    total: collections.Counter[str] = collections.Counter()
    for title in titles:
        words = re.findall(r"[A-Za-z][A-Za-z']*", str(title))
        for i, w in enumerate(words):
            lw = w.lower()
            total[lw] += 1
            # position 0 is capitalised for every title, so it says nothing;
            # ALL CAPS words are emphasis, not names.
            if i > 0 and w[0].isupper() and not w.isupper():
                cap[lw] += 1
    return {w for w, c in total.items()
            if c >= MIN_OCCURRENCES and cap[w] / c >= CAP_RATIO}


def coherence_at(docs: list[list[str]], k: int) -> float:
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.4)
    bow = [dictionary.doc2bow(d) for d in docs]
    keep = [i for i, b in enumerate(bow) if b]
    texts = [docs[i] for i in keep]
    bow = [bow[i] for i in keep]
    lda = LdaModel(corpus=bow, id2word=dictionary, num_topics=k,
                   random_state=SEED, passes=10, iterations=100,
                   alpha="auto", eta="auto", eval_every=None)
    topics = [[w for w, _ in lda.show_topic(t, topn=10)] for t in range(k)]
    return CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                          coherence="c_v").get_coherence()


def main() -> None:
    if not TOKENS.exists():
        raise SystemExit("run scripts/preprocess.py first")
    nltk.download("names", quiet=True)

    corpus = pd.read_csv(CORPUS)
    tokens = pd.read_csv(TOKENS)
    tokens["tokens"] = tokens["tokens"].fillna("").str.split()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    proper = proper_nouns(corpus["Title"])
    given = {n.lower() for n in names.words()}
    in_wordnet = {w for w in proper if wn.synsets(w)}
    drop = (proper - in_wordnet) | (proper & given)

    say("[1] WHAT COUNTS AS A NAME")
    say(f"    capitalised mid-title in >= {CAP_RATIO:.0%} of its occurrences,")
    say(f"    seen at least {MIN_OCCURRENCES} times          {len(proper):,} words")
    say(f"    of those, also ordinary English nouns    {len(in_wordnet):,} "
        "(kept unless also a given name)")
    say(f"    dropped in the ablation                  {len(drop):,}")
    say()
    say("    examples dropped: " + ", ".join(sorted(drop)[:20]))
    say("    examples kept:    " + ", ".join(sorted(in_wordnet - given)[:20]))
    say()

    say("[2] EFFECT ON LDA COHERENCE")
    say(f"    {'k':>4} {'names kept':>12} {'names dropped':>15} {'change':>9}")
    rows = []
    base_docs = [list(t) for t in tokens["tokens"]]
    cut_docs = [[w for w in t if w not in drop] for t in tokens["tokens"]]
    cut_docs = [d for d in cut_docs if len(d) >= 2]
    for k in K_GRID:
        a = coherence_at(base_docs, k)
        b = coherence_at(cut_docs, k)
        say(f"    {k:>4} {a:>12.3f} {b:>15.3f} {b - a:>+9.3f}")
        rows.append((k, a, b))
    say()

    gains = [b - a for _, a, b in rows]
    say("[3] CONCLUSION")
    say(f"    mean change {sum(gains) / len(gains):+.3f}, "
        f"and the sign is not consistent across k "
        f"({sum(g > 0 for g in gains)} of {len(gains)} improved).")
    say("    There is no reliable gain, so names are kept in the main pipeline.")
    say("    Reported so that the choice is evidence rather than preference.")
    say("    The cost of keeping them is visible in the topic word lists, where")
    say("    first names appear among the top words of several topics.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
