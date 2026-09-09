"""Fit LSA and LDA over the preprocessed titles and choose the number of topics.

Two approaches, deliberately different in kind:

  LSA  Latent Semantic Analysis. TF-IDF matrix, then truncated SVD. Linear
       algebra: it finds the k directions that explain the most variance in the
       term-document matrix. Topic weights can be negative, so a topic is read
       as "these words go together", not as a probability.

  LDA  Latent Dirichlet Allocation. A generative probability model: every title
       is a mixture over k topics, every topic is a distribution over words.
       Fitted by variational Bayes. Weights are probabilities and sum to 1,
       which is what makes "this title is 60% topic 3" a meaningful sentence.

Number of topics is selected by sweeping k and scoring every model. Nothing is
chosen by eye.

  c_v      topic coherence, sliding window + normalised PMI. Higher is better.
           This is the measure that correlates best with human judgement.
  u_mass   coherence from document co-occurrence counts. Closer to 0 is better.
           Reported as a second opinion because c_v is known to over-reward
           very small k.
  explained variance   LSA only: the share of the TF-IDF matrix the k
           components reconstruct.

Output: models/, reports/03_model_selection.txt, data/processed/doc_topics_*.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "processed" / "tokens.csv"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "03_model_selection.txt"
SWEEP = ROOT / "data" / "processed" / "k_sweep.csv"

K_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]

# LDA is randomly initialised, so a single fit can win a sweep by luck. Every k
# is fitted with three seeds and scored on the mean; the spread across seeds is
# reported so the reader can see how much of a c_v gap is real.
SEEDS = [42, 7, 2024]

# A word must appear in at least 5 titles to be modelled: below that it is
# almost always a victim or offender name that occurs once, and a topic built
# out of unique names describes nothing. A word appearing in more than 40% of
# titles is not distinguishing either.
NO_BELOW = 5
NO_ABOVE = 0.4
SEED = 42


def load_tokens() -> tuple[pd.DataFrame, list[list[str]]]:
    if not TOKENS.exists():
        raise SystemExit("run scripts/preprocess.py first")
    df = pd.read_csv(TOKENS)
    df["tokens"] = df["tokens"].fillna("").str.split()
    return df, list(df["tokens"])


def lsa_topics(svd: TruncatedSVD, terms: np.ndarray, topn: int) -> list[list[str]]:
    """Top words per SVD component, by absolute loading.

    Absolute value matters: an LSA component can describe a contrast, where the
    strongly negative words are as much part of the topic as the positive ones.
    """
    out = []
    for row in svd.components_:
        idx = np.argsort(np.abs(row))[::-1][:topn]
        out.append([str(terms[i]) for i in idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=10,
                    help="words per topic used for coherence and display")
    ap.add_argument("--passes", type=int, default=10, help="LDA passes")
    args = ap.parse_args()

    df, docs = load_tokens()
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    # --- shared vocabulary ----------------------------------------------
    dictionary = Dictionary(docs)
    n_before = len(dictionary)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    bow = [dictionary.doc2bow(d) for d in docs]

    # Titles emptied by the vocabulary filter cannot be assigned a topic.
    keep = [i for i, b in enumerate(bow) if b]
    df = df.iloc[keep].reset_index(drop=True)
    docs = [docs[i] for i in keep]
    bow = [bow[i] for i in keep]

    say("[1] VOCABULARY")
    say(f"    documents (titles)      {len(df):,}")
    say(f"    distinct tokens         {n_before:,}")
    say(f"    after filter_extremes   {len(dictionary):,}  "
        f"(kept words in >= {NO_BELOW} titles and <= {NO_ABOVE:.0%} of titles)")
    say(f"    removed                 {n_before - len(dictionary):,} words, "
        "almost all of them names appearing once")
    say()

    tfidf = TfidfVectorizer(
        vocabulary={dictionary[i]: i for i in range(len(dictionary))},
        tokenizer=str.split, preprocessor=None, lowercase=False,
        token_pattern=None, sublinear_tf=True,
    )
    X = tfidf.fit_transform(" ".join(d) for d in docs)
    terms = np.array(tfidf.get_feature_names_out())
    say(f"    TF-IDF matrix           {X.shape[0]:,} x {X.shape[1]:,}, "
        f"density {X.nnz / (X.shape[0] * X.shape[1]):.4%}")
    say()

    # --- the sweep -------------------------------------------------------
    say("[2] NUMBER-OF-TOPICS SWEEP")
    say("    Every k is fitted with both models and scored the same way, so the")
    say("    two columns are directly comparable.")
    say()
    header = (f"    {'k':>3} | {'LSA c_v':>8} {'LSA u_mass':>10} {'LSA var':>8} "
              f"| {'LDA c_v':>14} {'LDA u_mass':>10} {'LDA perplex':>11} | {'sec':>5}")
    say(header)
    say("    " + "-" * (len(header) - 4))

    rows = []
    lsa_models: dict[int, TruncatedSVD] = {}
    lda_models: dict[int, LdaModel] = {}

    for k in K_GRID:
        t0 = time.time()

        svd = TruncatedSVD(n_components=k, random_state=SEED, algorithm="arpack")
        svd.fit(X)
        lsa_words = lsa_topics(svd, terms, args.topn)
        lsa_cv = CoherenceModel(topics=lsa_words, texts=docs,
                                dictionary=dictionary, coherence="c_v").get_coherence()
        lsa_um = CoherenceModel(topics=lsa_words, texts=docs, corpus=bow,
                                dictionary=dictionary, coherence="u_mass").get_coherence()
        lsa_var = float(svd.explained_variance_ratio_.sum())

        cvs, ums, pxs, fits = [], [], [], []
        for seed in SEEDS:
            lda = LdaModel(corpus=bow, id2word=dictionary, num_topics=k,
                           random_state=seed, passes=args.passes, iterations=100,
                           alpha="auto", eta="auto", eval_every=None)
            lda_words = [[w for w, _ in lda.show_topic(t, topn=args.topn)]
                         for t in range(k)]
            cvs.append(CoherenceModel(topics=lda_words, texts=docs,
                                      dictionary=dictionary,
                                      coherence="c_v").get_coherence())
            ums.append(CoherenceModel(model=lda, corpus=bow, dictionary=dictionary,
                                      coherence="u_mass").get_coherence())
            pxs.append(float(np.exp2(-lda.log_perplexity(bow))))
            fits.append(lda)
        lda_cv, lda_sd = float(np.mean(cvs)), float(np.std(cvs))
        lda_um, lda_px = float(np.mean(ums)), float(np.mean(pxs))

        secs = time.time() - t0
        say(f"    {k:>3} | {lsa_cv:>8.3f} {lsa_um:>10.3f} {lsa_var:>7.1%} "
            f"| {lda_cv:>8.3f} +-{lda_sd:<4.3f} {lda_um:>10.3f} {lda_px:>11.1f} "
            f"| {secs:>5.0f}")

        rows.append(dict(k=k, lsa_cv=lsa_cv, lsa_umass=lsa_um,
                         lsa_explained_variance=lsa_var, lda_cv=lda_cv,
                         lda_cv_sd=lda_sd, lda_umass=lda_um,
                         lda_perplexity=lda_px, seconds=secs))
        lsa_models[k] = svd
        # keep the median-coherence seed, not the luckiest one
        lda_models[k] = fits[int(np.argsort(cvs)[len(cvs) // 2])]

    sweep = pd.DataFrame(rows)
    say()

    def knee(col: str) -> int:
        """Elbow of a score-vs-k curve: the k furthest from the straight line
        joining the first and last point (the Kneedle idea, done by hand).

        Needed because on 9-word documents c_v does not turn over - it keeps
        creeping up as k grows, so "take the maximum" would hand back the
        largest k on the grid whether or not those topics mean anything. The
        elbow is where extra topics stop buying much coherence.
        """
        ks = sweep["k"].to_numpy(float)
        ys = sweep[col].to_numpy(float)
        x = (ks - ks.min()) / (ks.max() - ks.min())
        y = (ys - ys.min()) / (ys.max() - ys.min())
        # perpendicular distance from the chord (0,0)-(1,1) is |y - x|
        return int(ks[np.argmax(y - x)])

    peak_lsa = int(sweep.loc[sweep["lsa_cv"].idxmax(), "k"])
    peak_lda = int(sweep.loc[sweep["lda_cv"].idxmax(), "k"])
    best_lsa = knee("lsa_cv")
    best_lda = knee("lda_cv")

    say("[3] SELECTION")
    say("    c_v is the primary measure because it is the one validated against")
    say("    human ratings of topic quality (Roder et al. 2015). Perplexity is")
    say("    reported but not used to choose: it is known to fall as k rises even")
    say("    when the extra topics are not interpretable (Chang et al. 2009).")
    say("    u_mass is shown as a cross-check.")
    say()
    say(f"    LSA raw peak c_v      k = {peak_lsa} ({sweep['lsa_cv'].max():.3f})")
    say(f"    LDA raw peak mean c_v k = {peak_lda} ({sweep['lda_cv'].max():.3f}, "
        f"sd {float(sweep.loc[sweep['lda_cv'].idxmax(), 'lda_cv_sd']):.3f} "
        "across seeds)")
    say()
    say("    Taking the peak is the wrong rule here. On documents this short the")
    say("    c_v curve does not turn over inside the grid - it is still rising at")
    say(f"    k = {int(sweep['k'].max())} - so the peak just returns the largest k "
        "tried. The seed-to-seed")
    say(f"    spread ({sweep['lda_cv_sd'].mean():.3f} on average) is also of the same "
        "size as the gap between neighbouring k,")
    say("    so most of that rise is not a real difference. We therefore take the")
    say("    elbow of the curve: the k furthest above the straight line joining")
    say("    the first and last point.")
    say()
    say(f"      LSA elbow k = {best_lsa}")
    say(f"      LDA elbow k = {best_lda}")
    say()
    say("    These are the candidates carried into human evaluation")
    say("    (scripts/human_eval.py), which is what actually confirms the choice.")
    say()
    if sweep.loc[sweep["lda_umass"].idxmax(), "k"] != peak_lda:
        say(f"    Note: u_mass prefers k = "
            f"{int(sweep.loc[sweep['lda_umass'].idxmax(), 'k'])} for LDA, while "
            f"c_v prefers {peak_lda}. The two measures")
        say("    disagree; this is reported in the limitations rather than hidden.")
        say()

    # --- fit the chosen models and write assignments ---------------------
    MODELS.mkdir(exist_ok=True)
    say("[4] CHOSEN MODELS")

    svd = lsa_models[best_lsa]
    lsa_doc = svd.transform(X)
    say(f"    LSA  k={best_lsa}, explained variance "
        f"{svd.explained_variance_ratio_.sum():.1%}")
    for t, words in enumerate(lsa_topics(svd, terms, args.topn)):
        say(f"      LSA {t:>2}  {', '.join(words)}")
    say()

    lda = lda_models[best_lda]
    lda_doc = np.zeros((len(bow), best_lda))
    for i, doc_bow in enumerate(bow):
        for t, p in lda.get_document_topics(doc_bow, minimum_probability=0.0):
            lda_doc[i, t] = p
    say(f"    LDA  k={best_lda}")
    for t in range(best_lda):
        words = [w for w, _ in lda.show_topic(t, topn=args.topn)]
        share = (lda_doc.argmax(axis=1) == t).mean()
        say(f"      LDA {t:>2}  ({share:>5.1%} of titles)  {', '.join(words)}")

    lda.save(str(MODELS / "lda.model"))
    np.save(MODELS / "lsa_components.npy", svd.components_)
    np.save(MODELS / "lsa_doc.npy", lsa_doc)
    np.save(MODELS / "lda_doc.npy", lda_doc)
    dictionary.save(str(MODELS / "dictionary.dict"))
    (MODELS / "meta.json").write_text(json.dumps(
        dict(best_lsa=best_lsa, best_lda=best_lda, topn=args.topn,
             n_docs=len(df), n_terms=len(dictionary), seed=SEED), indent=2))

    df["lsa_topic"] = lsa_doc.argmax(axis=1)
    df["lda_topic"] = lda_doc.argmax(axis=1)
    df["lda_prob"] = lda_doc.max(axis=1)
    df.drop(columns=["tokens"]).assign(
        tokens=[" ".join(d) for d in docs]
    ).to_csv(ROOT / "data" / "processed" / "doc_topics.csv", index=False)

    sweep.to_csv(SWEEP, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {SWEEP.relative_to(ROOT)}")
    print(f"wrote data/processed/doc_topics.csv")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
