"""Answer the research question: which kind of title attracts the most views?

Topic modelling on its own only says what the titles are about. This script
joins the topics back to the view counts.

The outcome variable is view_ratio, a video's views divided by the median video
on its own channel. Raw views cannot be used: a JCS video and a 48 Hours video
differ by four orders of magnitude for reasons that have nothing to do with the
title. view_ratio = 1.0 means "typical for this channel", 1.5 means "50% above
the channel's own norm".

Three views of the same question:

  [2] per topic       median view_ratio for the titles the model assigns to
                      each topic, with a bootstrap confidence interval so that
                      a topic with 60 videos is not read like one with 1,500.
                      Backed by a Kruskal-Wallis test across topics and a
                      Mann-Whitney post-hoc, because view_ratio is heavily
                      right-skewed and not normal, so ANOVA's assumptions do
                      not hold.
  [3] regression      all topic proportions entered at once, so each topic's
                      coefficient is its contribution holding the others fixed.
  [4] single words    the strongest individual title words, as a sanity check
                      that the topic story is not an artefact of the model.

Correlation, not causation: a topic's view_ratio co-varies with things this
script does not model - channel size (already partly controlled for by using
a ratio), recency, thumbnail, upload time. A topic effect here is evidence,
not proof, that the topic itself moves views.

Output: reports/05_views_by_topic.txt, data/processed/topic_views.csv,
        data/processed/topic_posthoc.csv
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DOCS = ROOT / "data" / "processed" / "doc_topics.csv"
REPORT = ROOT / "reports" / "05_views_by_topic.txt"
OUT = ROOT / "data" / "processed" / "topic_views.csv"

N_BOOT = 2000
SEED = 42
MIN_DOCS = 30  # a topic below this is not reported as a finding


def boot_ci(values: np.ndarray, stat=np.median, n=N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = stat(values[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    if not DOCS.exists():
        raise SystemExit("run scripts/topic_models.py first")

    df = pd.read_csv(DOCS)
    df["tokens"] = df["tokens"].fillna("").str.split()
    meta = json.loads((MODELS / "meta.json").read_text())
    lda = LdaModel.load(str(MODELS / "lda.model"))
    dictionary = Dictionary.load(str(MODELS / "dictionary.dict"))
    lda_doc = np.load(MODELS / "lda_doc.npy")
    comps = np.load(MODELS / "lsa_components.npy")
    terms = np.array([dictionary[i] for i in range(len(dictionary))])

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] OUTCOME VARIABLE")
    say("    view_ratio = views / median views of the same channel")
    say(f"    videos               {len(df):,}")
    say(f"    median view_ratio    {df['view_ratio'].median():.2f} (1.00 by "
        "construction, per channel)")
    say(f"    10th - 90th pct      {df['view_ratio'].quantile(0.10):.2f} - "
        f"{df['view_ratio'].quantile(0.90):.2f}")
    say("    The spread inside a single channel is already wide, so there is")
    say("    something for the title to explain.")
    say()

    # --- [2] per topic ---------------------------------------------------
    say("[2] VIEW RATIO BY LDA TOPIC  (topic = the title's highest-probability topic)")
    say()
    say(f"    {'topic':>5} {'videos':>7} {'median':>7} {'95% CI':>16} "
        f"{'>channel median':>16}  top words")
    rows = []
    for t in range(meta["best_lda"]):
        sub = df[df["lda_topic"] == t]
        if len(sub) == 0:
            continue
        vals = sub["view_ratio"].to_numpy()
        lo, hi = boot_ci(vals)
        words = ", ".join(w for w, _ in lda.show_topic(t, topn=6))
        rows.append(dict(model="LDA", topic=t, n=len(sub),
                         median_ratio=float(np.median(vals)), ci_lo=lo, ci_hi=hi,
                         above_median=float((vals > 1).mean()),
                         median_views=float(sub["Views"].median()),
                         n_channels=int(sub["Channel"].nunique()), words=words))
    table = pd.DataFrame(rows).sort_values("median_ratio", ascending=False)
    for _, r in table.iterrows():
        flag = " " if r.n >= MIN_DOCS else "*"
        say(f"    {int(r.topic):>5}{flag}{int(r.n):>6,} {r.median_ratio:>7.2f} "
            f"[{r.ci_lo:>5.2f}, {r.ci_hi:>5.2f}] {r.above_median:>15.0%}  {r.words}")
    if (table["n"] < MIN_DOCS).any():
        say(f"    * fewer than {MIN_DOCS} videos - not interpreted")
    say()

    best = table.iloc[0]
    worst = table.iloc[-1]
    say(f"    Highest: topic {int(best.topic)} at {best.median_ratio:.2f}x the "
        f"channel median ({int(best.n):,} videos,")
    say(f"             {int(best.n_channels)} channels) - {best.words}")
    say(f"    Lowest:  topic {int(worst.topic)} at {worst.median_ratio:.2f}x "
        f"({int(worst.n):,} videos) - {worst.words}")
    ratio = best.median_ratio / worst.median_ratio
    say(f"    The gap between the best and worst topic is {ratio:.1f}x.")
    say()
    say("    Confidence intervals that overlap mean the two topics are not")
    say("    distinguishable at this sample size. Read the table by bands, not")
    say("    by rank. The Kruskal-Wallis test below makes this precise.")
    say()

    # --- [2a] Kruskal-Wallis + post-hoc -----------------------------------
    say("[2a] KRUSKAL-WALLIS TEST ACROSS LDA TOPICS")
    say(f"    view_ratio is heavily right-skewed (median 1.00, 90th pct "
        f"{df['view_ratio'].quantile(0.9):.1f}x),")
    say("    so ANOVA's normal-residual assumption does not hold. "
        "Kruskal-Wallis is its non-parametric")
    say(f"    equivalent, run on the {MIN_DOCS}+-video topics from [2].")
    say()
    tested = table[table["n"] >= MIN_DOCS]
    topics_list = tested["topic"].astype(int).tolist()
    groups = [df.loc[df["lda_topic"] == t, "view_ratio"].to_numpy()
              for t in topics_list]
    h_stat, p_kw = kruskal(*groups)
    say(f"    H({len(groups) - 1}) = {h_stat:.1f}, p = {p_kw:.2e}  "
        f"({len(groups)} topics, {sum(len(g) for g in groups):,} videos)")
    if p_kw < 0.05:
        say("    At least one topic's view_ratio differs from the rest - the")
        say("    split in [2] is not sampling noise.")
    else:
        say("    No evidence the topics differ - the split in [2] could be")
        say("    sampling noise.")
    say()

    m = len(groups) * (len(groups) - 1) // 2
    say(f"    Post-hoc: all {m} topic pairs compared with a two-sided")
    say("    Mann-Whitney U test, Holm-Bonferroni corrected for running that")
    say("    many comparisons at once.")
    pairs, raw_p = [], []
    for (i, ti), (j, tj) in combinations(enumerate(topics_list), 2):
        _, p = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
        pairs.append((ti, tj))
        raw_p.append(p)
    order = np.argsort(raw_p)
    running_max = 0.0
    adj_p = np.empty(len(raw_p))
    for rank, idx in enumerate(order):
        running_max = max(running_max, raw_p[idx] * (len(raw_p) - rank))
        adj_p[idx] = min(running_max, 1.0)
    posthoc = pd.DataFrame({"topic_a": [p[0] for p in pairs],
                            "topic_b": [p[1] for p in pairs],
                            "p_raw": raw_p, "p_holm": adj_p})
    n_sig = int((posthoc["p_holm"] < 0.05).sum())
    say(f"    {n_sig} / {m} pairs differ significantly (adjusted p < .05).")

    best_t, worst_t = int(best.topic), int(worst.topic)
    bw_mask = ((posthoc.topic_a == best_t) & (posthoc.topic_b == worst_t)) | \
              ((posthoc.topic_a == worst_t) & (posthoc.topic_b == best_t))
    bw = posthoc[bw_mask]
    if len(bw):
        p_bw = float(bw["p_holm"].iloc[0])
        say(f"    Best vs worst (topic {best_t} vs {worst_t}): adjusted "
            f"p = {p_bw:.2e} - "
            f"{'differ' if p_bw < 0.05 else 'not distinguishable'}.")
    say()

    # --- [3] regression --------------------------------------------------
    say("[3] REGRESSION  (log2 view_ratio on all topic proportions at once)")
    say("    Assigning each title to its single best topic throws away the fact")
    say("    that most titles are a mixture. Here every topic proportion is a")
    say("    predictor, so a coefficient is the effect of a title being entirely")
    say("    that topic rather than the reference mixture, holding the rest fixed.")
    say()
    y = df["log_view_ratio"].to_numpy()
    # drop the last topic as reference: proportions sum to 1 and would be
    # perfectly collinear with the intercept
    Xt = np.column_stack([np.ones(len(df)), lda_doc[:, :-1]])
    coef, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    resid = y - Xt @ coef
    dof = len(y) - Xt.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xt.T @ Xt)
    se = np.sqrt(np.diag(cov))
    tstat = coef / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot

    reg = pd.DataFrame({
        "topic": list(range(meta["best_lda"] - 1)),
        "coef_log2": coef[1:], "t": tstat[1:],
    })
    reg["x_views"] = 2 ** reg["coef_log2"]
    reg["words"] = [", ".join(w for w, _ in lda.show_topic(t, topn=5))
                    for t in reg["topic"]]
    reg = reg.sort_values("coef_log2", ascending=False)

    say(f"    R2 = {r2:.3f}  - the topic of the title explains "
        f"{r2:.1%} of the variation in")
    say("    within-channel views. Small, and that is the honest headline: the")
    say("    topic matters, but most of what drives views is not in the title.")
    say()
    say(f"    {'topic':>5} {'coef':>7} {'x views':>8} {'t':>7}  top words")
    for _, r in pd.concat([reg.head(5), reg.tail(5)]).iterrows():
        say(f"    {int(r.topic):>5} {r.coef_log2:>7.2f} {r.x_views:>7.2f}x "
            f"{r.t:>7.1f}  {r.words}")
    say()
    say("    coef is in log2 units: +1.00 means the title gets twice the views of")
    say(f"    the reference topic ({meta['best_lda'] - 1}), -1.00 means half. "
        "|t| above 2 is the usual")
    say("    threshold for taking a coefficient seriously.")
    say()

    # --- [4] single words ------------------------------------------------
    say("[4] SINGLE WORDS  (sanity check, independent of both topic models)")
    say("    For every word appearing in at least 40 different titles: the")
    say("    median view_ratio of those titles. One title counts once per word.")
    say("    No model is involved, so if the topic story is real the same themes")
    say("    should appear here too.")
    say()
    # A word repeated inside one title must not count that title twice, so
    # each title contributes each of its words once.
    unique_words = df["tokens"].map(lambda t: sorted(set(t)))
    exploded = df[["view_ratio"]].join(
        unique_words.explode().rename("word")).dropna(subset=["word"])
    per_word = exploded.groupby("word").agg(
        n=("view_ratio", "size"), median_ratio=("view_ratio", "median"))
    per_word = per_word[per_word["n"] >= 40].sort_values(
        "median_ratio", ascending=False)
    say(f"    {len(per_word)} words qualify.")
    say()
    say("    strongest words")
    for word, r in per_word.head(15).iterrows():
        say(f"      {word:<24} {int(r.n):>5} titles   {r.median_ratio:>5.2f}x")
    say()
    say("    weakest words")
    for word, r in per_word.tail(10).iterrows():
        say(f"      {word:<24} {int(r.n):>5} titles   {r.median_ratio:>5.2f}x")
    say()

    # --- [5] LSA cross-check ---------------------------------------------
    say("[5] SAME QUESTION THROUGH LSA")
    say("    If the two models tell the same story the finding is about the data,")
    say("    not about the algorithm.")
    say()
    lsa_rows = []
    for t in range(meta["best_lsa"]):
        sub = df[df["lsa_topic"] == t]
        if len(sub) < MIN_DOCS:
            continue
        idx = np.argsort(np.abs(comps[t]))[::-1][:6]
        lsa_rows.append(dict(model="LSA", topic=t, n=len(sub),
                             median_ratio=float(sub["view_ratio"].median()),
                             words=", ".join(str(terms[i]) for i in idx)))
    lsa_table = pd.DataFrame(lsa_rows).sort_values("median_ratio",
                                                   ascending=False)
    say(f"    {'topic':>5} {'videos':>7} {'median':>7}  top words")
    for _, r in pd.concat([lsa_table.head(4), lsa_table.tail(4)]).iterrows():
        say(f"    {int(r.topic):>5} {int(r.n):>7,} {r.median_ratio:>7.2f}  "
            f"{r.words}")
    say()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table, lsa_table], ignore_index=True).to_csv(OUT, index=False)
    reg.to_csv(ROOT / "data" / "processed" / "topic_regression.csv", index=False)
    posthoc.to_csv(ROOT / "data" / "processed" / "topic_posthoc.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"wrote data/processed/topic_regression.csv")
    print(f"wrote data/processed/topic_posthoc.csv")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
