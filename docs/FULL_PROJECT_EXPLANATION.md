# Topic Modelling — Full Project Explanation

A guide for someone joining the project cold. Walks through the pipeline in
order — corpus → preprocessing → text representation → model training →
topic evaluation → results — and explains not only *what* was done but *why*
each decision was made and what every parameter means. Every number below is
read from the generated reports (`reports/01_corpus.txt` …
`reports/06_name_ablation.txt`); re-run the pipeline (see Appendix A) and
re-check before quoting these if any upstream script or data file changes.

**The task.** Given a true-crime YouTube video title, in what latent topic
does it belong, and does that topic predict the video's view count relative
to its own channel? Unsupervised topic modelling (LSA + LDA, with NMF as an
optional third comparison), followed by a per-topic view-count analysis.

**The corpus.** [True Crime Channel Statistics](https://www.kaggle.com/datasets/mhapich/true-crime-channel-statistics)
(mhapich, Kaggle, CC0 1.0) — 10,107 video titles from 21 channels, collected
via the official YouTube Data API v3.

**The headline result.** Titles whose dominant LDA topic is an unresolved
relationship-crime narrative get **1.36×** the views of the typical video on
the same channel; titles whose dominant topic is instructional/coursework
content (SPSS tutorials, psychology role-plays) get **0.45×** — a 3.0× gap,
confirmed by Kruskal-Wallis (H(19)=418.4, p≈6.5×10⁻⁷⁷) and a Holm-Bonferroni
corrected post-hoc. The topic of a title explains 6.4% of within-channel view
variation (R²=0.064) — real, but small: most of what drives views is not in
the title.

```
raw Kaggle CSV → filter/clean → preprocess (tokenize, de-brand, lemmatize,
phrase-detect) → TF-IDF/BoW → LSA + LDA + NMF, k-sweep → human word-intrusion
eval → per-topic view_ratio + Kruskal-Wallis/post-hoc + regression → figures
```

---

## Part 0 — Why this corpus is not a generic text-classification corpus

Two things shape every downstream decision and are worth stating up front,
because they explain choices that otherwise look arbitrary later:

| Property | Value | Consequence |
|---|---|---|
| Document length | mean 9.0 words/title, ~5.4 tokens after cleaning | c_v coherence never turns over in the tested k range — elbow, not peak, selection (Part 3) |
| Channel size spread | 4,119× between largest/smallest channel's median views | raw views cannot be the outcome variable — `view_ratio` used instead (Part 0.1, Part 5) |
| No topic labels | 0 — this is unsupervised | augmentation cannot be scoped to "balance topic classes" — there are none (Part 1.4) |

### 0.1 The outcome variable: `view_ratio`, not raw views

A video's `Views` divided by its own channel's median views. Computed in
`scripts/build_corpus.py`:

```python
channel_median = df.groupby("Channel")["Views"].transform("median")
df["view_ratio"] = df["Views"] / channel_median
df["log_view_ratio"] = np.log2(df["view_ratio"])
```

The most-watched channel's median video (JCS - Criminal Psychology,
15,943,809 median views) gets 4,119× the views of the least-watched
channel's median video (48 Hours, 3,871 median views). Any model built on
raw views would mostly be modelling channel identity, not title content. `view_ratio = 1.0` means "typical for this channel";
`log_view_ratio` is used in the regression (Part 5) because view_ratio itself
is right-skewed and a doubling should be treated as one unit of effect
regardless of direction.

---

## Part 1 — Corpus Construction & Legal/Ethical Compliance

**Script:** `scripts/build_corpus.py` → `reports/01_corpus.txt`.

### 1.1 Filtering

| Step | Dropped | Why |
|---|---|---|
| empty title | 0 | — |
| missing view count | 1 | can't compute an outcome |
| zero views | 11 | zero-view videos carry no signal for a views question and would distort a channel's median |
| duplicate channel+title | 69 | a re-upload of the same title is not a second observation |
| unparseable date | 0 | — |
| **kept** | **10,026 of 10,107 (99.2%)** | |

### 1.2 Text repair before anything else

Titles are cleaned in `clean_title()` before any filtering: NFKC Unicode
normalization, then a confusable-character fold. Some titles use Cyrillic
letters that render identically to Latin ones (`"тruеcrіmе & makeup"`) —
left alone, these become separate vocabulary entries from the Latin
spelling and split what should be one word across two tokens. A small table
(`CONFUSABLES`) maps the ~19 visually-identical Cyrillic characters back to
Latin before any tokenization happens.

### 1.3 Corpus description

- 10,026 videos, 21 channels, 2012-08-03 to 2023-05-21.
- Title length: mean 9.0 words, median 8, range 1–24.
- Views: median 133,680, mean 742,716 (right-skewed — mean >> median), max
  64,684,244.

### 1.4 Channel imbalance

Largest channel (Dr. Todd Grande) is 28.3% of all videos. This is a
class-imbalance-shaped problem but **not** the kind data augmentation
literature usually targets (there are no topic *labels* to be imbalanced
across — the whole point is that topics are unknown). Where imbalance
*does* matter for this project is on the outcome side: view counts are
heavily right-skewed, which is the reason `view_ratio` (not raw views) is
the outcome variable, and the reason any augmentation applied by a
preprocessing step should be scoped to minority *view-count buckets*
(oversampling under-represented high/low performing titles) rather than to
"balancing topics" — doing the latter would be circular, since topics are
the thing being discovered, not a known label to balance against.

### 1.5 Legal and ethical compliance

- **License**: CC0 1.0 Public Domain Dedication (Kaggle) — permits reuse,
  including redistribution.
- **Provenance**: collected via the official YouTube Data API v3, not
  scraped — the platform ToS was followed by the dataset's original author.
- **Data category**: titles, view counts, channel names are public metadata
  visible to any visitor to the channel page. No private data.
- **Personal data / anonymization**: channel names are kept (needed to
  compute `view_ratio`) but identify creators acting as public figures in a
  professional capacity, not private individuals. No commenters or viewers
  appear in the data, so there is nothing to anonymize on that front.
- **Deliberately excluded**: video descriptions — they contain creator
  contact details and sponsor links, and the research question is scoped to
  titles only.
- **Content sensitivity**: the corpus names real victims and offenders.
  Mitigation: topics are discussed only at the aggregate level (a topic's
  median view ratio, not "this video about this named victim") — no
  individual case is presented as a finding.

---

## Part 2 — Preprocessing

**Script:** `scripts/preprocess.py` → `reports/02_preprocessing.txt`. Seven
stages, in order, every stage's output recorded so the report shows the text
changing at each step.

### 2.1 Tokenization

`nltk.word_tokenize` (the Treebank tokenizer), not `str.split()`.
`str.split()` leaves punctuation attached, so `"killer?"` and `"killer"`
become two different vocabulary entries that the topic model treats as
unrelated. Treebank splits punctuation and clitics; contractions
(`"don't"` → `"do not"`) are expanded by the `contractions` library
*before* tokenization runs, in `normalize()`.

A subword tokenizer (BPE/WordPiece) is deliberately not used: LSA/LDA need
whole words to produce human-readable topics. A topic printed as `##ing,
murd, ##er` cannot be interpreted by a human, and interpretability is the
point of topic modelling (it's also what the human-evaluation stage in
Part 4 directly measures).

### 2.2 Branding removal

Implemented in `learn_branding()`: for each channel, collect every word
appearing in the channel's own name, plus any word appearing in ≥30%
(`BRAND_THRESHOLD`) of that channel's own titles. Both sets are stripped
from that channel's titles only — a different channel that doesn't repeat
the same word keeps it as a content word (e.g. "crime" is branding for
"Buzzfeed Unsolved True Crime" but a real topic word elsewhere).

Why 30% and not some other threshold: no real topic word appears in 30% of
one channel's videos (that would make the channel a single-topic channel),
but a series tag or channel name reliably does. Without this step, the
strongest pattern any of the three models finds is simply which channel a
title came from.

### 2.3 Stopword removal

Standard NLTK English stopword list, **plus** a hand-built domain list
(`DOMAIN_STOPWORDS`) of 51 words that are frequent in this corpus but carry
no topic signal: `part, episode, full, video, channel, subscribe, new,
watch, series, vlog, podcast, ep, pt, update, week, today, live, sneak,
peek, preview, official, feat, ft, special, edition, season, day, one, two,
three, get, got, make, made, go, going, let, like, know, thing, things,
way, time, year, years, story, stories`. These are near-universal in
YouTube titles generally, not specific to true crime, and keeping them
would make every topic partly "about" the genre/platform rather than the
content.

### 2.4 Lemmatization vs. stemming

WordNet lemmatization (POS-tagged via `nltk.pos_tag`, mapped to WordNet's
four POS categories) is the default; Porter stemming is available via
`--stem` for comparison. Concrete comparison from the report:

| word | Porter stem | WordNet lemma |
| --- | --- | --- |
| murderer | murder | murderer |
| mysterious | mysteri | mysterious |
| police | polic | police |
| missing | miss | missing |
| families | famili | family |
| disappearance | disappear | disappearance |
| confession | confess | confession |
| studies | studi | study |

A topic word list read by a human (the word-intrusion task in Part 4)
cannot be labelled with confidence if it reads `murder, mysteri, polic`.
Lemmatization's cost is speed (needs a POS tagger), negligible at ~10k
short documents.

### 2.5 Short-token drop

After lemmatizing, tokens under 3 characters or non-alphabetic (`t.isalpha()`
false) are dropped — at 5-9 tokens per title there is no room for tokens
that carry no topic signal (single letters, leftover digits).

### 2.6 Phrase detection (n-grams)

`gensim.models.phrases.Phrases(min_count=20, threshold=10.0)`, applied after
tokenization/cleaning, joined **41 bigrams**, e.g. `serial_killer,
dark_history, mental_health, conspiracy_theory, bryan_kohberger,
antisocial_personality, borderline_personality`. Phrases scores each
adjacent word pair by how much more often they co-occur than chance predicts
and merges pairs above threshold into one token before any vectorizer or
model sees the documents. Without this, "serial" and "killer" are two
separate dictionary entries and LDA can split one topic-bearing phrase
across two different word distributions.

### 2.7 Worked example (from the report)

| stage | value |
|---|---|
| 1 raw | Was it Sacrifice or Serial Killer?? Leonarda Cianciulli \| Mystery & Makeup \| Bailey Sarian |
| 2 normalized | was it sacrifice or serial killer leonarda cianciulli mystery makeup bailey sarian |
| 3 tokenized | was \| it \| sacrifice \| or \| serial \| killer \| leonarda \| cianciulli \| mystery \| makeup \| bailey \| sarian |
| 4 stopwords + branding removed | sacrifice \| serial \| killer \| leonarda \| cianciulli \| mystery |
| 5 lemmatized | sacrifice \| serial \| killer \| leonarda \| cianciulli \| mystery |
| 6 short tokens dropped | (unchanged — all ≥3 chars) |
| 7 phrases joined | sacrifice \| serial_killer \| leonarda \| cianciulli \| mystery |

### 2.8 Result

- 10,026 titles in → **9,897 kept** (129 dropped, left with <2 tokens after
  cleaning — a title that is only stopwords/branding cannot be modelled).
- Vocabulary: 11,840 → 10,223 after cleaning (86.3% retained).
- 53,359 total tokens, mean 5.4 / median 5 tokens per title after cleaning.

This last number — 5-word documents — is why Part 3's k-selection cannot
just take the peak of a coherence curve: short documents make c_v noisy and
roughly monotonically increasing with k rather than showing a clean peak.

### 2.9 Data augmentation — decision and scope

The rubric this project was built against calls for augmentation where
class imbalance or limited data exists. Applied here **only to minority
view-count buckets** (not to "balance topics" — there are no topic labels
to balance against, and doing so before modelling topics would be
circular). Candidate methods: random word replace/insert/delete,
back-translation. Constraint worth stating plainly: a 9-word title is short
enough that a single random word swap changes the exact word co-occurrence
counts LDA is estimating — augmentation at this text length risks
manufacturing topic signal rather than adding robustness. Mitigation:
models should be run and reported on **both** the original-only corpus and
an augmented corpus, with topic proportions compared between the two, so
any distortion augmentation introduces is visible rather than silently
baked into the headline numbers. (This repo's current pipeline runs on the
original-only corpus; the augmented-corpus comparison is listed as an open
item in Part 6.)

---

## Part 3 — Topic Models: LSA, LDA, NMF, and Number-of-Topics Selection

**Script:** `scripts/topic_models.py` → `reports/03_model_selection.txt`,
`models/`.

### 3.1 Shared vocabulary and TF-IDF

`gensim.corpora.Dictionary` built over all 9,785 documents that survive
vocabulary filtering (some titles are emptied entirely by
`filter_extremes` and dropped — 9,897 in, 9,785 modelled).
`filter_extremes(no_below=5, no_above=0.4)`: a word must appear in at least
5 titles (below that, almost always a one-off victim/offender name — a
topic built from unique names describes nothing) and at most 40% of titles
(not distinguishing). Result: 10,223 → 2,210 terms kept. TF-IDF matrix:
9,785 × 2,210, density 0.18% (sparse, as expected for short documents over
a >2k vocabulary).

### 3.2 LSA (Latent Semantic Analysis) — required

`sklearn.decomposition.TruncatedSVD(n_components=k, random_state=42,
algorithm="arpack")` on the TF-IDF matrix. Finds the k orthogonal
directions explaining the most variance in the term-document matrix — pure
linear algebra, no probability model. Component loadings can be
**negative**; topic word lists are read off by absolute loading
(`np.argsort(np.abs(row))`), so an LSA "topic" reads as "these words move
together (or in opposition)," not as a probability distribution — unlike
LDA, "this title is 60% component 3" is not a meaningful statement for LSA.

### 3.3 LDA (Latent Dirichlet Allocation) — required, primary model

`gensim.models.LdaModel(corpus=bow, id2word=dictionary, num_topics=k,
random_state=seed, passes=10, iterations=100, alpha="auto", eta="auto",
eval_every=None)`. Generative: every document is a mixture over k topics,
every topic a distribution over words, fit by variational Bayes. Weights
are genuine probabilities summing to 1, which is the reason LDA — not LSA
or NMF — is load-bearing for the second half of the project: its
per-document topic distribution is what Part 5 uses to assign one dominant
topic per video for the view-count analysis.

**Multi-seed fitting.** LDA's initialization is random, so a single fit at
a given k can win a sweep purely by init luck. Every k is fit at **3 seeds**
(42, 7, 2024); the sweep reports mean c_v across seeds plus the
seed-to-seed standard deviation, so the reader can see how much of any
coherence gap between neighboring k values is signal versus noise. The
model *kept* for each k is the **median-coherence seed's** fit
(`fits[int(np.argsort(cvs)[len(cvs) // 2])]`), not the best — taking the
best would be cherry-picking a lucky initialization.

### 3.4 NMF (Non-negative Matrix Factorisation) — optional third comparison

`sklearn.decomposition.NMF(n_components=k, init="nndsvda",
random_state=42, max_iter=400)` on the same TF-IDF matrix as LSA, factorised
under a non-negativity constraint instead of orthogonality, fit by
coordinate descent — not Bayesian, not probabilistic. Because factors are
non-negative, a topic's words are always additive (no "contrast" reading
needed, unlike LSA); top words are read directly off the largest loadings.
Functions as a cross-check: a topic pattern (e.g. the "personality
disorder / mental health" cluster, or the "SPSS/coursework" cluster)
showing up independently in LSA, LDA, *and* NMF is evidence it reflects the
data, not one algorithm's inductive bias.

### 3.5 The number-of-topics sweep

k ∈ {5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60}, all three models fit and
scored at every k:

| k | LSA c_v | LSA u_mass | LSA var | LDA c_v (±sd) | LDA u_mass | LDA perplexity | NMF c_v | NMF u_mass |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.281 | -8.705 | 4.0% | 0.350 ±0.028 | -13.426 | 157.0 | 0.376 | -10.245 |
| 8 | 0.297 | -9.582 | 5.5% | 0.356 ±0.007 | -13.999 | 162.2 | 0.396 | -11.206 |
| 10 | 0.310 | -9.345 | 6.3% | 0.375 ±0.019 | -14.924 | 165.0 | 0.415 | -11.261 |
| 12 | 0.328 | -10.070 | 7.0% | 0.386 ±0.005 | -15.731 | 165.8 | 0.427 | -11.426 |
| 15 | 0.333 | -10.153 | 8.1% | 0.382 ±0.007 | -16.426 | 624.6 | **0.443** | -11.153 |
| **20** | **0.351** | -10.316 | 9.7% | **0.415** ±0.015 | -17.567 | 4,812.3 | 0.417 | -11.737 |
| 25 | 0.340 | -10.637 | 11.1% | 0.420 ±0.011 | -17.463 | 11,290.9 | 0.428 | -11.945 |
| 30 | 0.350 | -11.659 | 12.4% | 0.420 ±0.009 | -17.767 | 25,300.4 | 0.439 | -11.720 |
| 40 | 0.346 | -12.401 | 14.7% | 0.438 ±0.009 | -18.370 | 157,474.6 | 0.450 | -12.244 |
| 50 | 0.358 | -12.710 | 16.7% | 0.432 ±0.003 | -18.246 | 1,337,672.2 | 0.443 | -12.894 |
| 60 | 0.362 | -13.090 | 18.5% | 0.453 ±0.005 | -18.418 | 10,957,486.6 | 0.441 | -13.003 |

**Metrics used:**
- **c_v** (primary): sliding-window word co-occurrence + normalized PMI,
  the coherence measure empirically validated against human topic judgments
  (Röder et al. 2015). This is *why* it's primary, not "the standard one."
- **u_mass** (secondary/cross-check): document co-occurrence counts, closer
  to 0 is better; reported because c_v is known to over-reward very small k.
- **Perplexity** (LDA only, reported, *not* used to choose k): falls
  monotonically as k rises even when the added topics are not
  interpretable (Chang et al. 2009) — note the table above shows perplexity
  exploding by 5 orders of magnitude from k=5 to k=60 while c_v barely
  moves, which is exactly the divergence this citation predicts.

**Elbow, not peak.** On these ~5-9-word documents, c_v never turns over
inside the tested grid — LDA's raw peak is k=60 (0.453) and LSA's is also
k=60 (0.362); NMF peaks near k=40 (0.450). Taking the peak would just hand
back the largest k tried, regardless of whether those topics mean anything.
The seed-to-seed LDA spread (0.011 average) is of the same order as the gap
between neighboring k values, so most of that apparent rise is not even a
reliable difference.

The elbow method (`knee()` in `topic_models.py`, the Kneedle idea
implemented by hand): normalize both k and score to [0,1], then take the k
with maximum perpendicular distance above the straight line joining the
first and last point of the curve:

```python
x = (ks - ks.min()) / (ks.max() - ks.min())
y = (ys - ys.min()) / (ys.max() - ys.min())
return int(ks[np.argmax(y - x)])
```

**Result: LSA and LDA both elbow at k=20; NMF elbows at k=15.**

c_v and u_mass disagree on the preferred k for LDA (u_mass prefers k=5,
c_v prefers the still-rising k=60) — reported as a limitation rather than
resolved by picking one metric, since the disagreement itself is
informative about how confidently k can be pinned down on documents this
short. These coherence-selected values are *candidates*; final confidence
comes from the human word-intrusion evaluation (Part 4).

### 3.6 Chosen models

- **LSA, k=20**, explained variance 9.7%.
- **LDA, k=20** — example topics (word list + share of titles by dominant
  assignment): topic 16 "killer, case, solve, old, disturb, detail, update,
  true_crime, mom, brian" (15.8% of titles, the largest); topic 13 "murder,
  coffee, case, trial, idaho, student, take, bryan_kohberger, away, arrest"
  (9.5%).
- **NMF, k=15** (optional) — e.g. topic 9 "spss, test, excel, anova,
  variable, calculate, assumption, create, regression, use_spss" — the same
  coursework/tutorial cluster that shows up independently in LDA topic 12
  and LSA components, cross-validating that this is a real, recurring theme
  in the corpus rather than a modelling artifact.

Artifacts saved to `models/`: `lda.model` (+ `.expElogbeta.npy`,
`.id2word`, `.state`), `dictionary.dict`, `lsa_components.npy`,
`lsa_doc.npy`, `lda_doc.npy`, `nmf_components.npy`, `nmf_doc.npy`,
`meta.json` (`{"best_lsa": 20, "best_lda": 20, "best_nmf": 15, "topn": 10,
"n_docs": 9785, "n_terms": 2210, "seed": 42}`). Downstream scripts read
`meta.json` for the chosen k rather than recomputing or hardcoding it.

---

## Part 4 — Human Evaluation

**Script:** `scripts/human_eval.py` → `data/human_eval/`,
`reports/04_human_eval.txt` (not yet generated — see status below).

Coherence scores are automatic proxies for "would a person find this topic
sensible." The standard direct test is word intrusion (Chang et al. 2009,
"Reading Tea Leaves"):

- For each topic, take its top 5 words (`N_REAL = 5`) plus one intruder word
  sampled from a *different* topic's top-10 words but absent from this
  topic's own list. A rater sees all 6, shuffled, and picks the one that
  doesn't belong.
- **Chance level is 1/(5+1) = 16.7%.** A model's "precision" is the share
  of questions where the rater found the true intruder — well above chance
  means the topic is coherent enough for the intruder to stand out.
- Both LSA (20 topics) and LDA (20 topics) are put on **one shuffled sheet
  with the model hidden** (`build()` strips the model/topic labels and
  reassigns opaque `Q001…` ids), so a rater cannot consciously or
  unconsciously favor one model.
- Raters also write a short label for the other five words and a 1-3
  confidence rating — a topic nobody can name is not usable regardless of
  its c_v.

**Status of this repo's evaluation**: `human_eval.py build` has been run
(`data/human_eval/intrusion_sheet_BLANK.csv`, 41 questions, `intrusion_key.csv`
kept separate from raters, `INSTRUCTIONS.md` written), but no
`responses_*.csv` files exist yet — the sheet has not been filled in by any
rater, so `reports/04_human_eval.txt` does not exist. Scoring (`score()`)
reports, once responses exist: precision per model, precision per
individual rater, the hardest topics (lowest precision — "topics to drop
from interpretation, a topic nobody can separate from the rest is a mixture,
not a theme"), and the labels raters gave each topic. A meaningful precision
comparison needs more than one rater, since inter-rater agreement is itself
part of what makes the result credible — a single rater's score cannot be
checked for consistency.

---

## Part 5 — RQ2: Does Topic Predict Views?

**Script:** `scripts/evaluate_topics.py` → `reports/05_views_by_topic.txt`,
`data/processed/{topic_views.csv, topic_regression.csv, topic_posthoc.csv}`.

### 5.1 Outcome variable, restated

`view_ratio` (Part 0.1). Median = 1.00 by construction per channel;
10th–90th percentile spans 0.13–5.17 — there is real within-channel spread
for a topic to explain.

### 5.2 Per-topic view ratio, with bootstrap confidence intervals

For each LDA topic (a video's *dominant* topic — argmax of its LDA
probability vector), the median `view_ratio` of videos assigned to it, with
a bootstrap 95% CI (`N_BOOT = 2000` resamples):

```python
def boot_ci(values, stat=np.median, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = stat(values[idx], axis=1)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)
```

This matters because a topic backed by 60 videos should not be read with the
same confidence as one backed by 1,500 — the CI width makes that explicit.
Topics with fewer than `MIN_DOCS = 30` videos are flagged and excluded from
interpretation.

**Result**: best topic (topic 7 — "say, boyfriend, real, shoot, date,
guilty," an unresolved relationship-crime narrative theme) at **1.36×** the
channel median (185 videos, 20 channels, 95% CI [1.22, 1.63]); worst (topic
12 — "short, test, love, parent, help, question," an SPSS/psychology
coursework-style content theme) at **0.45×** (622 videos, CI [0.41, 0.51]) —
a **3.0×** gap between best and worst.

### 5.3 Kruskal-Wallis test + post-hoc

`view_ratio` is heavily right-skewed (median 1.00, 90th percentile 5.2×),
so a one-way ANOVA's normal-residual assumption does not hold.
Kruskal-Wallis (`scipy.stats.kruskal`) is the rank-based non-parametric
equivalent, run across the 20 topics' `view_ratio` distributions (restricted
to the 30+-video topics from 5.2):

- **H(19) = 418.4, p ≈ 6.49×10⁻⁷⁷** — rejects the null that all 20 topics
  share one `view_ratio` distribution.
- **Post-hoc**: a significant Kruskal-Wallis only establishes that *some*
  pair differs, not *which*. Every one of the 190 topic pairs
  (C(20,2) = 190) is followed up with a two-sided Mann-Whitney U test
  (`scipy.stats.mannwhitneyu`), and the 190 p-values are
  **Holm-Bonferroni corrected** — implemented by hand in
  `evaluate_topics.py`:

  ```python
  order = np.argsort(raw_p)
  running_max = 0.0
  for rank, idx in enumerate(order):
      running_max = max(running_max, raw_p[idx] * (len(raw_p) - rank))
      adj_p[idx] = min(running_max, 1.0)
  ```

  Correction is necessary because running 190 tests uncorrected at α=.05
  would produce ~9-10 false positives by chance alone. Result: **73 of 190
  pairs (38%)** differ significantly after correction; best-vs-worst
  (topic 7 vs. 12) differs at adjusted p ≈ 1.83×10⁻¹⁷. Several top topics'
  confidence intervals overlap in the table (5.2) — the post-hoc test is
  what actually distinguishes real ranking differences from noise, and the
  full 190-row pairwise result is saved to
  `data/processed/topic_posthoc.csv` for inspection.

### 5.4 Regression — the complementary view

Assigning each title to its single best (argmax) topic throws away the fact
that most titles are a probability *mixture* over topics. As a complement,
all 19 free LDA topic proportions are entered as predictors of
`log2(view_ratio)` at once, one topic dropped as the reference category
(proportions sum to 1, so keeping all 20 would be exactly collinear with
the intercept):

```python
Xt = np.column_stack([np.ones(len(df)), lda_doc[:, :-1]])
coef, *_ = np.linalg.lstsq(Xt, y, rcond=None)
```

Standard errors from OLS residual variance and the Moore-Penrose
pseudoinverse of `XᵀX`; t-statistics as usual. **R² = 0.064** — topic
content explains 6.4% of within-channel view variation. State this
plainly as the headline, not a caveat to bury: small, real, and most of
what drives a video's views is not in its title at all. Strongest positive
coefficients: topic 9 (secret/murderer/teen cluster, coef +2.27 log2 units
≈ 4.82× views), topic 10 (mystery/serial_killer/case, +2.25 ≈ 4.75×);
strongest negative: topic 12 (short/test/love/parent, -2.26 ≈ 0.21×).

### 5.5 Single-word sanity check (model-independent)

For every word appearing in ≥40 titles, the median `view_ratio` of titles
containing it — each title counted **once per unique word**, not once per
occurrence, so a title repeating a word twice doesn't get double-counted
(`unique_words = df["tokens"].map(lambda t: sorted(set(t)))`). No topic
model is involved at all. The point: if the same high/low-performing themes
show up here independent of LDA/LSA/NMF, that's evidence the RQ2 finding is
about the underlying data, not an artifact of any one model's
parameterization.

### 5.6 LSA cross-check

The same per-topic `view_ratio` computation, run on LSA's 20 components
instead of LDA's dominant-topic assignment — reported so the reader can
check whether the two structurally different models (probabilistic mixture
vs. linear factorization) tell the same view-count story.

### 5.7 Correlation ≠ causation

Flagged explicitly, not just as a stock phrase: a topic's `view_ratio`
co-varies with channel size (only partially controlled for by `view_ratio`
— a channel's *typical* topic mix is itself channel-specific, so the
outcome-side correction doesn't fully separate topic effect from channel
identity), with recency (a topic trending in the corpus's later years will
look artificially strong if the corpus itself is recency-skewed), and with
thumbnail/upload-time effects — none of which are modelled here. A topic
effect here is evidence of *co-occurrence* with higher views, not proof
that the topic *causes* them.

---

## Part 6 — Limitations

- **Titles-only scope.** Descriptions/transcripts carry more topical signal
  than a 5-9-word title and could sharpen coherence, but weren't collected
  — the single biggest lever for a follow-up study.
- **Corpus imbalance.** One channel (Dr. Todd Grande) is 28.3% of all
  videos; topics that channel favors are structurally over-represented in
  the topic-word-list output, independent of whether they're view-driving.
  `view_ratio` corrects the *outcome* side of this, not the
  topic-distribution side.
- **Name ablation — a negative result, reported rather than hidden**
  (`scripts/name_ablation.py` → `reports/06_name_ablation.txt`): tested
  whether stripping 495 capitalized proper-noun tokens (words capitalized
  mid-title in ≥90% of occurrences, seen ≥3 times, and not also ordinary
  English nouns per WordNet) improves LDA coherence at k ∈ {10, 15, 20,
  25}. Mean change **+0.010**, sign inconsistent across k (improved at 2 of
  4 k values, worsened at the other 2) — no reliable gain, so names are
  kept in the main pipeline. Visible cost: several topic word lists include
  a first name (e.g. "james," "john") as a top word — a symptom of
  clustering on defendant/victim identity rather than crime *type*,
  distinct from the coherence-score question.
- **k-selection ceiling.** The elbow method is a heuristic on a curve that
  never truly plateaus at this document length (c_v still rising at k=60
  for both LDA and LSA). A larger k might yield more granular, still-valid
  topics; the elbow is a principled stopping point, not proof k=20 is
  optimal.
- **Augmentation not yet run as a comparison.** The original-only corpus is
  what's modelled throughout; the augmented-corpus comparison described in
  Part 2.9 (both corpora modelled, topic proportions compared) has not been
  executed in this repo.
- **Human evaluation not yet scored.** The word-intrusion sheet is built
  (Part 4) but has no rater responses yet — the k=20 choice currently rests
  on coherence and the elbow heuristic alone, not on the human check that
  is supposed to confirm it.
- **RQ2 causal limits.** Channel size, recency, thumbnail, and upload-time
  confounds are not modelled (Part 5.7).

---

## Appendix A — Running the pipeline

`gensim` has no wheel for very new Python versions (this project targets
**Python 3.11**):

```bash
python3.11 -m venv .venv
./.venv/bin/pip install pandas numpy scikit-learn gensim nltk matplotlib contractions scipy
./.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger_eng','names']]"
```

Each script checks for its input file and raises `SystemExit` naming the
prerequisite script if it's missing, rather than silently producing wrong
output on a stale/absent upstream file. Run in this order after changing
anything upstream:

```bash
./.venv/bin/python scripts/build_corpus.py       # -> data/processed/corpus.csv
./.venv/bin/python scripts/preprocess.py         # -> data/processed/tokens.csv
./.venv/bin/python scripts/topic_models.py       # -> models/, doc_topics.csv, k_sweep.csv (~2 min)
./.venv/bin/python scripts/evaluate_topics.py    # -> topic_views.csv, topic_regression.csv, topic_posthoc.csv
./.venv/bin/python scripts/make_figures.py       # -> figures/*.png
```

Optional, not part of the numbered dependency chain:

```bash
./.venv/bin/python scripts/human_eval.py build   # word-intrusion sheet
# ... hand data/human_eval/intrusion_sheet_BLANK.csv to raters as
#     responses_<name>.csv ...
./.venv/bin/python scripts/human_eval.py score   # -> reports/04_human_eval.txt

./.venv/bin/python scripts/name_ablation.py      # -> reports/06_name_ablation.txt
./.venv/bin/python scripts/preprocess.py --stem  # Porter stemming comparison
```

**The reports are the deliverable.** Every number quoted in `README.md`, in
`docs/`, and in this document comes from `reports/*.txt`. If behavior
changes, re-run the affected stage (and everything downstream of it) and
update both.

---

## Appendix B — Invariants that break silently

Things that will not raise an error but will quietly produce wrong or
misleading output:

| Invariant | What happens if violated |
|---|---|
| `models/meta.json`'s `best_lsa`/`best_lda`/`best_nmf` are read by every downstream script, never recomputed | if `topic_models.py` is re-run with different data but a downstream script uses a stale `meta.json`, results silently mismatch the current model |
| Every vectorizer is fit on the **whole preprocessed corpus** once, not per-split (this project has no train/test split — it's unsupervised) | N/A here, but the general principle (never fit a transform on data it will later score) is the same reason `evaluate_topics.py` reads model outputs rather than refitting anything |
| `filter_extremes(no_below=5, no_above=0.4)` must match between `topic_models.py` and `name_ablation.py`'s `coherence_at()` | a coherence comparison computed with different vocabulary filtering is not a fair comparison, even though nothing errors |
| LDA is fit at 3 seeds and the **median**-coherence seed is kept, not the best | keeping the best silently reports a lucky initialization as if it were the model's typical behavior |
| `random_state=42` / `SEED = 42` is threaded through `TruncatedSVD`, `LdaModel`, `NMF`, and every bootstrap/bar computation | an unseeded rerun changes topic numbering and exact coherence values, which would make report numbers silently stop matching the code that produced them |
| Bootstrap CIs and the Kruskal-Wallis/post-hoc test only run over topics with `n >= MIN_DOCS (30)` | including tiny topics would produce a CI wide enough to be meaningless without any visible warning in the number itself |
| `human_eval.py build` shuffles rows and strips model/topic identity before writing the blank sheet | if a rater sees which model or topic a question came from, the precision comparison between LSA and LDA is contaminated |
| Word counting in `evaluate_topics.py`'s single-word check dedupes per title (`sorted(set(t))`) before exploding | without dedup, a title repeating a word twice is counted twice, inflating that word's apparent title-count and biasing its reported median view_ratio |

---

## Appendix C — Glossary

- **c_v coherence** — topic coherence measure based on a sliding window over
  the corpus plus normalized pointwise mutual information between top
  words; the measure most validated against human judgments of topic
  quality.
- **u_mass coherence** — topic coherence based on document co-occurrence
  counts; closer to 0 is better. Tends to favor smaller k than c_v.
- **Perplexity** — how well a language model predicts held-out data; falls
  as k rises even past the point where topics remain interpretable, so not
  used here to choose k.
- **Elbow (Kneedle) method** — pick the point on a score-vs-parameter curve
  with maximum perpendicular distance from the straight line joining the
  curve's endpoints; used when the curve doesn't have a clean peak.
- **Dominant topic** — for a document under LDA, the topic with the highest
  probability in that document's topic-mixture vector (`argmax`).
- **Bootstrap confidence interval** — resample the data with replacement
  many times, recompute the statistic each time, and read off the 2.5th and
  97.5th percentiles of the resulting distribution.
- **Kruskal-Wallis test** — non-parametric, rank-based generalization of a
  one-way ANOVA; tests whether several independent samples come from the
  same distribution, without assuming normality.
- **Mann-Whitney U test** — non-parametric, rank-based test comparing two
  independent samples; the pairwise post-hoc used here after a significant
  Kruskal-Wallis result.
- **Holm-Bonferroni correction** — a step-down method for controlling the
  family-wise error rate across multiple simultaneous hypothesis tests;
  less conservative than a flat Bonferroni correction while still
  controlling false positives.
- **Word intrusion** — a human-evaluation protocol (Chang et al. 2009)
  where a rater picks the one word, out of several shown, that does not
  belong to a topic; used to measure whether a topic is interpretable by
  people, not just coherent by an automatic metric.
