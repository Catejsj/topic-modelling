# Presentation structure — 6 people

Maps the rubric's 7 criteria onto 6 parts, each backed by a specific
script/report/figure in this repo. Numbers are from the generated reports
(`reports/01_corpus.txt` … `reports/06_name_ablation.txt`) — re-check them if
the pipeline is re-run before the talk, since a re-run can shift them
slightly.

Total: ~20-25 min, 3-4 min per person + buffer. Order follows the pipeline's
data flow: corpus → preprocessing → models → evaluation → limitations.

---

## Person 1 — Problem Definition & Motivation (rubric: 5 pts)

No script — pure framing, sets up terminology the rest of the talk depends
on.

- **RQ1**: what latent topics exist in true-crime YouTube titles?
  Unsupervised — no ground-truth topic labels, which is the reason this is
  topic modelling rather than text classification.
- **RQ2**: does a video's topic predict its view count, relative to that
  channel's own norm?
- **Scope**: corpus is titles only, not descriptions or transcripts. Title
  is the text every viewer sees before clicking, making it the piece of
  text causally upstream of the click decision. Limitation revisited in
  Person 6's section.
- **Unit of analysis**: one video = one document. LDA/LSA need a document
  as the unit that "contains" a topic mixture; a title is the smallest
  coherent text tied to one measurable outcome (its own view count).
- **Dataset scale**: 10,107 videos, 21 channels, collected via the official
  YouTube Data API v3, via Kaggle (`data/raw/true_crime_channel_stats.csv`).

**Slides:** 1-2, no output to show yet.

---

## Person 2 — Corpus Collection & Legal/Ethical Compliance (rubric: 10 pts)

**Source:** `scripts/build_corpus.py` → `reports/01_corpus.txt`.

**Filtering pipeline** (drop counts):
- empty title: 0
- missing view count: 1
- zero views: 11
- duplicate channel+title: 69
- kept: 10,026 of 10,107 (99.2%)

**Corpus description**: 21 channels, dates 2012-08-03 to 2023-05-21, title
length mean 9.0 words (median 8, range 1-24). This length is the constraint
that shapes several downstream choices in Person 3/4's sections
(lemmatization over stemming, phrase detection, elbow-based k selection).

**Channel imbalance**: largest channel (Dr. Todd Grande) is 28.3% of all
videos; the most-watched channel's median video gets 4,119× the views of
the least-watched channel's median video. This is the reason the project
uses `view_ratio` (views ÷ that channel's own median) instead of raw views
as the outcome variable (Person 5), and the reason augmentation is scoped
to minority view-count buckets rather than topic classes (there are no
topic labels to be imbalanced across).

**Legal/ethical compliance** (10 of 100 rubric points):
- License: CC0 1.0 (Kaggle, public domain dedication) — permits reuse and
  redistribution.
- Provenance: official YouTube Data API v3, not scraped.
- Data category: titles, view counts, channel names are public metadata
  visible on the channel page — no private data.
- Personal data / anonymization: channel names are kept (needed for
  `view_ratio`) but identify creators as public figures in a professional
  capacity, not private individuals. No commenters or viewers appear in the
  data.
- Excluded by design: video descriptions (creator contact details, sponsor
  links; also out of scope per Person 1).
- Content sensitivity: corpus names real victims and offenders. Mitigation:
  topics are reported and discussed only at the aggregate level — no
  individual case is presented as a finding.

**Slides:** 2 — filtering funnel + channel balance table; legal/ethical
bullet list.

---

## Person 3 — Text Preprocessing & Augmentation (rubric: 25 pts)

**Source:** `scripts/preprocess.py` → `reports/02_preprocessing.txt`.

1. **Tokenization** — `nltk.word_tokenize` (Treebank tokenizer), not
   `str.split()`. `str.split()` leaves punctuation attached, so `"killer?"`
   and `"killer"` become distinct vocabulary entries. Treebank splits
   punctuation and clitics (`"don't"` → `do` + `n't`), handled by a
   contraction-expansion step that runs first. A subword tokenizer
   (BPE/WordPiece) is not used because LSA/LDA need whole words for
   human-readable topics — `##ing, murd, ##er` is not interpretable.

2. **Branding removal** — a word is stripped as branding, per channel, if
   it appears in ≥30% of that channel's own titles or is part of the
   channel name. Without this, the strongest pattern any model finds is
   the channel's own series tag. Examples from the report:
   - Bailey Sarian → `bailey, makeup, sarian`
   - JCS - Criminal Psychology → `case, criminal, jcs, psychology`
   Channel-scoped, not global: "crime" is branding for "Buzzfeed Unsolved
   True Crime" but stays a content word elsewhere.

3. **Stopword removal** — standard English list plus a custom domain list
   (`true, crime, story, episode, full, documentary, part`) and the
   per-channel branding tokens above. These terms are near-universal in
   true-crime titles and carry no discriminating topic signal.

4. **Lemmatization vs. stemming** — WordNet lemmatization (POS-tagged) is
   the default; Porter stemming was run for comparison
   (`preprocess.py --stem`):

   | word | Porter stem | WordNet lemma |
   |---|---|---|
   | murderer | murder | murderer |
   | mysterious | mysteri | mysterious |
   | police | polic | police |
   | disappearance | disappear | disappearance |

   A human-read topic word list (Person 5's word-intrusion task) is hard to
   label with confidence from `murder, mysteri, polic`. Cost of
   lemmatization is speed (POS tagging), negligible at ~10k short
   documents.

5. **Phrase detection (n-grams)** — `gensim.models.Phrases`
   (`min_count=20, threshold=10`) joined 41 bigrams before
   dictionary-building, e.g. `serial_killer, dark_history, mental_health,
   conspiracy_theory, bryan_kohberger`. Phrases scores adjacent word pairs
   by co-occurrence relative to chance and merges pairs above threshold
   into one token before any model sees the documents — otherwise "serial"
   and "killer" are separate dictionary entries and LDA can split one
   topic-bearing phrase across two word distributions.

6. **Data augmentation**:
   - Applied only to minority view-count buckets. View counts are heavily
     right-skewed (Person 2's 4,119× channel spread), so high-view outlier
     videos are under-represented as examples in any bucket-conditional
     analysis. Not applied to "balance topics" — there are no topic labels
     to balance by, and doing so would be circular.
   - Methods: random word replace/insert/delete, back-translation.
   - Constraint: a 9-word title is short enough that a single random word
     swap changes the exact word co-occurrence counts LDA estimates —
     augmentation at this text length risks manufacturing topic signal
     rather than adding robustness. Mitigation: models are run and reported
     on both the original-only corpus and the augmented corpus, with topic
     proportions compared between the two.

7. **Result**: 10,026 titles in → 9,897 kept (129 dropped, <2 tokens after
   cleaning); vocabulary 11,840 → 10,223 (86.3% retained); 53,359 total
   tokens, mean 5.4 / median 5 tokens per title after cleaning. This
   document length is the reason Person 4's k-selection uses an elbow
   rather than a coherence peak — short documents make c_v noisy and
   roughly monotonic in k.

**Slides:** 2-3 — pipeline diagram (raw → normalized → tokenized →
branding/stopwords → lemmatized → phrases); the report's worked-example
table (`[3] WORKED EXAMPLE`); augmentation scope and rationale.

---

## Person 4 — Model Design, Training & Number-of-Topics Selection (rubric: 25 pts)

**Source:** `scripts/topic_models.py` → `reports/03_model_selection.txt`,
`models/`.

**LSA (required)** — TF-IDF matrix, then truncated SVD
(`sklearn.decomposition.TruncatedSVD`, `algorithm="arpack"`). Finds the k
orthogonal directions explaining the most variance in the term-document
matrix; pure linear algebra, no probability model. Component loadings can
be negative — topic word lists are read by absolute loading
(`np.argsort(np.abs(row))`), so a topic reads as "these words move together
(or in opposition)," not as a probability distribution.

**LDA (required, primary model)** — generative: every document is a
mixture over k topics, every topic a distribution over words, fit by
variational Bayes (`gensim.models.LdaModel`, `alpha="auto", eta="auto"`).
Weights are probabilities summing to 1, so "this title is 60% topic 3" is a
valid statement only for LDA — that per-document distribution is what
Person 5 uses to assign a dominant topic per video for RQ2. This is why LDA
carries the second half of the project rather than LSA or NMF.
- **Multi-seed fitting**: LDA initialization is random, so a single fit can
  win a sweep by init luck. Every k is fit with 3 seeds (42, 7, 2024),
  scored on mean c_v; seed-to-seed spread (~0.01-0.03) is reported so the
  audience can gauge how much of a coherence gap between neighboring k is
  signal versus noise. The model kept per k is the median-coherence seed's
  fit, not the best.

**NMF (optional, third comparison)** — same TF-IDF matrix as LSA, factorised
under a non-negativity constraint instead of orthogonality
(`sklearn.decomposition.NMF`, `init="nndsvda"`), fit by coordinate descent —
not Bayesian, not probabilistic. Factors are non-negative, so a topic's
words are always additive; top words are read directly off the largest
loadings. Functions as a cross-check: a topic appearing in LSA, LDA, and
NMF independently is evidence the topic reflects the data rather than one
algorithm's inductive bias.

**Number-of-topics sweep** — k ∈ {5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60},
all three models fit and scored at every k.
- **c_v coherence** (primary): sliding-window co-occurrence + normalized
  PMI, the measure validated against human topic judgments (Röder et al.
  2015).
- **u_mass coherence** (secondary): document co-occurrence counts, closer
  to 0 is better; included because c_v over-rewards very small k.
- **Perplexity** (LDA only, reported, not used to choose k): falls
  monotonically as k rises even when added topics are not interpretable
  (Chang et al. 2009).
- **Elbow, not peak**: on these ~5-9-word documents, c_v never turns over
  inside the grid — still rising at k=60 for both LDA (0.453) and NMF
  (0.441). The seed-to-seed LDA spread (~0.011 average) is of the same
  order as the gap between neighboring k values. Elbow method: normalize
  both k and score to [0,1], take the k with maximum perpendicular distance
  above the line joining the first and last point (Kneedle idea,
  implemented by hand as `knee()`). Result: LSA and LDA both elbow at
  k=20; NMF elbows at k=15.
- c_v and u_mass occasionally disagree on the preferred k for LDA —
  reported as a limitation rather than resolved by picking one metric.
- These coherence-selected k values are candidates; confirmation comes from
  the human word-intrusion evaluation (Person 5).

**Other numbers**: vocabulary filtering via
`filter_extremes(no_below=5, no_above=0.4)` — words must appear in ≥5
titles (below that, almost always a one-off name) and ≤40% of titles (not
distinguishing); TF-IDF matrix shape and density; example chosen-LDA topics
with top words and share of titles (e.g. topic 16, "killer, case, solve,
old, disturb…", 15.8% of titles — the largest topic).

**Slides:** 2-3 — `fig1_coherence_vs_k.png` (all three models) as
centerpiece; a topic-word-list comparison across models at the chosen k
region.

---

## Person 5 — Topic Evaluation & Interpretation (rubric: 25 pts)

**Source:** `scripts/human_eval.py`, `scripts/evaluate_topics.py` →
`reports/04_human_eval.txt` + `reports/05_views_by_topic.txt`,
`data/human_eval/`.

**Human evaluation — word intrusion test.** Each topic's top words plus one
intruder word sampled from a different topic; a rater sees 6 words and
picks the one that doesn't belong (Chang et al. 2009 protocol). A coherent
topic makes the intruder easy to spot; an incoherent one makes it
indistinguishable from the rest.
- 40 questions, shuffled, blind to which model produced which topic, so no
  rater can favor LDA over LSA/NMF by knowing which is "supposed" to win.
- Raters record the intruder, a label for the remaining five words, and a
  1-3 confidence rating.
- Scoring (`human_eval.py score`): rater accuracy at picking the true
  intruder is the human-coherence metric; inter-rater agreement is a
  reliability check on top of it — both matter, since agreement can't be
  assessed from a single rater.
- This is what adjudicates between the coherence-selected k candidates
  (LSA/LDA at k=20, NMF at k=15) when the automatic metrics disagree.

**Topic coherence, quantitatively** — c_v and u_mass at the chosen k, all
three models side by side, from the full sweep table
(`reports/03_model_selection.txt`).

**RQ2 — does topic predict views?**

1. **Outcome variable**: `view_ratio` = views ÷ that video's own channel's
   median. Median = 1.00 by construction per channel; 10th-90th percentile
   spans 0.13-5.17.

2. **Per-topic view ratio, bootstrap CI**: median view_ratio of videos
   whose dominant LDA topic (argmax of the topic distribution) is each
   topic, with a bootstrap 95% CI (2,000 resamples) so a topic backed by 60
   videos isn't read with the confidence of one backed by 1,500. Best
   topic: topic 7 ("say, boyfriend, real, shoot, date, guilty") at 1.36×
   the channel median; worst: topic 12 ("short, test, love, parent, help,
   question") at 0.45× — a 3.0× gap.

3. **Kruskal-Wallis test + post-hoc** — `view_ratio` is heavily
   right-skewed, so ANOVA's normal-residual assumption doesn't hold;
   Kruskal-Wallis (`scipy.stats.kruskal`) is the non-parametric equivalent,
   run across the 20 topics' view_ratio distributions.
   - H(19) = 418.4, p ≈ 6.5×10⁻⁷⁷ — rejects the null that all topics share
     a distribution.
   - Post-hoc: all 190 topic pairs compared with two-sided Mann-Whitney U,
     Holm-Bonferroni corrected (uncorrected, 190 tests at α=.05 would
     produce ~9-10 false positives by chance). 73 of 190 pairs differ
     significantly after correction; best-vs-worst (topic 7 vs. 12) differs
     at adjusted p ≈ 1.8×10⁻¹⁷.
   - Several top topics' confidence intervals overlap — the post-hoc test
     is what distinguishes real ranking differences from noise.

4. **Regression**: all 19 free topic proportions as predictors of
   log2(view_ratio) at once (one topic dropped as reference to avoid
   collinearity with the intercept, since proportions sum to 1). Answers a
   different question than the per-topic medians: the marginal effect of
   more of a title being topic X, holding the rest fixed. R² = 0.064 —
   topic content explains 6.4% of within-channel view variation.

5. **Single-word check, independent of any model**: median view_ratio for
   words appearing in ≥40 titles (each title counted once per unique word).
   The same high/low-performing themes appearing here with no topic model
   involved is evidence the RQ2 finding is about the data, not an artifact
   of LDA's parameterization.

6. **Correlation ≠ causation**: topic co-varies with channel size
   (partially controlled via `view_ratio`, not perfectly — a channel's
   typical topic mix is itself channel-specific), recency, and
   thumbnail/upload-time effects, none of which are modelled. A topic
   effect is evidence of co-occurrence with higher views, not proof of
   causation.

**Slides:** 3 — word-intrusion mechanism + results; `fig2_views_by_topic.png`
as the RQ2 headline visual; Kruskal-Wallis/post-hoc numbers + regression R².

---

## Person 6 — Limitations, Future Studies, and Presentation Synthesis (rubric: Limitation 5 pts + Presentation & Report Clarity 5 pts)

**Source:** `scripts/make_figures.py`, `scripts/name_ablation.py` →
`reports/06_name_ablation.txt`, `figures/`.

**Limitations:**
- Titles-only scope (Person 1): descriptions/transcripts carry more
  topical signal than a 5-9-word title and could sharpen coherence, but
  weren't collected. The largest lever for a follow-up study.
- Corpus imbalance: one channel (Dr. Todd Grande) is 28.3% of all videos;
  topics that channel favors are structurally over-represented in the
  topic-word-list output, independent of whether they're view-driving.
  `view_ratio` corrects the outcome side of this, not the
  topic-distribution side.
- Name ablation (negative result): tested whether stripping 495
  capitalized proper-noun tokens (offender/victim names appearing ≥3
  times, ≥90% mid-title-capitalized, not also common English nouns)
  improves LDA coherence. Mean change +0.010, sign inconsistent across k
  (improved at 2 of 4 k values tested, worsened at the other 2) — no
  reliable gain, names kept. Visible cost: several topic word lists
  include a first name (e.g. "james," "john") as a top word, a symptom of
  clustering on defendant/victim identity rather than crime type —
  distinct from the coherence-score question.
- k-selection ceiling: the elbow method is a heuristic on a curve that
  never truly plateaus at this document length (c_v still rising at k=60).
  A larger k might yield more granular, still-valid topics; the elbow is a
  principled stopping point, not proof k=20 is optimal.
- Augmentation tradeoff (Person 3): helps sample balance but risks
  shifting topic proportions on already-short documents — both
  original-only and augmented-corpus results should be reported side by
  side to size that distortion rather than assert it's small.
- RQ2 causal limits (Person 5): channel size, recency, thumbnail, and
  upload-time confounds are not modelled.

**Future studies** (paired with the limitation each addresses):
- Extend the corpus to descriptions/transcripts as a second document type;
  compare topic coherence and RQ2 effect sizes against the titles-only
  model.
- Stratify or reweight by channel when reporting topic prevalence, on top
  of the outcome-side `view_ratio` correction.
- Named-entity resolution (map name instances to a `PERSON` placeholder)
  as a middle ground between the two options the ablation tested.
- A held-out temporal split (train on pre-2022 titles, evaluate coherence
  and RQ2 on 2022-2023 titles) to check topic and effect stability over
  time, addressing the recency confound directly.
- A multivariate RQ2 model including channel, upload recency, and
  thumbnail features alongside topic proportions, to partial out the
  confounds above.

**Presentation & Report Clarity** — cross-checking the other five parts for
consistency:
- Terminology: "topic" always an LDA/LSA/NMF component, not a colloquial
  subject; "topic" vs. "topic proportion" vs. "dominant topic" used
  precisely and consistently; `view_ratio`, channel share, and video counts
  matching across every slide that cites them.
- Every number in the talk traces to a specific report file and section
  (this document's citations are the map for that check).
- Q&A prep: consolidate the existing "not finished yet" / Q&A section in
  `docs/SLIDES_LSA_LDA.md` into one shared appendix covering augmentation,
  NMF, and the Kruskal-Wallis choice — the three points most likely to be
  interrogated.

**Slides:** 2 — limitations/future-work pairs; closing summary and Q&A
transition.

---

## Timing notes

- Persons 3, 4, and 5 carry the 25-point criteria — most rehearsal time and
  Q&A prep belongs there.
- Person 2's legal/ethics section is short in slide count but is 10 rubric
  points — worth deliberate pacing rather than a speed-run.
- Hand-offs follow the pipeline: Person 2 → 3 (9,897 of 10,026 titles
  survive preprocessing), Person 3 → 4 (those tokenized titles are
  factorised next), Person 4 → 5 (k=20 is the candidate, checked against
  human raters), Person 5 → 6 (what RQ2's answer leaves open).
