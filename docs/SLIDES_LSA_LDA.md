# Slide content — LSA and LDA (Seth)

Two slides. Everything below is either spoken or on screen; nothing else.
The numbers all come from `reports/` and are re-derived in the last section, so
if anyone asks "where did that come from" there is an answer.

Research question for the whole project:
**which kind of video title attracts the most views?**

My part is the modelling: turning 9,785 titles into a small number of themes,
choosing how many themes, and checking the themes are real.

---

## Slide 1 — Two models, and how many topics

### On screen (bullets, keep them short)

- **LSA** — TF-IDF matrix → truncated SVD. Linear algebra. Finds the directions
  that explain the most variance. Weights can be negative.
- **LDA** — every title is a *mixture* of topics, every topic a distribution
  over words. Probabilities, so "this title is 60% topic 10" means something.
- Same vocabulary for both: 2,210 words, appearing in ≥ 5 titles and ≤ 40%.
- k swept from 5 to 60. LDA fitted 3 times per k with different random seeds.
- **c_v coherence never turns over** — it is still rising at k = 60.
- So we took the **elbow**, not the peak. Both models chose **k = 20**.

### Visual

`figures/fig1_coherence_vs_k.png` — coherence against k, both models, shaded
seed band, dashed line at the chosen k.

### Say (about 90 seconds)

> We ran the two approaches from class on the same data and the same vocabulary,
> so the comparison is fair.
>
> LSA is the linear algebra one. We build a TF-IDF matrix — nine thousand titles
> down the side, two thousand words across — and use truncated SVD to compress it
> to twenty directions. Each direction is a topic. It is fast and deterministic,
> but the weights can be negative, so a topic is really "these words move
> together", which is harder to explain to a non-technical reader.
>
> LDA is the probability one. It assumes every title was written by picking a
> mixture of topics and then drawing words from those topics, and it works
> backwards to recover them. Everything comes out as a probability that sums to
> one, so we can say a title is sixty per cent topic ten.
>
> The real work was choosing k. The standard method is to plot coherence against
> k and take the maximum. That failed here, and the failure is the interesting
> part. Our documents are titles — nine words each, five after cleaning — so
> there is barely any word co-occurrence inside a document. On text that short
> the c_v score just keeps climbing as you add topics. We went all the way to
> sixty and it was still rising.
>
> We also fitted LDA three times at every k with different random seeds. The
> spread between seeds was about 0.011, which is the same size as the gap between
> neighbouring points on the curve. So most of that rise is noise, not a real
> improvement.
>
> Instead of the peak we took the elbow — the point furthest above the straight
> line from the first point to the last, which is where extra topics stop buying
> much coherence. LSA and LDA independently landed on the same answer, twenty.
> Twenty topics is also a number a person can actually read through, which
> matters because the whole point of topic modelling is interpretation.

---

## Slide 2 — What the topics say about views

### On screen (bullets)

- Outcome = **view_ratio**: a video's views ÷ the median video *on its own
  channel*. Raw views would just measure channel size.
- Best topic **1.36×** the channel norm — *say, boyfriend, real, shoot, guilty*.
- Worst topic **0.45×** — *short, test, love, parent, help* (SPSS and counselling
  tutorials). A **3.0× spread**.
- Regression on all topic proportions: **R² = 0.064**.
- Same split appears in raw single words, with no model involved:
  *dark history* 3.73×, *mystery* 3.12× / *session* 0.05×, *counsel diagnostic* 0.08×.
- LSA agrees with LDA. The finding is in the data, not in the algorithm.

### Visual

`figures/fig2_views_by_topic.png` — the twenty topics as bars above and below the
channel norm. If there is room, `figures/fig3_words_by_views.png` beside it.

### Say (about 90 seconds)

> Now the actual question: which titles get the views.
>
> First a warning about the outcome. If we used raw view counts we would just be
> measuring how big the channel is — the median JCS video has four thousand times
> the views of the median 48 Hours video, and that has nothing to do with the
> title. So we divide every video by the median video on its own channel. One
> point zero means typical for that channel.
>
> The top topic is at 1.36 times the channel norm: words like *say, boyfriend,
> real, shoot, guilty* — a specific case told as a story, with a courtroom angle.
> Just behind it is mystery, serial killer, unsolved, at 1.32. The bottom of the
> chart is completely different in kind: *short, test, love, parent, help* at 0.45,
> and *look, find, body, palette* at 0.46. Those are the instructional videos —
> SPSS tutorials, counselling role-plays, makeup tutorials — that these channels
> post alongside their crime content. Roughly a three-times spread from top to
> bottom.
>
> We checked this two more ways. A regression with all twenty topic proportions
> as predictors gives an R-squared of 0.064. I want to be straight about that
> number: the topic of the title explains about six per cent of the variation in
> views. It is real and it is highly significant, but ninety-four per cent is
> thumbnail, timing, the algorithm and how famous the case already is.
>
> And with no model at all — just the median view ratio of every title containing
> a given word — the same split appears. *Dark history* 3.7 times, *mystery* 3.1
> times at the top; *session*, *counselling diagnostic*, *review* at a twentieth
> of normal at the bottom. LSA and LDA also rank the topics the same way. So the
> answer is not an artefact of one algorithm.
>
> The one-line answer: **titles that promise a specific unresolved crime story do
> well; titles that promise instruction do badly — even on the same channel.**

---

## Q&A prep

**Why does LDA get a higher coherence score than LSA?**
LDA's top words come from a probability distribution over the whole vocabulary,
so they tend to co-occur in real documents. LSA's come from the largest absolute
loadings on an SVD direction, which can mix words that never appear together but
sit at opposite ends of the same contrast. Higher c_v does not automatically
mean LDA is more useful — our LSA topics are arguably easier to read, which is
what the human evaluation is there to settle.

**Why not just take the highest coherence?**
Because the curve never turns over. Taking the maximum would return whatever the
largest k on the grid happened to be — 60 in our case, and 100 if we had tried
100. It is a known failure of coherence measures on short text. The seed-to-seed
spread of 0.011 is the same size as the differences between neighbouring k, so
we would be choosing on noise.

**Why is R² so low? Doesn't that mean the finding is weak?**
It means the title is one factor among many, which is the honest picture. The
effect is not weak — the coefficients have t-statistics above 6, and the gap
between the best and worst topic is 3×. Low R² and a strong, reliable effect are
compatible: views are extremely variable for reasons no title could capture.

**Isn't this just measuring which channel posts what?**
That is exactly why the outcome is within-channel. Every video is compared to
the median video on the same channel, so a channel being popular cannot create
the effect. The instructional topics underperform *on their own channels*.

**Why did you remove the channel names from the titles?**
Because otherwise the first thing both models find is the channel. Every Bailey
Sarian title ends with "Mystery & Makeup, Bailey Sarian", so those words look
like a very strong theme while telling us nothing. We drop any word appearing in
30% or more of one channel's titles, and only for that channel.

**Why lemmatization and not stemming?**
Topic model output is read by a human. The Porter stemmer turns *mysterious*
into *mysteri* and *police* into *polic*, and a slide reading "murder, mysteri,
polic" cannot be labelled with confidence. Lemmatization returns real words. It
is slower, which does not matter at ten thousand short documents.

**Did you try removing people's names?**
Yes. Victim and offender names are a large part of the vocabulary and they
crowd the topics. We tested dropping them — 495 tokens identified as proper
names — and mean c_v did not improve consistently (k = 15 and 20 improved,
k = 25 got worse). Since there was no measurable gain, we kept them, and
`scripts/name_ablation.py` reproduces the test.

**What is the biggest limitation?**
Document length. Everything hard about this project comes from titles being
five tokens after cleaning. It is why coherence never peaks, why the topics
blur into each other, and why 28% of the corpus coming from one channel matters
so much.

---

## Where every number comes from

Run the pipeline and all of these regenerate. None are typed by hand.

| Number | Source | How |
|---|---|---|
| 10,107 raw rows | `reports/01_corpus.txt` §1 | rows in the Kaggle CSV |
| 10,026 kept (99.2%) | §2 | minus 1 missing view count, 11 zero-view, 69 duplicate channel+title |
| 21 channels, 2012–2023 | §3 | `nunique()` and min/max of PublishedDate |
| 9.0 words per title | §3 | mean of `Title.str.split().str.len()` |
| 4,119× channel gap | §4 | JCS median 15,943,809 ÷ 48 Hours median 3,871 |
| 28.3% one channel | §4 | Dr. Todd Grande, 2,838 ÷ 10,026 |
| 9,897 after preprocessing | `reports/02_preprocessing.txt` §6 | 129 titles left with < 2 tokens |
| vocabulary 11,840 → 10,223 | §6 | distinct tokens before and after cleaning |
| 5.4 tokens per title | §6 | mean tokens after cleaning |
| 41 bigrams | §5 | gensim Phrases, min_count 20, threshold 10 |
| 2,210 modelled words | `reports/03_model_selection.txt` §1 | after `filter_extremes(no_below=5, no_above=0.4)` |
| 9,785 documents | §1 | titles with at least one surviving vocabulary word |
| c_v still rising at k=60 | §2 | LDA 0.453 at k=60 vs 0.415 at k=20 |
| seed spread 0.011 | §2 | mean of the LDA c_v standard deviations |
| elbow k = 20 | §3 | max of (normalised c_v − normalised k) |
| LSA explained variance 9.7% | §4 | `svd.explained_variance_ratio_.sum()` at k=20 |
| top topic 1.36× | `reports/05_views_by_topic.txt` §2 | median view_ratio of the 185 titles assigned to LDA topic 7 |
| bottom topic 0.45× | §2 | same for the 622 titles in topic 12 |
| 3.0× spread | §2 | 1.36 ÷ 0.45 |
| R² = 0.064 | §3 | OLS of log2 view_ratio on 19 topic proportions + intercept |
| t up to 7.6 | §3 | coefficient ÷ its standard error |
| dark history 3.73× | §4 | median view_ratio of the 66 titles containing it |
| session 0.05× | §4 | median view_ratio of the 50 titles containing it |
| chance level 16.7% | `human_eval.py` | 1 intruder among 6 words |

**Bootstrap CIs** in §2 are 2,000 resamples of the titles in each topic, 2.5th
and 97.5th percentile of the median. Two topics whose intervals overlap are not
distinguishable — read the chart in bands, not as a ranking.

---

## Still to do before the presentation

1. **Human evaluation is built but not answered.** `data/human_eval/` has a
   40-question sheet (20 LSA topics + 20 LDA topics, shuffled, model hidden).
   Each group member copies `intrusion_sheet_BLANK.csv` to
   `responses_<name>.csv`, fills it in without opening `intrusion_key.csv`, then
   `python scripts/human_eval.py score` produces `reports/04_human_eval.txt`.
   That gives model precision per model — the number that decides whether LSA or
   LDA is the more interpretable of the two, and the rubric asks for it
   explicitly (Topic Evaluation, 25 points).
2. **Augmentation** is the preprocessing person's item. My position, ready if
   the teacher asks: the rubric asks for it "where class imbalance or limited
   data exists", and neither applies — 9,785 unlabelled documents, no classes to
   balance, and synonym replacement on a 9-word title changes the very
   co-occurrence counts LDA is estimating. Rewriting the data would make the
   topics worse, not more robust.
