# Every parameter we set, and why — question prep

Read this the night before. It is written as answers, not as documentation: each
row is a question a teacher can ask and a reply you can give out loud.

The honest general answer, if you are ever stuck:

> "We did not tune that one. We took the standard default and checked the result
> was not sensitive to it."

That is a good answer. Pretending everything was optimised is a bad answer,
because the next question is "show me the tuning".

---

# 1 — Building the corpus (`scripts/build_corpus.py`)

| Setting | Value | Why |
|---|---|---|
| Rows dropped: no view count | 1 | The outcome variable is missing, so the row cannot be used |
| Rows dropped: zero views | 11 | A video with 0 views tells us nothing about attracting views, and it would break the ratio |
| Rows dropped: duplicate channel + title | 69 | Same title on the same channel is a re-upload, not a second observation |
| Kept | 10,026 of 10,107 (99.2%) | |
| Columns used | Title, Views, Channel | The question is about titles |
| Columns dropped | Description, Likes, Comments | Descriptions contain sponsor links and contact details; likes and comments are outcomes, not causes |

**Why not use likes or comments as the outcome?**
Because they are downstream of views — you cannot like a video you did not watch.
Views is the cleanest measure of "did the title work".

**Why not deduplicate across channels?**
Two channels covering the same case is real competition, not duplication. That is
data, not noise.

## The one number people will ask about: `view_ratio`

```
view_ratio = this video's views ÷ the median video on the same channel
```

- **1.0** = a completely normal video for that channel
- **2.0** = double the channel's norm
- **0.5** = half

**Why divide by the channel median instead of using raw views?**
Because raw views mostly measure how big the channel is. The median JCS video has
**4,119×** the views of the median 48 Hours video. Any ranking built on raw views
would be a ranking of channels, not of titles.

**Why the median and not the mean?**
One viral video can be 56× the channel norm. A mean would let a single video drag
the whole channel's baseline up. The median ignores extremes.

**Why not subtract instead of divide?**
Views are multiplicative — a channel that normally gets 50,000 and one that gets
5,000,000 cannot be compared by subtraction. A ratio puts both on the same scale.

---

# 2 — Preprocessing (`scripts/preprocess.py`)

| Setting | Value | Why |
|---|---|---|
| Tokenizer | NLTK `word_tokenize` (Treebank) | `split()` leaves punctuation attached, so *killer?* and *killer* become two different words |
| Contractions | expanded **before** tokenizing | so *don't* becomes *do not*, not *do* + *n't* |
| Case | everything lowercased | *Murder* and *murder* are the same word |
| Stopwords | NLTK English (198) + our own list (47) = **245** | The NLTK list has no idea that *episode*, *part*, *full*, *sneak*, *peek* are format words in YouTube titles |
| Branding threshold | **30%** | A word in 30%+ of one channel's own titles is that channel's series tag |
| Word shortening | **lemmatization**, not stemming | Stemming gives *mysteri*, *polic* — unreadable on a slide |
| Minimum word length | **3 characters** | 1–2 letter leftovers carry no subject |
| Phrase detection | gensim `Phrases`, `min_count=20`, `threshold=10` | Joins pairs that appear together far more than chance: *serial_killer*, *mental_health*, *dark_history* |
| Minimum tokens to keep a title | **2** | A one-word document cannot contribute to a topic |
| Result | **9,897 titles**, 5.4 words each | |

### The branding rule — expect a question here

**What it does:** for each channel separately, any word appearing in **30% or
more** of that channel's own titles is deleted from that channel's titles.

**Example:** Bailey Sarian → *bailey*, *sarian*, *makeup* removed.
Dr. Todd Grande → *dr*, *todd*, *grande*, *analysis* removed.

**Why we had to:** every Bailey Sarian title ends with "| Mystery & Makeup |
Bailey Sarian". Without this rule, the strongest "topic" the models find is the
channel name, which answers nothing.

**Why 30% and not 50% or 10%?**
It has to be high enough that real subject words survive. No genuine subject word
appears in 30% of a channel's videos — even *murder*, our second most common word
overall, is in about 11% of all titles. A series tag is in 90%+. So 30% sits in
the empty gap between the two, and anything from about 25% to 60% would give the
same result.

**Why per-channel and not globally?**
So *crime* stays a real content word for the channels that do not put it in every
title. A global rule would delete it for everyone.

### Stemming vs lemmatization

If asked to justify, use the actual comparison from
`reports/02_preprocessing.txt` §4:

| word | stem | lemma |
|---|---|---|
| mysterious | *mysteri* | mysterious |
| police | *polic* | police |
| families | *famili* | family |
| disappearance | *disappear* | disappearance |
| studies | *studi* | study |

> "Our output is word lists a human has to read and label. 'murder, mysteri,
> polic' cannot be labelled with confidence. Lemmatization costs more time and we
> have ten thousand short documents, so time is not a constraint."

Note stemming would also merge *murderer* into *murder*, which loses the
difference between the crime and the person.

### Phrase detection

`min_count=20` — a pair must appear at least 20 times before we consider joining
it, otherwise a coincidence in three titles becomes a permanent term.
`threshold=10` — gensim's score for how much more often the pair appears together
than chance; 10 is the library's own default. It found **41** pairs.

**Why join them at all?** So LDA does not split one idea across two topics —
*serial* landing in one topic and *killer* in another.

---

# 3 — Vocabulary for the models (`scripts/topic_models.py`)

| Setting | Value | Why |
|---|---|---|
| `no_below` | **5** | A word in fewer than 5 titles is almost always one victim's name. A topic made of unique names describes nothing |
| `no_above` | **0.4** | A word in more than 40% of titles cannot distinguish one topic from another |
| Vocabulary after filtering | **2,210** words, from 10,223 | |
| Documents kept | **9,785** | 112 titles lost every word to the filter |
| TF-IDF | `sublinear_tf=True` | Uses 1+log(count) instead of raw count, so a word repeated three times in a title is not three times as important |

**Why 5 and not 2 or 10?**
At 2 the vocabulary fills with one-off names and the topics become lists of
strangers. At 10 we start losing real but uncommon subjects. 5 is the common
default and it left us 2,210 words, which is a sensible size for 9,785 short
documents.

**Why 40% at the top?**
Nothing actually hit that ceiling in our data — the most common word, *case*, is
in about 13% of titles. The filter is there as a safety net, and we report it
because it is part of the setup.

---

# 4 — The models

## LSA — `TruncatedSVD`

| Setting | Value | Why |
|---|---|---|
| Input | TF-IDF matrix, 9,785 × 2,210 | LSA needs weighted counts, not raw counts |
| `n_components` | 20 | The number of topics, chosen by the sweep |
| `algorithm` | `arpack` | Exact, and deterministic — the same input always gives the same answer. The faster `randomized` option would add randomness we do not need at this size |
| `random_state` | 42 | Reproducible |

**How we read an LSA topic:** by the **absolute** size of each word's loading,
because an LSA component can describe a contrast where the strongly negative
words matter as much as the positive ones.

**Explained variance at k=20: 9.7%.** Expect a question.

> "That sounds low, and for a dense dataset it would be. Our matrix is 99.7%
> empty — the average title uses 5 of 2,210 words — so no small number of
> components can reconstruct much of it. That is a property of short text, not a
> sign the model failed. It is also why we judge topics by coherence and by human
> reading rather than by variance explained."

## LDA — gensim `LdaModel`

| Setting | Value | Why |
|---|---|---|
| Input | bag of words counts | LDA is a count model; feeding it TF-IDF is not what it was derived for |
| `num_topics` | 20 | From the sweep |
| `passes` | **10** | How many times it reads the whole corpus. 1 is under-trained; beyond ~10 the topics stopped changing |
| `iterations` | **100** | How many times it refines a single document. gensim's own recommendation for short documents |
| `alpha` | **`auto`** | How mixed a title is allowed to be across topics. `auto` learns it from the data instead of us guessing |
| `eta` | **`auto`** | How concentrated a topic is over words. Also learned |
| `random_state` | 42, 7, 2024 | Three seeds — see below |
| `eval_every` | `None` | Turns off gensim's periodic perplexity logging, purely for speed |

**Why `alpha='auto'` rather than a fixed number?**
Because we had no principled value to pick, and letting the model estimate it is
the honest option. It is also the setting that lets some titles be almost purely
one topic while others are genuinely mixed.

**Why three seeds?**
LDA starts from a random guess, so one run can look better than another by luck.
Running every setting three times gave us a spread of about **0.011** in the
coherence score — and that turned out to be as large as the gaps we would have
been choosing between. Without this we would have been picking noise.

**Why bag of words for LDA but TF-IDF for LSA?**
LDA is a probability model over word counts; the maths assumes integers you could
actually have observed. LSA is a matrix factorisation with no such assumption, and
TF-IDF is the standard weighting for it. Both models see the same 2,210-word
vocabulary and the same 9,785 documents, so the comparison is still fair.

## NMF — the optional third model

| Setting | Value | Why |
|---|---|---|
| Input | the same TF-IDF matrix as LSA | So the three are comparable |
| `init` | `nndsvda` | Deterministic starting point based on the data's own SVD; avoids a random start |
| `max_iter` | 400 | The default 200 did not always converge |
| Elbow | **k = 15** | Slightly different from LSA and LDA, both at 20 |

**Why add a third model when the rubric only asks for two?**
The rubric says other algorithms are optional but recommended. NMF sits between
the other two: like LSA it factorises TF-IDF, but it forbids negative values, so
every topic is purely additive and easier to read. It is a cross-check, not a
replacement — our reported results are still LSA and LDA.

**Its coherence is higher than both (0.443 at k=15).** If asked why we did not
switch: NMF has no document-level probabilities, so "this title is 60% topic 3"
is not available, and our views analysis is built on exactly those mixtures.

---

# 5 — Choosing the number of topics

| Setting | Value | Why |
|---|---|---|
| Values of k tried | 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60 | Dense where the answer was likely, sparse above — enough to prove the curve never turns over |
| Words per topic for scoring | **10** | The standard in the coherence literature. Too few is unstable, too many drags in words nobody would read |
| Primary measure | **c_v** | The measure validated against human ratings (Röder et al. 2015) |
| Second measure | **u_mass** | A cross-check computed a different way |
| Reported but not used | perplexity | Known to keep improving as k grows even when the topics get worse (Chang et al. 2009) |
| Rule | the **elbow**, not the maximum | Because the curve never has a maximum here |

**How is the elbow computed?** Not by eye. Both axes are rescaled to 0–1, a
straight line is drawn from the first point to the last, and the point sitting
**furthest above** that line is the elbow. It is the standard knee method. Code:
the `knee()` function in `scripts/topic_models.py`.

**Why not just take the highest coherence?**
Because the highest was at 60, the largest value we tried, and it was still
rising. Taking the maximum would have returned 100 if we had tested 100. That is a
known failure of coherence measures on very short documents.

**Your two measures disagree — c_v likes many topics, u_mass likes 5. Which is
right?**

> "We report both rather than pick the flattering one. c_v is the measure with
> published evidence that it tracks human judgement, so it is our primary. u_mass
> is computed from raw document co-occurrence, and our documents are five words
> long, so it has very little to work with — it penalises any large number of
> topics almost automatically. The disagreement is real and it is in our
> limitations."

**Why 20 and not 15 or 25?**
The elbow. And practically: 20 topics is something a human can read through and
label, which is the point. At 60 nobody can interpret the output, so a higher
score would buy us nothing.

---

# 6 — Evaluation and statistics (`scripts/evaluate_topics.py`)

| Setting | Value | Why |
|---|---|---|
| Topic assignment | each title goes to its **highest-probability** topic | Simple and explainable. We also do the regression that uses all the mixtures, so nothing is lost |
| Bootstrap resamples | **2,000** | Enough for a stable 95% interval; 10,000 changes nothing visible |
| Confidence interval | 2.5th to 97.5th percentile | The standard 95% |
| Minimum topic size to report | **30 videos** | Below that the median jumps around too much to mean anything |
| Minimum titles for a word | **40** | Same reason, one level down |
| Regression outcome | **log2** of view_ratio | Views are multiplicative; a coefficient of +1 then means "double" |
| Reference topic | topic 19 | The 20 proportions add to 1, so one has to be left out or the maths breaks |

**Why the median everywhere instead of the mean?**
Because view counts are extremely skewed. Our best topic contains a video at 56×
the channel norm. Means would be reporting outliers; medians report the typical
video.

## The significance tests

| Test | Result | What it means |
|---|---|---|
| Kruskal-Wallis across the 20 topics | **H(19) = 418.4, p = 6.5 × 10⁻⁷⁷** | The topics genuinely differ in views. Not sampling noise |
| Mann-Whitney, all 190 pairs, Holm-corrected | **73 of 190 pairs differ** at p < .05 | Many pairs are distinguishable, many are not |
| Best vs worst (topic 7 vs 12) | **adjusted p = 1.8 × 10⁻¹⁷** | Our headline gap is real |

**Why Kruskal-Wallis and not ANOVA?**
ANOVA assumes the values are roughly normally distributed. Ours are strongly
right-skewed — the median is 1.0 but the 90th percentile is 5.2×. Kruskal-Wallis
is the non-parametric equivalent: it works on ranks, so the shape does not matter.

**What is Holm-Bonferroni and why?**
We ran 190 comparisons. At p < .05, about 9 of them would look significant by
chance alone. Holm-Bonferroni tightens the threshold to account for that. It is
less conservative than plain Bonferroni while still controlling the error.

**Why is R² only 6.4% when p is 10⁻⁷⁷?**
These answer different questions and this is worth having ready.

> "The p-value asks *is there an effect*. The R² asks *how big*. With 9,785
> videos we can detect a small effect with enormous confidence. So: the effect is
> certainly real, and it is also modest. Both statements are true and we say
> both."

---

# 7 — Human evaluation (`scripts/human_eval.py`)

| Setting | Value | Why |
|---|---|---|
| Real words shown | **5** | From Chang et al. (2009), the paper that invented the task |
| Intruders | 1 | Same |
| Chance level | **1/6 = 16.7%** | Six words, one is the intruder |
| Questions | **40** | All 20 LSA topics + all 20 LDA topics |
| Question ids | `Q001`… | Opaque on purpose, so the annotator cannot tell which model a question came from |
| Intruder choice | a word ranked high in a *different* topic and absent from this one | An intruder that is merely rare would make the task about word frequency instead of topic quality |
| `random_state` | 11 | So the sheet can be regenerated identically |

**Why bother when you already have coherence?**
Because coherence is a proxy for human judgement, not human judgement. The rubric
asks for human evaluation specifically, and it is the only way to compare LSA and
LDA on the thing that actually matters — can a person understand the topic.

**Why hide which model each question came from?**
So nobody unconsciously marks their own model's questions more generously.

---

# 8 — Things we tried that did not work (say these; they are strengths)

**Removing people's names.** Victim and offender names fill the word lists. We
detected 495 of them automatically — a word counts as a name if it is capitalised
mid-title in 90%+ of its appearances and is not an ordinary English noun — and
re-ran everything. Coherence improved at two settings and got worse at two
others, mean change **+0.010**. No reliable gain, so we kept the names.
Reproducible: `scripts/name_ablation.py`, `reports/06_name_ablation.txt`.

**Taking the best coherence score.** It returned the largest k on the grid every
time. Replaced with the elbow.

**Using raw view counts.** It ranked channels, not titles. Replaced with
view_ratio.

---

# 9 — The four numbers to have memorised

| | |
|---|---|
| **9,785** | titles that reached the models |
| **20** | topics, chosen at the elbow, agreed by LSA and LDA |
| **1.36× vs 0.45×** | best topic vs worst topic, a 3× gap |
| **6.4%** | how much of the view difference the topic explains |

If you remember nothing else, those four carry the whole part.

---

# 10 — If you genuinely do not know

Say so, then give the nearest thing you do know:

> "I did not set that one myself, it is the library default — but I can tell you
> what it controls and why it did not matter for our result."

Or:

> "That is in the repo, section 3 of the model selection report. I can show you
> after."

Both are far better than guessing a number. A wrong figure said confidently is
the only answer that actually costs marks.
