# Presentation plan — 7 people, 7 parts

The rubric has exactly **seven criteria**, so each person owns one. The grader
can then tick every box in order, and nobody has to explain someone else's work.

| # | Person | Rubric criterion | Points | Slides | Time |
|---|---|---|---|---|---|
| 1 | | Problem Definition & Motivation | 5 | 1 | 1 min |
| 2 | | Corpus Collection & Legal / Ethical | 10 | 2 | 2 min |
| 3 | | Text Preprocessing & Augmentation | 25 | 2 | 3 min |
| 4 | **Seth** | Model Design & Number of Topics | 25 | 2 | 3 min |
| 5 | | Topic Evaluation & Interpretation | 25 | 2 | 3 min |
| 6 | | Limitations & Future Studies | 5 | 1 | 1.5 min |
| 7 | | Conclusion + deck owner (Presentation Clarity) | 5 | 2 | 1.5 min |

**Total: 12 slides, about 15 minutes**, leaving room for questions.

Two parts are small on paper (1 and 6, worth 5 points each) and two are heavy
(3 and 5, worth 25 each). If someone in the group is nervous, give them part 1 or
6. If someone is confident, give them 3 or 5.

**Everything below is real.** Every number has a file next to it. Nobody should
invent a figure the night before.

---

# Part 1 — Problem Definition & Motivation (5 points)

**Rubric wants:** research questions clearly stated.

### Slide: "What we are asking"

- Main question: **what kind of YouTube video title attracts the most views?**
- Sub-question 1: what subjects do true crime channels actually make videos about?
- Sub-question 2: do some subjects reliably beat others *on the same channel*?
- Why it matters: creators choose a title in seconds; nobody knows what a title is
  worth
- Why topic modelling: 9,785 titles is far too many to read, and we do not have
  labels — so we need a method that finds the subjects by itself

### Say (about 60 seconds)

> Every YouTube creator writes a title and hopes it works, but nobody really knows
> which kind of title earns views. That is our question: what kind of title
> attracts the most views?
>
> We picked true crime because it is one topic area with many channels competing,
> so the titles are comparable. We have almost ten thousand of them, which is far
> too many to read by hand, and none of them come with a label saying what the
> video is about. That is exactly the situation topic modelling is built for — it
> reads all the titles and finds the recurring subjects on its own. Then we can
> check which subjects get watched.
>
> One thing we decided early and it shaped everything: we compare each video to
> the normal video on its own channel, not to the whole dataset. Otherwise we
> would just be measuring which channel is famous.

### Do not say

Do not promise we predict views. We explain a small part of them, and part 5
gives the honest number.

---

# Part 2 — Corpus Collection & Legal / Ethical (10 points)

**Rubric wants:** legitimate source, copyright / terms of service / personal data
addressed, corpus described with size, source and class balance.
**Source file:** `reports/01_corpus.txt`

### Slide A: "Our data"

- **True Crime Channel Statistics**, Kaggle, uploaded by mhapich
- **10,107 videos**, **21 channels**, **2012 to 2023**
- Collected through the **official YouTube Data API v3** — not scraped
- Licence **CC0 1.0** — public domain, reuse and redistribution allowed
- We use two columns only: **the title** and **the view count**

### Slide B: "Cleaning and balance"

- Removed: 1 with no view count, 11 with zero views, 69 duplicate re-uploads
- **10,026 kept — 99.2%**
- **Not balanced:** Dr. Todd Grande alone is **28.3%** of all videos
- Channel sizes are wildly different — the biggest channel's normal video gets
  **4,119×** the views of the smallest channel's normal video
- That imbalance is exactly why we measure views **relative to each channel**

### Say (about 2 minutes)

> Our data is a public Kaggle dataset called True Crime Channel Statistics —
> just over ten thousand videos from twenty-one channels, covering 2012 to 2023.
>
> On the legal side, three things. It was collected through YouTube's official
> API, so no scraping and no terms of service problem. It is released under CC0,
> which is public domain, so we are allowed to use and share it. And there is no
> personal data — titles, view counts and channel names are all public metadata
> anyone can see on the channel page. The channel names belong to creators acting
> professionally, so there is nothing to anonymise. We did drop the video
> descriptions, because those contain sponsor links and creator contact details
> and we do not need them.
>
> One content note: this is true crime, so the titles name real victims. We only
> ever report at the group level and make no claim about any individual case.
>
> On cleaning, we removed eighty-one videos out of ten thousand — one with no
> view count, eleven with zero views, and sixty-nine duplicate re-uploads. That
> leaves 10,026, ninety-nine per cent of what we downloaded.
>
> The important thing about balance: this corpus is not balanced at all. One
> channel, Dr. Todd Grande, is twenty-eight per cent of it. And the channels are
> completely different sizes — the biggest channel's normal video gets four
> thousand times the views of the smallest one's. So we never compare raw view
> counts. Every video is compared to the normal video on its own channel.

### Numbers, and where they live

| Number | File |
|---|---|
| 10,107 → 10,026 (99.2%) | `reports/01_corpus.txt` §2 |
| 21 channels, 2012–2023, 9 words per title | §3 |
| 28.3% one channel, 4,119× gap | §4 |
| licence, API, ethics wording | §1 and §5 |

---

# Part 3 — Text Preprocessing & Augmentation (25 points)

**Rubric wants:** tokenization explained, stopword removal, stemming or
lemmatization, normalization, and augmentation used *and justified* where class
imbalance or limited data exists.
**Source file:** `reports/02_preprocessing.txt`

### Slide A: "Cleaning a title, step by step"

Show the real example from the report — this one slide can carry the whole part:

```
1 raw          Was it Sacrifice or Serial Killer?? Leonarda Cianciulli
               | Mystery & Makeup | Bailey Sarian
2 normalized   was it sacrifice or serial killer leonarda cianciulli
               mystery makeup bailey sarian
3 tokenized    was | it | sacrifice | or | serial | killer | leonarda
               | cianciulli | mystery | makeup | bailey | sarian
4 stopwords +  sacrifice | serial | killer | leonarda | cianciulli
  branding     | mystery
  removed
5 lemmatized   (no change here)
6 phrases      sacrifice | serial_killer | leonarda | cianciulli | mystery
```

### Slide B: "The choices we made and why"

- **Tokenizer: NLTK word_tokenize**, not `split()` — `split()` leaves punctuation
  stuck on, so *killer?* and *killer* become two different words
- **Contractions expanded first** — *don't* → *do not*
- **Branding removed** — a word in **30%+ of one channel's own titles** is that
  channel's series tag, not a subject. Removing "bailey, sarian, makeup" for
  Bailey Sarian only
- **Lemmatization, not stemming** — stemming gives *mysteri*, *polic*; you cannot
  put that on a slide
- **41 two-word phrases** detected automatically: *serial_killer*, *mental_health*,
  *dark_history*
- Result: **9,897 titles**, average **5.4 words** each
- **No augmentation, and here is why** — see below

### Say (about 3 minutes)

> Our documents are titles, which makes cleaning unusually important, because
> after cleaning there are only about five words left to work with.
>
> Walk through the example. We start with a raw title. First we expand
> contractions, so "don't" becomes "do not", then lowercase everything and strip
> punctuation and emoji. Then we tokenize — we use NLTK's word tokenizer rather
> than just splitting on spaces, because splitting leaves the question mark
> attached and then "killer" with a question mark and "killer" without one count
> as two different words.
>
> Then two removals. Ordinary stopwords — "the", "was", "it" — plus a list of
> format words like "part", "episode", "full" that appear everywhere and mean
> nothing.
>
> And then the one we are most pleased with: branding removal. Every Bailey Sarian
> title ends with "Mystery and Makeup, Bailey Sarian". If we leave that in, the
> strongest subject the model finds is just the channel name. So any word that
> appears in thirty per cent or more of a single channel's own titles gets
> removed — but only for that channel, so "crime" stays a real word for channels
> that do not put it in every title.
>
> Then lemmatization, which reduces words to their dictionary form so "killers"
> and "killer" match. We chose lemmatization over stemming deliberately: stemming
> turns "mysterious" into "mysteri" and "police" into "polic", and our whole
> output is word lists that a human has to read.
>
> Last, we let the computer find pairs of words that almost always appear
> together and join them — it found forty-one, including serial killer, mental
> health and dark history. That stops one idea being split across two subjects.
>
> On augmentation: we did not use it, and that is a decision, not an oversight.
> The rubric asks for it where there is class imbalance or too little data. We
> have nearly ten thousand documents, no labels and no classes, so there is
> nothing to balance. And augmentation works by swapping or inserting words —
> but word co-occurrence is *exactly* what LDA counts. Rewriting our titles would
> change the answer rather than make it more reliable.

### Numbers, and where they live

| Number | File |
|---|---|
| the 6-step worked example | `reports/02_preprocessing.txt` §3 |
| stemming vs lemmatization comparison | §4 |
| 41 phrases found | §5 |
| 9,897 titles, 5.4 words each, vocabulary 11,840 → 10,223 | §6 |
| branding words per channel | §2 |

### If challenged on augmentation

> "We can show it if you want — the run is one flag. But on a nine-word title,
> replacing two words with synonyms changes almost a quarter of the document, and
> those word pairings are the only thing the model has to learn from."

---

# Part 4 — Model Design & Number of Topics (25 points) — **Seth**

**Rubric wants:** LSA, LDA, and how the number of topics was selected.

Full content, script, chart explanations and Q&A are in
**`docs/SLIDES_LSA_LDA.md`** — slides 1 and 2 there are parts 4 and 5 here.
Use Part A of that file for the slide text and Part B for the script.

**Two slides:** how the two models work, and the k = 20 decision with
`figures/fig1_coherence_vs_k.png`.

**Headline for the group:** the standard "pick the number of topics with the best
coherence score" method **failed** — the score never stopped rising, still going
up at 60 topics — so we took the bend in the curve instead. Both LSA and LDA
picked 20 independently.

---

# Part 5 — Topic Evaluation & Interpretation (25 points)

**Rubric wants:** human evaluation, topic coherence (c_v or u_mass), and a
comparison of the number of topics.
**Source files:** `reports/03_model_selection.txt`, `reports/04_human_eval.txt`
(after the group fills the sheet in), `reports/05_views_by_topic.txt`

This part has **three things to cover**, and the rubric names all three.

### Slide A: "Are the subjects any good?"

- **c_v coherence** — the automatic score. LDA **0.415** at 20 subjects,
  LSA **0.351**
- **u_mass** — a second automatic score, as a cross-check. It disagrees: it
  prefers 5 subjects. We say so rather than hide it
- **Human evaluation** — 40 word-intrusion questions, model hidden.
  *[fill in the result once the group has done the sheet]*
- Chance level is **16.7%** — one intruder among six words. Scoring above that
  means the subjects are real to a human

### Slide B: "Which subjects get the views"

- Use `figures/fig2_views_by_topic.png`
- Best **1.36×** normal — a real case, a boyfriend, a shooting, a verdict
- Worst **0.45×** normal — tutorials: SPSS, counselling practice, makeup
- **3× gap**, and it holds *inside* channels
- The subject explains **6%** of the difference in views
- No-model check: *dark history* **3.73×**, *session* **0.05×**

### Say (about 3 minutes)

> Evaluating topic models is genuinely hard, because there is no right answer to
> compare against. So we did it three ways.
>
> First, the automatic score, coherence. It asks whether the words inside a group
> actually appear near each other in real titles. LDA scored 0.415 and LSA 0.351
> at twenty subjects. We also ran a second measure, u_mass, and it disagrees — it
> prefers five subjects. We are reporting that rather than hiding it, because two
> measures disagreeing is itself a finding about short text.
>
> Second, human evaluation, which is the part a score cannot replace. We built
> forty questions. Each shows six words: five from one subject and one intruder
> from a different subject. If a subject is really one idea, a person spots the
> odd word immediately. Guessing would score seventeen per cent.
> *[state the result]*
>
> Third — and this is what answers our research question — we joined the subjects
> back to the view counts.
>
> [Explain the chart: middle line is normal for that channel, blue beats it, red
> loses to it.] The top subject gets 1.36 times normal views: a real case told as
> a story with a courtroom angle. The bottom gets 0.45: tutorials. SPSS lessons,
> counselling role-plays, makeup. Several of these channels post teaching videos
> beside their crime videos, and the teaching videos get less than half the views
> on their own channel. About a three times gap.
>
> One honest number. A statistical test using all twenty subjects at once says
> the subject explains about six per cent of the difference in views. Six per cent
> is small. The effect is real and it is not chance, but ninety-four per cent is
> thumbnail, timing and how famous the case already was.
>
> And a check with no model at all: titles containing "dark history" get 3.7 times
> normal, titles containing "session" get one twentieth. Same split, no LDA
> involved. So this is in the data, not in our modelling choices.

### The one trap on this slide

The top two subjects, 1.36 and 1.32, have **overlapping confidence ranges**.
They are joint top, not first and second. Say "the top group", never "the winner".

---

# Part 6 — Limitations & Future Studies (5 points)

**Rubric wants:** report clearly on the limitations of the project.

This part is short but it is where a good group separates itself. Every item
below is a real limitation of *our* work, not a generic one.

### Slide: "What our study cannot tell you"

- **Titles are too short.** Nine words, five after cleaning. This causes almost
  every other problem
- **The coherence score never peaked** — we had to use the bend instead, which is
  a judgement call, defensible but not automatic
- **Our two scores disagree** — c_v says many subjects, u_mass says five
- **One channel is 28% of the data**, so the subjects lean toward what that
  channel posts
- **We measure correlation, not cause.** A subject getting more views does not
  prove the title caused it — famous cases attract both good titles and viewers
- **Views keep growing.** An old video has had years to collect views; we do not
  know each video's age effect
- **We ignored the thumbnail**, which is probably more powerful than the title
  and is not in the dataset

### Future work

- Add the **description** text to give the model longer documents
- Try a model built for short text (**BERTopic**, or LDA with sentence embeddings)
- Control for **video age** and subscriber count at upload time
- Test one channel over time: did *changing* title style change its views?

### Say (about 90 seconds)

> Our biggest limitation is length, and nearly everything else follows from it.
> A title is nine words, five after cleaning, so there is very little for the
> model to learn from inside one document. That is why the coherence score never
> peaked, and why we had to pick the number of subjects by the bend in the curve
> instead of the maximum. It is defensible but it is a judgement call, and our two
> scoring measures do not agree with each other.
>
> Second, our data is unbalanced. One channel is twenty-eight per cent of it, so
> the subjects lean toward what that channel posts.
>
> Third, and most important: this is correlation, not cause. A subject getting
> more views does not prove the title caused it. Famous cases attract both careful
> titles and large audiences. And we could not account for video age, or for the
> thumbnail, which is probably stronger than the title and is not in the dataset
> at all.
>
> For future work, the obvious step is longer documents — add the video
> description — or use a method designed for short text such as BERTopic. The
> strongest test would be following one channel over time and seeing whether
> changing its title style changed its views.

---

# Part 7 — Conclusion + deck owner (5 points)

**Rubric wants:** report and slides well organised, clearly written, properly
cited, professionally presented. That is earned by the whole deck, so this person
owns the deck as well as speaking.

### Job before the presentation

- Build the actual slide file and keep one consistent template — same font, same
  colours, same title position on every slide
- Collect each person's bullets from this document. **Bullets only** — nobody
  pastes paragraphs onto a slide
- Build the references slide (below)
- Run one full timed rehearsal and cut whatever runs long
- Check every chart is readable from the back of the room

### Slide A: "What we found"

- **9,785 titles → 20 subjects**, using LSA and LDA
- **Answer: a title promising a real, unresolved crime story beats the channel
  norm by 1.36×; a title promising a lesson gets 0.45×**
- Holds across both models, inside individual channels, and with no model at all
- Honest limit: the subject explains **6%** of views — real, but one factor among
  many

### Slide B: "References"

- Dataset — mhapich, *True Crime Channel Statistics*, Kaggle, CC0 1.0
- Blei, Ng & Jordan (2003), *Latent Dirichlet Allocation* — the LDA method
- Deerwester et al. (1990), *Indexing by Latent Semantic Analysis* — LSA
- Röder, Both & Hinneburg (2015) — the c_v coherence measure
- Chang et al. (2009), *Reading Tea Leaves* — the word intrusion human evaluation
- Tools: gensim, scikit-learn, NLTK, pandas, matplotlib
- Code: `github.com/Catejsj/topic-modelling`

### Say (about 90 seconds)

> To close. We took nine thousand seven hundred true crime video titles, cleaned
> them, and used the two methods from class — LSA and LDA — to group them into
> twenty subjects. We chose twenty by the bend in the coherence curve, because on
> documents this short the score never peaks.
>
> Our answer: a title that promises a real, unresolved crime story gets about
> 1.36 times the views of the normal video on its own channel. A title that
> promises a lesson gets 0.45. That is a three times gap, and it holds across both
> models, inside individual channels, and even when we throw the models away and
> look at single words.
>
> The honest limit is that the subject of the title explains about six per cent of
> why a video gets views. The title matters. It is not the main thing.
>
> Everything — the code, the reports and every number in this presentation — is in
> our repository. Thank you, we are happy to take questions.

---

# Slide order for the deck builder

| Slide | Owner | Content |
|---|---|---|
| 1 | 7 | Title, group members, the research question in one line |
| 2 | 1 | What we are asking, and why topic modelling |
| 3 | 2 | Our data — source, size, licence, ethics |
| 4 | 2 | Cleaning and balance |
| 5 | 3 | Cleaning a title, step by step |
| 6 | 3 | Preprocessing choices, and why no augmentation |
| 7 | 4 | LSA and LDA — how they differ |
| 8 | 4 | Choosing 20 subjects + `fig1_coherence_vs_k.png` |
| 9 | 5 | Are the subjects any good? Coherence + human evaluation |
| 10 | 5 | Which subjects get the views + `fig2_views_by_topic.png` |
| 11 | 6 | Limitations and future work |
| 12 | 7 | Conclusion, then references |

**Backup slides after slide 12**, only shown if asked:

- `fig3_words_by_views.png` — the no-model check
- The full 20-subject word list from `reports/03_model_selection.txt` §4
- The stemming vs lemmatization comparison from `reports/02_preprocessing.txt` §4

---

# Rules for everyone

1. **Bullets on slides, sentences in your mouth.** If a slide has a paragraph,
   the audience reads instead of listening.
2. **Never quote a number you cannot source.** Every number in this document has
   a file next to it. If it is not here, do not say it.
3. **Say the weak parts first.** Six per cent, the unbalanced corpus, the two
   disagreeing scores — saying them yourself is worth more than being caught.
4. **Hand over by name.** "Now [name] will take you through the preprocessing."
   It looks rehearsed because it is.
5. **One person answers each question** — whoever owns that part. Nobody talks
   over anyone.
6. **The human evaluation must be done before the presentation.** It is part of a
   25-point criterion. Instructions are in `data/human_eval/INSTRUCTIONS.md`;
   ten minutes each.
