# Slides — LSA and LDA part (Seth)

Two slides.

- **Part A** is for whoever builds the slides. Copy the text in the boxes onto
  the slide. Nothing else goes on the slide.
- **Part B** is what I say out loud. Nobody needs to read it except me.
- **Part C** explains every picture: what the numbers mean and how we got them.
- **Part D** explains the technical words in case someone asks.
- **Parts E to G** are Q&A prep, where each number came from, and what is still
  unfinished.

Our group question: **what kind of video title gets the most views?**

My job: take 9,785 video titles, group them into a small number of subjects, and
then check which subjects get the most views.

---

# PART A — for the slide maker

## Slide 1

**Title of the slide:**

> How we grouped 9,785 video titles

**Put these bullets on the slide (short, do not add more):**

- Two methods from class: **LSA** and **LDA**
- Both read the same 9,785 titles and the same 2,210 words
- We had to choose how many groups to make: we tried 5, 10, 20, 30, 40, 60
- The score that measures group quality **never stopped going up** — so "pick the
  best score" does not work here
- We picked where the score **stops improving much**: **20 groups**
- Both methods chose 20 on their own

**Picture:** `figures/fig1_coherence_vs_k.png`
Full width, under the bullets. It is a line chart, blue line is LDA, orange line
is LSA, dashed vertical line marks 20.

**Nothing else on this slide.** No extra images, no formulas.

---

## Slide 2

**Title of the slide:**

> Which title subjects actually get the views

**Put these bullets on the slide:**

- We compare each video to **the normal video on its own channel** (a big channel
  is not "better titles", it is just a big channel)
- Best subject: **1.36×** normal — *a real case, a boyfriend, a shooting, a
  verdict*
- Worst subject: **0.45×** normal — *tutorials: SPSS, counselling practice,
  makeup*
- That is a **3× gap** between the best and the worst subject
- The subject of the title explains about **6%** of why a video gets views. The
  rest is thumbnail, timing, and how famous the case already is
- **One-line answer: promise a real unsolved crime story, not a lesson**

**Picture:** `figures/fig2_views_by_topic.png`
Right half of the slide, bullets on the left half. Blue bars are above normal,
red bars are below normal.



---

# PART B — what I say out loud

Roughly 90 seconds each. Say it in my own words, this is just the shape.

## Slide 1 script

> My part was grouping the titles by subject, using the two methods we learned in
> class.
>
> The first one is LSA. Think of a giant table: every video title is a row, every
> word is a column. Nine thousand rows, two thousand columns. That table is far
> too big to look at, so LSA squeezes it down to twenty columns and each of those
> twenty is a subject. It is fast and it gives the same answer every time.
>
> The second one is LDA. It works differently. It assumes each title is a
> *mixture* of subjects, not just one. So a title can be sixty per cent "missing
> person" and forty per cent "police investigation", the way a real title often
> is. That is closer to how titles actually work, and the numbers come out as
> percentages, which is easier to explain.
>
> The hard part was deciding how many subjects to make. There is a standard score
> for this that tells you how well the words in a group belong together, and the
> normal method is to try many numbers and keep the one with the best score.
>
> That did not work for us. Our documents are titles — nine words long, five
> after we clean them. There is almost nothing in one title for the computer to
> learn from. So the score just kept going up the more groups we added. We went
> all the way to sixty groups and it was still going up. If we had tried a
> hundred, it would have said a hundred.
>
> We also ran LDA three separate times for every number, because it starts from a
> random guess. The difference between those runs was about the same size as the
> difference between twenty groups and thirty groups. So most of that rise was
> just randomness, not real improvement.
>
> So instead of taking the best score we took the point where the score stops
> improving much — the bend in the line, which you can see on the chart. Both
> methods pointed at twenty. Twenty is also a number a human can actually sit and
> read through, which matters, because the whole point is that a person
> understands the groups.

## Slide 2 script

> Now the part everyone cares about: which titles get the views.
>
> One warning first. If we just ranked videos by view count we would only be
> measuring which channel is big. The typical JCS video gets four thousand times
> the views of the typical 48 Hours video, and that has nothing to do with the
> title. So we compare every video to the normal video *on its own channel*. One
> point zero means normal for that channel.
>
> At the top, at 1.36 times normal, is a subject about a real case told as a
> story — a boyfriend, a shooting, a verdict. Just under it, at 1.32, is mystery,
> serial killer, unsolved.
>
> At the bottom, at 0.45, is something completely different: tutorials. SPSS
> lessons, counselling practice videos, makeup tutorials. Several of these
> channels post teaching videos next to their crime videos, and those teaching
> videos get less than half the normal views *on their own channel*. So the gap
> from top to bottom is about three times.
>
> We checked this two more ways so it is not just one model's opinion. First, a
> statistical test using all twenty subjects at once. It says the subject of the
> title explains about six per cent of why a video gets views. I want to be
> honest about that number: six per cent is small. It is a real effect and it is
> not chance, but ninety-four per cent is thumbnail, timing, and how famous the
> case already was.
>
> Second, we threw the models away completely and just looked at single words.
> Titles containing "dark history" get 3.7 times normal views. Titles containing
> "session" get one twentieth. The same split, no model involved. And LSA and LDA
> put the subjects in the same order as each other.
>
> So the answer to our research question is: **a title that promises a real
> unsolved crime story does well; a title that promises a lesson does badly, even
> on the same channel.**

---

# PART C — the three pictures, explained properly

This is the part to read before presenting. If someone points at a chart and
asks "what is that number", the answer is here.

---

## Picture 1 — `fig1_coherence_vs_k.png` (goes on slide 1)

### What you are looking at

A line chart with two lines.

- **Bottom axis** — how many subjects we asked for. We tried 5, 8, 10, 12, 15,
  20, 25, 30, 40, 50, 60. Each dot is one of those attempts.
- **Left axis** — the coherence score, from about 0.28 to 0.45. This is a
  quality score for the word groups. Higher = the words in each group belong
  together better.
- **Blue line with circles** — LDA. **Orange line with squares** — LSA.
- **Pale blue shading** around the blue line — LDA gives a slightly different
  answer each time it runs, so we ran it **three times** at every setting. The
  shading is how much those three runs disagreed.
- **Dashed vertical line at 20** — the number we chose.

### What one dot means

Take the blue dot above 20. To make that single dot we:

1. asked LDA for exactly 20 subjects,
2. took the **top 10 words** of each of the 20 subjects,
3. for every pair of those words, checked how often they appear near each other
   in the real titles,
4. averaged it all into one number: **0.415**.

That is it. One dot = one whole model, boiled down to one quality number.

### Where the numbers come from

Everything on this chart is read from `data/processed/k_sweep.csv`, which is
written by `scripts/topic_models.py`. The full table is in
`reports/03_model_selection.txt` §2. The key rows:

| subjects (k) | LDA score | LSA score |
|---|---|---|
| 5 | 0.350 | 0.281 |
| 10 | 0.375 | 0.310 |
| **20** | **0.415** | **0.351** |
| 30 | 0.420 | 0.350 |
| 40 | 0.438 | 0.346 |
| 60 | 0.453 | 0.362 |

Coherence is calculated by gensim's `CoherenceModel`, the standard tool — we did
not invent the measure.

### Why this picture is on the slide

Because it shows the decision we had to make and why the textbook method failed.

Normally you make this chart, find the highest point, and that is your answer.
Look at our chart: **there is no highest point.** The line is still climbing at
60. If we had tested 100 settings the "best" would have been 100. That happens
with very short documents like titles, and it is worth showing rather than
hiding.

The pale shading is the second half of the argument. The three runs at each
setting differ by about **0.011**. Look at the gap between 20 (0.415) and 30
(0.420) — that is 0.005, *smaller than the disagreement between runs of the same
setting*. So that "improvement" is not real.

So instead of the highest point we took **the bend** — where the line stops
climbing steeply and flattens. That is at 20 for both lines.

### How the bend is worked out (in case the teacher asks)

Not by eye. We squash both axes onto a 0-to-1 scale, draw a straight line from
the first dot to the last dot, and find the dot that sits **furthest above** that
straight line. That dot is the bend. It is a standard method (the "elbow" or
"knee" method). The code is the `knee()` function in `scripts/topic_models.py`.

### If someone points at it

> "The blue line is the probability model, orange is the maths one. Both keep
> rising, which is why we could not just take the highest score. The shaded band
> is how much the model disagrees with itself between runs — it is as wide as the
> gaps we would be choosing between. So we took the bend at twenty."

---

## Picture 2 — `fig2_views_by_topic.png` (goes on slide 2, the important one)

### What you are looking at

20 horizontal bars, one for each subject LDA found.

- **Each row** is one subject. The label shows its number and its top 3 words,
  e.g. `10  mystery, serial_killer, murder`.
- **The line down the middle is 1.0** — a completely normal video for its
  channel.
- **Blue bars point right** = that subject gets **more** views than normal.
- **Red bars point left** = that subject gets **fewer** views than normal.
- **The number at the end of each bar** (1.36×, 0.45×) is how many times the
  normal views that subject gets.
- Bars are sorted best at the top, worst at the bottom.

### What one bar means — a real worked example

Take the top bar, subject 7, at **1.36×**.

LDA assigned **185 titles** to subject 7. For every one of those titles we
already know its view ratio. We sorted those 185 ratios and took the middle one.
The middle one is this video:

> **"How a Game Show Solved a Double Homicide"** — Joshua Miles, 111,924 views

Joshua Miles' normal video gets **82,150** views. So:

```
111,924 ÷ 82,150 = 1.36
```

That is the whole calculation. The bar is 1.36 because the middle video of that
subject got 1.36 times its channel's normal views.

We use the **middle value (median)**, not the average, on purpose. One viral
video can be 56× normal, and an average would let that single video drag the
whole subject up. The middle value ignores freaks like that.

### How a title ends up in a subject

LDA does not give a title one subject — it gives percentages, like "45% subject
10, 30% subject 16, 25% spread over the rest". For this chart we put each title
in **whichever subject has the highest percentage**. Simple, and easy to explain.

(This does throw information away, which is why we also did the statistical test
that uses all the percentages at once. That is the 6% number.)

### Where the numbers come from

`scripts/evaluate_topics.py` writes `data/processed/topic_views.csv`, and the
chart is drawn from that file by `scripts/make_figures.py`. The same table with
extra columns is in `reports/05_views_by_topic.txt` §2. Selected rows:

| subject | titles in it | view ratio | top words |
|---|---|---|---|
| 7 | 185 | **1.36×** | say, boyfriend, real, shoot, guilty |
| 10 | 465 | 1.32× | mystery, serial_killer, murder, unsolved |
| 8 | 304 | 1.29× | victim, case, chris, boy, robert |
| … | | ~1.0× | the middle of the chart |
| 11 | 354 | 0.59× | face, interview, tutorial, history |
| 3 | 302 | 0.46× | look, find, body, palette |
| 12 | 622 | **0.45×** | short, test, love, parent, help |

**3× gap** = 1.36 ÷ 0.45.

### Why this picture is on the slide

It *is* our answer. Everything else is method; this is the finding. And the shape
of it makes the point instantly — you can see the blue block at the top is all
crime storytelling and the red block at the bottom is all teaching content,
without reading a single number.

### The one thing to be careful about

In `reports/05_views_by_topic.txt` each row also has square brackets, like
`1.36 [1.22, 1.63]`. That is how sure we are about that bar. We got it by
re-drawing the 185 titles at random 2,000 times and seeing how much the middle
value moved.

**If two subjects' brackets overlap, we cannot say one really beats the other.**
Subject 7 (1.36) and subject 10 (1.32) overlap — they are joint top, not first
and second. Read the chart as *top group / middle / bottom group*, and never say
"subject 7 is the winner".

### If someone points at it

> "The line in the middle is a normal video for that channel. Blue beats normal,
> red loses to normal. Each bar is the middle video of that subject, so one viral
> hit cannot inflate it. The top bar means: the middle video about a real case
> with a courtroom angle got 36% more views than that channel's normal video."

---

## Picture 3 — `fig3_words_by_views.png` (backup slide only)

### What you are looking at

The same idea as picture 2, but with **no model at all** — just single words.

- **Each row is one word** from the titles. The 10 best at the top, the 10 worst
  at the bottom.
- Same middle line at 1× = normal for the channel.
- **`(n=62)`** means that word appears in 62 different titles.
- The bottom axis is squashed (a log scale) because the range is huge — from
  0.05× to 3.7×. Without squashing, everything on the left would be an invisible
  sliver.

### What one bar means

Take **`dark history`**, at the top, 3.73×.

We found every title containing "dark history" — **62** of them. We took the view
ratio of each, sorted them, and took the middle one: **3.73**. So the middle
video with "dark history" in its title got 3.7 times its channel's normal views.

At the other end, **`session`** appears in 49 titles and its middle video got
**0.05×** — one twentieth of normal. Those are counselling role-play videos.

A word needs at least **40 titles** to appear on this chart. Below that the
middle value bounces around too much to mean anything.

`dark history` is two words joined because our preprocessing detected that they
almost always appear together, so it treats them as one term. Same for
`mental health` and `serial killer`.

### Where the numbers come from

`scripts/evaluate_topics.py` §4, listed in `reports/05_views_by_topic.txt`.
`scripts/make_figures.py` redraws the same calculation for the chart.

### Why this picture exists

This is the honesty check, and it is the strongest thing in my part.

Someone can always say "you got that result because you chose LDA" or "because
you chose 20 subjects". This chart has **no model in it**. No LSA, no LDA, no
choice of k — just counting words and looking up view numbers. And it shows the
same split: crime and mystery words at the top, tutorial and therapy words at the
bottom.

So the finding is in the data, not in our modelling choices. That is worth one
sentence in the presentation even if the chart stays on a backup slide.

### If someone points at it

> "This one has no topic model in it at all. We just took each word, found the
> titles containing it, and looked at the middle view ratio. The same split
> appears, which tells us the result is not something our model invented."

---

## The one number that is NOT on any picture

**The 6% (R² = 0.064).** It comes from a statistical test, not a chart.

What we did: instead of forcing each title into one subject, we used all 20
percentages for all 9,785 titles at once, and asked how much of the difference in
views they can account for. The answer is 6%.

Why we mention it: it stops us overselling. The subject of a title genuinely
matters — the effect is far too consistent to be luck — but 94% of why a video
gets views is thumbnail, timing, and how famous the case already was. Saying this
before the teacher does is worth more than hoping nobody asks.

It is in `reports/05_views_by_topic.txt` §3.

---

# PART D — the words explained

Plain versions, in case someone in the group asks or the teacher does.

**Topic modelling** — the computer reads thousands of documents and finds which
words keep showing up together, then groups them. Nobody tells it the subjects in
advance; it finds them. We then look at the word groups and give them names.

**LSA** (Latent Semantic Analysis) — the maths approach. Big table of
titles × words, squeezed down to 20 columns using linear algebra. Each column is
a subject. Downside: some numbers come out negative, and "negative amount of a
subject" is hard to explain.

**LDA** (Latent Dirichlet Allocation) — the probability approach. It pretends
every title was written by picking a mix of subjects and then picking words from
each. Then it works backwards to figure out what the mix was. Everything is a
percentage, so "this title is 60% subject 10" makes sense.

**k** — just how many subjects we ask for. We chose k = 20.

**Coherence (c_v)** — a score from 0 to 1 for how well a group of words belongs
together. "murder, victim, police, case" scores high. "murder, palette, excel,
sister" scores low. Higher is better.

**The elbow** — on our chart the score climbs steeply and then flattens out. The
bend is where extra subjects stop being worth it. We took the bend.

**Seeds** — LDA starts from a random guess, so it can give slightly different
answers each run. We ran it three times per setting and used the average, so we
are not reporting a lucky run.

**view_ratio** — a video's views divided by the views of the normal (median)
video on the same channel. 1.0 = normal, 2.0 = double, 0.5 = half.

**R² (six per cent)** — how much of the difference in views the title subject can
explain. Six per cent means the title matters but is not the main thing.

**Stopwords** — words like "the", "a", "is" that appear everywhere and mean
nothing for the subject. We delete them.

**Lemmatization** — turning words back to their dictionary form so "killers" and
"killer" count as the same word.

---

# PART E — questions we might get

**Why not just take the best coherence score?**
Because the score never stopped rising. It was still going up at sixty groups. If
we had tested a hundred, it would have said a hundred. That is a known problem
with very short documents. Also, the difference between our three random runs was
as big as the difference between neighbouring settings, so we would be choosing
based on randomness.

**Why did LDA score higher than LSA?**
LDA picks its top words from a proper probability list, so those words really do
appear together in real titles. LSA can put words at opposite ends of the same
contrast into the same group, which scores worse even when the group is readable.
Higher score does not automatically mean more useful — that is what the human
check is for.

**Six per cent sounds terrible. Is the finding weak?**
No. It means the title is one factor out of many, which is the honest picture.
The effect itself is strong and reliable — the gap between the best and worst
subject is three times, and the statistics say it is nowhere near chance. A small
R² and a real effect can both be true, because view counts swing wildly for
reasons no title could capture.

**Isn't this just showing which channel posts what?**
No, and that is exactly why we compare each video to its own channel. A channel
being popular cannot create the effect. The tutorial videos do badly *on the
channels that post them*.

**Why did you delete the channel names from the titles?**
Because otherwise the strongest "subject" the computer finds is just the channel.
Every Bailey Sarian title ends with "Mystery & Makeup, Bailey Sarian". Those words
are everywhere and tell us nothing. So if a word appears in 30% or more of one
channel's titles, we remove it — but only for that channel.

**Why lemmatization instead of stemming?**
Stemming chops the ends off words, so "mysterious" becomes "mysteri" and "police"
becomes "polic". A slide reading "murder, mysteri, polic" is embarrassing and
hard to label. Lemmatization gives real words. It is slower but that does not
matter at ten thousand short titles.

**Did you try removing people's names?**
Yes, and we kept the test in the repo. Victim and offender names fill up the word
lists. We removed 495 of them and the score did not reliably improve — it got
better at two settings and worse at two others. Since there was no real gain, we
kept the names. `scripts/name_ablation.py` reproduces it.

**What is your biggest limitation?**
Length. Everything difficult here comes from titles being five words after
cleaning. It is why the score never peaks, why some subjects blur together, and
why one channel being 28% of the data matters so much.

---

# PART F — where every number comes from

If anyone asks "where did you get that", this is the answer. Everything
regenerates when the scripts are run; nothing was typed by hand.

| Number on the slide | Which file | How it was worked out |
|---|---|---|
| 10,107 videos downloaded | `reports/01_corpus.txt` §1 | rows in the Kaggle file |
| 10,026 usable (99.2%) | §2 | removed 1 with no view count, 11 with zero views, 69 duplicates |
| 21 channels, 2012–2023 | §3 | count of channels, earliest and latest date |
| 9 words per title | §3 | average words per title |
| 4,119× channel gap | §4 | JCS normal video 15,943,809 views ÷ 48 Hours normal video 3,871 |
| 28% from one channel | §4 | Dr. Todd Grande, 2,838 ÷ 10,026 |
| 9,897 titles after cleaning | `reports/02_preprocessing.txt` §6 | 129 titles had fewer than 2 words left |
| 5 words left per title | §6 | average after cleaning |
| 2,210 words used | `reports/03_model_selection.txt` §1 | words in at least 5 titles and at most 40% of titles |
| 9,785 titles modelled | §1 | titles with at least one of those words left |
| score still rising at 60 | §2 | 0.453 at k=60 vs 0.415 at k=20 |
| randomness of 0.011 | §2 | average spread across the three runs |
| chose 20 | §3 | the bend in the curve |
| 1.36× best subject | `reports/05_views_by_topic.txt` §2 | middle value of the 185 titles in that subject |
| 0.45× worst subject | §2 | middle value of the 622 titles in that subject |
| 3× gap | §2 | 1.36 ÷ 0.45 |
| 6% explained | §3 | statistical model using all 20 subjects at once |
| "dark history" 3.73× | §4 | middle value of the 62 titles containing it |
| "session" 0.05× | §4 | middle value of the 49 titles containing it |

**About the little brackets on the chart** (`[1.22, 1.63]`): those show how sure
we are. If two subjects' brackets overlap, we cannot really say one beats the
other — read the chart in groups of similar bars, not as a strict ranking.

---

# PART G — not finished yet

**1. The human check still needs the group.** The rubric gives 25 points for
topic evaluation and asks specifically for human evaluation. I built the test but
I cannot do it alone — the whole point is several people.

How it works: there are 40 questions. Each shows six words. Five come from one
subject and one is an intruder from a different subject. If the subject is a real
subject, a person spots the odd word easily. We also write a name for each group.

What each person does:

1. Open `data/human_eval/intrusion_sheet_BLANK.csv`
2. Save it as `responses_yourname.csv` in the same folder
3. Fill in the three empty columns. **Do not open `intrusion_key.csv`** — that is
   the answer sheet
4. Tell me when done, I run the scoring

It takes about ten minutes. The questions are shuffled and do not say which model
they came from, so nobody can accidentally favour one.

**2. Augmentation** is the preprocessing person's item, not mine. If the teacher
asks why we did not do it, my answer is ready: the rubric asks for it when there
is class imbalance or too little data. We have 9,785 titles, no labels and no
classes, so there is nothing to balance. And swapping words in a nine-word title
would change the exact word pairings the model is trying to count — it would
change our answer, not make it stronger. We should agree on this as a group
before the presentation.
