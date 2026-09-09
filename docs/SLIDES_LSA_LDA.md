# Slides — LSA and LDA part (Seth)

Two slides.

- **Part A** is for whoever builds the slides. Copy the text in the boxes onto
  the slide. Nothing else goes on the slide.
- **Part B** is what I say out loud. Nobody needs to read it except me.
- **Part C** explains the words in case someone asks.

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

**If there is space**, `figures/fig3_words_by_views.png` can go on a backup slide
at the end. Do not put it on slide 2 — two big charts on one slide is too much.

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

# PART C — the words explained

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

# PART D — questions we might get

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

# PART E — where every number comes from

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
| "dark history" 3.73× | §4 | middle value of the 66 titles containing it |
| "session" 0.05× | §4 | middle value of the 50 titles containing it |

**About the little brackets on the chart** (`[1.22, 1.63]`): those show how sure
we are. If two subjects' brackets overlap, we cannot really say one beats the
other — read the chart in groups of similar bars, not as a strict ranking.

---

# PART F — not finished yet

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
