# Topic modelling — what kind of true crime title attracts the most views?

Text Mining project. The corpus is 10,107 YouTube videos from 21 true crime
channels; the document is the **video title** and the outcome is the **view
count**. Two topic models are fitted — LSA and LDA — and their topics are joined
back to views.

Dataset: [True Crime Channel Statistics](https://www.kaggle.com/datasets/mhapich/true-crime-channel-statistics)
(mhapich, Kaggle, CC0 1.0), collected through the official YouTube Data API v3.

## Headline result

Titles that promise a specific unresolved crime story get about **1.36×** the
views of the typical video on the same channel; titles that promise instruction
(SPSS tutorials, counselling role-plays, makeup) get **0.45×** — a 3× spread,
and it holds inside individual channels, across both models, and in raw single
words with no model involved. The topic explains **6.4%** of the variance in
within-channel views; the rest is not in the title.

## Setup

`gensim` does not build on Python 3.14, so this project uses 3.11.

```bash
python3.11 -m venv .venv
./.venv/bin/pip install pandas numpy scikit-learn gensim nltk matplotlib contractions
./.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger_eng','names']]"
```

## Pipeline — run in this order

| # | Command | Produces |
|---|---|---|
| 1 | `./.venv/bin/python scripts/build_corpus.py` | `data/processed/corpus.csv`, `reports/01_corpus.txt` |
| 2 | `./.venv/bin/python scripts/preprocess.py` | `data/processed/tokens.csv`, `reports/02_preprocessing.txt` |
| 3 | `./.venv/bin/python scripts/topic_models.py` | `models/`, `data/processed/doc_topics.csv`, `reports/03_model_selection.txt` |
| 4 | `./.venv/bin/python scripts/evaluate_topics.py` | `reports/05_views_by_topic.txt` |
| 5 | `./.venv/bin/python scripts/make_figures.py` | `figures/*.png` |

Optional, and both worth running:

- `./.venv/bin/python scripts/human_eval.py build` — makes the word-intrusion
  sheet for the group; `... score` reads the completed sheets and writes
  `reports/04_human_eval.txt`.
- `./.venv/bin/python scripts/name_ablation.py` — tests whether deleting
  people's names improves coherence (it does not, reliably).
- `./.venv/bin/python scripts/preprocess.py --stem` — Porter stemming instead of
  lemmatization, for the comparison in the report.

Every script refuses to run and names its prerequisite rather than producing
wrong output. Step 3 takes about two minutes; the rest are seconds.

## Design decisions worth knowing

**The outcome is `view_ratio`, not views.** A video's views divided by the median
video on its own channel. Raw views mostly measure channel size — the median JCS
video has 4,119× the views of the median 48 Hours video — so any finding built on
them would be a finding about channels.

**Branding is removed before anything else.** Every title from one channel
repeats its series tag ("| Mystery & Makeup | Bailey Sarian"). Left in, the
strongest theme in the data is the channel name. Any word appearing in ≥ 30% of a
single channel's titles is dropped, for that channel only.

**k is chosen at the elbow, not the peak.** On 5-token documents the c_v
coherence curve never turns over — it was still rising at k = 60. LDA is fitted
with three random seeds at every k and the seed-to-seed spread (0.011) is as
large as the gaps between neighbouring k, so the peak would be noise. Both models
elbow at **k = 20**.

**No augmentation.** The rubric asks for it where class imbalance or limited data
exists. There are 9,785 unlabelled documents and no classes. Synonym replacement
on a 9-word title would alter the word co-occurrence counts that LDA is
estimating — it would change the answer rather than make it more robust.

## Layout

```
data/raw/          the Kaggle CSV, unmodified
data/processed/    corpus -> tokens -> doc_topics, plus the k sweep
data/human_eval/   word intrusion sheet, key, instructions
models/            fitted LDA, LSA components, document-topic matrices
reports/           numbered plain-text reports, one per stage
figures/           the three slide figures
docs/              slide content and speaking script
scripts/           the pipeline
```

`docs/PRESENTATION_PLAN_7_PEOPLE.md` splits the whole presentation across seven
speakers, one per rubric criterion, with each person's slide bullets, script and
sourced numbers. `docs/SLIDES_LSA_LDA.md` is the LSA/LDA part in full detail: two slides, a
word-for-word script, Q&A prep, and a table deriving every number quoted.
