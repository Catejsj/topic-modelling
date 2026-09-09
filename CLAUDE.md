# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A text-mining research project answering: which kind of true-crime YouTube
title attracts the most views? 10,107 video titles from 21 channels are
topic-modelled (LSA and LDA), and the resulting topics are joined back to a
per-channel view-ratio outcome. The deliverable is `docs/SLIDES_LSA_LDA.md`
plus the numbered reports in `reports/`.

## Setup

`gensim` does not build on Python 3.14, so this project pins **Python 3.11**.

```bash
python3.11 -m venv .venv
./.venv/bin/pip install pandas numpy scikit-learn gensim nltk matplotlib contractions
./.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger_eng','names']]"
```

There is no requirements.txt/pyproject.toml — the package list above is the
full dependency set. Always invoke scripts via `./.venv/bin/python`, not a
bare `python`/`python3`.

## Pipeline — must run in this order

Each script checks for its input file and raises `SystemExit` naming the
prerequisite script if it's missing, rather than producing wrong output on a
stale/absent upstream file. When re-running after an edit, re-run every
downstream step too.

| # | Command | Produces |
|---|---|---|
| 1 | `./.venv/bin/python scripts/build_corpus.py` | `data/processed/corpus.csv`, `reports/01_corpus.txt` |
| 2 | `./.venv/bin/python scripts/preprocess.py` | `data/processed/tokens.csv`, `reports/02_preprocessing.txt` |
| 3 | `./.venv/bin/python scripts/topic_models.py` | `models/`, `data/processed/doc_topics.csv`, `reports/03_model_selection.txt` |
| 4 | `./.venv/bin/python scripts/evaluate_topics.py` | `reports/05_views_by_topic.txt` |
| 5 | `./.venv/bin/python scripts/make_figures.py` | `figures/*.png` |

Optional side branches, not part of the numbered dependency chain:

- `./.venv/bin/python scripts/human_eval.py build` — generates the
  word-intrusion sheet (`data/human_eval/`); `... score` reads completed
  sheets and writes `reports/04_human_eval.txt`.
- `./.venv/bin/python scripts/name_ablation.py` — one-off test of whether
  stripping people's names improves coherence (writes `reports/06_name_ablation.txt`).
- `./.venv/bin/python scripts/preprocess.py --stem` — Porter stemming instead
  of lemmatization, for the comparison documented in the report. Other flags:
  `--no-phrases`, `--min-tokens N`.

There is no test suite. Correctness is checked by re-running the pipeline and
reading the generated `reports/NN_*.txt` files — every script prints the same
content it writes to its report. Step 3 (`topic_models.py`, which fits LDA at
3 seeds × 11 values of k) takes ~2 minutes; the rest run in seconds.

## Architecture

**Everything is a CLI script under `scripts/`, run in sequence, communicating
only through files on disk** (`data/processed/*.csv`, `models/*`). There is no
importable library code and no shared module — each script is self-contained
with its own `ROOT = Path(__file__).resolve().parents[1]` and hardcoded
in/out paths at the top. When changing a step's output schema, grep the
downstream scripts for the column/file name rather than assuming a single
source of truth.

Common shape across every script: a `say()` closure that both `print()`s and
appends to a `lines` list, so console output and the written `reports/NN_*.txt`
file are always identical; `main()` does the work; outputs are written at the
very end.

Pipeline data flow:

```
data/raw/true_crime_channel_stats.csv (Kaggle export, checked into git)
  -> build_corpus.py    -> data/processed/corpus.csv          (title cleaning, view_ratio)
  -> preprocess.py      -> data/processed/tokens.csv           (tokenized, branding/stopwords stripped)
  -> topic_models.py    -> models/{lda.model,dictionary.dict,lsa_components.npy,lsa_doc.npy,lda_doc.npy,meta.json}
                         -> data/processed/{doc_topics.csv,k_sweep.csv}
  -> evaluate_topics.py -> data/processed/{topic_views.csv,topic_regression.csv}
  -> make_figures.py    -> figures/*.png
```

`models/meta.json` is the handoff point for the chosen `k` for each model
(`best_lsa`, `best_lda`) — scripts after `topic_models.py` read it rather than
recomputing or hardcoding k.

### Key design decisions (don't relitigate without reading the README rationale)

- **Outcome variable is `view_ratio`** (views / that video's channel's own
  median), not raw views — raw views mostly reflect channel size (up to
  4,119× spread between channels), which would swamp any title-content effect.
- **Per-channel branding removal** happens in `preprocess.py`: any word in
  ≥30% of one channel's titles is dropped for that channel only, before
  stopwording. Skipping this makes the channel name the dominant "topic".
- **k (number of topics) is chosen by elbow, not peak c_v coherence**
  (`topic_models.py`, function `knee()`) — on these short (~9-word) titles c_v
  never turns over within the tested grid, and LDA's seed-to-seed spread is as
  large as the gap between neighboring k, so both models land at k=20.
- **LDA is fit with 3 seeds per k** (`SEEDS` in `topic_models.py`) and scored
  on the mean, because a single fit can win a sweep by init luck; the
  median-coherence seed's model is the one kept.
- Lemmatization (WordNet, POS-tagged) is the default over stemming because
  topic word lists are read by humans; `--stem` exists only for the
  documented comparison.
