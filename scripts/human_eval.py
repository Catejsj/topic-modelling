"""Build and score the human evaluation of the topics.

Coherence scores are automatic proxies for "would a person find this topic
sensible". The rubric asks for human evaluation as well, so this script runs the
two standard tasks from Chang et al. (2009), "Reading Tea Leaves":

  word intrusion   Six words are shown: five are the top words of one topic and
                   one is an intruder - a word that is rare in this topic but
                   prominent in another. If the topic really describes one idea,
                   a person spots the odd word out easily. Model precision is
                   the share of topics where the annotator finds the intruder.

  topic labelling  The annotator writes a short label for the topic and rates
                   how confident they are (1-3). A topic nobody can name is not
                   usable no matter what its c_v says.

Both LSA and LDA are evaluated with the same sheet, shuffled together and with
the model hidden, so the annotator cannot favour one.

  build   python scripts/human_eval.py build
  score   python scripts/human_eval.py score
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EVAL = ROOT / "data" / "human_eval"
REPORT = ROOT / "reports" / "04_human_eval.txt"

N_REAL = 5  # real words shown per question
SEED = 11


def topic_word_lists() -> dict[str, list[list[str]]]:
    """Top words per topic for both models, from the saved artefacts."""
    meta = json.loads((MODELS / "meta.json").read_text())
    dictionary = Dictionary.load(str(MODELS / "dictionary.dict"))
    lda = LdaModel.load(str(MODELS / "lda.model"))

    lda_words = [[w for w, _ in lda.show_topic(t, topn=20)]
                 for t in range(meta["best_lda"])]

    comps = np.load(MODELS / "lsa_components.npy")
    terms = np.array([dictionary[i] for i in range(len(dictionary))])
    lsa_words = []
    for row in comps:
        idx = np.argsort(np.abs(row))[::-1][:20]
        lsa_words.append([str(terms[i]) for i in idx])

    return {"LSA": lsa_words, "LDA": lda_words}


def build() -> None:
    rng = random.Random(SEED)
    topics = topic_word_lists()
    rows, key = [], []

    for model, word_lists in topics.items():
        for t, words in enumerate(word_lists):
            real = words[:N_REAL]
            # An intruder must be prominent somewhere else and absent here,
            # otherwise the question is unfair rather than diagnostic.
            pool = [w for other, ws in enumerate(word_lists) if other != t
                    for w in ws[:10] if w not in words]
            if not pool:
                continue
            intruder = rng.choice(pool)
            shown = real + [intruder]
            rng.shuffle(shown)
            rows.append({
                "model": model, "topic": t,
                "words": " | ".join(shown),
                "intruder_you_think_is_odd": "",
                "your_label_for_the_rest": "",
                "confidence_1_to_3": "",
            })
            key.append({"model": model, "topic": t,
                        "intruder": intruder, "real_words": " ".join(real)})

    # Shuffle and give opaque ids so the annotator cannot see which model a
    # question came from and unconsciously favour one.
    rng.shuffle(rows)
    for i, row in enumerate(rows, start=1):
        qid = f"Q{i:03d}"
        model, topic = row.pop("model"), row.pop("topic")
        entry = next(k for k in key
                     if k["model"] == model and k["topic"] == topic)
        entry["question_id"] = qid
        row["question_id"] = qid
    rows = [{"question_id": r.pop("question_id"), **r} for r in rows]
    key = [{"question_id": k["question_id"], **{x: v for x, v in k.items()
                                                if x != "question_id"}}
           for k in key]
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EVAL / "intrusion_sheet_BLANK.csv", index=False)
    pd.DataFrame(key).to_csv(EVAL / "intrusion_key.csv", index=False)

    (EVAL / "INSTRUCTIONS.md").write_text(
        "# Topic evaluation sheet\n\n"
        "Copy `intrusion_sheet_BLANK.csv` to `responses_<yourname>.csv` and "
        "fill in the three empty columns. Do not open `intrusion_key.csv`.\n\n"
        "For each row you see six words.\n\n"
        "1. **intruder_you_think_is_odd** - five of the words belong together "
        "and one does not. Type the one that does not belong. Copy it exactly.\n"
        "2. **your_label_for_the_rest** - a short name for what the other five "
        "words are about, two or three words, e.g. `missing person cases`. "
        "If you cannot name it, write `unclear`.\n"
        "3. **confidence_1_to_3** - 1 = guessing, 2 = fairly sure, 3 = certain.\n\n"
        "Answer every row. There is no time limit, but do not look up the "
        "videos - the point is whether the words alone make sense.\n",
        encoding="utf-8")

    n = len(rows)
    print(f"wrote {(EVAL / 'intrusion_sheet_BLANK.csv').relative_to(ROOT)} "
          f"({n} questions: "
          f"{len(topics['LSA'])} LSA + {len(topics['LDA'])} LDA topics)")
    print(f"wrote {(EVAL / 'intrusion_key.csv').relative_to(ROOT)}  "
          "(do not share with annotators)")
    print(f"wrote {(EVAL / 'INSTRUCTIONS.md').relative_to(ROOT)}")
    print(f"\nchance level for word intrusion is 1/{N_REAL + 1} = "
          f"{1 / (N_REAL + 1):.1%}")


def score() -> None:
    key = pd.read_csv(EVAL / "intrusion_key.csv")
    responses = sorted(EVAL.glob("responses_*.csv"))
    if not responses:
        raise SystemExit(
            f"no responses_*.csv in {EVAL.relative_to(ROOT)} - "
            "run `build`, hand the blank sheet out, then run `score`")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] ANNOTATORS")
    frames = []
    for path in responses:
        name = path.stem.replace("responses_", "")
        r = pd.read_csv(path)
        r["annotator"] = name
        frames.append(r)
        say(f"    {name:<16} {len(r)} rows")
    resp = pd.concat(frames, ignore_index=True)
    merged = resp.merge(key, on="question_id", how="inner")
    say(f"    matched to key   {len(merged)} of {len(resp)}")
    say()

    def norm(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().str.lower()

    merged["correct"] = norm(merged["intruder_you_think_is_odd"]) == \
        norm(merged["intruder"])
    merged["named"] = ~norm(merged["your_label_for_the_rest"]).isin(
        {"", "unclear", "n/a", "none"})
    merged["confidence_1_to_3"] = pd.to_numeric(
        merged["confidence_1_to_3"], errors="coerce")

    say("[2] MODEL PRECISION  (word intrusion)")
    say(f"    chance level is {1 / (N_REAL + 1):.1%}")
    say()
    say(f"    {'model':<8} {'topics':>7} {'precision':>10} {'named':>8} "
        f"{'mean conf':>10}")
    for model, g in merged.groupby("model"):
        say(f"    {model:<8} {g['topic'].nunique():>7} "
            f"{g['correct'].mean():>9.1%} {g['named'].mean():>7.1%} "
            f"{g['confidence_1_to_3'].mean():>10.2f}")
    say()
    say("    Model precision is the share of questions where the annotator")
    say("    picked the true intruder. Higher means the topic held together.")
    say()

    say("[3] PER-ANNOTATOR")
    for (model, ann), g in merged.groupby(["model", "annotator"]):
        say(f"    {model:<5} {ann:<16} {g['correct'].mean():>6.1%} "
            f"({int(g['correct'].sum())}/{len(g)})")
    say()

    say("[4] HARDEST TOPICS  (nobody found the intruder)")
    per_topic = merged.groupby(["model", "topic"]).agg(
        precision=("correct", "mean"), conf=("confidence_1_to_3", "mean"),
        words=("real_words", "first"))
    worst = per_topic.sort_values("precision").head(8)
    for (model, t), row in worst.iterrows():
        say(f"    {model} {t:>2}  precision {row.precision:>5.0%}  "
            f"conf {row.conf:.1f}  {row.words}")
    say()
    say("    These are the topics to drop from the interpretation. A topic that")
    say("    no annotator can separate from the rest is a mixture, not a theme.")
    say()

    say("[5] LABELS GIVEN")
    for (model, t), g in merged.groupby(["model", "topic"]):
        labels = [l for l in g["your_label_for_the_rest"].dropna().astype(str)
                  if l.strip()]
        if labels:
            say(f"    {model} {t:>2}  " + "  /  ".join(labels))

    per_topic.to_csv(EVAL / "topic_scores.csv")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(EVAL / 'topic_scores.csv').relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    args = ap.parse_args()
    if not (MODELS / "meta.json").exists():
        raise SystemExit("run scripts/topic_models.py first")
    build() if args.mode == "build" else score()
