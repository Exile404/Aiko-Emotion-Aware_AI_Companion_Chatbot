#!/usr/bin/env python3
"""Probe: do 1024-dim emotion2vec voice EMBEDDINGS carry emotion signal that the
7-dim score-level fusion threw away? Cross-validated on the MELD test set (a
feasibility check before extracting the 9,988 train embeddings).

One pipeline (StandardScaler + LogisticRegression), 5-fold stratified
cross_val_predict, reporting weighted-F1 / macro-F1 / accuracy across feature sets:

  text-scores (7)                 <- the strong score-level head
  voice-scores (7)
  voice-emb (1024)                <- can voice embeddings alone predict emotion?
  text-scores + voice-scores (14) <- the earlier fusion features
  text-scores + voice-emb (1031)  <- KEY: does voice-emb add to text?

    .venv/bin/python paper/meld_emb_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiko import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = config.UNIFIED_LABELS
IDX = {e: i for i, e in enumerate(LABELS)}


def load(scores_path, emb_path):
    scores, order = {}, []
    for line in open(scores_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        scores[(r["dia"], r["utt"])] = r
        order.append((r["dia"], r["utt"]))

    embs = {}
    for line in open(emb_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        embs[(r["dia"], r["utt"])] = np.asarray(r["emb"], dtype=np.float32)

    keys = [k for k in order if k in embs and scores[k]["gold"] in IDX]
    y = np.array([IDX[scores[k]["gold"]] for k in keys])
    tvec = np.array([[scores[k]["text_scores"].get(e, 0.0) for e in LABELS] for k in keys], dtype=np.float32)
    vvec = np.array([[scores[k]["voice"].get(e, 0.0) for e in LABELS] for k in keys], dtype=np.float32)
    evec = np.stack([embs[k] for k in keys])
    return y, tvec, vvec, evec, len(keys)


def evaluate(name, X, y, cv, n_jobs):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=n_jobs)
    return (name, X.shape[1],
            f1_score(y, pred, average="weighted", zero_division=0) * 100,
            f1_score(y, pred, average="macro", zero_division=0) * 100,
            accuracy_score(y, pred) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--emb", default="paper/meld_test_emb.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="cross_val parallelism (1 = RAM-safe)")
    args = ap.parse_args()

    y, tvec, vvec, evec, n = load(args.scores, args.emb)
    print(f"{n} utterances aligned (scores + embeddings), {len(LABELS)} classes\n")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    feature_sets = [
        ("text-scores (7)", tvec),
        ("voice-scores (7)", vvec),
        ("voice-emb (1024)", evec),
        ("text-scores + voice-scores (14)", np.concatenate([tvec, vvec], axis=1)),
        ("text-scores + voice-emb (1031)", np.concatenate([tvec, evec], axis=1)),
    ]
    rows = [evaluate(name, X, y, cv, args.jobs) for name, X in feature_sets]

    print(f"| {'Feature set':<34} | Dim  | Weighted-F1 | Macro-F1 | Accuracy |")
    print("|" + "-" * 36 + "|" + "-" * 6 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    for name, dim, wf1, mf1, acc in rows:
        print(f"| {name:<34} | {dim:>4} | {wf1:>10.2f} | {mf1:>7.2f} | {acc:>7.2f} |")

    base = next(r for r in rows if r[0].startswith("text-scores (7)"))
    combo = next(r for r in rows if r[0].startswith("text-scores + voice-emb"))
    print(f"\nvoice-emb contribution over text-scores alone: "
          f"WF1 {combo[2]-base[2]:+.2f}, macro {combo[3]-base[3]:+.2f}, acc {combo[4]-base[4]:+.2f}")


if __name__ == "__main__":
    main()