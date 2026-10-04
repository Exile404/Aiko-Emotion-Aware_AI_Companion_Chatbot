#!/usr/bin/env python3
"""Is the voice contribution seen under random-fold CV an artifact of speaker leakage?

meld_emb_probe.py used random StratifiedKFold over utterances, so utterances from the
SAME dialogue (hence the same speakers) land in both train and test folds. Voice features
encode speaker identity and MELD's six main speakers have skewed emotion priors, so random
folds can credit voice for what is really speaker memorisation. The leakage-free
train->test protocol disagreed with CV about voice (-0.48 vs +2.34 WF1): that is the tell.

This re-runs the same feature sets under random StratifiedKFold and dialogue-disjoint
GroupKFold(Dialogue_ID) side by side. The gap between the two IS the leakage estimate.

    .venv/bin/python paper/meld_leak_check.py
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
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
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
    dia = np.array([k[0] for k in keys])
    tvec = np.array([[scores[k]["text_scores"].get(e, 0.0) for e in LABELS] for k in keys], dtype=np.float32)
    vvec = np.array([[scores[k]["voice"].get(e, 0.0) for e in LABELS] for k in keys], dtype=np.float32)
    evec = np.stack([embs[k] for k in keys])
    return y, dia, tvec, vvec, evec, len(keys)


def evaluate(X, y, cv, groups=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pred = cross_val_predict(clf, X, y, cv=cv, groups=groups, n_jobs=1)
    return (f1_score(y, pred, average="weighted", zero_division=0) * 100,
            f1_score(y, pred, average="macro", zero_division=0) * 100,
            accuracy_score(y, pred) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--emb", default="paper/meld_test_emb.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    y, dia, tvec, vvec, evec, n = load(args.scores, args.emb)
    print(f"{n} utterances, {len(set(dia.tolist()))} dialogues\n")

    feature_sets = [
        ("text-scores (7)", tvec),
        ("+ voice-scores (14)", np.concatenate([tvec, vvec], axis=1)),
        ("+ voice-emb (1031)", np.concatenate([tvec, evec], axis=1)),
    ]

    protocols = [
        ("random folds (leaky)", StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0), None),
        ("dialogue-disjoint", GroupKFold(n_splits=args.folds), dia),
    ]

    results = {}
    for pname, cv, groups in protocols:
        for fname, X in feature_sets:
            results[(pname, fname)] = evaluate(X, y, cv, groups)

    print(f"| {'Feature set':<21} | {'Protocol':<21} | Weighted-F1 | Macro-F1 | Accuracy |")
    print("|" + "-" * 23 + "|" + "-" * 23 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    for pname, _, _ in protocols:
        for fname, _ in feature_sets:
            wf1, mf1, acc = results[(pname, fname)]
            print(f"| {fname:<21} | {pname:<21} | {wf1:>10.2f} | {mf1:>7.2f} | {acc:>7.2f} |")

    print("\nvoice contribution over text-scores alone (WF1):")
    for pname, _, _ in protocols:
        base = results[(pname, "text-scores (7)")][0]
        sc = results[(pname, "+ voice-scores (14)")][0] - base
        em = results[(pname, "+ voice-emb (1031)")][0] - base
        print(f"  {pname:<21} voice-scores {sc:>+6.2f}   voice-emb {em:>+6.2f}")
    print("\nIf voice's gain shrinks or flips under dialogue-disjoint folds, the random-fold\n"
          "gain was speaker leakage, and the leakage-free train->test result stands.")


if __name__ == "__main__":
    main()