#!/usr/bin/env python3
"""Why do the two protocols disagree about whether voice helps?

Dialogue-disjoint CV on test: voice scores add +3.25 WF1 (and the gain GREW when random-fold
leakage was removed, so that leakage was inflating TEXT, not voice).
Train->test ablation: voice adds -0.48 WF1, i.e. nothing.

Both are dialogue-disjoint, so leakage is not the explanation. The two runs differ in TWO
ways at once:
  - classifier:     CV used StandardScaler+LogReg;  the ablation used HistGradientBoosting
  - training data:  CV trained on ~2.1k test-split utts; the ablation on 9,988 train-split

This isolates both. For each model in {LogReg, HGB} x each feature set in
{voice-only, text-only, text+voice}: train on MELD train, report dev and test.

Reading the result:
  (a) LogReg train->test ALSO shows a voice gain, HGB does not
      -> voice's information is redundant once a NONLINEAR head squeezes the text scores.
         Honest nuance ("voice helps weak heads only"), but still not a strong C1.
  (b) NEITHER model shows a voice gain train->test
      -> the CV gain comes from training on test-split data => train/test DOMAIN SHIFT in
         the voice features. The domain probe below confirms it.
  (c) BOTH show a voice gain
      -> the original ablation was an artifact and C1 is alive.

Domain probe: can a classifier tell MELD train from MELD test using voice scores alone vs
text scores alone? AUC well above 0.5 for voice but not text => voice features do not
transfer across splits, which would explain the whole discrepancy.

    .venv/bin/python paper/meld_protocol_check.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiko import config
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = config.UNIFIED_LABELS
IDX = {e: i for i, e in enumerate(LABELS)}


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("gold") in IDX:
                rows.append(r)
    rows.sort(key=lambda r: (r["dia"], r["utt"]))
    y = np.array([IDX[r["gold"]] for r in rows])
    tvec = np.array([[r["text_scores"].get(e, 0.0) for e in LABELS] for r in rows], dtype=np.float32)
    vvec = np.array([[r["voice"].get(e, 0.0) for e in LABELS] for r in rows], dtype=np.float32)
    return y, tvec, vvec


def make_model(kind):
    if kind == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return HistGradientBoostingClassifier(random_state=0)


def scores(y, pred):
    return (f1_score(y, pred, average="weighted", zero_division=0) * 100,
            f1_score(y, pred, average="macro", zero_division=0) * 100,
            accuracy_score(y, pred) * 100)


def boot(y, pred_a, pred_b, B=1000, seed=0):
    """Paired bootstrap on (b - a): WF1 and accuracy deltas with 95% CI and two-sided p."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_wf1, d_acc = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        a_w, _, a_a = scores(ys, pred_a[idx])
        b_w, _, b_a = scores(ys, pred_b[idx])
        d_wf1.append(b_w - a_w)
        d_acc.append(b_a - a_a)
    out = {}
    for name, d in (("WF1", np.array(d_wf1)), ("Acc", np.array(d_acc))):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[name] = (d.mean(), lo, hi, min(p, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="paper/meld_train_scores.jsonl")
    ap.add_argument("--dev", default="paper/meld_dev_scores.jsonl")
    ap.add_argument("--test", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    ytr, ttr, vtr = load(args.train)
    yde, tde, vde = load(args.dev)
    yte, tte, vte = load(args.test)
    print(f"train {len(ytr)}  dev {len(yde)}  test {len(yte)}\n")

    feats = {
        "voice-only (7)":   (vtr, vde, vte),
        "text-only (7)":    (ttr, tde, tte),
        "text+voice (14)":  (np.concatenate([ttr, vtr], 1),
                             np.concatenate([tde, vde], 1),
                             np.concatenate([tte, vte], 1)),
    }

    preds, rows = {}, []
    for kind in ("LogReg", "HGB"):
        for fname, (Xtr, Xde, Xte) in feats.items():
            m = make_model(kind).fit(Xtr, ytr)
            pde, pte = m.predict(Xde), m.predict(Xte)
            preds[(kind, fname)] = pte
            rows.append((kind, fname, scores(yde, pde), scores(yte, pte)))

    print(f"| {'Model':<7} | {'Features':<16} | {'dev WF1':>7} | {'dev acc':>7} "
          f"| {'test WF1':>8} | {'test macro':>10} | {'test acc':>8} |")
    print("|" + "-" * 9 + "|" + "-" * 18 + "|" + "-" * 9 + "|" + "-" * 9 + "|"
          + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 10 + "|")
    for kind, fname, dv, ts in rows:
        print(f"| {kind:<7} | {fname:<16} | {dv[0]:>7.2f} | {dv[2]:>7.2f} "
              f"| {ts[0]:>8.2f} | {ts[1]:>10.2f} | {ts[2]:>8.2f} |")

    print("\nvoice contribution on TEST (text+voice minus text-only), paired bootstrap "
          f"B={args.boot}:")
    for kind in ("LogReg", "HGB"):
        b = boot(yte, preds[(kind, "text-only (7)")], preds[(kind, "text+voice (14)")],
                 B=args.boot)
        for metric in ("WF1", "Acc"):
            mean, lo, hi, p = b[metric]
            sig = "significant" if (lo > 0 or hi < 0) else "not significant"
            print(f"  {kind:<7} d{metric} {mean:>+6.2f}  95% CI [{lo:>+6.2f}, {hi:>+6.2f}]  "
                  f"p={p:.3f}  ({sig})")

    # ---- domain probe: are voice features train/test separable? --------------------
    print("\ndomain-shift probe (AUC for discriminating MELD train vs MELD test; 0.5 = "
          "no shift):")
    for fname, tr, te in (("voice scores", vtr, vte), ("text scores", ttr, tte)):
        X = np.concatenate([tr, te], 0)
        d = np.concatenate([np.zeros(len(tr)), np.ones(len(te))])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        prob = cross_val_predict(clf, X, d, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        print(f"  {fname:<13} AUC {roc_auc_score(d, prob):.3f}")


if __name__ == "__main__":
    main()