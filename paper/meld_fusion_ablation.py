#!/usr/bin/env python3
"""C1 closers: learned-fusion feature ablation (train-trained) + paired bootstrap significance.

    .venv/bin/python paper/meld_fusion_ablation.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meld_eval as me  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

LABELS = me.LABELS


def load(p):
    r = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    r.sort(key=lambda x: (x["dia"], x["utt"]))
    return r


def feats(recs, use_voice=True, use_text=True):
    rows = []
    for r in recs:
        row = []
        if use_voice:
            row += [r["voice"].get(e, 0.0) for e in LABELS]
        if use_text:
            row += [r["text_scores"].get(e, 0.0) for e in LABELS]
        rows.append(row)
    return np.array(rows), [r["gold"] for r in recs]


def boot(yte, a, b, name, B=1000, seed=0):
    rng = random.Random(seed)
    n = len(yte)
    a_ok = np.array([t == p for t, p in zip(yte, a)])
    b_ok = np.array([t == p for t, p in zip(yte, b)])
    da, dw = [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        da.append(a_ok[idx].mean() - b_ok[idx].mean())
        yt = [yte[i] for i in idx]
        dw.append(me.metrics(yt, [a[i] for i in idx])["weighted_f1"]
                  - me.metrics(yt, [b[i] for i in idx])["weighted_f1"])
    da, dw = np.array(da) * 100, np.array(dw) * 100
    print(f"proposed - {name:10s}: dAcc={da.mean():+.2f} "
          f"[{np.percentile(da, 2.5):+.2f}, {np.percentile(da, 97.5):+.2f}]  "
          f"dWF1={dw.mean():+.2f} [{np.percentile(dw, 2.5):+.2f}, {np.percentile(dw, 97.5):+.2f}]  "
          f"p(dAcc<=0)={np.mean(da <= 0):.3f}")


def main():
    train, test = load("paper/meld_train_scores.jsonl"), load("paper/meld_test_scores.jsonl")
    yte = [r["gold"] for r in test]

    abl, preds = [], {}
    for label, kw in [("voice + text (14)", dict(use_voice=True, use_text=True)),
                      ("voice only (7)", dict(use_voice=True, use_text=False)),
                      ("text only (7)", dict(use_voice=False, use_text=True))]:
        Xtr, ytr = feats(train, **kw)
        Xte, _ = feats(test, **kw)
        p = list(HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).predict(Xte))
        preds[label] = p
        abl.append((label, me.metrics(yte, p)))
    print(me.fmt_table("## Feature ablation - learned fusion (HGB, train -> test)", abl, delta_from=0))

    print("\n## Significance (paired bootstrap, B=1000, 95% CI of the difference)\n")
    boot(yte, preds["voice + text (14)"], me.run_condition(test, me.text_only), "text-only")
    boot(yte, preds["voice + text (14)"], me.run_condition(test, me.make_fuse()), "heuristic")


if __name__ == "__main__":
    main()