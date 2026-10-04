#!/usr/bin/env python3
"""Learned decision-level fusion: train on MELD dev, evaluate on MELD test (leakage-free).

The fusion head sees only the dev split; every reported number is on the untouched test
split. Baselines and the deployed heuristic are reused from meld_eval.py.

    .venv/bin/python paper/meld_fusion_train.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meld_eval as me  # noqa: E402  (LABELS, metrics, fmt_table, fmt_per_class, baselines)

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

LABELS = me.LABELS


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: (r["dia"], r["utt"]))
    return recs


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


def main():
    dev = load("paper/meld_dev_scores.jsonl")
    test = load("paper/meld_test_scores.jsonl")
    yte = [r["gold"] for r in test]
    print(f"train (dev): {len(dev)} utts | test: {len(test)} utts\n")

    # baselines + deployed heuristic, on test
    base = {
        "Voice only (emotion2vec)": me.voice_only,
        "Text only (DistilRoBERTa)": me.text_only,
        "Naive fusion (equal weight)": me.naive,
        "Heuristic fusion (deployed)": me.make_fuse(),
    }
    rows = [(name, me.metrics(yte, me.run_condition(test, fn))) for name, fn in base.items()]

    # learned fusion: fit on dev (14-dim voice+text), predict test
    Xtr, ytr = feats(dev)
    Xte, _ = feats(test)
    best = (None, None, -1.0)
    for name, clf in {"Learned fusion (LogReg)": LogisticRegression(max_iter=2000),
                      "Learned fusion (HistGBM)": HistGradientBoostingClassifier(random_state=0)}.items():
        clf.fit(Xtr, ytr)
        pred = list(clf.predict(Xte))
        m = me.metrics(yte, pred)
        rows.append((name, m))
        if m["weighted_f1"] > best[2]:
            best = (name, pred, m["weighted_f1"])

    oracle = sum((me.argmax(r["voice"]) == r["gold"]) or (me.argmax(r["text_scores"]) == r["gold"])
                 for r in test) / len(test)

    out = [f"# Learned fusion: train on MELD dev ({len(dev)}), test on MELD test ({len(test)})\n",
           me.fmt_table("## Table 1 - Baselines vs heuristic vs learned fusion (test set)", rows),
           f"\n(oracle ceiling, either modality right: Acc = {oracle*100:.2f})\n"]

    # feature ablation of the learned head (HistGBM)
    abl = []
    for label, kw in [("voice + text (14)", dict(use_voice=True, use_text=True)),
                      ("voice features only (7)", dict(use_voice=True, use_text=False)),
                      ("text features only (7)", dict(use_voice=False, use_text=True))]:
        Xtr2, ytr2 = feats(dev, **kw)
        Xte2, _ = feats(test, **kw)
        clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr2, ytr2)
        abl.append((label, me.metrics(yte, list(clf.predict(Xte2)))))
    out.append(me.fmt_table("## Table 2 - Learned-fusion feature ablation (HistGBM, test)", abl, delta_from=0))

    out.append("\n" + me.fmt_per_class(f"## Per-class - {best[0]} (test set)", me.metrics(yte, best[1])))

    report = "\n".join(out)
    print(report)
    with open("paper/fusion_results.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n(written to paper/fusion_results.md)")


if __name__ == "__main__":
    main()