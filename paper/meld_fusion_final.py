#!/usr/bin/env python3
"""Final C1: learned voice+text fusion. Train on MELD train, select on dev, test on test.

Reports two proposed variants (standard = best dev weighted-F1; balanced = best dev macro-F1
via class weighting) vs single-modality baselines and the deployed heuristic. Leakage-free:
test is used only for the final report; model choice is made on dev.

    .venv/bin/python paper/meld_fusion_final.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meld_eval as me  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

LABELS = me.LABELS


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: (r["dia"], r["utt"]))
    return recs


def feats(recs):
    X = np.array([[r["voice"].get(e, 0.0) for e in LABELS]
                  + [r["text_scores"].get(e, 0.0) for e in LABELS] for r in recs])
    return X, [r["gold"] for r in recs]


def fit_predict(kind, balanced, Xtr, ytr, Xte):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None)
        clf.fit(Xtr, ytr)
    else:  # histgbm
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr) if balanced else None)
    return list(clf.predict(Xte))


def main():
    train, dev, test = (load("paper/meld_train_scores.jsonl"),
                        load("paper/meld_dev_scores.jsonl"),
                        load("paper/meld_test_scores.jsonl"))
    Xtr, ytr = feats(train)
    Xdev, ydev = feats(dev)
    _, yte = feats(test)
    print(f"train {len(train)} | dev {len(dev)} | test {len(test)}\n")

    candidates = [("logreg", False), ("histgbm", False), ("logreg", True), ("histgbm", True)]
    cache, dev_rows = {}, []
    for kind, bal in candidates:
        m = me.metrics(ydev, fit_predict(kind, bal, Xtr, ytr, Xdev))
        cache[(kind, bal)] = m
        dev_rows.append((f"{kind}{' (balanced)' if bal else ''}", m))

    std_key = max([c for c in candidates if not c[1]], key=lambda c: cache[c]["weighted_f1"])
    bal_key = max([c for c in candidates if c[1]], key=lambda c: cache[c]["macro_f1"])

    Xte, _ = feats(test)
    std_m = me.metrics(yte, fit_predict(*std_key, Xtr, ytr, Xte))
    bal_m = me.metrics(yte, fit_predict(*bal_key, Xtr, ytr, Xte))

    base = {
        "Voice only (emotion2vec)": me.voice_only,
        "Text only (DistilRoBERTa)": me.text_only,
        "Naive fusion (equal weight)": me.naive,
        "Heuristic fusion (deployed)": me.make_fuse(),
    }
    rows = [(n, me.metrics(yte, me.run_condition(test, fn))) for n, fn in base.items()]
    rows.append((f"Learned fusion, standard ({std_key[0]})", std_m))
    rows.append((f"Learned fusion, balanced ({bal_key[0]})", bal_m))

    oracle = sum((me.argmax(r["voice"]) == r["gold"]) or (me.argmax(r["text_scores"]) == r["gold"])
                 for r in test) / len(test)

    out = [f"# C1 final: train MELD train ({len(train)}), select on dev, test on test ({len(test)})\n",
           me.fmt_table("## Model selection on dev", dev_rows),
           f"\n  -> standard = {std_key[0]}, balanced = {bal_key[0]} (balanced)\n",
           me.fmt_table("## Table 1 - Headline (TEST)", rows),
           f"\n(oracle ceiling, either modality right: Acc = {oracle*100:.2f})\n",
           me.fmt_per_class("## Per-class - standard (TEST)", std_m), "",
           me.fmt_per_class("## Per-class - balanced (TEST)", bal_m)]

    report = "\n".join(out)
    print(report)
    with open("paper/fusion_final.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n(written to paper/fusion_final.md)")


if __name__ == "__main__":
    main()