#!/usr/bin/env python3
"""Step 3: evaluate the emotion-fusion method on the cached MELD test scores.

Reads paper/meld_test_scores.jsonl (voice + text score dicts per utterance) and scores
several predictors using the DEPLOYED fusion logic and constants from aiko/config.py, so
the paper numbers come straight from the shipped code:

  Table 1  baselines vs proposed (voice-only, text-only, naive fusion, reliability-weighted)
  Table 2  ablation (proposed minus one component at a time)
  Table 3  short-utterance subset (<= SHORT_UTTERANCE_WORDS), the C1 robustness claim
  plus a per-class breakdown for the proposed method

Metrics: weighted-F1 (the MELD standard), macro-F1, accuracy. Models are never re-run.

    .venv/bin/python paper/meld_eval.py --cache paper/meld_test_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiko import config

LABELS = config.UNIFIED_LABELS


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def argmax(scores: dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def fuse(voice: dict[str, float], text: dict[str, float], n_words: int, history: deque,
         *, use_reliability=True, use_short=True, use_agreement=True,
         use_corrections=True, use_recency=True, use_floor=True) -> str:
    """Faithful reimplementation of aiko/emotion.py::EmotionDetector._fuse, with each
    component behind a flag so it can be ablated. All flags True == deployed behaviour."""
    short = n_words <= config.SHORT_UTTERANCE_WORDS
    voice_weight = config.VOICE_SHORT_WEIGHT if (use_short and short) else 1.0

    if use_reliability:
        vrel, trel = config.VOICE_RELIABILITY, config.TEXT_RELIABILITY
    else:                                            # equal trust in both modalities
        vrel = {e: 0.5 for e in LABELS}
        trel = {e: 0.5 for e in LABELS}

    fused = {
        e: voice.get(e, 0.0) * vrel[e] * voice_weight + text.get(e, 0.0) * trel[e]
        for e in LABELS
    }

    voice_top, text_top = argmax(voice), argmax(text)
    text_conf = text.get(text_top, 0.0)

    if use_agreement and voice_top == text_top:
        fused[voice_top] *= 1.20
    if use_corrections:
        if text_top in ("fearful", "sad") and text_conf > 0.40 and voice_top in ("neutral", "happy"):
            fused[text_top] *= 1.35
            fused[voice_top] *= 0.80
        if not short and voice_top in ("angry", "happy") and voice.get(voice_top, 0.0) > 0.70 and text_top == "neutral":
            fused[voice_top] *= 1.25
    if use_recency:
        for recent in list(history)[-2:]:
            if recent != "neutral" and recent in fused:
                fused[recent] *= 1.05

    total = sum(fused.values())
    if total:
        fused = {k: v / total for k, v in fused.items()}

    best = argmax(fused)
    if use_floor and best != "neutral" and fused[best] < config.NEUTRAL_CONF_FLOOR:
        best = "neutral"
    history.append(best)
    return best


def make_fuse(**flags):
    def predict(r, history):
        return fuse(r["voice"], r["text_scores"], r["n_words"], history, **flags)
    return predict


def voice_only(r, history):
    return argmax(r["voice"])


def text_only(r, history):
    return argmax(r["text_scores"])


def naive(r, history):
    fused = {e: 0.5 * r["voice"].get(e, 0.0) + 0.5 * r["text_scores"].get(e, 0.0) for e in LABELS}
    return argmax(fused)


def run_condition(records, predictor) -> list[str]:
    """Predict every utterance in dialogue order, resetting recency history per dialogue."""
    history: deque = deque(maxlen=config.EMOTION_HISTORY)
    prev_dia = None
    preds = []
    for r in records:
        if r["dia"] != prev_dia:
            history.clear()
            prev_dia = r["dia"]
        preds.append(predictor(r, history))
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    total = len(y_true)
    per = {}
    for c in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t != c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        sup = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
    macro = sum(per[c]["f1"] for c in LABELS) / len(LABELS)
    weighted = sum(per[c]["f1"] * per[c]["support"] for c in LABELS) / total if total else 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total else 0.0
    return {"weighted_f1": weighted, "macro_f1": macro, "accuracy": acc, "per_class": per}


def subset(values, mask):
    return [v for v, m in zip(values, mask) if m]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_table(title, rows, delta_from=None):
    """rows: list of (label, metrics-dict). delta_from: index of the reference row for a Δ column."""
    lines = [title, ""]
    header = f"| {'Method':<34} | Weighted-F1 | Macro-F1 | Accuracy |"
    if delta_from is not None:
        header += " ΔWF1  |"
    lines.append(header)
    sep = "|" + "-" * 36 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
    if delta_from is not None:
        sep += "-" * 7 + "|"
    lines.append(sep)
    ref = rows[delta_from][1]["weighted_f1"] if delta_from is not None else None
    for label, m in rows:
        row = (f"| {label:<34} | {m['weighted_f1']*100:>10.2f} | "
               f"{m['macro_f1']*100:>7.2f} | {m['accuracy']*100:>7.2f} |")
        if delta_from is not None:
            d = (m["weighted_f1"] - ref) * 100
            row += f" {d:>+5.2f} |" if abs(d) > 1e-9 else f" {'—':>5} |"
        lines.append(row)
    return "\n".join(lines)


def fmt_per_class(title, m):
    lines = [title, "", f"| {'Emotion':<11} | Precision | Recall | F1     | Support |",
             "|" + "-" * 13 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|"]
    for c in LABELS:
        pc = m["per_class"][c]
        lines.append(f"| {c:<11} | {pc['precision']*100:>8.2f} | {pc['recall']*100:>6.2f} | "
                     f"{pc['f1']*100:>6.2f} | {pc['support']:>7} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--out", default="paper/emotion_results.md")
    args = ap.parse_args()

    records = []
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: (r["dia"], r["utt"]))

    gold = [r["gold"] for r in records]
    short_mask = [r["n_words"] <= config.SHORT_UTTERANCE_WORDS for r in records]
    n_short = sum(short_mask)

    # Run every predictor once (predictions aligned to `records` order).
    conditions = {
        "voice_only": voice_only,
        "text_only": text_only,
        "naive": naive,
        "full": make_fuse(),
        "no_reliability": make_fuse(use_reliability=False),
        "no_short": make_fuse(use_short=False),
        "no_floor": make_fuse(use_floor=False),
        "no_agreement": make_fuse(use_agreement=False),
        "no_corrections": make_fuse(use_corrections=False),
        "no_recency": make_fuse(use_recency=False),
    }
    preds = {name: run_condition(records, fn) for name, fn in conditions.items()}
    full_m = {name: metrics(gold, yp) for name, yp in preds.items()}
    short_m = {name: metrics(subset(gold, short_mask), subset(yp, short_mask))
               for name, yp in preds.items()}

    out = []
    out.append(f"# Emotion fusion on MELD test ({len(records)} utterances, "
               f"{n_short} short <= {config.SHORT_UTTERANCE_WORDS} words)\n")

    out.append(fmt_table(
        "## Table 1 - Baselines vs proposed (all utterances)",
        [("Voice only (emotion2vec)", full_m["voice_only"]),
         ("Text only (DistilRoBERTa)", full_m["text_only"]),
         ("Naive fusion (equal weight)", full_m["naive"]),
         ("Proposed (reliability-weighted)", full_m["full"])]))
    out.append("")

    out.append(fmt_table(
        "## Table 2 - Ablation (remove one component from proposed)",
        [("Proposed (full)", full_m["full"]),
         ("  - per-emotion reliability", full_m["no_reliability"]),
         ("  - short-utterance down-weight", full_m["no_short"]),
         ("  - neutral-confidence floor", full_m["no_floor"]),
         ("  - agreement boost", full_m["no_agreement"]),
         ("  - heuristic corrections", full_m["no_corrections"]),
         ("  - recency smoothing", full_m["no_recency"])],
        delta_from=0))
    out.append("")

    out.append(fmt_table(
        f"## Table 3 - Short-utterance subset ({n_short} utterances, <= "
        f"{config.SHORT_UTTERANCE_WORDS} words)",
        [("Voice only (emotion2vec)", short_m["voice_only"]),
         ("Text only (DistilRoBERTa)", short_m["text_only"]),
         ("Proposed (full)", short_m["full"]),
         ("Proposed - short down-weight", short_m["no_short"])],
        delta_from=2))
    out.append("")

    out.append(fmt_per_class("## Per-class - Proposed (all utterances)", full_m["full"]))
    out.append("")

    report = "\n".join(out)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()
