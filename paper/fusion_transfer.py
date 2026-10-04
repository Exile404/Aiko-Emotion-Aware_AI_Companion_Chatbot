#!/usr/bin/env python3
"""Does fusion tuned on one recording regime transfer to the other? (deployed encoders)

iemocap_results.md asserts that a fusion rule tuned on one acoustic regime does not
transfer to the other, and that Aiko's text-leading heuristic is calibrated for the wrong
regime. Both were INFERRED from unimodal accuracy. This script measures them directly,
on the deployed encoders' cached scores, so nothing is re-extracted:

  F-1  the deployed heuristic on IEMOCAP against voice-only and text-only, next to MELD
  F-2  each heuristic component ablated on BOTH corpora; a component whose effect flips
       sign between corpora was tuned to one regime
  F-3  a learned fusion head trained on one corpus and tested on the other, against the
       in-domain head and a head trained on both
  F-4  the same pooled head given a blind per-utterance audio-quality estimate
       (snr_proxy.py), which is computable from the microphone alone at runtime
  F-5  within each corpus, voice and text accuracy by audio-quality quartile; if audio
       quality drives the voice collapse, voice should improve with quality INSIDE a
       corpus too, where corpus identity is held fixed

Caveat carried by every number here: both deployed encoders have seen these corpora in
training (see clean_probe_eval.py for the contamination-free replication).

Protocols. P4 = the four classes common to all corpora (angry, happy, neutral, sad),
scored by balanced accuracy and macro F1. P7 = the full deployed seven-class scheme by
weighted F1, matching the MELD tables in emotion_results.md and fusion_final.md. In F-1
and F-2 every predictor emits any of the seven classes (the heuristic cannot be
restricted), so single-modality rows there use the unrestricted seven-way argmax.

    .venv/bin/python paper/fusion_transfer.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

import meld_eval as me
from iemocap_extract_scores import read_labels

COMMON4 = ["angry", "happy", "neutral", "sad"]
LABELS = me.LABELS
IEMOCAP_ROOT = "IEMOCAP_full_release/IEMOCAP_full_release"
B_BOOT = 1000
OUT = "paper/fusion_transfer_results.md"


# ----------------------------------------------------------------------------- data
def load_snr() -> dict[tuple[str, str], float]:
    out = {}
    for line in open("paper/snr_proxy.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[(r["corpus"], r["clip"])] = r["snr_db"]
    return out


def load_iemocap(snr: dict) -> list[dict]:
    """Scoreable IEMOCAP utterances in chronological order within each dialogue, so the
    heuristic's recency smoothing sees turns in the order they were spoken."""
    starts = read_labels(IEMOCAP_ROOT)
    rows = []
    for r in (json.loads(l) for l in open("paper/iemocap_scores.jsonl", encoding="utf-8")):
        if not r.get("gold") or not r.get("text_scores"):
            continue
        r["dia"] = r["dialog"]                      # meld_eval keys history on "dia"
        r["start"] = starts[r["clip"]]["start"]
        r["group"] = r["dialog"]
        r["snr"] = snr.get(("iemocap", r["clip"]))
        rows.append(r)
    rows.sort(key=lambda r: (r["dialog"], r["start"]))
    return rows


def load_meld(split: str, snr: dict) -> list[dict]:
    rows = [json.loads(l) for l in open(f"paper/meld_{split}_scores.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: (r["dia"], r["utt"]))
    for r in rows:
        r["clip"] = f"{split}_dia{r['dia']}_utt{r['utt']}"
        r["group"] = f"{split}_{r['dia']}"
        r["snr"] = snr.get(("meld", r["clip"]))
    return rows


# -------------------------------------------------------------------------- metrics
def p4(gold: list[str], pred: list[str]) -> dict:
    idx = [i for i, g in enumerate(gold) if g in COMMON4]
    g = [gold[i] for i in idx]
    p = [pred[i] for i in idx]
    return {"bal": 100 * balanced_accuracy_score(g, p),
            "macro": 100 * f1_score(g, p, labels=COMMON4, average="macro", zero_division=0),
            "n": len(g)}


def p7_wf1(gold: list[str], pred: list[str]) -> float:
    return 100 * me.metrics(gold, pred)["weighted_f1"]


def boot_diff(gold, pa, pb, groups, b=B_BOOT, seed=0) -> tuple[float, float, float]:
    """Dialogue-clustered paired bootstrap of balanced accuracy, A minus B."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    gold, pa, pb = np.array(gold), np.array(pa), np.array(pb)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(b):
        sel = np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        diffs.append(100 * (balanced_accuracy_score(gold[sel], pa[sel])
                            - balanced_accuracy_score(gold[sel], pb[sel])))
    point = 100 * (balanced_accuracy_score(gold, pa) - balanced_accuracy_score(gold, pb))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


# --------------------------------------------------------------- F-1 / F-2 heuristic
def heuristic_tables(meld: list[dict], iemo: list[dict]) -> list[str]:
    preds = {
        "Voice only (7-way argmax)": me.voice_only,
        "Text only (7-way argmax)": me.text_only,
        "Naive fusion (equal weight)": me.naive,
        "Heuristic fusion (deployed)": me.make_fuse(),
    }
    gm = [r["gold"] for r in meld]
    gi = [r["gold"] for r in iemo]
    rows, res = [], {}
    for name, fn in preds.items():
        pm, pi = me.run_condition(meld, fn), me.run_condition(iemo, fn)
        res[name] = (pm, pi)
        a, b = p4(gm, pm), p4(gi, pi)
        rows.append([name, f"{a['bal']:.2f}", f"{a['macro']:.2f}", f"{p7_wf1(gm, pm):.2f}",
                     f"{b['bal']:.2f}", f"{b['macro']:.2f}", f"{p7_wf1(gi, pi):.2f}"])
    out = [fmt("## Table F-1 - Deployed heuristic on both corpora", rows,
               ["Predictor", "MELD P4 bal acc", "MELD P4 macro F1", "MELD P7 WF1",
                "IEMOCAP P4 bal acc", "IEMOCAP P4 macro F1", "IEMOCAP P7 WF1"])]

    h = res["Heuristic fusion (deployed)"]
    lines = ["\nPaired dialogue-clustered bootstrap, P4 balanced accuracy, heuristic minus "
             "each single modality:"]
    for corpus, gold, recs, k in (("MELD test", gm, meld, 0), ("IEMOCAP", gi, iemo, 1)):
        mask = [i for i, g in enumerate(gold) if g in COMMON4]
        g4 = [gold[i] for i in mask]
        grp = [recs[i]["group"] for i in mask]
        for other in ("Voice only (7-way argmax)", "Text only (7-way argmax)"):
            d, lo, hi = boot_diff(g4, [h[k][i] for i in mask],
                                  [res[other][k][i] for i in mask], grp)
            lines.append(f"- {corpus}: heuristic minus {other.split(' (')[0].lower()} = "
                         f"{d:+.2f} points, 95 % CI [{lo:+.2f}, {hi:+.2f}]")
    out.append("\n".join(lines) + "\n")

    abl = [("per-emotion reliability", "use_reliability"),
           ("short-utterance voice down-weight", "use_short"),
           ("neutral-confidence floor", "use_floor"),
           ("agreement boost", "use_agreement"),
           ("heuristic corrections", "use_corrections"),
           ("recency smoothing", "use_recency")]
    full_m = me.run_condition(meld, me.make_fuse())
    full_i = me.run_condition(iemo, me.make_fuse())
    base_m, base_i = p4(gm, full_m)["bal"], p4(gi, full_i)["bal"]
    wm, wi = p7_wf1(gm, full_m), p7_wf1(gi, full_i)
    arows = []
    for name, flag in abl:
        pm = me.run_condition(meld, me.make_fuse(**{flag: False}))
        pi = me.run_condition(iemo, me.make_fuse(**{flag: False}))
        dm, di = p4(gm, pm)["bal"] - base_m, p4(gi, pi)["bal"] - base_i
        flip = "FLIPS" if (dm > 0.25 and di < -0.25) or (dm < -0.25 and di > 0.25) else ""
        arows.append([f"remove {name}", f"{dm:+.2f}", f"{p7_wf1(gm, pm) - wm:+.2f}",
                      f"{di:+.2f}", f"{p7_wf1(gi, pi) - wi:+.2f}", flip])
    out.append(fmt("## Table F-2 - Removing each heuristic component, both corpora", arows,
                   ["Ablation", "MELD d bal acc (P4)", "MELD d WF1 (P7)",
                    "IEMOCAP d bal acc (P4)", "IEMOCAP d WF1 (P7)", "sign"]))
    out.append("\nPositive = the component was HURTING on that corpus. A component that "
               "helps on one corpus and hurts on the other (marked FLIPS, threshold 0.25 "
               "points each way) is calibrated to one recording regime.\n")
    return out


# ------------------------------------------------------------ F-3 / F-4 learned heads
def X(rows: list[dict], with_snr: bool = False) -> np.ndarray:
    base = [[r["voice"].get(e, 0.0) for e in LABELS]
            + [r["text_scores"].get(e, 0.0) for e in LABELS] for r in rows]
    if with_snr:
        base = [b + [r["snr"] if r["snr"] is not None else np.nan] for b, r in zip(base, rows)]
    return np.array(base, dtype=float)


def c4(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["gold"] in COMMON4]


def fit_pred(tr: list[dict], te: list[dict], with_snr=False) -> list[str]:
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X(tr, with_snr), [r["gold"] for r in tr])
    return list(clf.predict(X(te, with_snr)))


def loso(iemo: list[dict], extra: list[dict] | None = None, with_snr=False) -> list[str]:
    """IEMOCAP predictions, each session predicted by a head that never saw it."""
    pred = [None] * len(iemo)
    for s in sorted({r["session"] for r in iemo}):
        tr = [r for r in iemo if r["session"] != s] + (extra or [])
        te_idx = [i for i, r in enumerate(iemo) if r["session"] == s]
        for i, p in zip(te_idx, fit_pred(tr, [iemo[i] for i in te_idx], with_snr)):
            pred[i] = p
    return pred


def transfer_tables(mtr: list[dict], mte: list[dict], iemo: list[dict]) -> list[str]:
    mtr, mte, iemo = c4(mtr), c4(mte), c4(iemo)
    gm, gi = [r["gold"] for r in mte], [r["gold"] for r in iemo]
    grp_m, grp_i = [r["group"] for r in mte], [r["group"] for r in iemo]

    def argmax4(rows, key):
        return [max(COMMON4, key=lambda e: r[key].get(e, 0.0)) for r in rows]

    P = {
        "voice argmax": (argmax4(mte, "voice"), argmax4(iemo, "voice")),
        "text argmax": (argmax4(mte, "text_scores"), argmax4(iemo, "text_scores")),
        "head trained on MELD": (fit_pred(mtr, mte), fit_pred(mtr, iemo)),
        "head trained on IEMOCAP": (fit_pred(iemo, mte), loso(iemo)),
        "head trained on both": (fit_pred(mtr + iemo, mte), loso(iemo, extra=mtr)),
        "head trained on both + audio quality": (fit_pred(mtr + iemo, mte, True),
                                                 loso(iemo, extra=mtr, with_snr=True)),
    }
    rows = []
    for name, (pm, pi) in P.items():
        a, b = p4(gm, pm), p4(gi, pi)
        rows.append([name, f"{a['bal']:.2f}", f"{a['macro']:.2f}",
                     f"{b['bal']:.2f}", f"{b['macro']:.2f}"])
    out = [fmt("## Table F-3 - Learned fusion across corpora (P4, deployed scores)", rows,
               ["Fusion", "MELD test bal acc", "MELD test macro F1",
                "IEMOCAP bal acc", "IEMOCAP macro F1"])]
    out.append("\nHistGradientBoosting on the 14 deployed scores (seven voice, seven text), "
               "default settings as selected on MELD dev in fusion_final.md. IEMOCAP "
               "in-corpus predictions are leave-one-session-out, so no head is scored on a "
               "session it was trained on. MELD heads train on MELD train only.\n")

    tests = [
        ("MELD test", gm, grp_m, 0, "head trained on IEMOCAP", "head trained on MELD"),
        ("IEMOCAP", gi, grp_i, 1, "head trained on MELD", "head trained on IEMOCAP"),
        ("MELD test", gm, grp_m, 0, "head trained on both", "head trained on MELD"),
        ("IEMOCAP", gi, grp_i, 1, "head trained on both", "head trained on IEMOCAP"),
        ("MELD test", gm, grp_m, 0, "head trained on both + audio quality",
         "head trained on both"),
        ("IEMOCAP", gi, grp_i, 1, "head trained on both + audio quality",
         "head trained on both"),
    ]
    lines = ["Paired dialogue-clustered bootstrap, balanced accuracy:"]
    for corpus, gold, grp, k, a, b in tests:
        d, lo, hi = boot_diff(gold, P[a][k], P[b][k], grp)
        lines.append(f"- {corpus}: {a} minus {b} = {d:+.2f}, 95 % CI [{lo:+.2f}, {hi:+.2f}]")
    out.append("\n".join(lines) + "\n")
    return out


# ------------------------------------------------------------------- F-5 quality bins
def quality_bins(mte: list[dict], iemo: list[dict]) -> list[str]:
    rows = []
    for corpus, recs in (("MELD test", c4(mte)), ("IEMOCAP", c4(iemo))):
        recs = [r for r in recs if r["snr"] is not None]
        q = np.percentile([r["snr"] for r in recs], [25, 50, 75])
        for b in range(4):
            lo = -np.inf if b == 0 else q[b - 1]
            hi = np.inf if b == 3 else q[b]
            sub = [r for r in recs if lo <= r["snr"] < hi]
            g = [r["gold"] for r in sub]
            pv = [max(COMMON4, key=lambda e: r["voice"].get(e, 0.0)) for r in sub]
            pt = [max(COMMON4, key=lambda e: r["text_scores"].get(e, 0.0)) for r in sub]
            v, t = 100 * balanced_accuracy_score(g, pv), 100 * balanced_accuracy_score(g, pt)
            span = (f"< {q[0]:.1f}" if b == 0 else f">= {q[2]:.1f}" if b == 3
                    else f"{q[b - 1]:.1f} to {q[b]:.1f}")
            rows.append([corpus, f"Q{b + 1} ({span} dB)", len(sub), f"{v:.2f}", f"{t:.2f}",
                         f"{v - t:+.2f}"])
    out = [fmt("## Table F-5 - Accuracy by audio-quality quartile, within each corpus", rows,
               ["Corpus", "SNR-proxy quartile", "n", "voice bal acc", "text bal acc",
                "voice - text"])]
    out.append("\nQuartiles are computed within each corpus, so corpus identity is held "
               "fixed and only estimated audio quality varies. If degraded audio drives the "
               "voice collapse, voice accuracy should rise from Q1 to Q4 inside each corpus. "
               "Text accuracy is a control: it does not hear the audio.\n")
    return out


def main() -> None:
    snr = load_snr()
    iemo = load_iemocap(snr)
    mtr, mte = load_meld("train", snr), load_meld("test", snr)
    print(f"IEMOCAP {len(iemo)} | MELD train {len(mtr)} | MELD test {len(mte)}", flush=True)

    report = ["# Fusion transfer across recording regimes (deployed encoders)\n",
              "emotion2vec+ voice scores and DistilRoBERTa text scores as cached by the "
              "extraction scripts. Both encoders have seen these corpora in training; the "
              "contamination-free replication is in clean_probe_results.md.\n"]
    report += heuristic_tables(mte, iemo)
    report += transfer_tables(mtr, mte, iemo)
    report += quality_bins(mte, iemo)
    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
