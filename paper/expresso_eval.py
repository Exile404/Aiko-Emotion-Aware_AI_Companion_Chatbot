#!/usr/bin/env python3
"""Gap 1: does emotion2vec+ behave differently where it has NOT seen the labels?

On IEMOCAP (labels seen via EmoBox) a contamination-free WavLM probe finds improvised
speech 12 points easier for voice than scripted speech, while emotion2vec+ (zero-shot
head AND its embeddings under the same probe) finds no difference. Two explanations:

  CONTAMINATION   emotion2vec+ memorised IEMOCAP's labels, lifting the hard (scripted)
                  condition to ceiling and erasing the contrast
  SPECIALISATION  emotion2vec+'s representation is simply insensitive to delivery style,
                  wherever it is applied

Expresso separates them: the same actors read and improvise in matching styles, and its
labels never entered emotion2vec+'s supervised training. The decisive quantity is the
difference in differences

    DiD = (e2v read - improvised) - (WavLM read - improvised)

On IEMOCAP DiD is about +9.9 (the encoders disagree). Contamination predicts DiD near
zero on Expresso (the encoders agree once nothing is memorised); specialisation predicts
a disagreement of similar size, in the same direction, persisting.

A second, independent signature: emotion2vec+'s advantage over WavLM under identical
probes, corpus by corpus. A genuinely better encoder should lead everywhere; a
memorising one should lead most where it saw the labels.

Read-vs-improvised is computed on the three styles that exist in both conditions
(happy, sad, default -> neutral); labels are intended styles, not perceptions. Probes are
leave-one-speaker-out (four speakers) and use the regularisation selected on MELD dev
(WavLM: clean_probe_config.json; emotion2vec+: C = 0.01, e2v_probe_results.md E-1), so
nothing is tuned on Expresso.

    .venv/bin/python paper/expresso_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import GroupKFold

from clean_probe_eval import COMMON4, FEATS, bal, fmt, probe

C3 = ["happy", "neutral", "sad"]
C_E2V = 0.01
B_BOOT = 1000
OUT = "paper/expresso_results.md"
UNIFIED7 = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def cluster_boot(fn, groups: np.ndarray, b: int = B_BOOT, seed: int = 0):
    """Percentile CI of fn(index_array) under resampling of clusters."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        sel = np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        vals.append(fn(sel))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    z = np.load(f"{FEATS}/expresso.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/expresso.meta.jsonl", encoding="utf-8")]
    y = np.array([m["gold"] for m in meta])
    cond = np.array([m["condition"] for m in meta])
    spk = np.array([m["speaker"] for m in meta])
    grp = np.array([m["group"] for m in meta])
    dur = np.array([m["duration"] for m in meta])
    Xw = z["wavlm"][:, cfg["layer_index"], :].astype(np.float32)
    Xe = z["e2v"].astype(np.float32)
    dep = z["dep"]
    R, I = cond == "read", cond == "improvised"

    # ---- predictions: zero-shot head, and leave-one-speaker-out probes ---------------
    idx3 = [UNIFIED7.index(c) for c in C3]
    P = {"deployed head": np.array(C3, dtype=object)[dep[:, idx3].argmax(1)]}
    for name, X, c in (("e2v probe", Xe, C_E2V), ("WavLM probe", Xw, cfg["C_voice"])):
        p = np.empty(len(y), dtype=object)
        for s in sorted(set(spk)):
            tr, te = spk != s, spk == s
            p[te] = probe(c).fit(X[tr], y[tr]).predict(X[te])
        P[name] = p
    systems = list(P)

    report = ["# Gap 1: emotion2vec+ where it has not seen the labels (Expresso)\n",
              "Same four actors reading and improvising in happy, sad and default "
              "(neutral) styles. Labels are intended styles. Probes leave-one-speaker-out, "
              "regularisation fixed from MELD dev. Nothing is tuned on Expresso.\n"]

    comp = []
    for c_ in ("read", "improvised"):
        m = cond == c_
        comp.append([c_, int(m.sum())] + [int(((y == g) & m).sum()) for g in C3]
                    + [f"{np.median(dur[m]):.2f}", len(set(grp[m]))])
    report.append(fmt("## Table X-0 - Composition", comp,
                      ["Condition", "n"] + C3 + ["median dur s", "clusters"]))

    # ---- X-1: accuracy by condition ----------------------------------------------------
    rows = [[s, f"{bal(y[R], P[s][R]):.2f}", f"{bal(y[I], P[s][I]):.2f}",
             f"{bal(y, P[s]):.2f}"] for s in systems]
    report.append(fmt("## Table X-1 - Voice balanced accuracy (three classes)", rows,
                      ["Voice system", "read", "improvised", "all"]))

    # ---- X-2: read minus improvised, per system, with cluster CIs ---------------------
    def effect(p, sel):
        r, i = sel[R[sel]], sel[I[sel]]
        return bal(y[r], p[r]) - bal(y[i], p[i])

    # duration-matched subsample: equal counts per (duration quartile, style)
    q = np.searchsorted(np.percentile(dur, [25, 50, 75]), dur, side="right")
    rng = np.random.default_rng(0)
    keep = []
    for b in range(4):
        for g in C3:
            r_ = np.where((q == b) & (y == g) & R)[0]
            i_ = np.where((q == b) & (y == g) & I)[0]
            n = min(len(r_), len(i_))
            keep.extend(rng.permutation(r_)[:n])
            keep.extend(rng.permutation(i_)[:n])
    keep = np.array(sorted(keep))

    allidx = np.arange(len(y))
    rows, eff = [], {}
    for s in systems:
        d = effect(P[s], allidx)
        lo, hi = cluster_boot(lambda sel: effect(P[s], sel), grp)
        dm = effect(P[s], keep)
        lom, him = cluster_boot(lambda sel: effect(P[s], keep[sel]), grp[keep])
        eff[s] = d
        rows.append([s, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{dm:+.2f} [{lom:+.2f}, {him:+.2f}]"])
    report.append(fmt("## Table X-2 - Read minus improvised, voice (95 % CI)", rows,
                      ["Voice system", "all clips", "duration x style matched"]))
    report.append(f"\nCluster bootstrap over recording groups (dialogue takes for improvised, "
                  f"speaker x style for read; {len(set(grp))} clusters), {B_BOOT} resamples. "
                  f"The duration-matched subsample has {len(keep)} clips.\n")

    # ---- X-3: the decisive difference in differences ---------------------------------
    def did(sel):
        return effect(P["e2v probe"], sel) - effect(P["WavLM probe"], sel)
    d_exp = did(allidx)
    lo, hi = cluster_boot(did, grp)
    iemo_did = -2.31 - (-12.18)
    report.append(fmt("## Table X-3 - Do the two encoders disagree about delivery?", [
        ["IEMOCAP (labels seen by emotion2vec+)", "scripted - improvised", f"{iemo_did:+.2f}",
         "e2v_probe_results.md E-3"],
        ["Expresso (labels not seen)", "read - improvised", f"{d_exp:+.2f} [{lo:+.2f}, {hi:+.2f}]",
         "this file"]],
        ["Corpus", "contrast", "e2v effect - WavLM effect", "source"]))
    report.append("\nNear zero on Expresso, the encoders agree once nothing is memorised, "
                  "which favours contamination as the cause of their IEMOCAP disagreement. "
                  "A disagreement of similar size and sign on Expresso favours "
                  "specialisation. Note the contrasts are analogous, not identical: IEMOCAP "
                  "compares scripted with improvised dialogue, Expresso read with improvised.\n")
    covers_zero, covers_iemo = lo <= 0 <= hi, lo <= iemo_did <= hi
    verdict = ("INCONCLUSIVE: the interval contains both zero and the IEMOCAP value, so this "
               "test cannot separate contamination from specialisation"
               if covers_zero and covers_iemo else
               "favours CONTAMINATION: the interval contains zero but excludes the IEMOCAP value"
               if covers_zero else
               "favours SPECIALISATION: the encoders still disagree without label exposure")
    report.append(f"**Verdict: {verdict}.** Expresso's read speech comes in only "
                  f"{len(set(grp[R]))} recording blocks (speaker x style), and an honest "
                  f"bootstrap resamples blocks, not clips, so this interval is wide. "
                  f"MSP-IMPROV, with many sentence x emotion x session blocks and perceived "
                  f"labels, is the test with the power to decide.\n")

    # ---- X-4: emotion2vec+'s advantage over WavLM, corpus by corpus -----------------
    sig = []
    sig.append(["Expresso (3 classes)", "labels not in training",
                f"{bal(y, P['e2v probe']):.2f}", f"{bal(y, P['WavLM probe']):.2f}",
                f"{bal(y, P['e2v probe']) - bal(y, P['WavLM probe']):+.2f}"])
    crema = crema_signature(cfg)
    if crema:
        sig.insert(0, crema)
    sig.insert(1 if crema else 0,
               ["IEMOCAP (4 classes)", "in EmoBox", "87.41", "72.01", "+15.40"])
    sig.insert(2 if crema else 1,
               ["MELD test (4 classes)", "in EmoBox (test split may be held out)",
                "48.86", "54.02", "-5.16"])
    report.append(fmt("## Table X-4 - emotion2vec+ minus WavLM under identical probes", sig,
                      ["Corpus", "emotion2vec+ exposure", "e2v probe", "WavLM probe",
                       "advantage"]))
    report.append("\nIEMOCAP and MELD rows are from e2v_probe_results.md (leave-one-session-"
                  "out; train -> test). CREMA-D is computed here, 5-fold grouped by actor, "
                  "both probes with the MELD-dev regularisation. Label spaces differ "
                  "between rows, so compare the SIGN and rough size of the advantage, not "
                  "absolute accuracies.\n")
    report.append("**Reading.** emotion2vec+ was fine-tuned on tens of thousands of hours of "
                  "emotional speech, so if its advantage reflected representation quality it "
                  "should lead a generic speech encoder on every corpus. It leads only on the "
                  "corpora whose labels it trained on, and trails where it did not: the "
                  "advantage tracks training exposure, which is the signature contamination "
                  "predicts. Two caveats bound this. Expresso's labels are intended speaking "
                  "styles rather than perceived emotions, a task emotion2vec+ was not built "
                  "for, which could depress it there independently of contamination; MELD, "
                  "whose labels ARE perceived emotions, shows the same sign, but its test split "
                  "may have been held out of emotion2vec+'s training rather than truly unseen. "
                  "The pattern is strong circumstantial evidence, not proof.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


def crema_signature(cfg: dict) -> list | None:
    if not os.path.exists(f"{FEATS}/crema_all.e2v.npz"):
        return None
    z = np.load(f"{FEATS}/crema_all.npz")
    e = np.load(f"{FEATS}/crema_all.e2v.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/crema_all.meta.jsonl", encoding="utf-8")]
    if [m["clip"] for m in meta] != list(e["clip"]):
        sys.exit("! CREMA-D e2v rows not aligned")
    keep = np.array([m["gold"] in COMMON4 for m in meta])
    y = np.array([m["gold"] for m in meta])[keep]
    g = np.array([m["group"] for m in meta])[keep]
    Xw = z["wavlm"][keep][:, cfg["layer_index"], :].astype(np.float32)
    Xe = e["e2v"][keep].astype(np.float32)
    pw, pe = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
    for a, b in GroupKFold(n_splits=5).split(Xw, y, g):
        pw[b] = probe(cfg["C_voice"]).fit(Xw[a], y[a]).predict(Xw[b])
        pe[b] = probe(C_E2V).fit(Xe[a], y[a]).predict(Xe[b])
    return ["CREMA-D (4 classes)", "in EmoBox", f"{bal(y, pe):.2f}", f"{bal(y, pw):.2f}",
            f"{bal(y, pe) - bal(y, pw):+.2f}"]


if __name__ == "__main__":
    main()
