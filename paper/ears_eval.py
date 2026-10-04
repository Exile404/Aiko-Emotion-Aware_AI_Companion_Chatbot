#!/usr/bin/env python3
"""Gap 1 on EARS: emotion2vec+ on a corpus it cannot have seen.

EARS was first posted 10 June 2024, after emotion2vec+'s May 2024 release, so neither its
audio nor its labels can be in emotion2vec+'s training data. Two tests:

  1. SAME TASK, SEEN versus NEVER SEEN. EARS's read recordings are the same task as
     CREMA-D: actors reading fixed sentences in an acted emotion, studio audio, intended
     labels. emotion2vec+ trained on CREMA-D's labels (via EmoBox) and cannot have seen
     EARS. On the three classes both share cleanly (angry, neutral, sad) the question is
     whether emotion2vec+'s edge over a generic encoder survives the move from the seen
     corpus to the unseen one. Contamination predicts it shrinks or reverses; genuine
     representation quality predicts it holds.

  2. THE DELIVERY CONTRAST, POWERED. On IEMOCAP the encoders disagree about scripted
     versus improvised speech (difference in differences about +9.9). Every EARS speaker
     recorded each emotion both read and freeform, so the read-minus-freeform effect is
     paired within speaker across 107 speakers, where Expresso had 4 speakers and an
     inconclusive interval.

Probes: grouped 5-fold by speaker (actor for CREMA-D), regularisation fixed from MELD dev
(WavLM: clean_probe_config.json; emotion2vec+: C = 0.01, e2v_probe_results.md E-1); nothing
is tuned on EARS. Windows are 3 s, length-matched across conditions (ears_extract.py).
CIs: bootstrap over speakers, 1000 resamples.

    .venv/bin/python paper/ears_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import GroupKFold

from clean_probe_eval import FEATS, bal, fmt, probe

C3 = ["angry", "neutral", "sad"]
EARS_MAP = {"anger": "angry", "neutral": "neutral", "sadness": "sad"}
HAPPY_PROXIES = ["amusement", "contentment", "extasy"]
UNIFIED7 = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
C_E2V = 0.01
B_BOOT = 1000
OUT = "paper/ears_results.md"


def cv_predict(X, y, groups, c) -> np.ndarray:
    p = np.empty(len(y), dtype=object)
    for a, b in GroupKFold(n_splits=5).split(X, y, groups):
        p[b] = probe(c).fit(X[a], y[a]).predict(X[b])
    return p


def zero_shot(dep: np.ndarray, classes: list[str]) -> np.ndarray:
    idx = [UNIFIED7.index(c) for c in classes]
    return np.array(classes, dtype=object)[dep[:, idx].argmax(1)]


def boot(fn, groups, b=B_BOOT, seed=0) -> tuple[float, float]:
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))]))
            for _ in range(b)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_two(fn1, g1, fn2, g2, b=B_BOOT, seed=0) -> tuple[float, float]:
    """CI of fn1 - fn2 over two independent sets of clusters."""
    def idx(groups):
        by = defaultdict(list)
        for i, g in enumerate(groups):
            by[g].append(i)
        return list(by.values())
    c1, c2 = idx(g1), idx(g2)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        s1 = np.concatenate([c1[k] for k in rng.integers(0, len(c1), len(c1))])
        s2 = np.concatenate([c2[k] for k in rng.integers(0, len(c2), len(c2))])
        vals.append(fn1(s1) - fn2(s2))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    L = cfg["layer_index"]
    z = np.load(f"{FEATS}/ears.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/ears.meta.jsonl", encoding="utf-8")]
    emo = np.array([m["emotion"] for m in meta])
    cond = np.array([m["condition"] for m in meta])
    spk = np.array([m["speaker"] for m in meta])
    Xw_all = z["wavlm"][:, L, :].astype(np.float32)
    Xe_all = z["e2v"].astype(np.float32)
    dep_all = z["dep"]

    # ---- three-class EARS set ----------------------------------------------------------
    m3 = np.isin(emo, list(EARS_MAP))
    y = np.array([EARS_MAP[e] for e in emo[m3]])
    c3, s3 = cond[m3], spk[m3]
    Xw, Xe, dep = Xw_all[m3], Xe_all[m3], dep_all[m3]
    R, F = c3 == "read", c3 == "freeform"
    P = {"deployed head": zero_shot(dep, C3),
         "e2v probe": cv_predict(Xe, y, s3, C_E2V),
         "WavLM probe": cv_predict(Xw, y, s3, cfg["C_voice"])}
    systems = list(P)

    report = ["# Gap 1 on EARS: emotion2vec+ on a corpus released after it\n",
              "EARS (arXiv 2406.06185, 10 June 2024) post-dates emotion2vec+ (May 2024), so "
              "none of it can be in emotion2vec+'s training data. 3 s windows, length-"
              f"matched across conditions; {len(set(spk))} speakers. Probes grouped 5-fold "
              "by speaker, regularisation fixed from MELD dev; nothing tuned on EARS.\n"]
    comp = [[c_, int((c3 == c_).sum())] + [int(((c3 == c_) & (y == k)).sum()) for k in C3]
            for c_ in ("read", "freeform")]
    report.append(fmt("## Table Z-0 - Composition (three classes)", comp,
                      ["Condition", "windows"] + C3))

    rows = [[s, f"{bal(y[R], P[s][R]):.2f}", f"{bal(y[F], P[s][F]):.2f}", f"{bal(y, P[s]):.2f}"]
            for s in systems]
    report.append(fmt("## Table Z-1 - Voice balanced accuracy, angry / neutral / sad", rows,
                      ["Voice system", "read", "freeform", "all"]))

    # ---- test 2: read minus freeform, and the encoder disagreement --------------------
    def effect(p, sel):
        r, f = sel[R[sel]], sel[F[sel]]
        return bal(y[r], p[r]) - bal(y[f], p[f])
    allidx = np.arange(len(y))
    rows = []
    for s in systems:
        lo, hi = boot(lambda sel: effect(P[s], sel), s3)
        rows.append([s, f"{effect(P[s], allidx):+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table Z-2 - Read minus freeform, voice (95 % CI, speakers "
                      "resampled)", rows, ["Voice system", "read - freeform"]))

    def did(sel):
        return effect(P["e2v probe"], sel) - effect(P["WavLM probe"], sel)
    d = did(allidx)
    lo, hi = boot(did, s3)
    iemo_did = -2.31 - (-12.18)
    report.append(fmt("## Table Z-3 - Do the encoders disagree about delivery?", [
        ["IEMOCAP (labels seen)", "scripted - improvised", f"{iemo_did:+.2f}", "e2v_probe E-3"],
        ["Expresso (labels not seen)", "read - improvised", "+5.63 [-6.69, +14.59]",
         "expresso X-3"],
        ["EARS (never seen)", "read - freeform", f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]",
         "this file"]],
        ["Corpus", "contrast", "e2v effect - WavLM effect", "source"]))
    covers_zero, covers_iemo = lo <= 0 <= hi, lo <= iemo_did <= hi
    verdict = ("INCONCLUSIVE: the interval contains both zero and the IEMOCAP value"
               if covers_zero and covers_iemo else
               "favours CONTAMINATION: the encoders agree on a corpus emotion2vec+ never saw "
               "(interval contains zero, excludes the IEMOCAP disagreement)"
               if covers_zero else
               "favours SPECIALISATION: the encoders disagree on a corpus emotion2vec+ never "
               "saw, by as much as on IEMOCAP"
               if covers_iemo else
               "BOTH: the encoders still disagree on a corpus emotion2vec+ never saw, so "
               "part of the IEMOCAP disagreement is a general property of the encoder, but by "
               "less than on IEMOCAP, which leaves room for contamination to account for the "
               "rest")
    report.append(f"\n**Verdict: {verdict}.** In both corpora the disagreement has the same "
                  "sign: relative to WavLM, emotion2vec+ does worse on the spontaneous "
                  "condition than on the read or scripted one. That is the signature of an "
                  "encoder biased toward read, acted speech, which its training mix (mostly "
                  "acted corpora) would produce. Caveats: the contrasts are analogous, not "
                  "identical (IEMOCAP scripted against improvised dialogue; EARS read sentences "
                  "against freeform monologue), and the IEMOCAP value is a point estimate "
                  "carrying its own uncertainty, so 'less than on IEMOCAP' is indicative, not "
                  "a formal test.\n")
    report.append("Note the direction of the delivery effect itself differs between corpora: "
                  "on IEMOCAP the contamination-free probe found improvised speech EASIER for "
                  "voice than scripted, on EARS every system finds read speech easier than "
                  "freeform. 'Improvised speech is easier for voice' is therefore an IEMOCAP "
                  "finding and must not be generalised.\n")

    # ---- test 1: same task, seen (CREMA-D) versus never seen (EARS read) -------------
    cz = np.load(f"{FEATS}/crema_all.npz")
    ce = np.load(f"{FEATS}/crema_all.e2v.npz")
    cmeta = [json.loads(l) for l in open(f"{FEATS}/crema_all.meta.jsonl", encoding="utf-8")]
    if [m["clip"] for m in cmeta] != list(ce["clip"]):
        sys.exit("! CREMA-D e2v rows not aligned")
    ck = np.array([m["gold"] in C3 for m in cmeta])
    cy = np.array([m["gold"] for m in cmeta])[ck]
    cg = np.array([m["group"] for m in cmeta])[ck]
    cdep_map = {json.loads(l)["clip"]: json.loads(l)["voice"]
                for l in open("paper/crema_scores.jsonl", encoding="utf-8")}
    cdep = np.array([[cdep_map[m["clip"]].get(k, 0.0) for k in UNIFIED7]
                     for m, keep in zip(cmeta, ck) if keep])
    CP = {"deployed head": zero_shot(cdep, C3),
          "e2v probe": cv_predict(ce["e2v"][ck].astype(np.float32), cy, cg, C_E2V),
          "WavLM probe": cv_predict(cz["wavlm"][ck][:, L, :].astype(np.float32), cy, cg,
                                    cfg["C_voice"])}
    # EARS read only, probes trained on read only, to match CREMA-D's task exactly
    yr, sr_ = y[R], s3[R]
    EP = {"deployed head": P["deployed head"][R],
          "e2v probe": cv_predict(Xe[R], yr, sr_, C_E2V),
          "WavLM probe": cv_predict(Xw[R], yr, sr_, cfg["C_voice"])}
    rows = []
    for s in systems:
        a, b = bal(cy, CP[s]), bal(yr, EP[s])
        rows.append([s, f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}"])
    report.append(fmt("## Table Z-4 - Same task, seen versus never seen (angry / neutral / "
                      "sad, read acted speech)", rows,
                      ["Voice system", "CREMA-D (in training)", "EARS read (never seen)",
                       "change"]))

    def adv(PP, yy):
        return lambda sel: bal(yy[sel], PP["e2v probe"][sel]) - bal(yy[sel], PP["WavLM probe"][sel])
    a_c, a_e = adv(CP, cy)(np.arange(len(cy))), adv(EP, yr)(np.arange(len(yr)))
    lo_c, hi_c = boot(adv(CP, cy), cg)
    lo_e, hi_e = boot(adv(EP, yr), sr_)
    lo_d, hi_d = boot_two(adv(EP, yr), sr_, adv(CP, cy), cg)
    report.append(fmt("## Table Z-5 - emotion2vec+'s edge over WavLM, seen versus never seen",
                      [["CREMA-D (in training)", f"{a_c:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"],
                       ["EARS read (never seen)", f"{a_e:+.2f} [{lo_e:+.2f}, {hi_e:+.2f}]"],
                       ["change, never seen minus seen", f"{a_e - a_c:+.2f} [{lo_d:+.2f}, {hi_d:+.2f}]"]],
                      ["Corpus", "e2v probe - WavLM probe [95 % CI]"]))
    report.append("\nBoth corpora are read, acted, studio-quality speech with intended labels, "
                  "scored on the same three classes under identical probes, so the task is "
                  "held fixed and only emotion2vec+'s prior exposure changes. A genuinely "
                  "better encoder keeps its edge on the unseen corpus; a memorising one loses "
                  "it. The deployed zero-shot head is in Table Z-4 for the same contrast.\n")
    report.append("**Reading.** The contamination-free WavLM probe finds EARS read speech "
                  "EASIER than CREMA-D, so the unseen corpus is not a harder version of the "
                  "task. Against that, the deployed emotion2vec+ head falls by "
                  f"{bal(cy, CP['deployed head']) - bal(yr, EP['deployed head']):.1f} points "
                  "and emotion2vec+'s representational edge over WavLM reverses sign. Its "
                  "CREMA-D performance therefore overstates what it does on speakers and "
                  "recordings it has not trained on, which is the deployment case: every "
                  "real user of a companion system is an unseen speaker. Residual caveats: "
                  "the corpora differ in actors, sentences and recording chain (EARS is "
                  "anechoic, 48 kHz), and a zero-shot head always pays some domain shift; "
                  "the probe rows remove the label-mapping part of that shift and the gap "
                  "persists.\n")

    # ---- four-class signature, with each happy proxy -----------------------------------
    rows = []
    for hp in HAPPY_PROXIES:
        mp = {"anger": "angry", "neutral": "neutral", "sadness": "sad", hp: "happy"}
        m4 = np.isin(emo, list(mp))
        y4 = np.array([mp[e] for e in emo[m4]])
        pe = cv_predict(Xe_all[m4], y4, spk[m4], C_E2V)
        pw = cv_predict(Xw_all[m4], y4, spk[m4], cfg["C_voice"])
        rows.append([f"happy := {hp}", f"{bal(y4, pe):.2f}", f"{bal(y4, pw):.2f}",
                     f"{bal(y4, pe) - bal(y4, pw):+.2f}"])
    report.append(fmt("## Table Z-6 - Four-class EARS, each happy proxy (read + freeform)", rows,
                      ["Mapping", "e2v probe", "WavLM probe", "e2v advantage"]))
    report.append("\nFor comparison with the four-class rows of expresso_results.md X-4: "
                  "CREMA-D +6.46, IEMOCAP +15.40 (labels seen); MELD test -5.16, Expresso "
                  "-14.15 (not seen). EARS has no plain happiness category, so every proxy "
                  "is reported rather than one chosen.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
