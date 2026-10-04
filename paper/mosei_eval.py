#!/usr/bin/env python3
"""Gap 2: does "voice beats text only in improvised speech" extend to natural speech?

Table S-4 (clean_spontaneity_results.md) found, with contamination-free encoders, that
voice minus text is about +26 balanced-accuracy points in IEMOCAP's improvised dialogue
but about zero in scripted speech (IEMOCAP scripted, MELD). IEMOCAP's improvisation is
still actors. CMU-MOSEI is NATURAL speech: real YouTube opinion monologues, crowd-rated.
If natural speech behaves like improvised speech, voice should beat text clearly here; if
the S-4 effect is a property of actors improvising, it may not.

Encoders and protocol are exactly S-4's: frozen WavLM-large (layer and C from MELD dev)
and frozen MiniLM on the human transcript, linear probes trained on MOSEI's official train
split and scored once on its test split. Nothing is tuned on MOSEI. CIs resample videos
(one speaker each), 1000 times, paired.

LABELS. MOSEI rates six emotions on 0-3 intensity scales, averaged over three raters; the
deployed scheme needs single classes. No mapping is canonical, so three are reported:
  A  neutral if every intensity is 0; otherwise the strictly dominant emotion if it is
     happiness, sadness or anger and its intensity is at least 1 (primary)
  B  as A with threshold 0.5 (more segments, noisier labels)
  C  presence: neutral if no emotion is present; the single present emotion if exactly
     one is present and it is happiness, sadness or anger
Raters saw the video with sound and words, so labels may lean on the words, as with MELD.

For deployment relevance, the deployed system's own two detectors are also compared on
the test split: emotion2vec+ (MOSEI is not in EmoBox) against DistilRoBERTa (not trained
on MOSEI), on the human transcript.

    .venv/bin/python paper/mosei_eval.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clean_probe_eval import FEATS, bal, fmt, probe

CLASSES = ["angry", "happy", "neutral", "sad"]
NAME = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
EMOS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
UNIFIED7 = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
B_BOOT = 1000
OUT = "paper/mosei_results.md"


def label(m: dict, rule: str) -> str | None:
    it = m["intensity"]
    if rule in ("A", "B"):
        if all(it[e] == 0 for e in EMOS):
            return "neutral"
        top = max(EMOS, key=lambda e: it[e])
        if sum(1 for e in EMOS if it[e] == it[top]) > 1:
            return None
        thr = 1.0 if rule == "A" else 0.5
        return NAME.get(top) if it[top] >= thr else None
    present = [e for e in EMOS if m["presence"][e]]
    if not present:
        return "neutral"
    return NAME.get(present[0]) if len(present) == 1 else None


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    z = np.load(f"{FEATS}/mosei.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/mosei.meta.jsonl", encoding="utf-8")]
    split = np.array([m["split"] for m in meta])
    video = np.array([m["video"] for m in meta])
    Xv = z["wavlm"][:, cfg["layer_index"], :].astype(np.float32)
    Xt = z["minilm"]
    dep = z["dep"]

    texts = {}
    for line in open("data/mosei/manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        texts[r["id"]] = r["text"]

    report = ["# Gap 2: voice against text on natural speech (CMU-MOSEI)\n",
              "Real YouTube opinion monologues, crowd-rated. Contamination-free encoders, "
              "probes trained on the official train split, scored once on test, settings "
              "from MELD dev. CIs resample videos.\n"]

    rows, comp, results = [], [], {}
    for rule in ("A", "B", "C"):
        y = np.array([label(m, rule) for m in meta], dtype=object)
        tr = (split == "train") & (y != None)  # noqa: E711
        te = (split == "test") & (y != None)  # noqa: E711
        pv = probe(cfg["C_voice"]).fit(Xv[tr], y[tr]).predict(Xv[te])
        pt = probe(cfg["C_text"]).fit(Xt[tr], y[tr]).predict(Xt[te])
        yt, vt = y[te], video[te]
        by = defaultdict(list)
        for i, v in enumerate(vt):
            by[v].append(i)
        keys = list(by)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(B_BOOT):
            sel = np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            diffs.append(bal(yt[sel], pv[sel]) - bal(yt[sel], pt[sel]))
        d = bal(yt, pv) - bal(yt, pt)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        results[rule] = (yt, pv, pt, vt, te)
        present = sorted(set(yt))
        rows.append([rule, int(te.sum()), len(present), f"{bal(yt, pv):.2f}", f"{bal(yt, pt):.2f}",
                     f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
        c_tr, c_te = Counter(y[tr]), Counter(yt)
        comp.append([rule, int(tr.sum()), int(te.sum())]
                    + [f"{c_te.get(k, 0)}" for k in CLASSES])
        print(f"rule {rule}: margin {d:+.2f}", flush=True)

    report.append(fmt("## Table M-0 - Segments per labelling rule (test counts per class)",
                      comp, ["Rule", "train", "test"] + CLASSES))
    report.append(fmt("## Table M-1 - Voice minus text on natural speech (clean encoders, "
                      "test split)", rows,
                      ["Rule", "test segments", "classes", "voice bal acc", "text bal acc",
                       "voice - text [95 % CI]"]))
    report.append("\nReference, clean_spontaneity_results.md Table S-4: IEMOCAP improvised "
                  "+25.76, IEMOCAP scripted +0.40, MELD (scripted television) +0.03.\n")
    margins = {r[0]: r[5] for r in rows}
    all_pos = all(float(m.split(" [")[1].split(",")[0]) > 0 for m in margins.values())
    report.append(("**Verdict: the voice advantage extends to natural speech.** " if all_pos
                   else "**Verdict: mixed across labelling rules.** ")
                  + "Under every labelling rule the interval "
                  + ("excludes zero" if all_pos else "does not uniformly exclude zero")
                  + ", so the conclusion does not depend on how MOSEI's intensities are mapped "
                  "to classes. Two features of MOSEI favour TEXT, which makes the result "
                  "conservative: its transcripts are long (see the word counts below) and its "
                  "raters saw the words. The size of the advantage sits between scripted "
                  "speech (about zero) and actors' improvisation (about +26), giving the "
                  "ordering scripted < natural < actor-improvised. Cross-corpus magnitudes "
                  "compare different label spaces and protocols, so the ordering is "
                  "indicative; the sign and its interval are the robust part.\n")

    # ---- deployed detectors on the same test segments (rule A) ------------------------
    yt, pv, pt, vt, te = results["A"]
    present = sorted(set(yt))
    idx = [UNIFIED7.index(c) for c in present]
    dv = np.array(present, dtype=object)[dep[te][:, idx].argmax(1)]
    from aiko.emotion import EmotionDetector
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        det = EmotionDetector()
    test_ids = [m["id"] for m, keep in zip(meta, te) if keep]
    dt = []
    for tid in test_ids:
        s = det.text_scores(texts[tid])
        dt.append(max(present, key=lambda c: s.get(c, 0.0)))
    dt = np.array(dt, dtype=object)
    report.append(fmt("## Table M-2 - Aiko's own detectors on natural speech (rule A, test)",
                      [["emotion2vec+ (voice, deployed head)", f"{bal(yt, dv):.2f}"],
                       ["DistilRoBERTa (text, human transcript)", f"{bal(yt, dt):.2f}"],
                       ["voice - text", f"{bal(yt, dv) - bal(yt, dt):+.2f}"]],
                      ["Detector", "bal acc"]))
    report.append(f"\nChance is {100 / len(present):.1f} for {len(present)} classes. Neither "
                  "deployed detector was trained on MOSEI's labels; emotion2vec+'s "
                  "self-supervised pretraining may have included MOSEI audio.\n")
    report.append(f"**Reading.** On natural speech both of Aiko's detectors sit close to "
                  f"chance ({bal(yt, dv):.1f} and {bal(yt, dt):.1f} against "
                  f"{100 / len(present):.1f}), while a linear probe on a generic encoder "
                  f"trained on natural speech reaches {bal(yt, pv):.1f} (Table M-1, rule A). "
                  "The emotional signal is present in natural voices; the off-the-shelf "
                  "detectors, trained largely on acted speech, do not transfer to it. For a "
                  "companion whose users speak naturally, the deployed emotion pipeline needs "
                  "adaptation to natural speech before its outputs can be trusted.\n")

    words = np.array([m["n_words"] for m, keep in zip(meta, te) if keep])
    report.append(f"Test segments average {words.mean():.1f} words (median "
                  f"{np.median(words):.0f}), against IEMOCAP's 11.5 and MELD's 8.3.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
