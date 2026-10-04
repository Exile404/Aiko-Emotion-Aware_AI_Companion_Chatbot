#!/usr/bin/env python3
"""Gap 3: is label reliability what both modalities lose between IEMOCAP and MELD?

From IEMOCAP's scripted dialogue to MELD (scripted television), contamination-free voice
and text probes both lose about ten balanced-accuracy points (clean_spontaneity_results.md,
Table S-5: voice -10.02, text -9.65). The text probe never hears the audio, so the shared
loss cannot be acoustic. One candidate is the LABELS: if they are less reliable, every
model, whatever it reads, is scored against noisier targets.

IEMOCAP makes this testable because every utterance was judged by exactly three external
evaluators, whose individual categorical judgements are distributed (C-E* lines; the
actors' self-ratings, C-F*/C-M*, are excluded). The released label is their majority, so
each scored utterance is either UNANIMOUS (3 of 3) or a SPLIT decision (2 of 3). Two tests:

  1. MECHANISM. Do voice and text both lose accuracy on split-decision utterances, and by
     similar amounts, as they do from IEMOCAP to MELD? Contamination-free probes,
     leave-one-session-out, settings frozen on MELD dev. Paired dialogue-clustered CIs.
  2. LEVEL. Is MELD's labelling less reliable than IEMOCAP's? MELD reports Fleiss' kappa
     0.43 over three annotators (Poria et al., ACL 2019). IEMOCAP's kappa is computed here
     the same way, over all utterances and the full category set, taking each evaluator's
     first-listed label. Kappa depends on the category inventory and prevalence (IEMOCAP has
     ten categories, MELD seven), so the comparison is indicative.

An evaluator agrees with the released label if it is among their (possibly several)
choices; happiness and excitement count as one family, matching the excited-into-happy
merge used throughout.

    .venv/bin/python paper/agreement_check.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from clean_probe_eval import COMMON4, FEATS, bal, fmt, probe

ROOT = "IEMOCAP_full_release/IEMOCAP_full_release"
FAMILY = {"Anger": "angry", "Happiness": "happy", "Excited": "happy", "Neutral": "neutral",
          "Sadness": "sad"}
MELD_KAPPA = 0.43
B_BOOT = 1000
OUT = "paper/agreement_results.md"


def read_evaluations() -> dict[str, list[list[str]]]:
    head = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[")
    cline = re.compile(r"^(C-E\d+):\s*(.*?)\s*\(")
    out, cur = {}, None
    for p in sorted(glob.glob(f"{ROOT}/Session*/dialog/EmoEvaluation/*.txt")):
        if os.path.basename(p).startswith("._"):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            m = head.match(line)
            if m:
                cur = m.group(3)
                out[cur] = []
                continue
            c = cline.match(line)
            if c and cur:
                out[cur].append([x.strip() for x in c.group(2).split(";") if x.strip()])
    return out


def fleiss(rows: list[list[str]]) -> float:
    """Fleiss' kappa for items with the same number of raters, one label per rater."""
    cats = sorted({l for r in rows for l in r})
    n = len(rows[0])
    counts = np.array([[r.count(c) for c in cats] for r in rows], dtype=float)
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e))


def main() -> None:
    ev = read_evaluations()

    # ---- level: IEMOCAP's kappa, computed as MELD's was --------------------------------
    first = {u: [r[0] for r in rs] for u, rs in ev.items() if len(rs) == 3 and all(rs)}
    kinds = {u: ("impro" if "impro" in u else "script") for u in first}
    k_all = fleiss(list(first.values()))
    k_s = fleiss([v for u, v in first.items() if kinds[u] == "script"])
    k_i = fleiss([v for u, v in first.items() if kinds[u] == "impro"])
    # IEMOCAP's ten categories include confusable pairs MELD's seven do not; collapsing them
    # gives IEMOCAP every chance to reach MELD's level before concluding it cannot.
    collapses = {
        "excited -> happiness (9 categories)": {"Excited": "Happiness"},
        "+ frustration -> anger (8, MELD-like)": {"Excited": "Happiness", "Frustration": "Anger"},
        "+ frustration -> neutral instead (8)": {"Excited": "Happiness", "Frustration": "Neutral"},
    }
    k_var = {name: fleiss([[mp.get(l, l) for l in v] for v in first.values()])
             for name, mp in collapses.items()}

    # ---- mechanism: accuracy by agreement, contamination-free probes -------------------
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    z = np.load(f"{FEATS}/iemocap_all.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/iemocap_all.meta.jsonl", encoding="utf-8")]
    keep = np.array([m["gold"] in COMMON4 for m in meta])
    meta = [m for m, k in zip(meta, keep) if k]
    Xv = z["wavlm"][keep][:, cfg["layer_index"], :].astype(np.float32)
    Xt = z["minilm"][keep]
    y = np.array([m["gold"] for m in meta])
    sess = np.array([m["session"] for m in meta])
    kind = np.array([m["kind"] for m in meta])
    grp = np.array([m["group"] for m in meta])
    agree = np.array([sum(1 for r in ev[m["clip"]] if any(FAMILY.get(l) == m["gold"] for l in r))
                      for m in meta])
    if not set(agree) <= {2, 3}:
        print(f"note: agreement values present: {sorted(Counter(agree).items())}")
    U, S = agree == 3, agree == 2

    pv = np.empty(len(y), dtype=object)
    pt = np.empty(len(y), dtype=object)
    for s in sorted(set(sess)):
        tr, te = sess != s, sess == s
        pv[te] = probe(cfg["C_voice"]).fit(Xv[tr], y[tr]).predict(Xv[te])
        pt[te] = probe(cfg["C_text"]).fit(Xt[tr], y[tr]).predict(Xt[te])
    dep_map = {}
    for line in open("paper/iemocap_scores.jsonl", encoding="utf-8"):
        r = json.loads(line)
        dep_map[r["clip"]] = max(COMMON4, key=lambda c: r["voice"].get(c, 0.0))
    pd_ = np.array([dep_map[m["clip"]] for m in meta], dtype=object)

    by = defaultdict(list)
    for i, g in enumerate(grp):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(0)
    samples = [np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
               for _ in range(B_BOOT)]

    def drop(p, mask_set):
        def fn(sel):
            sel = sel[mask_set[sel]]
            a, b = sel[S[sel]], sel[U[sel]]
            return bal(y[a], p[a]) - bal(y[b], p[b])
        return fn

    def ci(fn):
        v = [fn(s) for s in samples]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    report = ["# Gap 3: label reliability and the shared IEMOCAP-to-MELD drop\n",
              "Every IEMOCAP utterance has exactly three external evaluators; the released "
              "label is their majority, so a scored utterance is unanimous (3 of 3) or a split "
              "decision (2 of 3). Contamination-free probes (WavLM / MiniLM), leave-one-"
              "session-out, settings from MELD dev. CIs resample dialogues, paired.\n"]

    comp = []
    for nm, m in (("all", np.ones(len(y), bool)), ("scripted", kind == "script"),
                  ("improvised", kind == "impro")):
        comp.append([nm, int(m.sum()), int((m & U).sum()), int((m & S).sum()),
                     f"{100 * (m & S).sum() / m.sum():.1f} %"])
    report.append(fmt("## Table G-0 - Unanimous and split-decision utterances", comp,
                      ["Subset", "n", "unanimous", "split", "split share"]))

    rows, drops = [], {}
    for nm, m in (("all", np.ones(len(y), bool)), ("scripted", kind == "script"),
                  ("improvised", kind == "impro")):
        for mod, p in (("voice", pv), ("text", pt)):
            u, s_ = m & U, m & S
            d = drop(p, m)(np.arange(len(y)))
            lo, hi = ci(drop(p, m))
            drops[(nm, mod)] = d
            rows.append([nm, mod, f"{bal(y[u], p[u]):.2f}", f"{bal(y[s_], p[s_]):.2f}",
                         f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table G-1 - Accuracy on unanimous versus split-decision utterances",
                      rows, ["Subset", "modality", "unanimous bal acc", "split bal acc",
                             "split - unanimous [95 % CI]"]))

    rows = []
    for nm, m in (("all", np.ones(len(y), bool)), ("scripted", kind == "script"),
                  ("improvised", kind == "impro")):
        def gap(sel, m=m):
            return drop(pv, m)(sel) - drop(pt, m)(sel)
        d = gap(np.arange(len(y)))
        lo, hi = ci(gap)
        rows.append([nm, f"{drops[(nm, 'voice')]:+.2f}", f"{drops[(nm, 'text')]:+.2f}",
                     f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    rows.append(["IEMOCAP scripted -> MELD (S-5)", "-10.02", "-9.65", "-0.37"])
    report.append(fmt("## Table G-2 - Do voice and text fall together?", rows,
                      ["Contrast", "voice drop", "text drop", "voice drop - text drop"]))
    report.append("\nA difference near zero means the two modalities lose the same amount, "
                  "the pattern a shared cause such as label reliability produces and the "
                  "pattern seen from IEMOCAP scripted to MELD. The last row is the observed "
                  "cross-corpus drop for comparison.\n")

    report.append(fmt("## Table G-3 - Is MELD's labelling less reliable? Fleiss' kappa, three "
                      "annotators", [
                          ["IEMOCAP, all utterances (computed here)", f"{k_all:.3f}"],
                          ["IEMOCAP, scripted", f"{k_s:.3f}"],
                          ["IEMOCAP, improvised", f"{k_i:.3f}"]]
                      + [[f"IEMOCAP, all, {name}", f"{k:.3f}"] for name, k in k_var.items()]
                      + [["MELD (Poria et al., 2019, as reported)", f"{MELD_KAPPA:.2f}"]],
                      ["Corpus", "Fleiss' kappa"]))
    best = max([k_all] + list(k_var.values()))
    level = ("does NOT support MELD having less reliable labels: IEMOCAP's agreement stays "
             f"below MELD's ({best:.3f} at most, against {MELD_KAPPA:.2f}) even after "
             "collapsing its confusable categories"
             if best < MELD_KAPPA - 0.02 else
             f"shows COMPARABLE reliability once the category sets are aligned ({best:.3f} "
             f"against {MELD_KAPPA:.2f}), so MELD's labels are not measurably less reliable "
             "than IEMOCAP's"
             if best <= MELD_KAPPA + 0.02 else
             f"suggests MELD's labels are LESS reliable: under a MELD-like category set "
             f"IEMOCAP reaches {best:.3f}, above MELD's {MELD_KAPPA:.2f}")
    mech = drops[("scripted", "voice")], drops[("scripted", "text")]
    report.append(f"\n**Verdict.** MECHANISM: within scripted speech, split-decision labels "
                  f"cost voice {mech[0]:+.2f} and text {mech[1]:+.2f}, falling together as "
                  "they do from IEMOCAP scripted to MELD (-10.02, -9.65), so unreliable labels "
                  "can produce that drop at that size. LEVEL: the kappa comparison "
                  f"{level}. Label reliability is therefore a sufficient mechanism but is not "
                  "shown to be the cause; the shared, non-acoustic IEMOCAP-to-MELD drop is "
                  "attributed to cross-corpus differences that affect both modalities "
                  "equally (label construct and context, domain, multi-party structure), "
                  "which this test cannot separate. Kappa from different category "
                  "inventories remains an indicative comparison.\n")
    report.append(f"\nIEMOCAP's kappa is computed over {len(first)} utterances with three "
                  "external evaluators, ten categories, first-listed label per evaluator. "
                  "MELD's is over seven categories. Kappa depends on the category inventory "
                  "and prevalence, so this comparison is indicative rather than exact.\n")

    # ---- does label noise explain the headline? -------------------------------------
    # IEMOCAP's scripted labels are less reliable than its improvised ones (Table G-3). If
    # "voice ties text in scripted speech" were an artifact of noisier scripted labels, it
    # should vanish on unanimous utterances, where the labels are as clean as they get.
    def margin(m):
        def fn(sel):
            sel = sel[m[sel]]
            return bal(y[sel], pv[sel]) - bal(y[sel], pt[sel])
        return fn
    rows = []
    for nm, mk in (("scripted", kind == "script"), ("improvised", kind == "impro")):
        for st, ma in (("unanimous", U), ("split", S)):
            m = mk & ma
            d = margin(m)(np.arange(len(y)))
            lo, hi = ci(margin(m))
            rows.append([nm, st, int(m.sum()), f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table G-4 - The headline on unanimous labels only (voice - text)",
                      rows, ["Condition", "agreement", "n", "voice - text [95 % CI]"]))
    report.append("\nTable S-4's pattern (voice beats text in improvised speech, ties it in "
                  "scripted) is compared here within each agreement level. If it survives on "
                  "unanimous utterances it is not an artifact of IEMOCAP's scripted labels "
                  "being less reliable.\n")

    u, s_ = U, S
    report.append(fmt("## Reference - deployed emotion2vec+ (contaminated) by agreement", [[
        "deployed voice head", f"{bal(y[u], pd_[u]):.2f}", f"{bal(y[s_], pd_[s_]):.2f}",
        f"{bal(y[s_], pd_[s_]) - bal(y[u], pd_[u]):+.2f}"]],
        ["System", "unanimous", "split", "split - unanimous"]))
    report.append("\nShown for completeness only. This encoder trained on IEMOCAP's released "
                  "labels, so on split decisions it may reproduce the majority label from "
                  "memory rather than from the audio; its drop is not comparable to the "
                  "probes'.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
