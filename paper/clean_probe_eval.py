#!/usr/bin/env python3
"""Contamination-free replication of the C1 boundary condition and the modality inversion.

Every headline in iemocap_results.md rests on two encoders that have seen the test
corpora: emotion2vec+ (fine-tuned via EmoBox, which contains IEMOCAP, MELD and CREMA-D)
and DistilRoBERTa (trained on MELD). This script asks whether the SHAPE of those results
survives when neither encoder has seen any emotion label:

    voice: frozen WavLM-large + linear probe     text: frozen MiniLM + linear probe

Probes are trained only on each corpus's training partition and scored on data they
never saw: MELD official train -> test; IEMOCAP leave-one-session-out; CREMA-D 5-fold
grouped by actor. All hyperparameters (WavLM layer, regularisation) are chosen ONCE on
MELD dev and then frozen for every corpus, so no IEMOCAP or CREMA-D test fold informs
any choice. A per-layer sensitivity table shows the conclusions do not hinge on that
choice. Probes use balanced class weights because balanced accuracy is the metric and the
corpora's priors differ sharply.

What a probe result means, and does not. A probe trained in-domain measures how much
emotion information the AUDIO (or text) of a corpus carries in a form a generic encoder
exposes. It is not the deployed zero-shot setting, so the two are reported side by side:
the gap between them bounds what contamination could have contributed.

Tables (all on the four classes common to every corpus):
  K-1  hyperparameters selected on MELD dev
  K-2  the ladder, clean probe beside deployed zero-shot
  K-3  the voice/text inversion, clean, with dialogue-clustered CIs
  K-4  script against improvised within IEMOCAP, clean
  K-5  transfer: the whole pipeline (probes + fusion) trained on one corpus, run on the other
  K-6  WavLM layer sensitivity

    .venv/bin/python paper/clean_probe_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMMON4 = ["angry", "happy", "neutral", "sad"]          # alphabetical == predict_proba order
LAYER_NAMES = ["L3", "L6", "L9", "L12", "L15", "L18", "L21", "L24", "mean"]
CS_VOICE = [0.001, 0.01, 0.1]
CS_TEXT = [0.1, 1.0, 10.0]
B_BOOT = 1000
FEATS = "paper/clean_feats"
OUT = "paper/clean_probe_results.md"


# ----------------------------------------------------------------------------- data
class Set:
    """Common-four utterances of one feature file, with per-row metadata."""

    def __init__(self, name: str):
        z = np.load(f"{FEATS}/{name}.npz")
        meta = [json.loads(l) for l in open(f"{FEATS}/{name}.meta.jsonl", encoding="utf-8")]
        keep = np.array([m["gold"] in COMMON4 for m in meta])
        self.name = name
        self.wavlm = z["wavlm"][keep]                   # (N, 9, 1024) float16
        self.text = z["minilm"][keep]                   # (N, 384)
        self.meta = [m for m, k in zip(meta, keep) if k]
        self.y = np.array([m["gold"] for m in self.meta])
        self.groups = np.array([str(m["group"]) for m in self.meta])

    def voice(self, layer: int) -> np.ndarray:
        return self.wavlm[:, layer, :].astype(np.float32)

    def subset(self, mask: np.ndarray) -> "Set":
        s = object.__new__(Set)
        s.name, s.wavlm, s.text = self.name, self.wavlm[mask], self.text[mask]
        s.meta = [m for m, k in zip(self.meta, mask) if k]
        s.y, s.groups = self.y[mask], self.groups[mask]
        return s

    def __len__(self) -> int:
        return len(self.y)


def concat(a: Set, b: Set) -> Set:
    s = object.__new__(Set)
    s.name = f"{a.name}+{b.name}"
    s.wavlm = np.concatenate([a.wavlm, b.wavlm])
    s.text = np.concatenate([a.text, b.text])
    s.meta = a.meta + b.meta
    s.y = np.concatenate([a.y, b.y])
    s.groups = np.concatenate([a.groups, b.groups])
    return s


# -------------------------------------------------------------------------- helpers
def probe(c: float):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))


def bal(y, p) -> float:
    return 100 * balanced_accuracy_score(y, p)


def macro(y, p) -> float:
    return 100 * f1_score(y, p, labels=COMMON4, average="macro", zero_division=0)


def argmax4(proba: np.ndarray) -> np.ndarray:
    return np.array(COMMON4)[proba.argmax(1)]


def boot(y, pa, pb, groups, b=B_BOOT, seed=0) -> tuple[float, float, float]:
    """Dialogue-clustered paired bootstrap of balanced accuracy, A minus B."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(b):
        sel = np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        d.append(bal(y[sel], pa[sel]) - bal(y[sel], pb[sel]))
    return bal(y, pa) - bal(y, pb), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_unpaired(y1, p1, g1, y2, p2, g2, b=B_BOOT, seed=0) -> tuple[float, float, float]:
    """Balanced-accuracy difference between two disjoint sets of dialogues."""
    def idx(groups):
        by = defaultdict(list)
        for i, g in enumerate(groups):
            by[g].append(i)
        return list(by.values())
    c1, c2 = idx(g1), idx(g2)
    y1, p1, y2, p2 = map(np.asarray, (y1, p1, y2, p2))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(b):
        s1 = np.concatenate([c1[k] for k in rng.integers(0, len(c1), len(c1))])
        s2 = np.concatenate([c2[k] for k in rng.integers(0, len(c2), len(c2))])
        d.append(bal(y1[s1], p1[s1]) - bal(y2[s2], p2[s2]))
    return bal(y1, p1) - bal(y2, p2), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------- pipeline
class Pipe:
    """Voice probe + text probe + learned fusion over their probabilities.

    Fusion is trained on OUT-OF-FOLD probe probabilities (grouped by dialogue), never on
    the probes' in-sample outputs, which would be overconfident and teach the fusion head
    to trust whichever probe overfits most.
    """

    def __init__(self, layer: int, cv: float, ct: float):
        self.layer, self.cv, self.ct = layer, cv, ct

    def fit(self, tr: Set) -> "Pipe":
        Xv, Xt, y, g = tr.voice(self.layer), tr.text, tr.y, tr.groups
        ov, ot = np.zeros((len(y), 4)), np.zeros((len(y), 4))
        for a, b in GroupKFold(n_splits=5).split(Xv, y, g):
            ov[b] = probe(self.cv).fit(Xv[a], y[a]).predict_proba(Xv[b])
            ot[b] = probe(self.ct).fit(Xt[a], y[a]).predict_proba(Xt[b])
        self.pv = probe(self.cv).fit(Xv, y)
        self.pt = probe(self.ct).fit(Xt, y)
        self.fusion = HistGradientBoostingClassifier(random_state=0).fit(np.hstack([ov, ot]), y)
        return self

    def predict(self, te: Set) -> dict[str, np.ndarray]:
        qv = self.pv.predict_proba(te.voice(self.layer))
        qt = self.pt.predict_proba(te.text)
        return {"voice": argmax4(qv), "text": argmax4(qt),
                "fusion": self.fusion.predict(np.hstack([qv, qt]))}


def loso(iemo: Set, cfg: tuple, extra: Set | None = None) -> dict[str, np.ndarray]:
    sessions = np.array([m["session"] for m in iemo.meta])
    out = {k: np.empty(len(iemo), dtype=object) for k in ("voice", "text", "fusion")}
    for s in sorted(set(sessions)):
        tr = iemo.subset(sessions != s)
        if extra is not None:
            tr = concat(tr, extra)
        pred = Pipe(*cfg).fit(tr).predict(iemo.subset(sessions == s))
        for k in out:
            out[k][sessions == s] = pred[k]
    return out


def crema_cv(crema: Set, layer: int, cv: float, ct: float) -> dict[str, np.ndarray]:
    out = {"voice": np.empty(len(crema), dtype=object), "text": np.empty(len(crema), dtype=object)}
    for a, b in GroupKFold(n_splits=5).split(crema.text, crema.y, crema.groups):
        out["voice"][b] = probe(cv).fit(crema.voice(layer)[a], crema.y[a]).predict(
            crema.voice(layer)[b])
        out["text"][b] = probe(ct).fit(crema.text[a], crema.y[a]).predict(crema.text[b])
    return out


# ---------------------------------------------------------------- deployed reference
def deployed_preds() -> dict[str, dict[str, str]]:
    """Zero-shot deployed predictions (argmax restricted to the common four), by clip."""
    def am(s):
        return max(COMMON4, key=lambda e: (s or {}).get(e, 0.0))
    out: dict[str, dict[str, str]] = {"voice": {}, "text": {}}
    for r in (json.loads(l) for l in open("paper/crema_scores.jsonl", encoding="utf-8")):
        out["voice"][r["clip"]] = am(r["voice"])
    for r in (json.loads(l) for l in open("paper/iemocap_scores.jsonl", encoding="utf-8")):
        out["voice"][r["clip"]] = am(r["voice"])
        if r.get("text_scores"):
            out["text"][r["clip"]] = am(r["text_scores"])
    for r in (json.loads(l) for l in open("paper/meld_test_scores.jsonl", encoding="utf-8")):
        clip = f"test_dia{r['dia']}_utt{r['utt']}"
        out["voice"][clip] = am(r["voice"])
        out["text"][clip] = am(r["text_scores"])
    return out


# ------------------------------------------------------------------------------ main
def main() -> None:
    crema, iemo = Set("crema_all"), Set("iemocap_all")
    mtr, mdev, mte = Set("meld_train"), Set("meld_dev"), Set("meld_test")
    print(f"common-four: CREMA-D {len(crema)} | IEMOCAP {len(iemo)} | MELD train {len(mtr)} "
          f"dev {len(mdev)} test {len(mte)}", flush=True)
    report = ["# Contamination-free replication: frozen WavLM and MiniLM with linear probes\n"]

    # ---- K-1: select once on MELD dev ----------------------------------------------
    best_v, best_vs = None, -1.0
    for li in range(len(LAYER_NAMES)):
        for c in CS_VOICE:
            s = bal(mdev.y, probe(c).fit(mtr.voice(li), mtr.y).predict(mdev.voice(li)))
            if s > best_vs:
                best_v, best_vs = (li, c), s
    best_t, best_ts = None, -1.0
    for c in CS_TEXT:
        s = bal(mdev.y, probe(c).fit(mtr.text, mtr.y).predict(mdev.text))
        if s > best_ts:
            best_t, best_ts = c, s
    layer, cv, ct = best_v[0], best_v[1], best_t
    cfg = (layer, cv, ct)
    print(f"selected on MELD dev: voice {LAYER_NAMES[layer]} C={cv} ({best_vs:.2f}), "
          f"text C={ct} ({best_ts:.2f})", flush=True)
    with open("paper/clean_probe_config.json", "w", encoding="utf-8") as f:
        json.dump({"layer_index": layer, "layer_name": LAYER_NAMES[layer], "C_voice": cv,
                   "C_text": ct, "selected_on": "MELD dev, balanced accuracy"}, f, indent=2)
    report.append(fmt("## Table K-1 - Hyperparameters, selected once on MELD dev", [
        ["voice (WavLM-large)", f"layer {LAYER_NAMES[layer]}, C = {cv}", f"{best_vs:.2f}"],
        ["text (MiniLM-L6)", f"C = {ct}", f"{best_ts:.2f}"]],
        ["Probe", "Selected", "MELD dev bal acc"]))
    report.append("\nSearched: every stored WavLM layer x C in "
                  f"{CS_VOICE} for voice; C in {CS_TEXT} for text. These choices are then "
                  "frozen for every corpus.\n")

    # ---- predictions ----------------------------------------------------------------
    p_crema = crema_cv(crema, layer, cv, ct)
    p_iemo = loso(iemo, cfg)
    meld_pipe = Pipe(*cfg).fit(mtr)
    p_mte = meld_pipe.predict(mte)
    print("in-domain predictions done", flush=True)

    dep = deployed_preds()
    kind = np.array([m.get("kind") for m in iemo.meta])
    clips = {"crema": [m["clip"] for m in crema.meta], "iemo": [m["clip"] for m in iemo.meta],
             "meld": [m["clip"] for m in mte.meta]}

    # ---- K-2: the ladder ------------------------------------------------------------
    def dep_bal(clip_list, y, mod="voice"):
        pairs = [(yy, dep[mod].get(c)) for c, yy in zip(clip_list, y)]
        pairs = [(a, b) for a, b in pairs if b is not None]
        return bal([a for a, _ in pairs], [b for _, b in pairs]) if pairs else float("nan")

    ladder = [
        ("CREMA-D (actor-grouped 5-fold)", crema.y, p_crema["voice"], clips["crema"]),
        ("IEMOCAP script (LOSO)", iemo.y[kind == "script"], p_iemo["voice"][kind == "script"],
         np.array(clips["iemo"])[kind == "script"]),
        ("IEMOCAP impro (LOSO)", iemo.y[kind == "impro"], p_iemo["voice"][kind == "impro"],
         np.array(clips["iemo"])[kind == "impro"]),
        ("IEMOCAP all (LOSO)", iemo.y, p_iemo["voice"], clips["iemo"]),
        ("MELD test (train -> test)", mte.y, p_mte["voice"], clips["meld"]),
    ]
    rows = []
    for name, y, p, cl in ladder:
        c = bal(y, p)
        d = dep_bal(list(cl), y)
        rows.append([name, len(y), f"{c:.2f}", f"{macro(y, p):.2f}", f"{d:.2f}", f"{d - c:+.2f}"])
    report.append(fmt("## Table K-2 - Voice across the ladder: clean probe beside deployed "
                      "zero-shot", rows,
                      ["Condition", "n", "clean bal acc", "clean macro F1",
                       "deployed bal acc", "deployed - clean"]))
    report.append("\n`deployed - clean` is NOT a contamination estimate by itself: the "
                  "deployed model is zero-shot while the probe trains in-domain, which "
                  "favours the probe, and emotion2vec+ is a stronger audio model than a "
                  "linear probe on WavLM, which favours the deployed model. What survives "
                  "contamination is the ORDERING of the rungs under the clean encoder.\n")

    # ---- K-3: the inversion ---------------------------------------------------------
    inv = []
    for name, y, pv, pt, grp, cl in (
            ("IEMOCAP (LOSO)", iemo.y, p_iemo["voice"], p_iemo["text"], iemo.groups,
             clips["iemo"]),
            ("MELD test", mte.y, p_mte["voice"], p_mte["text"], mte.groups, clips["meld"])):
        d, lo, hi = boot(y, pv, pt, grp)
        dv, dt = dep_bal(list(cl), y, "voice"), dep_bal(list(cl), y, "text")
        inv.append([name, len(y), f"{bal(y, pv):.2f}", f"{bal(y, pt):.2f}",
                    f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{dv - dt:+.2f}"])
    report.append(fmt("## Table K-3 - Voice against text, clean encoders", inv,
                      ["Corpus", "n", "clean voice bal acc", "clean text bal acc",
                       "clean voice - text [95 % CI]", "deployed voice - text"]))
    report.append("\nCIs are dialogue-clustered paired bootstraps. If the inversion were an "
                  "artifact of each deployed encoder having memorised a different corpus, the "
                  "clean margins would not change sign between the corpora.\n")

    # ---- K-4: spontaneity -----------------------------------------------------------
    s_, i_ = kind == "script", kind == "impro"
    d, lo, hi = boot_unpaired(iemo.y[s_], p_iemo["voice"][s_], iemo.groups[s_],
                              iemo.y[i_], p_iemo["voice"][i_], iemo.groups[i_])
    dt_, lot, hit = boot_unpaired(iemo.y[s_], p_iemo["text"][s_], iemo.groups[s_],
                                  iemo.y[i_], p_iemo["text"][i_], iemo.groups[i_])
    report.append(fmt("## Table K-4 - Script against improvised within IEMOCAP, clean", [
        ["voice", f"{bal(iemo.y[s_], p_iemo['voice'][s_]):.2f}",
         f"{bal(iemo.y[i_], p_iemo['voice'][i_]):.2f}", f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"],
        ["text", f"{bal(iemo.y[s_], p_iemo['text'][s_]):.2f}",
         f"{bal(iemo.y[i_], p_iemo['text'][i_]):.2f}", f"{dt_:+.2f} [{lot:+.2f}, {hit:+.2f}]"]],
        ["Modality", "script bal acc", "impro bal acc", "script - impro [95 % CI]"]))
    report.append("\nDialogue-clustered bootstrap, script and improvised dialogues resampled "
                  "independently.\n")

    # ---- K-5: transfer --------------------------------------------------------------
    iemo_pipe = Pipe(*cfg).fit(iemo)
    p_m2i = meld_pipe.predict(iemo)
    p_i2m = iemo_pipe.predict(mte)
    p_both_m = Pipe(*cfg).fit(concat(mtr, iemo)).predict(mte)
    p_both_i = loso(iemo, cfg, extra=mtr)
    print("transfer predictions done", flush=True)
    trows, tests = [], []
    for comp in ("voice", "text", "fusion"):
        trows.append([comp,
                      f"{bal(mte.y, p_mte[comp]):.2f}", f"{bal(mte.y, p_i2m[comp]):.2f}",
                      f"{bal(mte.y, p_both_m[comp]):.2f}",
                      f"{bal(iemo.y, p_iemo[comp]):.2f}", f"{bal(iemo.y, p_m2i[comp]):.2f}",
                      f"{bal(iemo.y, p_both_i[comp]):.2f}"])
        tests.append((comp, boot(mte.y, p_i2m[comp], p_mte[comp], mte.groups),
                      boot(iemo.y, p_m2i[comp], p_iemo[comp], iemo.groups)))
    report.append(fmt("## Table K-5 - Whole pipeline trained on one corpus, run on the other "
                      "(clean, balanced accuracy)", trows,
                      ["Component", "MELD: in-domain", "MELD: from IEMOCAP", "MELD: from both",
                       "IEMOCAP: in-domain", "IEMOCAP: from MELD", "IEMOCAP: from both"]))
    lines = ["\nCross-corpus minus in-domain, dialogue-clustered paired bootstrap:"]
    for comp, (a, alo, ahi), (b, blo, bhi) in tests:
        lines.append(f"- {comp}: on MELD {a:+.2f} [{alo:+.2f}, {ahi:+.2f}]; "
                     f"on IEMOCAP {b:+.2f} [{blo:+.2f}, {bhi:+.2f}]")
    report.append("\n".join(lines) + "\n")

    # ---- K-6: layer sensitivity -----------------------------------------------------
    sessions = np.array([m["session"] for m in iemo.meta])
    lrows = []
    for li, ln in enumerate(LAYER_NAMES):
        pi = np.empty(len(iemo), dtype=object)
        for s in sorted(set(sessions)):
            tr, te = sessions != s, sessions == s
            pi[te] = probe(cv).fit(iemo.voice(li)[tr], iemo.y[tr]).predict(iemo.voice(li)[te])
        pm = probe(cv).fit(mtr.voice(li), mtr.y).predict(mte.voice(li))
        lrows.append([ln, f"{bal(iemo.y, pi):.2f}", f"{bal(mte.y, pm):.2f}",
                      f"{bal(iemo.y, pi) - bal(mte.y, pm):+.2f}"])
        print(f"  layer {ln} done", flush=True)
    report.append(fmt("## Table K-6 - WavLM layer sensitivity (voice probe, C fixed)", lrows,
                      ["Layer", "IEMOCAP LOSO bal acc", "MELD test bal acc",
                       "IEMOCAP - MELD"]))
    report.append("\nIf the clean-against-wild gap held only at the selected layer it would be "
                  "a selection artifact; it should hold across layers.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
