#!/usr/bin/env python3
"""Do the voice-versus-text conclusions survive a text encoder LARGER than the voice one?

Every voice-minus-text margin in paper/ used a MiniLM-L6 text probe (~22M parameters)
against a WavLM-large voice probe (~316M). This re-runs them with xlm-roberta-large
(~560M), selected and probed under exactly the WavLM protocol: frozen, layer and
regularisation chosen once on MELD dev, linear probes, the same splits.

Claims under test, from clean_spontaneity_results.md (S-4), agreement_results.md (G-4) and
mosei_results.md (M-1):
  1. voice beats text in UNSCRIPTED speech: IEMOCAP improvised; MOSEI natural speech
  2. voice and text TIE in SCRIPTED speech: IEMOCAP scripted; MELD
  3. the improvised margin exceeds the scripted margin within IEMOCAP
If text loses only because MiniLM is small, claim 1 should shrink or vanish here.

CIs: 1000 resamples of dialogues (IEMOCAP, MELD) or videos (MOSEI), paired.

    .venv/bin/python paper/text_encoder_check.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from agreement_check import FAMILY, read_evaluations
from clean_probe_eval import COMMON4, FEATS, bal, fmt, probe
from mosei_eval import label as mosei_label

XLMR_LAYERS = ["L6", "L12", "L16", "L20", "L24", "mean"]
CS = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 1000
OUT = "paper/text_encoder_results.md"


def load(group: str, layer: int, keep_fn) -> dict:
    z = np.load(f"{FEATS}/{group}.npz")
    x = np.load(f"{FEATS}/{group}.xlmr.npz")["xlmr"]
    meta = [json.loads(l) for l in open(f"{FEATS}/{group}.meta.jsonl", encoding="utf-8")]
    keep = np.array([keep_fn(m) for m in meta])
    return {"voice": z["wavlm"][keep][:, layer, :].astype(np.float32),
            "minilm": z["minilm"][keep].astype(np.float32),
            "xlmr_all": x[keep], "meta": [m for m, k in zip(meta, keep) if k]}


def clustered(groups):
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(0)
    return [np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            for _ in range(B_BOOT)]


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    L, Cv, Ct = cfg["layer_index"], cfg["C_voice"], cfg["C_text"]
    c4 = lambda m: m.get("gold") in COMMON4  # noqa: E731

    mtr, mdev, mte = (load(g, L, c4) for g in ("meld_train", "meld_dev", "meld_test"))
    ytr = np.array([m["gold"] for m in mtr["meta"]])
    ydev = np.array([m["gold"] for m in mdev["meta"]])

    # ---- select the XLM-R layer and C once on MELD dev, as WavLM's were -------------
    best, best_s, grid = None, -1.0, []
    for li, ln in enumerate(XLMR_LAYERS):
        for c in CS:
            s = bal(ydev, probe(c).fit(mtr["xlmr_all"][:, li].astype(np.float32), ytr)
                    .predict(mdev["xlmr_all"][:, li].astype(np.float32)))
            grid.append((ln, c, s))
            if s > best_s:
                best, best_s = (li, c), s
    XL, XC = best
    minilm_dev = bal(ydev, probe(Ct).fit(mtr["minilm"], ytr).predict(mdev["minilm"]))
    print(f"XLM-R selected {XLMR_LAYERS[XL]} C={XC} (MELD dev {best_s:.2f}; "
          f"MiniLM {minilm_dev:.2f})", flush=True)
    for d in (mtr, mdev, mte):
        d["xlmr"] = d["xlmr_all"][:, XL].astype(np.float32)

    report = ["# Text-encoder strength check: XLM-RoBERTa-large against MiniLM-L6\n",
              "Voice is WavLM-large (~316M parameters) throughout. Text is MiniLM-L6 (~22M, as "
              "in every earlier table) or xlm-roberta-large (~560M, larger than the voice "
              "encoder). Both text encoders and the voice encoder are frozen, selected once "
              "on MELD dev, and probed with the same linear probes and splits.\n",
              fmt("## Table H-1 - Selection on MELD dev (balanced accuracy)", [
                  ["MiniLM-L6 (~22M)", f"C = {Ct}", f"{minilm_dev:.2f}"],
                  ["XLM-RoBERTa-large (~560M)", f"layer {XLMR_LAYERS[XL]}, C = {XC}",
                   f"{best_s:.2f}"],
                  ["WavLM-large voice (~316M), for reference",
                   f"layer {cfg['layer_name']}, C = {Cv}", "57.36"]],
                  ["Encoder", "selected", "MELD dev bal acc"])]

    # ---- IEMOCAP, leave-one-session-out ------------------------------------------------
    iemo = load("iemocap_all", L, c4)
    iemo["xlmr"] = iemo["xlmr_all"][:, XL].astype(np.float32)
    y = np.array([m["gold"] for m in iemo["meta"]])
    sess = np.array([m["session"] for m in iemo["meta"]])
    kind = np.array([m["kind"] for m in iemo["meta"]])
    grp = np.array([m["group"] for m in iemo["meta"]])
    P = {k: np.empty(len(y), dtype=object) for k in ("voice", "minilm", "xlmr")}
    for s in sorted(set(sess)):
        tr, te = sess != s, sess == s
        for k, c in (("voice", Cv), ("minilm", Ct), ("xlmr", XC)):
            P[k][te] = probe(c).fit(iemo[k][tr], y[tr]).predict(iemo[k][te])

    # unique scripted lines (exact transcript not seen in another session)
    text = {}
    for line in open("paper/.text_backup/iemocap_scores.jsonl", encoding="utf-8"):
        r = json.loads(line)
        text[r["clip"]] = " ".join(re.sub(r"[^a-z0-9' ]+", " ", (r.get("text") or "").lower()).split())
    nt = np.array([text.get(m["clip"], "") for m in iemo["meta"]])
    sess_of = defaultdict(set)
    for t, s in zip(nt, sess):
        if t:
            sess_of[t].add(s)
    unique = np.array([not (t and len(sess_of[t] - {s}) > 0) for t, s in zip(nt, sess)])
    ev = read_evaluations()
    unanim = np.array([sum(1 for r in ev[m["clip"]] if any(FAMILY.get(l) == m["gold"] for l in r)) == 3
                       for m in iemo["meta"]])

    # ---- MELD train -> test ------------------------------------------------------------
    ym = np.array([m["gold"] for m in mte["meta"]])
    gm = np.array([str(m["group"]) for m in mte["meta"]])
    PM = {k: probe(c).fit(mtr[k], ytr).predict(mte[k])
          for k, c in (("voice", Cv), ("minilm", Ct), ("xlmr", XC))}

    # ---- MOSEI train -> test, three labelling rules ----------------------------------
    mos = load("mosei", L, lambda m: True)
    mos["xlmr"] = mos["xlmr_all"][:, XL].astype(np.float32)
    msplit = np.array([m["split"] for m in mos["meta"]])
    mvideo = np.array([m["video"] for m in mos["meta"]])
    full_meta = {}
    for line in open("data/mosei/manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        full_meta[r["id"]] = r
    mos_rows = {}
    for rule in ("A", "B", "C"):
        yy = np.array([mosei_label(full_meta[m["id"]], rule) for m in mos["meta"]], dtype=object)
        tr = (msplit == "train") & (yy != None)  # noqa: E711
        te = (msplit == "test") & (yy != None)  # noqa: E711
        preds = {k: probe(c).fit(mos[k][tr], yy[tr]).predict(mos[k][te])
                 for k, c in (("voice", Cv), ("minilm", Ct), ("xlmr", XC))}
        mos_rows[rule] = (yy[te], preds, mvideo[te])
    print("all predictions done", flush=True)

    # ---- margins ---------------------------------------------------------------------
    conditions = [
        ("IEMOCAP improvised", "unscripted", y, P, grp, kind == "impro"),
        ("IEMOCAP improvised, unanimous", "unscripted", y, P, grp, (kind == "impro") & unanim),
        ("MOSEI natural, rule A", "unscripted") + (None,) * 4,
        ("MOSEI natural, rule B", "unscripted") + (None,) * 4,
        ("MOSEI natural, rule C", "unscripted") + (None,) * 4,
        ("IEMOCAP scripted", "scripted", y, P, grp, kind == "script"),
        ("IEMOCAP scripted, unique lines", "scripted", y, P, grp, (kind == "script") & unique),
        ("IEMOCAP scripted, unanimous", "scripted", y, P, grp, (kind == "script") & unanim),
        ("MELD test (scripted TV)", "scripted", ym, PM, gm, np.ones(len(ym), bool)),
    ]
    rows, verdicts = [], []
    for name, kind_, yy, PP, gg, mask in conditions:
        if name.startswith("MOSEI"):
            yy, PP, gg = mos_rows[name[-1]]
            mask = np.ones(len(yy), bool)
        yy, gg = yy[mask], gg[mask]
        PP = {k: v[mask] for k, v in PP.items()}
        samples = clustered(gg)
        cells = [name, int(mask.sum()), f"{bal(yy, PP['voice']):.2f}",
                 f"{bal(yy, PP['minilm']):.2f}", f"{bal(yy, PP['xlmr']):.2f}"]
        res = {}
        for k in ("minilm", "xlmr"):
            fn = (lambda sel, k=k: bal(yy[sel], PP["voice"][sel]) - bal(yy[sel], PP[k][sel]))
            v = [fn(s) for s in samples]
            d = fn(np.arange(len(yy)))
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            res[k] = (d, lo, hi)
            cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        rows.append(cells)
        d, lo, hi = res["xlmr"]
        if kind_ == "unscripted":
            ok = lo > 0
            verdicts.append([name, "voice beats text", "HOLDS" if ok else
                             ("text wins" if hi < 0 else "becomes a tie")])
        else:
            ok = lo <= 0 <= hi
            verdicts.append([name, "voice ties text", "HOLDS" if ok else
                             ("text wins" if hi < 0 else "voice wins")])
    report.append(fmt("## Table H-2 - Voice minus text with each text encoder (balanced "
                      "accuracy; 95 % CI)", rows,
                      ["Condition", "n", "voice", "text MiniLM", "text XLM-R",
                       "voice - MiniLM", "voice - XLM-R"]))

    # claim 3: improvised margin minus scripted margin within IEMOCAP
    s_, i_ = kind == "script", kind == "impro"
    def diff(k, sel):
        a, b = sel[s_[sel]], sel[i_[sel]]
        return ((bal(y[b], P["voice"][b]) - bal(y[b], P[k][b]))
                - (bal(y[a], P["voice"][a]) - bal(y[a], P[k][a])))
    samples = clustered(grp)
    c3 = []
    for k, nm in (("minilm", "MiniLM"), ("xlmr", "XLM-R")):
        v = [diff(k, s) for s in samples]
        d = diff(k, np.arange(len(y)))
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        c3.append([nm, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
        if k == "xlmr":
            verdicts.append(["IEMOCAP, improvised margin - scripted margin",
                             "positive", "HOLDS" if lo > 0 else "does not hold"])
    report.append(fmt("## Table H-3 - Improvised margin minus scripted margin (IEMOCAP)", c3,
                      ["Text encoder", "difference [95 % CI]"]))

    report.append(fmt("## Table H-4 - Verdict per claim with the larger text encoder", verdicts,
                      ["Condition", "claim", "with XLM-RoBERTa-large"]))
    held = sum(1 for v in verdicts if v[2] == "HOLDS")
    report.append(f"\n{held} of {len(verdicts)} claims hold with a text encoder larger than the "
                  "voice encoder. Any claim that does not is reported as such and must be "
                  "narrowed in the paper accordingly.\n")

    with open("paper/text_encoder_config.json", "w", encoding="utf-8") as f:
        json.dump({"xlmr_layer": XLMR_LAYERS[XL], "C_xlmr": XC, "meld_dev_bal_acc": best_s,
                   "grid": [{"layer": a, "C": b, "bal_acc": c} for a, b, c in grid]}, f, indent=2)
    out = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
