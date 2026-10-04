#!/usr/bin/env python3
"""Contamination check: is the deployed spontaneity null due to the encoder or the read-out?

Three voice systems are compared on identical utterances and splits:

  deployed  emotion2vec+ with its own zero-shot classification head (what Aiko runs)
  e2v probe emotion2vec+ EMBEDDINGS with a linear probe, the WavLM protocol exactly
  WavLM     frozen WavLM-large L18 with a linear probe (contamination-free)

deployed vs e2v probe differ only in the read-out; e2v probe vs WavLM differ only in the
encoder. So:

  - if the e2v probe shows WavLM's 12-14 point improvised advantage, the deployed null
    came from the zero-shot head, NOT from the encoder having seen IEMOCAP;
  - if the e2v probe is null like the deployed head, the flattening lives in the
    encoder's representation. emotion2vec+ is documented to have trained on IEMOCAP
    labels (via EmoBox), which makes contamination the leading explanation, though an
    emotion-specialised encoder could in principle be flat for other reasons; separating
    those needs the same encoder on an uncontaminated corpus with the same structure
    (MSP-IMPROV).

The e2v probe's regularisation is selected on MELD dev, as WavLM's was. Text, where used,
is the MiniLM probe from clean_probe_eval.py.

    .venv/bin/python paper/e2v_probe_eval.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from clean_probe_eval import (COMMON4, FEATS, bal, boot, boot_unpaired, deployed_preds, fmt,
                              probe)

CS_E2V = [0.0001, 0.001, 0.01, 0.1]
OUT = "paper/e2v_probe_results.md"


def load(name: str, layer: int) -> dict:
    z = np.load(f"{FEATS}/{name}.npz")
    e = np.load(f"{FEATS}/{name}.e2v.npz")
    meta = [json.loads(l) for l in open(f"{FEATS}/{name}.meta.jsonl", encoding="utf-8")]
    if [m["clip"] for m in meta] != list(e["clip"]):
        sys.exit(f"! {name}: e2v rows are not aligned with clean_feats meta")
    keep = np.array([m["gold"] in COMMON4 for m in meta])
    meta = [m for m, k in zip(meta, keep) if k]
    return {
        "e2v": e["e2v"][keep].astype(np.float32),
        "wavlm": z["wavlm"][keep][:, layer, :].astype(np.float32),
        "text": z["minilm"][keep],
        "y": np.array([m["gold"] for m in meta]),
        "g": np.array([str(m["group"]) for m in meta]),
        "meta": meta,
    }


def loso(d: dict, key: str, c: float) -> np.ndarray:
    sess = np.array([m["session"] for m in d["meta"]])
    out = np.empty(len(d["y"]), dtype=object)
    for s in sorted(set(sess)):
        tr, te = sess != s, sess == s
        out[te] = probe(c).fit(d[key][tr], d["y"][tr]).predict(d[key][te])
    return out


def matched_idx(d: dict, kind: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    dur = np.array([m.get("duration") or 0.0 for m in d["meta"]])
    q = np.searchsorted(np.percentile(dur, [25, 50, 75]), dur, side="right")
    rng = np.random.default_rng(seed)
    ms, mi = [], []
    for b in range(4):
        for e in COMMON4:
            s = np.where((q == b) & (d["y"] == e) & (kind == "script"))[0]
            i = np.where((q == b) & (d["y"] == e) & (kind == "impro"))[0]
            n = min(len(s), len(i))
            ms.extend(rng.permutation(s)[:n])
            mi.extend(rng.permutation(i)[:n])
    return np.array(ms), np.array(mi)


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    L = cfg["layer_index"]
    iemo, mtr, mdev, mte = (load(n, L) for n in
                            ("iemocap_all", "meld_train", "meld_dev", "meld_test"))
    print(f"IEMOCAP {len(iemo['y'])} | MELD train {len(mtr['y'])} dev {len(mdev['y'])} "
          f"test {len(mte['y'])}", flush=True)

    # ---- E-1: select the e2v probe's C once on MELD dev -----------------------------
    scores = {c: bal(mdev["y"], probe(c).fit(mtr["e2v"], mtr["y"]).predict(mdev["e2v"]))
              for c in CS_E2V}
    c_e2v = max(scores, key=scores.get)
    print(f"e2v probe C={c_e2v} (MELD dev {scores[c_e2v]:.2f})", flush=True)

    # ---- predictions ------------------------------------------------------------------
    P_i = {"e2v probe": loso(iemo, "e2v", c_e2v),
           "WavLM probe": loso(iemo, "wavlm", cfg["C_voice"]),
           "text": loso(iemo, "text", cfg["C_text"])}
    P_m = {"e2v probe": probe(c_e2v).fit(mtr["e2v"], mtr["y"]).predict(mte["e2v"]),
           "WavLM probe": probe(cfg["C_voice"]).fit(mtr["wavlm"], mtr["y"]).predict(mte["wavlm"]),
           "text": probe(cfg["C_text"]).fit(mtr["text"], mtr["y"]).predict(mte["text"])}
    dep = deployed_preds()["voice"]
    P_i["deployed head"] = np.array([dep[m["clip"]] for m in iemo["meta"]], dtype=object)
    P_m["deployed head"] = np.array([dep[m["clip"]] for m in mte["meta"]], dtype=object)
    print("predictions done", flush=True)

    kind = np.array([m["kind"] for m in iemo["meta"]])
    S, I = kind == "script", kind == "impro"
    y, g = iemo["y"], iemo["g"]
    systems = ["deployed head", "e2v probe", "WavLM probe"]

    report = ["# Contamination check: emotion2vec+ embeddings under the clean-probe protocol\n",
              "Voice systems compared on identical utterances and splits (IEMOCAP "
              "leave-one-session-out, MELD train -> test, common four classes). "
              "`deployed head` and `e2v probe` share the encoder and differ in read-out; "
              "`e2v probe` and `WavLM probe` share the read-out and differ in encoder.\n"]
    report.append(fmt("## Table E-1 - e2v probe regularisation, selected on MELD dev",
                      [[f"C = {c}", f"{s:.2f}", "selected" if c == c_e2v else ""]
                       for c, s in scores.items()], ["Setting", "MELD dev bal acc", ""]))

    # ---- E-2: voice by condition -------------------------------------------------------
    rows = []
    for sysname in systems:
        pi, pm = P_i[sysname], P_m[sysname]
        rows.append([sysname, f"{bal(y[S], pi[S]):.2f}", f"{bal(y[I], pi[I]):.2f}",
                     f"{bal(y, pi):.2f}", f"{bal(mte['y'], pm):.2f}"])
    report.append(fmt("## Table E-2 - Voice balanced accuracy by condition", rows,
                      ["Voice system", "IEMOCAP scripted", "IEMOCAP improvised",
                       "IEMOCAP all", "MELD test"]))

    # ---- E-3: the spontaneity contrast, per system -------------------------------------
    ms, mi = matched_idx(iemo, kind)
    rows = []
    for sysname in systems:
        p = P_i[sysname]
        d, lo, hi = boot_unpaired(y[S], p[S], g[S], y[I], p[I], g[I])
        dm, lom, him = boot_unpaired(y[ms], p[ms], g[ms], y[mi], p[mi], g[mi])
        rows.append([sysname, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                     f"{dm:+.2f} [{lom:+.2f}, {him:+.2f}]"])
    report.append(fmt("## Table E-3 - Scripted minus improvised, voice (95 % CI)", rows,
                      ["Voice system", "all utterances", "duration x emotion matched"]))
    report.append("\nDialogue-clustered bootstrap, 1000 resamples; scripted and improvised "
                  "dialogues resampled independently. Negative = improvised is easier.\n")

    # ---- E-4: voice minus text, per condition ------------------------------------------
    rows = []
    for sysname in ("e2v probe", "WavLM probe"):
        pi, pm = P_i[sysname], P_m[sysname]
        cells = []
        for m in (S, I):
            d, lo, hi = boot(y[m], pi[m], P_i["text"][m], g[m])
            cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        d, lo, hi = boot(mte["y"], pm, P_m["text"], mte["g"])
        cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        rows.append([sysname] + cells)
    report.append(fmt("## Table E-4 - Voice minus text (MiniLM probe), by condition", rows,
                      ["Voice system", "IEMOCAP scripted", "IEMOCAP improvised",
                       "MELD test"]))
    report.append("\nPaired dialogue-clustered bootstrap. The WavLM row reproduces Table S-4. "
                  "If the improvised-only voice advantage reappears with emotion2vec+ "
                  "embeddings, it is not a WavLM artifact either.\n")

    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
