#!/usr/bin/env python3
"""Stress-test the clean-encoder spontaneity effect (Table K-4) before it is reported.

Under the deployed encoder, script against improvised was null (iemocap_results.md,
Tables C1-12/13). Under the contamination-free WavLM probe it becomes a 12-point voice
advantage for IMPROVISED speech, and a 13-point text advantage for SCRIPTED speech.
A reversal that large needs the same controls the deployed analysis had, plus one the
deployed analysis did not need:

  S-1  paired within-speaker, within-emotion cells (Wilcoxon), as in Table C1-12
  S-2  duration-quartile x emotion matched subsample, as in Table C1-13
  S-3  SCRIPT-LINE REPETITION. IEMOCAP's scripts are performed in every session, so
       under leave-one-session-out a held-out scripted line has usually been seen,
       verbatim, in the training sessions. A text probe can then memorise line ->
       label rather than read emotion from language. This splits scripted utterances
       by whether their exact normalised transcript occurs in another session.

Transcripts come from the local-only paper/.text_backup/; only aggregates are written.

    .venv/bin/python paper/clean_spontaneity_check.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMMON4 = ["angry", "happy", "neutral", "sad"]
OUT = "paper/clean_spontaneity_results.md"
B_BOOT = 1000


def bal(y, p) -> float:
    return 100 * balanced_accuracy_score(y, p)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower()).split())


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def boot_unpaired(y1, p1, g1, y2, p2, g2, b=B_BOOT, seed=0):
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


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    z = np.load("paper/clean_feats/iemocap_all.npz")
    meta = [json.loads(l) for l in open("paper/clean_feats/iemocap_all.meta.jsonl",
                                        encoding="utf-8")]
    keep = np.array([m["gold"] in COMMON4 for m in meta])
    meta = [m for m, k in zip(meta, keep) if k]
    Xv = z["wavlm"][keep][:, cfg["layer_index"], :].astype(np.float32)
    Xt = z["minilm"][keep]
    y = np.array([m["gold"] for m in meta])
    sess = np.array([m["session"] for m in meta])
    kind = np.array([m["kind"] for m in meta])
    spk = np.array([m["speaker"] for m in meta])
    grp = np.array([m["group"] for m in meta])
    dur = np.array([m.get("duration") or 0.0 for m in meta])
    text = {json.loads(l)["clip"]: json.loads(l).get("text") or ""
            for l in open("paper/.text_backup/iemocap_scores.jsonl", encoding="utf-8")}
    nt = np.array([norm(text.get(m["clip"], "")) for m in meta])

    def probe(c):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced"))

    pv = np.empty(len(y), dtype=object)
    pt = np.empty(len(y), dtype=object)
    for s in sorted(set(sess)):
        tr, te = sess != s, sess == s
        pv[te] = probe(cfg["C_voice"]).fit(Xv[tr], y[tr]).predict(Xv[te])
        pt[te] = probe(cfg["C_text"]).fit(Xt[tr], y[tr]).predict(Xt[te])
        print(f"  session {s} done", flush=True)

    report = ["# Stress test of the clean-encoder spontaneity effect (IEMOCAP, LOSO)\n",
              f"WavLM {cfg['layer_name']} / MiniLM probes, settings frozen on MELD dev, "
              "identical to clean_probe_eval.py.\n"]

    # ---- S-1 paired cells -----------------------------------------------------------
    rows = []
    for name, p in (("voice", pv), ("text", pt)):
        cells = []
        for sp in sorted(set(spk)):
            for e in COMMON4:
                a = (spk == sp) & (y == e) & (kind == "script")
                b = (spk == sp) & (y == e) & (kind == "impro")
                if a.sum() >= 5 and b.sum() >= 5:
                    cells.append((100 * np.mean(p[a] == y[a]), 100 * np.mean(p[b] == y[b])))
        c = np.array(cells)
        w, pval = stats.wilcoxon(c[:, 0], c[:, 1])
        d = c[:, 0] - c[:, 1]
        rows.append([name, len(c), f"{d.mean():+.2f}", f"{int((d > 0).sum())}/{len(d)}",
                     f"{w:.0f}", f"{pval:.3g}"])
    report.append(fmt("## Table S-1 - Paired within-speaker, within-emotion cells", rows,
                      ["Modality", "cells", "mean script - impro", "script higher",
                       "Wilcoxon W", "p"]))

    # ---- S-2 duration x emotion matched ----------------------------------------------
    edges = np.percentile(dur, [25, 50, 75])
    q = np.searchsorted(edges, dur, side="right")
    rng = np.random.default_rng(0)
    ms, mi = [], []
    for b_ in range(4):
        for e in COMMON4:
            s_idx = np.where((q == b_) & (y == e) & (kind == "script"))[0]
            i_idx = np.where((q == b_) & (y == e) & (kind == "impro"))[0]
            n = min(len(s_idx), len(i_idx))
            ms.extend(rng.permutation(s_idx)[:n])
            mi.extend(rng.permutation(i_idx)[:n])
    ms, mi = np.array(ms), np.array(mi)
    rows = []
    for name, p in (("voice", pv), ("text", pt)):
        d, lo, hi = boot_unpaired(y[ms], p[ms], grp[ms], y[mi], p[mi], grp[mi])
        rows.append([name, len(ms), f"{bal(y[ms], p[ms]):.2f}", f"{bal(y[mi], p[mi]):.2f}",
                     f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table S-2 - Matched on duration quartile and emotion", rows,
                      ["Modality", "n per side", "script bal acc", "impro bal acc",
                       "script - impro [95 % CI]"]))

    # ---- S-3 script-line repetition --------------------------------------------------
    by_text = defaultdict(set)
    for t, s in zip(nt, sess):
        if t:
            by_text[t].add(s)
    seen = np.array([bool(t) and len(by_text[t] - {s}) > 0 for t, s in zip(nt, sess)])
    rows = []
    for k in ("script", "impro"):
        m = kind == k
        rows.append([k, int(m.sum()), f"{100 * seen[m].mean():.1f} %"])
    report.append(fmt("## Table S-3a - Utterances whose exact transcript occurs in another "
                      "session", rows, ["kind", "n", "transcript seen in another session"]))
    rows = []
    for label, m in (("script, line seen elsewhere", (kind == "script") & seen),
                     ("script, line unique", (kind == "script") & ~seen),
                     ("impro (all)", kind == "impro")):
        if m.sum() < 20:
            continue
        rows.append([label, int(m.sum()), f"{bal(y[m], pt[m]):.2f}", f"{bal(y[m], pv[m]):.2f}"])
    report.append(fmt("## Table S-3b - Accuracy by whether the line was seen in training",
                      rows, ["subset", "n", "text bal acc", "voice bal acc"]))
    report.append("\nUnder leave-one-session-out, a 'seen elsewhere' scripted line appeared "
                  "verbatim in the training sessions. If the text probe's scripted advantage "
                  "is concentrated there, it is memorised script content, not emotion read "
                  "from language, and the text side of Table K-4 must not be reported as a "
                  "property of scripted speech. Voice is reported alongside: a voice probe "
                  "cannot memorise a line it hears in a different speaker's voice nearly as "
                  "easily, so it acts as a control.\n")

    # ---- S-4 the voice-text margin by condition, IEMOCAP and MELD ---------------------
    def load_set(name):
        zz = np.load(f"paper/clean_feats/{name}.npz")
        mm = [json.loads(l) for l in open(f"paper/clean_feats/{name}.meta.jsonl",
                                          encoding="utf-8")]
        kk = np.array([m["gold"] in COMMON4 for m in mm])
        mm = [m for m, k in zip(mm, kk) if k]
        return (zz["wavlm"][kk][:, cfg["layer_index"], :].astype(np.float32), zz["minilm"][kk],
                np.array([m["gold"] for m in mm]), np.array([str(m["group"]) for m in mm]))
    mv_tr, mt_tr, my_tr, _ = load_set("meld_train")
    mv_te, mt_te, my_te, mg_te = load_set("meld_test")
    mpv = probe(cfg["C_voice"]).fit(mv_tr, my_tr).predict(mv_te)
    mpt = probe(cfg["C_text"]).fit(mt_tr, my_tr).predict(mt_te)

    def paired(yy, a, b, gg, bb=B_BOOT, seed=0):
        """Dialogue-clustered bootstrap of bal(a) - bal(b) on the same utterances."""
        by = defaultdict(list)
        for i, g in enumerate(gg):
            by[g].append(i)
        keys = list(by)
        r = np.random.default_rng(seed)
        d = []
        for _ in range(bb):
            sel = np.concatenate([by[keys[k]] for k in r.integers(0, len(keys), len(keys))])
            d.append(bal(yy[sel], a[sel]) - bal(yy[sel], b[sel]))
        return bal(yy, a) - bal(yy, b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    conds = [("IEMOCAP scripted (clean audio)", kind == "script"),
             ("IEMOCAP scripted, unique lines only", (kind == "script") & ~seen),
             ("IEMOCAP improvised (clean audio)", kind == "impro")]
    rows = []
    for label, m in conds:
        d, lo, hi = paired(y[m], pv[m], pt[m], grp[m])
        rows.append([label, int(m.sum()), f"{bal(y[m], pv[m]):.2f}", f"{bal(y[m], pt[m]):.2f}",
                     f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    d, lo, hi = paired(my_te, mpv, mpt, mg_te)
    rows.append(["MELD test (scripted TV, broadcast audio)", len(my_te),
                 f"{bal(my_te, mpv):.2f}", f"{bal(my_te, mpt):.2f}",
                 f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table S-4 - Voice minus text, by condition (clean encoders)", rows,
                      ["Condition", "n", "voice bal acc", "text bal acc",
                       "voice - text [95 % CI]"]))

    # difference of margins, improvised minus scripted, dialogues resampled per side
    s_, i_ = kind == "script", kind == "impro"

    def margin_diff(bb=B_BOOT, seed=0):
        def idx(m):
            by = defaultdict(list)
            for i in np.where(m)[0]:
                by[grp[i]].append(i)
            return list(by.values())
        cs, ci = idx(s_), idx(i_)
        r = np.random.default_rng(seed)
        d = []
        for _ in range(bb):
            a = np.concatenate([cs[k] for k in r.integers(0, len(cs), len(cs))])
            b = np.concatenate([ci[k] for k in r.integers(0, len(ci), len(ci))])
            d.append((bal(y[b], pv[b]) - bal(y[b], pt[b])) - (bal(y[a], pv[a]) - bal(y[a], pt[a])))
        point = (bal(y[i_], pv[i_]) - bal(y[i_], pt[i_])) - (bal(y[s_], pv[s_]) - bal(y[s_], pt[s_]))
        return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    md, mlo, mhi = margin_diff()
    report.append(f"\nImprovised margin minus scripted margin, within IEMOCAP (same actors, "
                  f"same recording chain): {md:+.2f} points, 95 % CI [{mlo:+.2f}, {mhi:+.2f}].\n")

    # ---- S-5 do both modalities fall together from IEMOCAP scripted to MELD? ---------
    rows = []
    for name, ip, mp in (("voice", pv, mpv), ("text", pt, mpt)):
        d, lo, hi = boot_unpaired(my_te, mp, mg_te, y[s_], ip[s_], grp[s_])
        rows.append([name, f"{bal(y[s_], ip[s_]):.2f}", f"{bal(my_te, mp):.2f}",
                     f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    report.append(fmt("## Table S-5 - IEMOCAP scripted to MELD, each modality", rows,
                      ["Modality", "IEMOCAP scripted", "MELD test",
                       "MELD - IEMOCAP scripted [95 % CI]"]))
    report.append("\nBoth conditions are scripted performance; they differ in recording "
                  "conditions, multi-party context, corpus and annotation. The text probe "
                  "never hears the audio, so any drop it shares with the voice probe cannot "
                  "be acoustic. A voice drop that the text probe matches points to something "
                  "both modalities share (label reliability, domain), not to degraded audio.\n")

    text_out = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text_out)
    print(text_out)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()
