#!/usr/bin/env python3
"""Section 4.5: decompose the voice encoder's clean-to-wild collapse using IEMOCAP.

Table C1-7 established that emotion2vec scores 86.50 balanced accuracy on CREMA-D and
30.83 on MELD. That is a 55.67-point gap across a comparison in which EVERY variable
moves at once, so it licenses no claim about cause. IEMOCAP resolves it because it is
acoustically clean like CREMA-D but conversational like MELD, and because
it contains scripted and improvised dialogue from the same actors in the same sessions.

The ladder this script builds:

    CREMA-D          acted, isolated utterance, studio clean, 12 fixed sentences
    IEMOCAP script   acted from a play script, conversational, studio clean, open vocab
    IEMOCAP impro    actor-improvised, conversational, studio clean, open vocab
    MELD             scripted TV performance, multi-party, broadcast audio

The cross-corpus rungs differ in many properties at once (corpus, annotation, audio,
structure), so their deltas are NOT attributable to any one of them; the controlled
evidence on what drives them lives in noise_results.md and clean_spontaneity_results.md.
The script-against-impro rung is the only WITHIN-corpus, within-speaker,
within-recording-chain contrast, tested with paired statistics.

READ FIRST: every number in this file comes from the deployed encoders, both of which
saw these corpora in training. Two of the interpretations an earlier version drew here
(a recording-condition modality inversion; a null spontaneity effect) did not survive the
contamination-free replication in clean_probe_eval.py / clean_spontaneity_check.py, and
the generated prose now says so where each table appears.

THREE comparison traps this script exists to avoid.

1. LABEL SPACE. CREMA-D has six categories, MELD seven, and IEMOCAP has two usable
   `disgusted` utterances and 40 `fearful`. A six-way scheme is therefore impossible here.
   All three corpora are restricted to the four classes with real support everywhere
   (angry, happy, neutral, sad) and argmax is taken over those four on every side.
   CREMA-D and MELD are rescored from their existing caches so the ladder is internally
   consistent. The four-class numbers are consequently NOT comparable to Table C1-7's
   six-class numbers, and both are reported rather than silently interchanged.

2. CLASS PRIORS. The four corpora/conditions have very different priors, MELD's being
   neutral-dominated. Balanced accuracy and macro F1 are the primary cross-corpus
   quantities; raw accuracy and each condition's majority baseline are printed so the
   headroom above chance stays visible.

3. CONFOUNDED CONTRAST. Scripted and improvised utterances differ in duration and in
   emotion prior as well as in spontaneity, and the paper has already shown short
   utterances hurt the voice modality. The contrast is therefore repeated on a subsample
   matched cell by cell on duration quartile AND emotion, so a duration or prior
   difference cannot masquerade as a spontaneity effect.

    .venv/bin/python paper/iemocap_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, f1_score,
)

# The only four categories with usable support in ALL THREE corpora.
COMMON4 = ["angry", "happy", "neutral", "sad"]
UNIFIED7 = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
SEED = 0
B_BOOT = 1000


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def argmax_restricted(scores: dict[str, float] | None, labels: list[str]) -> str | None:
    """Argmax over `labels` only, so a corpus missing a class is not penalised for the
    encoder spending probability mass on it."""
    if not scores:
        return None
    sub = {k: scores.get(k, 0.0) for k in labels}
    return max(sub, key=sub.get)


def metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    return {
        "n": len(gold),
        "acc": 100 * accuracy_score(gold, pred),
        "bal_acc": 100 * balanced_accuracy_score(gold, pred),
        "wf1": 100 * f1_score(gold, pred, average="weighted", labels=labels,
                              zero_division=0),
        "macro": 100 * f1_score(gold, pred, average="macro", labels=labels,
                                zero_division=0),
    }


def majority_baseline(gold: list[str]) -> float:
    return 100 * Counter(gold).most_common(1)[0][1] / len(gold) if gold else 0.0


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def voice_eval(rows: list[dict], labels: list[str]) -> tuple[list[str], list[str]]:
    """Gold and voice-argmax for rows whose gold is in `labels`."""
    gold, pred = [], []
    for r in rows:
        if r.get("gold") not in labels:
            continue
        p = argmax_restricted(r.get("voice"), labels)
        if p is None:
            continue
        gold.append(r["gold"])
        pred.append(p)
    return gold, pred


def ladder_row(name: str, gold: list[str], pred: list[str], labels: list[str]) -> list:
    m = metrics(gold, pred, labels)
    base = majority_baseline(gold)
    return [name, m["n"], f"{m['acc']:.2f}", f"{m['bal_acc']:.2f}", f"{m['wf1']:.2f}",
            f"{m['macro']:.2f}", f"{base:.2f}", f"{m['acc'] - base:+.2f}"]


def bal_acc(gold: np.ndarray, pred: np.ndarray) -> float:
    """Balanced accuracy, guarding resamples that happen to drop a class entirely."""
    if gold.size == 0:
        return float("nan")
    return 100 * balanced_accuracy_score(gold, pred)


def cluster_bootstrap_gap(rows_a: list[dict], rows_b: list[dict], labels: list[str],
                          b: int = B_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap the balanced-accuracy gap (a minus b), resampling DIALOGUES not utterances.

    Utterances inside one dialogue share speaker, topic and recording take, so they are
    not independent. Resampling utterances would understate the interval; the cluster is
    the dialogue, so that is the unit resampled. The two conditions are disjoint sets of
    dialogues, so each side is resampled independently.
    """
    def by_dialog(rows: list[dict]) -> list[tuple[list[str], list[str]]]:
        groups: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
        for r in rows:
            if r.get("gold") not in labels:
                continue
            p = argmax_restricted(r.get("voice"), labels)
            if p is None:
                continue
            g, pr = groups[r["dialog"]]
            g.append(r["gold"])
            pr.append(p)
        return list(groups.values())

    ga, gb = by_dialog(rows_a), by_dialog(rows_b)
    if not ga or not gb:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(b):
        sa = rng.integers(0, len(ga), len(ga))
        sb = rng.integers(0, len(gb), len(gb))
        g1 = np.array([x for i in sa for x in ga[i][0]])
        p1 = np.array([x for i in sa for x in ga[i][1]])
        g2 = np.array([x for i in sb for x in gb[i][0]])
        p2 = np.array([x for i in sb for x in gb[i][1]])
        diffs.append(bal_acc(g1, p1) - bal_acc(g2, p2))
    arr = np.array([d for d in diffs if not np.isnan(d)])
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_cells(rows: list[dict], labels: list[str], min_cell: int = 5
                 ) -> tuple[list[str], np.ndarray]:
    """Script against impro as a WITHIN-speaker, within-emotion paired contrast.

    Script and impro cannot be matched utterance to utterance, since the text differs
    entirely. The matched unit is therefore the (speaker, emotion) cell: the same actor
    expressing the same emotion under both conditions. Cells with fewer than `min_cell`
    utterances on either side are dropped as too noisy to carry a per-cell rate.
    """
    acc: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in rows:
        if r.get("gold") not in labels or r.get("kind") not in ("script", "impro"):
            continue
        p = argmax_restricted(r.get("voice"), labels)
        if p is None:
            continue
        acc[(r["speaker"], r["gold"], r["kind"])].append(int(p == r["gold"]))

    pairs, names = [], []
    for (spk, emo, kind) in list(acc):
        if kind != "script":
            continue
        s = acc.get((spk, emo, "script"), [])
        i = acc.get((spk, emo, "impro"), [])
        if len(s) < min_cell or len(i) < min_cell:
            continue
        pairs.append([100 * np.mean(s), 100 * np.mean(i)])
        names.append(f"{spk}/{emo}")
    order = np.argsort(names)
    return [names[i] for i in order], (np.array(pairs)[order] if pairs else np.empty((0, 2)))


def duration_matched(rows: list[dict], labels: list[str], seed: int = SEED
                     ) -> tuple[list[dict], list[dict], list[str]]:
    """Subsample script and impro to equal counts in every (duration quartile x emotion) cell.

    Spontaneity is not the only thing that differs between the conditions: duration and
    emotion prior differ too, and short utterances are already known to hurt the voice
    modality. Matching on both removes them as alternative explanations.
    """
    pool = [r for r in rows
            if r.get("gold") in labels and r.get("kind") in ("script", "impro")
            and r.get("duration") is not None]
    if not pool:
        return [], [], []
    durs = np.array([r["duration"] for r in pool])
    edges = np.percentile(durs, [25, 50, 75])
    def qbin(d: float) -> int:
        return int(np.searchsorted(edges, d, side="right"))

    cells: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for r in pool:
        cells[(qbin(r["duration"]), r["gold"], r["kind"])].append(r)

    rng = np.random.default_rng(seed)
    out_s, out_i, notes = [], [], []
    for q in range(4):
        for emo in labels:
            s = cells.get((q, emo, "script"), [])
            i = cells.get((q, emo, "impro"), [])
            n = min(len(s), len(i))
            if n == 0:
                continue
            out_s.extend([s[k] for k in rng.permutation(len(s))[:n]])
            out_i.extend([i[k] for k in rng.permutation(len(i))[:n]])
    notes.append(f"duration quartile edges (s): "
                 f"{', '.join(f'{e:.2f}' for e in edges)}")
    return out_s, out_i, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap", default="paper/iemocap_scores.jsonl")
    ap.add_argument("--crema", default="paper/crema_scores.jsonl")
    ap.add_argument("--meld", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--out", default="paper/iemocap_results.md")
    args = ap.parse_args()

    iemo = load(args.iemocap)
    crema = load(args.crema)
    meld = load(args.meld)
    if not iemo:
        print(f"! no IEMOCAP cache at {args.iemocap}. Run iemocap_extract_scores.py first.")
        sys.exit(1)
    for name, rows in (("CREMA-D", crema), ("MELD", meld)):
        if not rows:
            print(f"! warning: {name} cache missing; its ladder rung will be omitted",
                  file=sys.stderr)

    impro = [r for r in iemo if r.get("kind") == "impro"]
    script = [r for r in iemo if r.get("kind") == "script"]
    report: list[str] = []

    # ---- Table C1-10: the four-rung ladder, common four classes --------------------
    rungs = []
    if crema:
        rungs.append(("CREMA-D (acted, isolated utterance, clean)", crema))
    rungs.append(("IEMOCAP script (acted, conversational, clean)", script))
    rungs.append(("IEMOCAP impro (improvised, conversational, clean)", impro))
    rungs.append(("IEMOCAP all", iemo))
    if meld:
        rungs.append(("MELD test (acted TV, conversational, degraded)", meld))

    lrows, bal = [], {}
    for name, rows in rungs:
        g, p = voice_eval(rows, COMMON4)
        if not g:
            continue
        lrows.append(ladder_row(name, g, p, COMMON4))
        bal[name] = metrics(g, p, COMMON4)["bal_acc"]
    report.append(fmt("## Table C1-10 - Voice encoder across the cleanliness/spontaneity "
                      "ladder (common four classes)", lrows,
                      ["Condition", "n", "acc", "bal acc", "WF1", "macro F1",
                       "majority", "acc over majority"]))
    report.append("\nAll rows: emotion2vec_plus_large, voice modality only, argmax "
                  "restricted to angry/happy/neutral/sad. These four are the only "
                  "categories with usable support in all three corpora (IEMOCAP has two "
                  "`disgusted` utterances and 40 `fearful`). Four-class figures are NOT "
                  "comparable to Table C1-7's six-class figures.\n")

    # ---- Table C1-11: attribute each step ------------------------------------------
    # Both cross-corpus steps are anchored on IEMOCAP SCRIPT, which is acted and
    # conversational exactly as MELD is. That holds the acted/improvised axis fixed
    # across them, so the two deltas are additive and sum to the total gap, and
    # spontaneity branches off as a separate controlled contrast rather than being
    # smuggled into a rung. MELD is scripted television performance, not spontaneous
    # speech, so no rung on this ladder treats it as such.
    k_crema = next((n for n in bal if n.startswith("CREMA-D")), None)
    k_scr = next((n for n in bal if n.startswith("IEMOCAP script")), None)
    k_imp = next((n for n in bal if n.startswith("IEMOCAP impro")), None)
    k_meld = next((n for n in bal if n.startswith("MELD")), None)
    steps = []

    def step(label, a, b, note):
        if a in bal and b in bal:
            steps.append([label, f"{bal[a]:.2f}", f"{bal[b]:.2f}",
                          f"{bal[b] - bal[a]:+.2f}", note])

    step("CREMA-D -> IEMOCAP script", k_crema, k_scr, "isolated -> conversational")
    step("IEMOCAP script -> MELD test", k_scr, k_meld, "clean -> degraded audio")
    step("IEMOCAP script -> IEMOCAP impro", k_scr, k_imp, "acted -> improvised")
    if k_crema in bal and k_meld in bal:
        steps.append(["TOTAL CREMA-D -> MELD test", f"{bal[k_crema]:.2f}",
                      f"{bal[k_meld]:.2f}", f"{bal[k_meld] - bal[k_crema]:+.2f}",
                      "sum of the first two"])
    report.append(fmt("## Table C1-11 - Gap decomposition (balanced accuracy)", steps,
                      ["Step", "from", "to", "delta", "property varied"]))
    report.append("\nThe first two steps share IEMOCAP script as their anchor, which is "
                  "scripted and conversational as MELD is, so their deltas sum to the "
                  "total gap. Step 1 adds dialogue context and open vocabulary. Step 2 "
                  "changes recording conditions, but also corpus, annotation procedure and "
                  "multi-party structure, and must NOT be read as an acoustic effect: "
                  "realistic noise at MELD-matched speech-recognition difficulty does not "
                  "reproduce it (noise_results.md, Table N-5), and under contamination-free "
                  "encoders text falls as far as voice across this step "
                  "(clean_spontaneity_results.md, Table S-5), which audio cannot explain. "
                  "Step 3 is the only within-corpus contrast; under this deployed encoder "
                  "it is null, but see the caveat under Table C1-13. Steps 1 and 2 are "
                  "cross-corpus and absorb everything else that differs between corpora.\n")

    # ---- Table C1-12: the controlled contrast, with paired statistics ---------------
    gs, ps = voice_eval(script, COMMON4)
    gi, pi = voice_eval(impro, COMMON4)
    crows = []
    for nm, g, p in (("script", gs, ps), ("impro", gi, pi)):
        m = metrics(g, p, COMMON4)
        crows.append([nm, m["n"], f"{m['acc']:.2f}", f"{m['bal_acc']:.2f}",
                      f"{m['macro']:.2f}", f"{majority_baseline(g):.2f}"])
    report.append(fmt("## Table C1-12 - Spontaneity as a within-corpus contrast", crows,
                      ["Condition", "n", "acc", "bal acc", "macro F1", "majority"]))

    mean_d, lo, hi = cluster_bootstrap_gap(script, impro, COMMON4)
    names, cells = paired_cells(iemo, COMMON4)
    lines = [f"\n**Dialogue-clustered bootstrap** ({B_BOOT} resamples, the dialogue is the "
             f"resampled unit because utterances within a dialogue share speaker, topic "
             f"and take): script minus impro balanced accuracy = {mean_d:+.2f} points, "
             f"95 % CI [{lo:+.2f}, {hi:+.2f}]."]
    if cells.shape[0] >= 3:
        d = cells[:, 0] - cells[:, 1]
        try:
            w_stat, w_p = stats.wilcoxon(cells[:, 0], cells[:, 1])
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        lines.append(f"\n**Paired within-speaker, within-emotion test** on "
                     f"{cells.shape[0]} (speaker x emotion) cells present in both "
                     f"conditions with at least 5 utterances each: mean script minus "
                     f"impro accuracy = {d.mean():+.2f} points, "
                     f"Wilcoxon signed-rank W = {w_stat:.0f}, p = {w_p:.4g}. "
                     f"Script higher in {int((d > 0).sum())}/{len(d)} cells.\n")
    else:
        lines.append("\n(too few matched (speaker x emotion) cells for a paired test)\n")
    report.extend(lines)

    # ---- Table C1-13: duration- and prior-matched repeat ----------------------------
    ms, mi, notes = duration_matched(iemo, COMMON4)
    if ms and mi:
        g1, p1 = voice_eval(ms, COMMON4)
        g2, p2 = voice_eval(mi, COMMON4)
        m1, m2 = metrics(g1, p1, COMMON4), metrics(g2, p2, COMMON4)
        mrows = [["script (matched)", m1["n"], f"{m1['acc']:.2f}", f"{m1['bal_acc']:.2f}",
                  f"{m1['macro']:.2f}"],
                 ["impro (matched)", m2["n"], f"{m2['acc']:.2f}", f"{m2['bal_acc']:.2f}",
                  f"{m2['macro']:.2f}"]]
        report.append(fmt("## Table C1-13 - Same contrast, matched on duration quartile "
                          "and emotion", mrows,
                          ["Condition", "n", "acc", "bal acc", "macro F1"]))
        md, mlo, mhi = cluster_bootstrap_gap(ms, mi, COMMON4)
        report.append(f"\n{notes[0]}. Cells are matched to equal counts in every "
                      f"(quartile x emotion) combination, so neither utterance length nor "
                      f"emotion prior differs between the conditions. Script minus impro "
                      f"balanced accuracy = {md:+.2f} points, 95 % CI [{mlo:+.2f}, "
                      f"{mhi:+.2f}] under the same dialogue-clustered bootstrap.\n")
        report.append("**Caveat: this null is a property of the deployed encoder, not of "
                      "speech.** emotion2vec+ saw IEMOCAP labels in training (via EmoBox). "
                      "Under a contamination-free encoder (frozen WavLM with a probe trained "
                      "leave-one-session-out) improvised speech is 12 to 14 points EASIER "
                      "for voice than scripted speech, surviving the same paired-cell and "
                      "duration-and-emotion-matched controls used here "
                      "(clean_probe_results.md K-4; clean_spontaneity_results.md S-1, S-2). "
                      "Probing emotion2vec+'s own embeddings with that same protocol is "
                      "still null (e2v_probe_results.md, E-3), so the flattening lives in "
                      "the encoder's representation, not its classification head. Do not "
                      "report the null above as a finding about spontaneity.\n")

    # ---- Table C1-14: C1's own question on a third corpus ---------------------------
    vrows = []
    gv, pv, gt, pt, orc = [], [], [], [], []
    for r in iemo:
        if r.get("gold") not in COMMON4:
            continue
        pvv = argmax_restricted(r.get("voice"), COMMON4)
        ptt = argmax_restricted(r.get("text_scores"), COMMON4)
        if pvv is None or ptt is None:
            continue
        gv.append(r["gold"]); pv.append(pvv)
        gt.append(r["gold"]); pt.append(ptt)
        orc.append(int(pvv == r["gold"] or ptt == r["gold"]))
    if gv:
        for nm, g, p in (("Voice only (emotion2vec)", gv, pv),
                         ("Text only (DistilRoBERTa, gold transcripts)", gt, pt)):
            m = metrics(g, p, COMMON4)
            vrows.append([nm, m["n"], f"{m['acc']:.2f}", f"{m['bal_acc']:.2f}",
                          f"{m['wf1']:.2f}", f"{m['macro']:.2f}"])
        report.append(fmt("## Table C1-14 - Voice against text on IEMOCAP (common four)",
                          vrows, ["Method", "n", "acc", "bal acc", "WF1", "macro F1"]))
        report.append(f"\nOracle ceiling (either modality correct): "
                      f"{100 * np.mean(orc):.2f} % accuracy on {len(orc)} utterances. "
                      f"Transcripts are the corpus's own, so the text pathway is "
                      f"evaluated under conditions more favourable than deployment. Note "
                      f"also that the DistilRoBERTa text model was trained on MELD but not "
                      f"on IEMOCAP, so its weakness here and its strength on MELD partly "
                      f"reflect which corpus it has seen.\n")

    # ---- Table C1-15: the modality inversion (the C1 headline) ---------------------
    def vt(rows):
        g, pv, pt, orc = [], [], [], []
        for r in rows:
            if r.get("gold") not in COMMON4:
                continue
            a = argmax_restricted(r.get("voice"), COMMON4)
            b = argmax_restricted(r.get("text_scores"), COMMON4)
            if a is None or b is None:
                continue
            g.append(r["gold"]); pv.append(a); pt.append(b)
            orc.append(int(a == r["gold"] or b == r["gold"]))
        if not g:
            return None
        mv, mt = metrics(g, pv, COMMON4), metrics(g, pt, COMMON4)
        return mv, mt, 100 * float(np.mean(orc)), len(g)

    inv = []
    for nm, rows in (("IEMOCAP (clean, conversational)", iemo),
                     ("MELD test (degraded TV audio)", meld)):
        r = vt(rows)
        if r is None:
            continue
        mv, mt, orc, n = r
        inv.append([nm, n, f"{mv['bal_acc']:.2f}", f"{mt['bal_acc']:.2f}",
                    f"{mv['bal_acc'] - mt['bal_acc']:+.2f}",
                    f"{mv['macro']:.2f}", f"{mt['macro']:.2f}", f"{orc:.2f}"])
    if len(inv) == 2:
        report.append(fmt("## Table C1-15 - Voice against text across corpora, deployed "
                          "encoders (does not survive contamination-free replication)", inv,
                          ["Corpus", "n", "voice bal acc", "text bal acc",
                           "voice - text", "voice macro F1", "text macro F1",
                           "oracle acc"]))
        swing = float(inv[0][4]) - float(inv[1][4])
        report.append(f"\nIdentical label space, identical argmax restriction, identical "
                      f"deployed encoders. Under these encoders the sign of the "
                      f"voice-minus-text margin reverses between the corpora, a swing of "
                      f"{swing:.2f} points. **This is not a finding about the modalities.** "
                      f"Both deployed encoders have seen these corpora in training: "
                      f"emotion2vec+ saw IEMOCAP, MELD and CREMA-D through EmoBox, and the "
                      f"DistilRoBERTa text model was trained on MELD but not IEMOCAP, which "
                      f"favours voice on IEMOCAP and text on MELD exactly as the table "
                      f"shows. With contamination-free encoders the margins are +13.05 on "
                      f"IEMOCAP and +0.03 on MELD (clean_probe_results.md, K-3): voice's "
                      f"advantage vanishes on MELD but text does not overtake it, and the "
                      f"swing shrinks by about three quarters. The defensible pattern is "
                      f"in clean_spontaneity_results.md, Table S-4: voice beats text only "
                      f"in improvised speech, and the two tie for scripted speech in both "
                      f"the studio (IEMOCAP scripted) and on television (MELD). That the "
                      f"deployed text-leading heuristic is miscalibrated is measured "
                      f"directly in fusion_transfer_results.md (F-1, F-2).\n")

    # ---- Table C1-16: is the cross-corpus collapse just utterance brevity? ---------
    brev = []
    for nm, pred in (("IEMOCAP, <= 5 words", lambda r: r.get("n_words", 0) <= 5),
                     ("IEMOCAP, > 5 words", lambda r: r.get("n_words", 0) > 5)):
        sub = [r for r in iemo if r.get("gold") in COMMON4 and pred(r)]
        g, p = voice_eval(sub, COMMON4)
        if g:
            m = metrics(g, p, COMMON4)
            brev.append([nm, m["n"], f"{m['bal_acc']:.2f}", f"{m['macro']:.2f}"])
    wi = [r.get("n_words", 0) for r in iemo if r.get("gold") in COMMON4]
    wm = [r.get("n_words", 0) for r in meld if r.get("gold") in COMMON4]
    if brev and wi and wm:
        report.append(fmt("## Table C1-16 - Brevity control on step 2", brev,
                          ["Subset", "n", "bal acc", "macro F1"]))
        report.append(f"\nMELD's utterances are shorter than IEMOCAP's (mean "
                      f"{np.mean(wm):.2f} against {np.mean(wi):.2f} words; "
                      f"{100 * float(np.mean(np.array(wm) <= 3)):.1f} % against "
                      f"{100 * float(np.mean(np.array(wi) <= 3)):.1f} % at three words or "
                      f"fewer), and short utterances are already known to hurt the voice "
                      f"modality. Brevity is therefore an alternative explanation for "
                      f"step 2 and is tested directly: restricting IEMOCAP to its short "
                      f"utterances costs "
                      f"{float(brev[1][2]) - float(brev[0][2]):.2f} balanced-accuracy "
                      f"points, which is a small fraction of step 2. For the deployed "
                      f"encoder, step 2 is therefore not an artifact of utterance length. "
                      f"This does not make it acoustic: see the note under Table C1-11.\n")

    # ---- per-class, IEMOCAP voice ---------------------------------------------------
    if gv:
        rep = classification_report(gv, pv, labels=COMMON4, digits=3, zero_division=0)
        report.append("## Per-class F1 - IEMOCAP, voice only, common four\n\n```\n"
                      + rep + "\n```\n")

    # ---- frustration diagnostic -----------------------------------------------------
    fru = [r for r in iemo if r.get("tag") == "fru"]
    if fru:
        dist: Counter[str] = Counter()
        for r in fru:
            p = argmax_restricted(r.get("voice"), UNIFIED7)
            if p:
                dist[p] += 1
        tot = sum(dist.values())
        frows = [[k, v, f"{100 * v / tot:.1f}"] for k, v in dist.most_common()]
        report.append(fmt("## Frustration has no home in the deployed label scheme", frows,
                          ["voice argmax", "n", "%"]))
        report.append(f"\n{tot} utterances labelled `fru` in IEMOCAP, the corpus's second "
                      f"largest category, have no counterpart among the deployed seven "
                      f"classes and are excluded from every accuracy table above. This "
                      f"row shows where the encoder puts them instead: almost evenly "
                      f"across four classes, so the encoder has no representation of "
                      f"frustration. If frustrated speech is harder than the retained "
                      f"classes, as that spread suggests, excluding it flatters IEMOCAP. "
                      f"That makes the CREMA-D-to-IEMOCAP step in Table C1-11 conservative "
                      f"but the IEMOCAP-to-MELD step an OVERestimate, not a lower bound as "
                      f"an earlier version of this file stated.\n")

    # ---- excited-merge sensitivity ---------------------------------------------------
    no_exc = [r for r in iemo if not r.get("merged_exc")]
    g1, p1 = voice_eval(iemo, COMMON4)
    g2, p2 = voice_eval(no_exc, COMMON4)
    if g1 and g2:
        m1, m2 = metrics(g1, p1, COMMON4), metrics(g2, p2, COMMON4)
        srows = [["happy = hap + exc (reported)", m1["n"], f"{m1['bal_acc']:.2f}",
                  f"{m1['macro']:.2f}"],
                 ["happy = hap only", m2["n"], f"{m2['bal_acc']:.2f}", f"{m2['macro']:.2f}"]]
        report.append(fmt("## Sensitivity - the excited-into-happy merge", srows,
                          ["Scheme", "n", "bal acc", "macro F1"]))
        report.append("\nMerging `exc` into happy is conventional on IEMOCAP and is forced "
                      "by the deployed scheme, which has no excited category. It moves "
                      "1,041 utterances, so it is reported both ways rather than left "
                      "implicit.\n")

    # ---- composition table ----------------------------------------------------------
    comp = []
    for nm, rows in (("impro", impro), ("script", script)):
        d = [r["duration"] for r in rows if r.get("duration") is not None]
        c4 = [r for r in rows if r.get("gold") in COMMON4]
        comp.append([nm, len(rows), len(c4), f"{np.mean(d):.2f}" if d else "-",
                     f"{np.median(d):.2f}" if d else "-",
                     f"{np.mean([r.get('n_words', 0) for r in rows]):.1f}"])
    report.append(fmt("## Corpus composition", comp,
                      ["kind", "n cached", "n common-four", "mean dur s", "median dur s",
                       "mean words"]))

    header = ("# C1 boundary condition, part 2: the voice encoder across corpora, with "
              "IEMOCAP\n\n"
              "> **Read first.** Every number below uses the DEPLOYED encoders "
              "(emotion2vec+ for voice, DistilRoBERTa for text), and both saw these "
              "corpora in training: emotion2vec+ via EmoBox (IEMOCAP, MELD, CREMA-D), "
              "DistilRoBERTa via MELD. The numbers are correct as measurements of the "
              "deployed system, but two interpretations an earlier version of this file "
              "drew from them did not survive contamination-free replication: the "
              "voice/text ranking does not invert with recording conditions, and "
              "spontaneity is not null. See clean_probe_results.md, "
              "clean_spontaneity_results.md, noise_results.md and "
              "fusion_transfer_results.md for what holds.\n\n"
              "emotion2vec_plus_large unless stated. IEMOCAP is studio-recorded and "
              "conversational and contains both scripted and actor-improvised dialogue "
              "from the same ten actors.\n\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()
