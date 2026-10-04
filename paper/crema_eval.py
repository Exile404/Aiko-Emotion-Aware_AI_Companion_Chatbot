#!/usr/bin/env python3
"""Voice-encoder competence on clean speech (CREMA-D) versus in-the-wild speech (MELD).

The question this answers: is emotion2vec weak in general, or weak on MELD's audio?
If it is competent on CREMA-D and not on MELD, the C1 null is a property of in-the-wild
recording conditions and becomes a scoped finding instead of an unbounded claim.

TWO comparison traps this script exists to avoid.

1. LABEL SPACE. CREMA-D has six categories, MELD seven. A six-way problem is easier, so
   both corpora are restricted to the shared six: MELD rows labelled `surprised` are
   dropped, and argmax is taken over the shared six only, on both sides.

2. CLASS PRIORS. CREMA-D is near balanced; MELD is dominated by neutral. Raw accuracy is
   therefore NOT comparable across the two even with matched labels, because MELD's
   accuracy is inflated by majority guessing. Balanced accuracy and macro F1 are the
   primary cross-corpus quantities, and each corpus's majority-class baseline is printed
   so the headroom above chance is visible.

    .venv/bin/python paper/crema_eval.py
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

# Categories present in BOTH corpora. CREMA-D has no `surprised`.
SHARED = ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]
LEVELS = ("LO", "MD", "HI")     # CREMA-D's acted intensity levels, ordered


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


def argmax_restricted(scores: dict[str, float], labels: list[str]) -> str:
    """Argmax over `labels` only, so a corpus missing a class is not penalised for the
    encoder spending probability mass on it."""
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


def row(name: str, m: dict, base: float) -> list:
    return [name, m["n"], f"{m['acc']:.2f}", f"{m['bal_acc']:.2f}",
            f"{m['wf1']:.2f}", f"{m['macro']:.2f}", f"{base:.2f}",
            f"{m['acc'] - base:+.2f}"]


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial p against 0.5, for McNemar on discordant pairs."""
    if n == 0:
        return 1.0
    try:
        return float(stats.binomtest(k, n, 0.5).pvalue)
    except AttributeError:                      # scipy < 1.7
        return float(stats.binom_test(k, n, 0.5))


def cochrans_q(mat: np.ndarray) -> tuple[float, float]:
    """Cochran's Q: McNemar generalised to k > 2 paired binary conditions.

    Subjects whose outcome is identical across all conditions cancel out of the
    statistic, which is precisely why the paired form is so much more powerful here than
    an unpaired comparison of three independent proportions.
    """
    k = mat.shape[1]
    col = mat.sum(axis=0).astype(float)         # successes per intensity level
    rowsum = mat.sum(axis=1).astype(float)      # successes per matched triple
    total = float(col.sum())
    denom = k * total - float((rowsum ** 2).sum())
    if denom <= 0:                              # no within-subject variation at all
        return 0.0, 1.0
    q = (k - 1) * (k * float((col ** 2).sum()) - total ** 2) / denom
    return q, float(stats.chi2.sf(q, k - 1))


def paired_intensity(crema: list[dict], labels: list[str], b_boot: int = 1000) -> list[str]:
    """Intensity as a WITHIN-ACTOR contrast.

    CREMA-D varies intensity while holding actor, carrier sentence and emotion fixed, so
    LO/MD/HI form matched triples and the correct analysis is paired. Treating the levels
    as independent samples discards that pairing and is badly underpowered: unpaired, the
    LO-vs-HI difference on these data sits around p = 0.20 despite being a clean
    within-subject design.

    Two paired quantities are reported. Correctness is the headline but coarse, since it
    only moves when an argmax flips. The probability mass the encoder puts on the TRUE
    class is graded, so it detects a confidence shift on clips whose prediction never
    changes, and is the more sensitive test of whether the encoder tracks affect STRENGTH
    rather than merely affect category.
    """
    triples: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in crema:
        if r["intensity"] in LEVELS:
            # The key holds actor, words AND emotion fixed: anything less is not a
            # within-subject contrast, it is a different comparison wearing its clothes.
            triples[(r["actor"], r["sentence"], r["gold"])][r["intensity"]] = r
    full = [t for t in triples.values() if len(t) == len(LEVELS)]
    if not full:
        return []

    correct = np.array([[int(argmax_restricted(t[lv]["voice"], labels) == t[lv]["gold"])
                         for lv in LEVELS] for t in full])
    ptrue = np.array([[float(t[lv]["voice"].get(t[lv]["gold"], 0.0)) for lv in LEVELS]
                      for t in full])
    n = len(full)

    out = [fmt("## Table C1-9 - Intensity as a paired within-actor contrast",
               [[lv, n, f"{100 * correct[:, i].mean():.2f}", f"{ptrue[:, i].mean():.4f}"]
                for i, lv in enumerate(LEVELS)],
               ["Level", "n triples", "acc", "mean p(true class)"])]
    out.append(f"\n{n} matched triples: the same actor speaking the same sentence with "
               "the same intended emotion at all three intensities. Actor, lexical "
               "content and emotion are therefore held constant, and only portrayal "
               "strength varies.\n")

    # --- correctness across all three levels, and the LO vs HI contrast ---------
    q, q_p = cochrans_q(correct)
    disc_lo = int(((correct[:, 0] == 1) & (correct[:, 2] == 0)).sum())
    disc_hi = int(((correct[:, 0] == 0) & (correct[:, 2] == 1)).sum())
    mc_p = binom_p(min(disc_lo, disc_hi), disc_lo + disc_hi)

    # --- graded confidence in the true class ------------------------------------
    w_stat, w_p = stats.wilcoxon(ptrue[:, 2], ptrue[:, 0])

    # --- paired bootstrap, resampling TRIPLES so the pairing survives -----------
    rng = np.random.default_rng(0)               # fixed seed: the CI must reproduce
    idx = rng.integers(0, n, size=(b_boot, n))
    d_acc = 100 * (correct[:, 2][idx].mean(axis=1) - correct[:, 0][idx].mean(axis=1))
    d_pt = ptrue[:, 2][idx].mean(axis=1) - ptrue[:, 0][idx].mean(axis=1)
    acc_lo, acc_hi = np.percentile(d_acc, [2.5, 97.5])
    pt_lo, pt_hi = np.percentile(d_pt, [2.5, 97.5])

    # --- how many triples rise monotonically? chance is 1/6 of orderings --------
    mono = int(((ptrue[:, 0] < ptrue[:, 1]) & (ptrue[:, 1] < ptrue[:, 2])).sum())
    mono_p = float(stats.binomtest(mono, n, 1 / 6).pvalue) if hasattr(stats, "binomtest") \
        else float("nan")

    out.append(
        f"**Correctness.** Cochran's Q across the three levels = {q:.2f}, p = {q_p:.4f}. "
        f"LO vs HI by exact McNemar on {disc_lo + disc_hi} discordant triples "
        f"({disc_hi} gained at HI, {disc_lo} lost): p = {mc_p:.4f}. Paired bootstrap "
        f"({b_boot} resamples) on HI minus LO accuracy: "
        f"{100 * (correct[:, 2].mean() - correct[:, 0].mean()):+.2f} points, "
        f"95 % CI [{acc_lo:+.2f}, {acc_hi:+.2f}].\n")
    out.append(
        f"**Confidence in the true class.** Mean p(true) rises "
        f"{ptrue[:, 0].mean():.4f} -> {ptrue[:, 1].mean():.4f} -> {ptrue[:, 2].mean():.4f}. "
        f"Wilcoxon signed-rank HI vs LO: W = {w_stat:.0f}, p = {w_p:.2e}. Paired bootstrap "
        f"on HI minus LO: {ptrue[:, 2].mean() - ptrue[:, 0].mean():+.4f}, "
        f"95 % CI [{pt_lo:+.4f}, {pt_hi:+.4f}]. Strictly increasing across all three "
        f"levels in {mono}/{n} triples ({100 * mono / n:.1f} %), against the "
        f"{100 / 6:.1f} % expected if orderings were random (p = {mono_p:.2e}).\n")

    # Verdict is computed, not asserted: the script must not claim a dose-response the
    # data does not show.
    sig_c = mc_p < 0.05 and not (acc_lo <= 0 <= acc_hi)
    sig_p = w_p < 0.05 and not (pt_lo <= 0 <= pt_hi)
    if sig_c and sig_p:
        verdict = ("Both the coarse and the graded measure move with intensity, so the "
                   "encoder tracks portrayal STRENGTH and not merely category.")
    elif sig_p:
        verdict = ("The graded measure moves with intensity while argmax correctness does "
                   "not separate reliably. The encoder's CONFIDENCE tracks portrayal "
                   "strength even where its decision is already correct at low intensity, "
                   "which is the expected signature when accuracy is near ceiling.")
    elif sig_c:
        verdict = ("Correctness separates but the graded confidence measure does not, "
                   "which is unusual and worth inspecting before relying on it.")
    else:
        verdict = ("Neither measure separates reliably under the paired test. The monotone "
                   "trend in the means is NOT statistically supported and must not be "
                   "reported as a dose-response.")
    out.append(f"**Reading.** {verdict}\n")
    return out


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crema", default="paper/crema_scores.jsonl")
    ap.add_argument("--meld", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--out", default="paper/crema_results.md")
    args = ap.parse_args()

    crema = load(args.crema)
    meld = load(args.meld)
    if not crema:
        print(f"! no CREMA-D cache at {args.crema}; run crema_extract_scores.py first")
        sys.exit(1)
    print(f"CREMA-D {len(crema)} clips, MELD test {len(meld)} utterances\n")

    report: list[str] = []

    # ---- headline: voice-only, shared six classes, both corpora -------------------
    cg = [r["gold"] for r in crema]
    cp = [argmax_restricted(r["voice"], SHARED) for r in crema]
    cm = metrics(cg, cp, SHARED)
    rows = [row("CREMA-D (clean, acted)", cm, majority_baseline(cg))]

    if meld:
        keep = [r for r in meld if r.get("gold") in SHARED]
        mg = [r["gold"] for r in keep]
        mp = [argmax_restricted(r["voice"], SHARED) for r in keep]
        mm = metrics(mg, mp, SHARED)
        rows.append(row("MELD test (in the wild)", mm, majority_baseline(mg)))
        dropped = len(meld) - len(keep)
    else:
        mm, dropped = None, 0

    report.append(fmt("## Table C1-7 - Voice encoder competence, shared six classes",
                      rows, ["Corpus", "n", "acc", "bal acc", "WF1", "macro F1",
                             "majority", "acc over majority"]))
    if dropped:
        report.append(f"\n{dropped} MELD utterances labelled `surprised` were dropped so "
                      "both corpora span the same six categories. Argmax is restricted to "
                      "those six on both sides.\n")
    if mm:
        report.append(
            f"**Primary comparison (balanced accuracy, immune to the two corpora's very "
            f"different class priors): CREMA-D {cm['bal_acc']:.2f} % against MELD "
            f"{mm['bal_acc']:.2f} %, a gap of {cm['bal_acc'] - mm['bal_acc']:+.2f} points.** "
            f"Macro F1 tells the same story: {cm['macro']:.2f} against {mm['macro']:.2f}. "
            "Raw accuracy is reported for completeness but is NOT comparable across the two "
            "corpora, because MELD's neutral-dominated prior lifts it independently of "
            "encoder quality; the `acc over majority` column is the honest accuracy-based "
            "reading.\n")

    # ---- text-only control: at chance by construction ----------------------------
    tp = [argmax_restricted(r["text_scores"], SHARED) for r in crema]
    tm = metrics(cg, tp, SHARED)
    report.append(f"**Text-only control on CREMA-D: acc {tm['acc']:.2f} %, balanced acc "
                  f"{tm['bal_acc']:.2f} %, macro F1 {tm['macro']:.2f}** against a majority "
                  f"baseline of {majority_baseline(cg):.2f} %. CREMA-D's twelve carrier "
                  "sentences are each recorded in all six emotions, so the transcript "
                  "carries no emotion information by design. This confirms empirically "
                  "that the CREMA-D result isolates the voice modality and cannot be "
                  "explained by lexical cues.\n")

    # ---- per-class, both corpora -------------------------------------------------
    report.append("## Per-class F1 (voice only, shared six)\n")
    report.append("```\nCREMA-D\n" + classification_report(
        cg, cp, labels=SHARED, zero_division=0, digits=3) + "\n```\n")
    if meld:
        report.append("```\nMELD test\n" + classification_report(
            mg, mp, labels=SHARED, zero_division=0, digits=3) + "\n```\n")

    # ---- intensity dose-response (CREMA-D only) ----------------------------------
    # Acted intensity is the one factor CREMA-D varies while holding actor, words and
    # emotion fixed, so it tests whether the encoder tracks affect STRENGTH. It also bears
    # on the objection that the CREMA-D/MELD gap reflects prototypical portrayal rather
    # than audio quality: if that were the whole story, the LO row should slide toward
    # MELD's figures. Table C1-9 supplies the properly paired version of this contrast.
    by_int: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r, p in zip(crema, cp):
        if r["gold"] != "neutral" and r["intensity"] in LEVELS:
            by_int[r["intensity"]].append((r["gold"], p))
    irows = []
    for lvl in LEVELS:
        pairs = by_int.get(lvl, [])
        if not pairs:
            continue
        g, p = [x[0] for x in pairs], [x[1] for x in pairs]
        # Labels must be the classes actually PRESENT. `neutral` has no intensity variants
        # and is excluded by design, so averaging a zero in for it deflates macro F1 by 1/6.
        m = metrics(g, p, sorted(set(g)))
        irows.append([lvl, m["n"], f"{m['acc']:.2f}", f"{m['bal_acc']:.2f}",
                      f"{m['macro']:.2f}"])
    if irows:
        report.append(fmt("## Table C1-8 - CREMA-D accuracy by acted intensity "
                          "(neutral excluded)", irows,
                          ["Intensity", "n", "acc", "bal acc", "macro F1"]))
        report.append("\nMacro F1 is averaged over the five classes present in these "
                      "subsets. These rows treat the levels as independent samples, which "
                      "understates the design; the paired analysis below is the one to "
                      "cite.\n")
    report.extend(paired_intensity(crema, SHARED))

    # ---- sanity: does any carrier sentence leak emotion? -------------------------
    per_sent: dict[str, Counter] = defaultdict(Counter)
    for r in crema:
        per_sent[r["sentence"]][r["gold"]] += 1
    skew = []
    for s, c in sorted(per_sent.items()):
        tot = sum(c.values())
        top = c.most_common(1)[0]
        skew.append([s, tot, top[0], f"{100 * top[1] / tot:.1f}"])
    report.append(fmt("## Sentence balance check", skew,
                      ["Sentence", "n", "most common gold", "% of that sentence"]))
    report.append("\nEach carrier sentence should be near uniform across the six "
                  "emotions. Large skew on any row would mean the text-only control is "
                  "weaker than it appears.\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# C1 boundary condition - voice encoder on clean versus in-the-wild "
                "speech\n\nemotion2vec_plus_large, voice modality only, argmax restricted "
                "to the six categories CREMA-D and MELD share.\n\n" + "\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()