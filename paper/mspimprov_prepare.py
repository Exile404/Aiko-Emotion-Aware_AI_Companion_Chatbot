#!/usr/bin/env python3
"""Prepare MSP-IMPROV for the definitive gap-1 test (run when the licence arrives).

Why MSP-IMPROV. Expresso (expresso_eval.py) tests emotion2vec+ where its LABELS were never
trained on, but its labels are intended styles and only three classes overlap. MSP-IMPROV
(Busso et al., IEEE TAC 2017) is absent from EmoBox, has PERCEIVED labels on exactly the
four deployed classes, and its key contrast is cleaner than anything in IEMOCAP: the same
target sentences, same actors, same intended emotion, recorded once READ (alone, on a
different day) and once IMPROVISED (inside a dyadic scenario). Only delivery changes.

Documented structure (paper, Tables 3 and 4), used to VALIDATE parsing, not assumed:

    condition            turns   angry  sad  happy neutral other no-agreement
    Target - improvised    652    115   106   136    283      1     11
    Target - read          620    169    80    88    241      3     39
    Other - improvised   4,381    470   633 1,048  1,789     70    371
    Natural interaction  2,785     38    66 1,372  1,164     11    134

Six dyadic sessions, twelve actors (eight of whom recorded the read set). Audio was
captured at 48 kHz; it is resampled to 16 kHz mono here.

The file-naming code for the condition, and the exact label-file format, are not given
in the paper, so this script DISCOVERS them and then checks itself: the condition code
is mapped by matching per-code file counts to the turn counts above, and labels are
accepted only if the parsed per-condition class counts reproduce Table 4 within
tolerance. If either check fails it stops and prints what it found, rather than guessing.

    .venv/bin/python paper/mspimprov_prepare.py --root <MSP-IMPROV dir> --inspect
    .venv/bin/python paper/mspimprov_prepare.py --root <MSP-IMPROV dir>

Output under data/mspimprov/ (gitignored): wav/<id>.wav and manifest.jsonl in the same
format as data/expresso/manifest.jsonl, so the Expresso extraction and evaluation
scripts run on it unchanged apart from paths and the class list.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

OUT = "data/mspimprov"
DOCUMENTED = {                       # condition -> (turns, A, S, H, N)
    "target_improvised": (652, 115, 106, 136, 283),
    "target_read": (620, 169, 80, 88, 241),
    "other_improvised": (4381, 470, 633, 1048, 1789),
    "natural": (2785, 38, 66, 1372, 1164),
}
LETTER = {"A": "angry", "S": "sad", "H": "happy", "N": "neutral"}
TOL = 0.06                           # relative tolerance for count validation
# e.g. MSP-IMPROV-S01A-F01-P-FM01: sentence+target emotion, gender+session, condition, turn
NAME = re.compile(r"(?:MSP|UTD)-IMPROV-S(\d{2})([AHNS])-([FM])(\d{2})-([A-Z])-([A-Z]{1,2}\d{2,3})",
                  re.IGNORECASE)


def canon(m: re.Match) -> str:
    """Prefix-free utterance key. Audio may be named MSP-IMPROV-... while label files name
    the video, UTD-IMPROV-...; keying on the fields alone lets the two meet."""
    sent, temo, gender, sess, code, turn = m.groups()
    return f"S{sent}{temo}-{gender}{sess}-{code}-{turn}".upper()


def find_audio(root: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
                  if not os.path.basename(p).startswith("._"))


def parse(path: str) -> dict | None:
    m = NAME.search(os.path.basename(path))
    if not m:
        return None
    sent, temo, gender, sess, code, turn = m.groups()
    return {"id": canon(m), "sentence": int(sent), "target_emotion": temo.upper(),
            "gender": gender.upper(), "session": int(sess), "code": code.upper(),
            "turn": turn.upper(), "wav_src": path}


def map_codes(rows: list[dict]) -> dict[str, str]:
    """Assign each condition code to the documented condition with the matching count."""
    counts = Counter(r["code"] for r in rows)
    mapping, used = {}, set()
    for code, n in counts.most_common():
        best = min((c for c in DOCUMENTED if c not in used),
                   key=lambda c: abs(DOCUMENTED[c][0] - n) / DOCUMENTED[c][0], default=None)
        if best is None or abs(DOCUMENTED[best][0] - n) / DOCUMENTED[best][0] > TOL:
            sys.exit(f"! condition code {code!r} has {n} files, matching no documented "
                     f"condition within {TOL:.0%}. Counts by code: {dict(counts)}. "
                     f"Run --inspect and adjust map_codes().")
        mapping[code] = best
        used.add(best)
    return mapping


def find_label_lines(root: str) -> dict[str, str]:
    """Scan text files for lines naming an utterance id and a single consensus letter.

    Accepts formats like 'MSP-IMPROV-S01A-F01-R-FM01.avi; A; A:3.0; ...' or CSV/TSV with
    the id and a label column. The first standalone A/S/H/N/O/X token after the id is
    taken as the consensus label. Validated downstream against Table 4.
    """
    labels = {}
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path) or os.path.basename(path).startswith("._"):
            continue
        if os.path.splitext(path)[1].lower() not in (".txt", ".csv", ".tsv", ".lab", ""):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for line in text.splitlines():
            m = NAME.search(line)
            if not m:
                continue
            rest = line[m.end():]
            tok = re.search(r"(?<![A-Za-z:])([ASHNOX])(?![A-Za-z:])", rest)
            if tok:
                labels.setdefault(canon(m), tok.group(1))
    return labels


def validate_labels(rows: list[dict], mapping: dict, labels: dict) -> None:
    by = defaultdict(Counter)
    for r in rows:
        by[mapping[r["code"]]][labels.get(r["id"], "?")] += 1
    bad = []
    for cond, (n, a, s, h, nn) in DOCUMENTED.items():
        got = by[cond]
        for letter, want in zip("ASHN", (a, s, h, nn)):
            if want and abs(got[letter] - want) / want > TOL:
                bad.append(f"{cond} {letter}: parsed {got[letter]}, documented {want}")
    print("parsed label counts per condition:")
    for cond in DOCUMENTED:
        print(f"  {cond:18s} {dict(by[cond])}")
    if bad:
        sys.exit("! labels do not reproduce Table 4:\n  " + "\n  ".join(bad)
                 + "\nRun --inspect and adjust find_label_lines().")


def inspect(root: str) -> None:
    wavs = find_audio(root)
    print(f"{len(wavs)} wav files under {root}")
    for p in wavs[:5]:
        print("  e.g.", os.path.relpath(p, root))
    parsed = [parse(p) for p in wavs]
    ok = [p for p in parsed if p]
    print(f"{len(ok)} match the naming pattern; {len(wavs) - len(ok)} do not")
    print("files per condition code:", dict(Counter(p["code"] for p in ok)))
    print("sessions:", sorted(Counter(p["session"] for p in ok).items()))
    texts = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".txt", ".csv", ".tsv")]
    print(f"{len(texts)} text files; first lines of up to five that mention an utterance id:")
    shown = 0
    for t in texts:
        lines = [l for l in open(t, encoding="utf-8", errors="replace").read().splitlines()
                 if NAME.search(l)]
        if lines:
            print(f"--- {os.path.relpath(t, root)} ({len(lines)} id lines)")
            for l in lines[:3]:
                print("   ", l[:160])
            shown += 1
        if shown == 5:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.root)
        return

    import soundfile as sf
    from scipy.signal import resample_poly

    rows = [p for p in (parse(w) for w in find_audio(args.root)) if p]
    if not rows:
        sys.exit("! no files match the naming pattern; run --inspect")
    mapping = map_codes(rows)
    print("condition codes:", mapping)
    labels = find_label_lines(args.root)
    validate_labels(rows, mapping, labels)

    os.makedirs(f"{OUT}/wav", exist_ok=True)
    kept = Counter()
    with open(f"{OUT}/manifest.jsonl", "w", encoding="utf-8") as man:
        for r in rows:
            lab = labels.get(r["id"])
            if lab not in LETTER:
                continue                         # other / no agreement: not in the scheme
            x, sr = sf.read(r["wav_src"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr == 48000:
                x = resample_poly(x, 1, 3).astype(np.float32)
            elif sr == 44100:
                x = resample_poly(x, 160, 441).astype(np.float32)
            elif sr != 16000:
                sys.exit(f"! unexpected sample rate {sr}")
            sf.write(f"{OUT}/wav/{r['id']}.wav", x, 16000, subtype="PCM_16")
            cond = mapping[r["code"]]
            man.write(json.dumps({
                "id": r["id"], "condition": cond, "speaker": f"S{r['session']:02d}{r['gender']}",
                "session": r["session"], "sentence": r["sentence"],
                "target_emotion": LETTER.get(r["target_emotion"]), "gold": LETTER[lab],
                "duration": round(len(x) / 16000, 3),
                "group": f"S{r['session']:02d}_{cond}_{r['sentence']:02d}{r['target_emotion']}",
                "text": "",
            }) + "\n")
            kept[(cond, LETTER[lab])] += 1
    print(f"kept {sum(kept.values())} clips:", dict(sorted(kept.items())))


if __name__ == "__main__":
    main()
