#!/usr/bin/env python3
"""Export CMU-MOSEI's segment table to plain JSON for the gap-2 natural-speech test.

Runs in the ISOLATED audb environment, never the project .venv: audb requires pandas >= 2.1,
which breaks the project's pinned pandas 1.5.3 (TTS needs < 2.0). Installing audb into
.venv once silently upgraded pandas; that was reverted, and audb now lives only in
data/.venv-audb (gitignored). Everything downstream reads the JSON written here.

Why MOSEI. The headline pattern, that voice beats text only in improvised speech, rests on
actors (IEMOCAP). Whether it holds for NATURAL speech, which is what a companion's users
produce, needs non-acted recordings with human emotion labels. CMU-MOSEI (Zadeh et al.,
ACL 2018; audEERING packaging cmu-mosei 1.2.4, CC-BY-NC-4.0) has 23,259 sentence-level
segments of real YouTube opinion monologues from over 1,000 speakers, crowd-rated for six
emotions, with human transcripts and an official train / dev / test split by video.

Setup (once):
    python3.11 -m venv data/.venv-audb && data/.venv-audb/bin/python -m pip install audb
    data/.venv-audb/bin/python -c "import audb; audb.load('cmu-mosei', version='1.2.4',
        cache_root='data/mosei/audb_cache', num_workers=8)"

    data/.venv-audb/bin/python paper/mosei_prepare.py

Output: data/mosei/manifest.jsonl (gitignored; holds transcripts).
"""
from __future__ import annotations

import json
import os

import audb

EMOS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]


def main() -> None:
    db = audb.load("cmu-mosei", version="1.2.4", cache_root="data/mosei/audb_cache",
                   only_metadata=False, verbose=False)
    split = {}
    for s in ("train", "dev", "test"):
        for f in db[s].index:
            split[f] = s
    emo = db["emotion"].get()
    pres = db["emotion.presence"].get()
    tr = db["transcription"].get()
    sent = db["sentiment"].get()
    n = 0
    os.makedirs("data/mosei", exist_ok=True)
    with open("data/mosei/manifest.jsonl", "w", encoding="utf-8") as out:
        for k, (idx, row) in enumerate(emo.iterrows()):
            f, start, end = idx
            video = os.path.splitext(os.path.basename(f))[0]
            out.write(json.dumps({
                "id": f"mosei_{k:05d}", "file": f, "video": video,
                "start": round(start.total_seconds(), 4), "end": round(end.total_seconds(), 4),
                "split": split.get(f, "none"),
                "intensity": {e: float(row[e]) for e in EMOS},
                "presence": {e: bool(pres.loc[idx, e]) for e in EMOS},
                "sentiment": float(sent.loc[idx, "sentiment"]),
                "text": str(tr.loc[idx, "transcription"]),
            }) + "\n")
            n += 1
    counts = {s: sum(1 for v in split.values() if v == s) for s in ("train", "dev", "test")}
    print(f"{n} segments -> data/mosei/manifest.jsonl; videos per split {counts}")


if __name__ == "__main__":
    main()
