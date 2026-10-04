#!/usr/bin/env python3
"""MELD smoke test: verify the eval pipeline on a few utterances before the full run.

Extracts audio from the first N MELD test clips, runs the DEPLOYED EmotionDetector
(aiko/emotion.py) on each, and prints gold vs predicted so we can confirm the data
path, ffmpeg, model loading, and label mapping all work before scoring ~2.6k clips.

    .venv/bin/python paper/meld_smoke.py \
        --csv    data/.../test_sent_emo.csv \
        --videos data/.../output_repeated_splits_test --n 5
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

# make the `aiko` package importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiko.emotion import EmotionDetector

# MELD "Emotion" column -> Aiko's UNIFIED_LABELS
MELD_TO_UNIFIED = {
    "neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry",
    "surprise": "surprised", "fear": "fearful", "disgust": "disgusted",
}


def extract_wav(mp4: str, wav: str) -> bool:
    """Extract 16 kHz mono wav from an MELD clip (cached; skip if present)."""
    if os.path.exists(wav):
        return True
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
         "-ac", "1", "-ar", "16000", "-vn", wav],
        capture_output=True, text=True,
    )
    if proc.returncode:
        print(f"  ffmpeg failed on {os.path.basename(mp4)}: {proc.stderr.strip()[:120]}")
    return proc.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to test_sent_emo.csv")
    ap.add_argument("--videos", required=True, help="dir with dia*_utt*.mp4 clips")
    ap.add_argument("--n", type=int, default=5, help="how many utterances to test")
    ap.add_argument("--cache", default="./paper/meld_wavs", help="extracted-wav cache dir")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    print("Loading EmotionDetector (voice + text models)...")
    det = EmotionDetector()

    with open(args.csv, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} utterances in {os.path.basename(args.csv)}; testing first {args.n}\n")

    hits = seen = 0
    for row in rows[: args.n]:
        did, uid = row["Dialogue_ID"], row["Utterance_ID"]
        gold = MELD_TO_UNIFIED.get(row["Emotion"].strip().lower())
        text = row["Utterance"].strip()
        mp4 = os.path.join(args.videos, f"dia{did}_utt{uid}.mp4")
        wav = os.path.join(args.cache, f"dia{did}_utt{uid}.wav")

        if not os.path.exists(mp4):
            print(f"  MISSING clip: {os.path.basename(mp4)}")
            continue
        if not extract_wav(mp4, wav):
            continue

        res = det.detect(wav, text)
        seen += 1
        hit = res.emotion == gold
        hits += hit
        print(f"{'OK ' if hit else 'xx '}gold={gold:<9} pred={res.emotion:<9} "
              f"conf={res.confidence:.2f} | voice_top={res.voice_top:<9} "
              f"text_top={res.text_top:<9} short={res.short}")
        print(f"    text : {text[:80]!r}")
        print(f"    top3 : {[(e, round(p, 2)) for e, p in res.top3]}\n")

    if seen:
        print(f"smoke accuracy: {hits}/{seen} = {hits / seen:.1%}")


if __name__ == "__main__":
    main()