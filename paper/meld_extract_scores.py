#!/usr/bin/env python3
"""Step 2: cache per-utterance emotion scores over the full MELD test split.

Runs the DEPLOYED models once (the expensive GPU pass) and writes a JSONL cache of
each utterance's voice + text score dicts, gold label, and the deployed fused
prediction. paper/meld_eval.py reads this cache, so we never re-run the models while
iterating on baselines/ablations.

Grouped by dialogue in utterance order; the detector's recency history is reset at
each dialogue boundary so the cached prediction matches within-conversation behavior.

    .venv/bin/python paper/meld_extract_scores.py \
        --csv    data/MELD.Raw/test_sent_emo.csv \
        --videos data/MELD.Raw/output_repeated_splits_test \
        --out    paper/meld_test_scores.jsonl
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aiko.emotion import EmotionDetector

MELD_TO_UNIFIED = {
    "neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry",
    "surprise": "surprised", "fear": "fearful", "disgust": "disgusted",
}

# MELD's CSV is UTF-8 but stores Windows-1252 punctuation as C1 control codepoints
# (e.g. an apostrophe as U+0092). Map those back to the characters they were meant to be.
_CP1252_C1 = {
    0x80: "€", 0x82: "‚", 0x83: "ƒ", 0x84: "„", 0x85: "…",
    0x86: "†", 0x87: "‡", 0x88: "ˆ", 0x89: "‰", 0x8a: "Š",
    0x8b: "‹", 0x8c: "Œ", 0x8e: "Ž", 0x91: "‘", 0x92: "’",
    0x93: "“", 0x94: "”", 0x95: "•", 0x96: "–", 0x97: "—",
    0x98: "˜", 0x99: "™", 0x9a: "š", 0x9b: "›", 0x9c: "œ",
    0x9e: "ž", 0x9f: "Ÿ",
}


def fix_text(s: str) -> str:
    return s.translate(_CP1252_C1)


def extract_wav(mp4: str, wav: str) -> bool:
    if os.path.exists(wav):
        return True
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
         "-ac", "1", "-ar", "16000", "-vn", wav],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def rss_mb() -> float:
    """Resident memory of this process, in MB (for OOM/leak monitoring)."""
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--cache", default="paper/meld_wavs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping utterances already cached (survives OOM restarts)")
    ap.add_argument("--max-new", type=int, default=0,
                    help="exit (code 3) after this many new records; a fresh process per chunk "
                         "releases funasr's leaked RAM, so the run can never OOM")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # UTF-8 file with Windows-1252 punctuation smuggled in as C1 controls; fix_text() repairs it.
    with open(args.csv, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    # group by dialogue, ordered by utterance, so recency smoothing stays within-conversation
    rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
    if args.limit:
        rows = rows[: args.limit]

    # Resume support: skip utterances already cached so an OOM restart never loses work.
    done: set[tuple[int, int]] = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["dia"], r["utt"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        mode = "a"
        print(f"resuming: {len(done)} utterances already cached", flush=True)

    print("Loading EmotionDetector...", flush=True)
    det = EmotionDetector()

    n_ok = n_missing = n_ffmpeg = 0
    stopped_early = False
    # buffering=1 -> line-buffered, so each record is flushed to disk (safe against OOM kills)
    with open(args.out, mode, encoding="utf-8", buffering=1) as out:
        for i, row in enumerate(rows):
            dia, utt = int(row["Dialogue_ID"]), int(row["Utterance_ID"])
            if (dia, utt) in done:
                continue

            gold = MELD_TO_UNIFIED.get(row["Emotion"].strip().lower())
            text = fix_text(row["Utterance"].strip())
            mp4 = os.path.join(args.videos, f"dia{dia}_utt{utt}.mp4")
            wav = os.path.join(args.cache, f"dia{dia}_utt{utt}.wav")

            if not os.path.exists(mp4):
                n_missing += 1
                continue
            if not extract_wav(mp4, wav):
                n_ffmpeg += 1
                continue

            # Cache only the two model outputs; all fusion/ablation happens in Step 3 from these
            # scores, so extraction is order-independent and safe to resume after a kill.
            with torch.inference_mode():
                voice = det.voice_scores(wav)
                text_scores = det.text_scores(text)
            n_words = len(text.split())

            out.write(json.dumps({
                "dia": dia, "utt": utt, "gold": gold, "text": text,
                "n_words": n_words, "voice": voice, "text_scores": text_scores,
            }) + "\n")
            n_ok += 1

            del voice, text_scores
            if (i + 1) % 25 == 0:            # keep memory bounded on a RAM-tight box
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(rows)}] new={n_ok} missing={n_missing} "
                      f"rss={rss_mb():.0f}MB", flush=True)

            if args.max_new and n_ok >= args.max_new:
                stopped_early = True
                print(f"  chunk limit {args.max_new} reached at [{i + 1}/{len(rows)}] "
                      f"(rss={rss_mb():.0f}MB); exiting for a fresh process", flush=True)
                break

    # Final summary reflects the whole cache file (this run plus any resumed records).
    gold_dist: Counter[str] = Counter()
    total = 0
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                gold_dist[json.loads(line)["gold"]] += 1
                total += 1
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"\n{'Chunk done' if stopped_early else 'Done'}. cache now holds {total} records -> {args.out}")
    print(f"  this run: new={n_ok} missing={n_missing} ffmpeg_failures={n_ffmpeg}")
    print(f"  gold distribution: {dict(gold_dist)}")
    if stopped_early:
        sys.exit(3)   # tell the wrapper loop there is more to do


if __name__ == "__main__":
    main()