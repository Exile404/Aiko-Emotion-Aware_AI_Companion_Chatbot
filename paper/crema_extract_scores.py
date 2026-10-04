#!/usr/bin/env python3
"""Cache emotion2vec voice scores over CREMA-D (clean acted studio speech).

Why: the C1 finding is that voice adds nothing on MELD. Before that can be attributed
to MELD's audio (laugh track, background music, overlapping speakers) rather than to
our pipeline, the voice encoder has to be shown competent on clean audio. CREMA-D is
free, needs no license, and its 12 fixed sentences mean text carries no emotion
information by construction, so it isolates the voice modality.

CREMA-D has SIX emotion categories, not seven: there is no `surprised`. Any comparison
against MELD must restrict both corpora to the shared six, which crema_eval.py does,
or the CREMA-D number is flattered by a smaller label space.

Filenames are ActorID_Sentence_Emotion_Intensity.wav, e.g. 1001_DFA_ANG_XX.wav.

funasr leaks RAM, so run in chunks, each in a fresh process:

    while :; do .venv/bin/python paper/crema_extract_scores.py \
        --audio data/CREMA-D/AudioWAV --resume --max-new 500; \
        [ $? -ne 3 ] && break; done
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
import torch

from aiko.emotion import EmotionDetector

CREMA_TO_UNIFIED = {
    "ANG": "angry", "DIS": "disgusted", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# The 12 fixed carrier sentences. Wording is approximate and immaterial: every sentence
# is recorded in all six emotions, so text is at chance by design regardless of exact
# transcription. These exist only to produce an EMPIRICAL text-at-chance control rather
# than asserting one.
SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


def parse_name(stem: str) -> tuple[str, str, str, str] | None:
    """1001_DFA_ANG_XX -> (actor, sentence, emotion, intensity); None if unparseable."""
    parts = stem.split("_")
    if len(parts) != 4 or parts[2] not in CREMA_TO_UNIFIED:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def ensure_16k_mono(src: str, cache_dir: str) -> str | None:
    """emotion2vec expects 16 kHz mono. Convert only when the source is not already
    that, so an already-conformant corpus costs zero ffmpeg calls."""
    try:
        info = sf.info(src)
        if info.samplerate == 16000 and info.channels == 1:
            return src
    except Exception:
        return None
    dst = os.path.join(cache_dir, os.path.basename(src))
    if os.path.exists(dst):
        return dst
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", "16000", "-vn", dst],
        capture_output=True, text=True,
    )
    return dst if proc.returncode == 0 else None


def is_uniform(scores: dict[str, float]) -> bool:
    """voice_scores() swallows exceptions and returns a uniform distribution, which would
    otherwise be silently counted as a real prediction."""
    if not scores:
        return True
    vals = list(scores.values())
    return max(vals) - min(vals) < 1e-9


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="CREMA-D AudioWAV directory")
    ap.add_argument("--out", default="paper/crema_scores.jsonl")
    ap.add_argument("--cache", default="paper/crema_wavs_16k")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping clips already cached")
    ap.add_argument("--max-new", type=int, default=0,
                    help="exit (code 3) after this many new records so a fresh process "
                         "releases funasr's leaked RAM")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    files = sorted(f for f in os.listdir(args.audio) if f.lower().endswith(".wav"))
    if not files:
        print(f"! no .wav files in {args.audio}")
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    done: set[str] = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["clip"])
                except (json.JSONDecodeError, KeyError):
                    continue
        mode = "a"
        print(f"resuming: {len(done)} clips already cached", flush=True)

    print("Loading EmotionDetector...", flush=True)
    det = EmotionDetector()

    # Only 12 distinct strings, so the text model runs 12 times, not 7,442.
    text_cache: dict[str, dict[str, float]] = {}

    n_ok = n_bad = n_conv = n_fail = 0
    stopped_early = False
    with open(args.out, mode, encoding="utf-8", buffering=1) as out:
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            if stem in done:
                continue
            meta = parse_name(stem)
            if meta is None:
                n_bad += 1
                continue
            actor, sent, emo, intensity = meta

            wav = ensure_16k_mono(os.path.join(args.audio, fn), args.cache)
            if wav is None:
                n_conv += 1
                continue

            with torch.inference_mode():
                voice = det.voice_scores(wav)
                if sent not in text_cache:
                    text_cache[sent] = det.text_scores(SENTENCES.get(sent, ""))
            if is_uniform(voice):
                n_fail += 1          # model failed on this clip; do not cache as data
                continue

            out.write(json.dumps({
                "clip": stem, "actor": actor, "sentence": sent,
                "intensity": intensity, "gold": CREMA_TO_UNIFIED[emo],
                "voice": voice, "text_scores": text_cache[sent],
            }) + "\n")
            n_ok += 1

            del voice
            if (i + 1) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(files)}] new={n_ok} rss={rss_mb():.0f}MB", flush=True)

            if args.max_new and n_ok >= args.max_new:
                stopped_early = True
                print(f"  chunk limit {args.max_new} reached at [{i + 1}/{len(files)}] "
                      f"(rss={rss_mb():.0f}MB); exiting for a fresh process", flush=True)
                break

    gold_dist: Counter[str] = Counter()
    total = 0
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                gold_dist[json.loads(line)["gold"]] += 1
                total += 1
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"\n{'Chunk done' if stopped_early else 'Done'}. cache holds {total} -> {args.out}")
    print(f"  this run: new={n_ok} unparseable={n_bad} convert_fail={n_conv} "
          f"model_fail={n_fail}")
    print(f"  gold distribution: {dict(gold_dist)}")
    if stopped_early:
        sys.exit(3)


if __name__ == "__main__":
    main()