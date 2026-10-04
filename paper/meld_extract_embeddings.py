#!/usr/bin/env python3
"""Extract 1024-dim emotion2vec utterance embeddings for cached MELD clips.

Reuses the wav cache and the (dia,utt) list from a scores JSONL. OOM-proof: chunked
(--max-new) + resumable (--resume), run in a while-loop.

    .venv/bin/python paper/meld_extract_embeddings.py \
        --scores paper/meld_test_scores.jsonl --cache paper/meld_wavs \
        --out paper/meld_test_emb.jsonl --resume --max-new 200
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from funasr import AutoModel

from aiko import config


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores JSONL providing the (dia,utt) list")
    ap.add_argument("--cache", required=True, help="wav cache dir for this split")
    ap.add_argument("--out", required=True, help="output embeddings JSONL")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=20.0,
                    help="truncate audio to this many seconds before embedding; caps memory "
                         "on pathologically long MELD clips (e.g. dia38_utt4 = 305s)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scores, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: (r["dia"], r["utt"]))

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
        print(f"resuming: {len(done)} already embedded", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading emotion2vec ({device})...", flush=True)
    model = AutoModel(model=config.VOICE_EMOTION_MODEL, device=device)

    n_ok = n_missing = 0
    stopped_early = False
    with open(args.out, mode, encoding="utf-8", buffering=1) as out:
        for i, r in enumerate(rows):
            dia, utt = r["dia"], r["utt"]
            if (dia, utt) in done:
                continue
            wav = os.path.join(args.cache, f"dia{dia}_utt{utt}.wav")
            if not os.path.exists(wav):
                n_missing += 1
                continue
            audio, sr = sf.read(wav, dtype="float32")
            if audio.ndim > 1:                       # safety: force mono
                audio = audio.mean(axis=1)
            cap = int(args.max_sec * sr)
            if audio.shape[0] > cap:                 # bound memory on pathologically long clips
                audio = audio[:cap]
            with torch.inference_mode():
                res = model.generate(audio, granularity="utterance", extract_embedding=True)
            emb = np.asarray(res[0]["feats"], dtype=np.float32)
            out.write(json.dumps({"dia": dia, "utt": utt,
                                  "emb": [round(float(x), 5) for x in emb]}) + "\n")
            n_ok += 1

            del res, emb
            if (i + 1) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(rows)}] new={n_ok} missing={n_missing} rss={rss_mb():.0f}MB", flush=True)
            if args.max_new and n_ok >= args.max_new:
                stopped_early = True
                print(f"  chunk limit {args.max_new} reached; exiting for a fresh process", flush=True)
                break

    total = sum(1 for _ in open(args.out, encoding="utf-8"))
    print(f"\n{'Chunk done' if stopped_early else 'Done'}. {total} embeddings -> {args.out}")
    print(f"  this run: new={n_ok} missing={n_missing}")
    if stopped_early:
        sys.exit(3)


if __name__ == "__main__":
    main()