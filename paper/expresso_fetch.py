#!/usr/bin/env python3
"""Fetch the Expresso clips needed for the label-uncontaminated read-vs-improvised test.

Why Expresso. On IEMOCAP, emotion2vec+ and a contamination-free WavLM probe disagree
about scripted versus improvised speech (e2v_probe_results.md, E-3), and probing
emotion2vec+'s own embeddings shows the disagreement lives in its representation. The
question is whether that is CONTAMINATION (it saw IEMOCAP's labels via EmoBox) or a
property of the encoder. That needs a corpus with the same read/improvised structure
whose labels emotion2vec+ never trained on. Expresso (Meta, CC BY-NC 4.0) is not in
EmoBox, and has the same four actors both READING and IMPROVISING in matching styles.

Limits, stated up front: labels are the INTENDED style, not perceived emotion; only
three styles map onto the deployed scheme (happy, sad, default -> neutral); four
speakers; and Expresso's audio, though not its labels, could sit in emotion2vec+'s
undisclosed pseudo-labelled pool. MSP-IMPROV remains the definitive test.

Sources (streamed; only matching rows are kept, resampled 48 kHz -> 16 kHz mono):
  ylacombe/expresso                  read speech, 11,615 rows
  nytopop/expresso-conversational    improvised dialogue, machine-segmented, 31,150 rows

Output under data/expresso/ (gitignored): wav/<id>.wav and manifest.jsonl.

    .venv/bin/python paper/expresso_fetch.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT = "data/expresso"
KEEP = {"default": "neutral", "happy": "happy", "sad": "sad"}
SOURCES = [("read", "ylacombe/expresso"), ("improvised", "nytopop/expresso-conversational")]


def conv_key(r: dict) -> str:
    """Speaker-order-independent id of the dialogue a conversational segment came from.

    ids look like ex04-ex01_animal-animaldir_007_312480_1378320: pair, styles, take
    index, then sample offsets. Both partners' segments map to the same key.
    """
    take = r["id"].split("_")[2]
    pair = "-".join(sorted([r["speaker_id"], r["other_speaker_id"]]))
    styles = "-".join(sorted([r["style"], r["other_style"]]))
    return f"{pair}_{styles}_{take}"


def main() -> None:
    from datasets import Audio, load_dataset

    os.makedirs(f"{OUT}/wav", exist_ok=True)
    man_path = f"{OUT}/manifest.jsonl"
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"resuming with {len(done)} clips already kept", flush=True)

    with open(man_path, "a", encoding="utf-8", buffering=1) as man:
        for condition, repo in SOURCES:
            ds = load_dataset(repo, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            t0, seen, kept = time.time(), 0, 0
            for r in ds:
                seen += 1
                if seen % 2000 == 0:
                    print(f"  {condition}: scanned {seen}, kept {kept} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                if r["style"] not in KEEP or r["id"] in done:
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr == 48000:
                    x = resample_poly(x, 1, 3).astype(np.float32)
                elif sr != 16000:
                    sys.exit(f"! unexpected sample rate {sr} in {r['id']}")
                sf.write(f"{OUT}/wav/{r['id']}.wav", x, 16000, subtype="PCM_16")
                man.write(json.dumps({
                    "id": r["id"], "condition": condition, "speaker": r["speaker_id"],
                    "style": r["style"], "gold": KEEP[r["style"]],
                    "duration": round(len(x) / 16000, 3),
                    "group": conv_key(r) if condition == "improvised"
                    else f"{r['speaker_id']}_{r['style']}_read",
                    "text": r.get("text") or "",
                }) + "\n")
                kept += 1
            print(f"{condition}: scanned {seen}, kept {kept} in {time.time() - t0:.0f}s",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
