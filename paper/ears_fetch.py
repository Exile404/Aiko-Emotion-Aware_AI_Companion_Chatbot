#!/usr/bin/env python3
"""Fetch the EARS emotion recordings for the gap-1 contamination test.

Why EARS. The contamination question is whether emotion2vec+'s behaviour on IEMOCAP and
CREMA-D reflects having trained on their labels. That needs a corpus it cannot have seen.
EARS (Richter et al., Interspeech 2024; arXiv 2406.06185, first posted 10 June 2024) was
released AFTER emotion2vec+ (May 2024), so neither its audio nor its labels can be in
emotion2vec+'s training data. Unlike Expresso, whose labels are speaking styles, EARS has
acted EMOTIONS, the same kind of label as CREMA-D. And every one of its 107 speakers
recorded each emotion twice, once READING fixed sentences and once speaking FREEFORM, so
the read-versus-spontaneous contrast is paired within speaker with real statistical
power (Expresso had 4 speakers and 12 read recording blocks).

Per speaker, one zip (~600 MB, ~64 GB for all 107) is downloaded, only the needed emotion
recordings are kept (resampled 48 kHz float -> 16 kHz PCM16), and the zip is deleted, so
disk use stays near a few GB.

Emotions kept: anger, sadness, neutral (clean maps onto the deployed scheme), plus
amusement, contentment and extasy as candidate "happy" proxies for a sensitivity
analysis; EARS has no plain happiness category.

Output under data/ears/ (gitignored): wav/<speaker>_<emotion>_<read|freeform>.wav and
manifest.jsonl. Licence: CC-NC 4.0.

GitHub caps each connection at roughly 10 MB/s, so speakers are split across parallel
workers (speaker n goes to worker (n - 1) % workers), each with its own manifest part;
--merge folds the parts into manifest.jsonl. At most one zip per worker is on disk.

    for i in 0 1 2 3; do .venv/bin/python paper/ears_fetch.py --worker $i --workers 4 & done; wait
    .venv/bin/python paper/ears_fetch.py --merge
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT = "data/ears"
URL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{:03d}.zip"
EMOTIONS = ["anger", "sadness", "neutral", "amusement", "contentment", "extasy"]
TASKS = {"sentences": "read", "freeform": "freeform"}


def speakers_in(paths: list[str]) -> set[str]:
    done = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["speaker"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def merge() -> None:
    main_path = f"{OUT}/manifest.jsonl"
    rows, seen = [], set()
    for path in [main_path] + sorted(glob.glob(f"{OUT}/manifest.part*.jsonl")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    rows.sort(key=lambda r: r["id"])
    with open(main_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for path in glob.glob(f"{OUT}/manifest.part*.jsonl"):
        os.remove(path)
    spk = {r["speaker"] for r in rows}
    print(f"merged: {len(rows)} recordings from {len(spk)} speakers", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.makedirs(f"{OUT}/wav", exist_ok=True)
    os.makedirs(f"{OUT}/zips", exist_ok=True)
    if args.merge:
        merge()
        return
    man_path = f"{OUT}/manifest.part{args.worker}.jsonl"
    done = speakers_in([f"{OUT}/manifest.jsonl"] + glob.glob(f"{OUT}/manifest.part*.jsonl"))
    print(f"worker {args.worker}: {len(done)} speakers already processed", flush=True)

    with open(man_path, "a", encoding="utf-8", buffering=1) as man:
        for n in range(1, 108):
            spk = f"p{n:03d}"
            if spk in done or (n - 1) % args.workers != args.worker:
                continue
            zpath = f"{OUT}/zips/{spk}.zip"
            if not os.path.exists(zpath):
                tmp = zpath + ".part"
                rc = subprocess.run(["curl", "-sSL", "--retry", "5", "--retry-delay", "5",
                                     "-o", tmp, URL.format(n)]).returncode
                if rc != 0:
                    print(f"! {spk}: download failed (curl exit {rc}); will retry on rerun",
                          flush=True)
                    continue
                os.replace(tmp, zpath)
            rows = []
            try:
                with zipfile.ZipFile(zpath) as z:
                    names = set(z.namelist())
                    for emo in EMOTIONS:
                        for task, cond in TASKS.items():
                            member = f"{spk}/emo_{emo}_{task}.wav"
                            if member not in names:
                                continue
                            x, sr = sf.read(io.BytesIO(z.read(member)), dtype="float32")
                            if x.ndim > 1:
                                x = x.mean(axis=1)
                            if sr == 48000:
                                x = resample_poly(x, 1, 3).astype(np.float32)
                            elif sr != 16000:
                                sys.exit(f"! {member}: unexpected sample rate {sr}")
                            peak = float(np.max(np.abs(x))) if len(x) else 0.0
                            if peak > 0.999:            # float source may exceed PCM16 range
                                x = x / peak * 0.999
                            cid = f"{spk}_{emo}_{cond}"
                            sf.write(f"{OUT}/wav/{cid}.wav", x, 16000, subtype="PCM_16")
                            rows.append({"id": cid, "speaker": spk, "emotion": emo,
                                         "condition": cond,
                                         "duration": round(len(x) / 16000, 3)})
            except zipfile.BadZipFile:
                os.remove(zpath)
                print(f"! {spk}: corrupt zip removed; will retry on rerun", flush=True)
                continue
            for r in rows:
                man.write(json.dumps(r) + "\n")
            os.remove(zpath)
            print(f"{spk}: kept {len(rows)} recordings", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
