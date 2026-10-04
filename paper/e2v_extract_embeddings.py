#!/usr/bin/env python3
"""emotion2vec+ utterance embeddings, row-aligned with paper/clean_feats, for the
contamination check in e2v_probe_eval.py.

The deployed encoder showed no script/improvised voice difference on IEMOCAP; a
contamination-free WavLM probe showed a 12-14 point one. Those two setups differ in TWO
ways at once: the encoder (emotion2vec+, which saw IEMOCAP labels through EmoBox, versus
WavLM, which saw no emotion labels) and the read-out (emotion2vec+'s own zero-shot
classification head versus a linear probe trained in-domain). Probing emotion2vec+'s
EMBEDDINGS with exactly the WavLM protocol holds the read-out fixed, so any remaining
difference belongs to the encoder.

Rows follow clean_extract_features.manifest(), so index i here is index i there; clip
ids are stored and checked downstream.

Audio is cropped to the first 30 s, exactly as clean_extract_features.py crops it for
WavLM. MELD's test split contains two whole-scene "utterances" (235 s and 305 s); fed to
emotion2vec+ uncropped they exhausted system RAM and got the worker OOM-killed.
`--repair-long` recomputes, cropped, any rows of already-merged groups whose source
audio exceeds 30 s, so every row in every group is produced the same way.

funasr leaks RAM, so each call computes one part and exits 3 while parts remain:

    while :; do .venv/bin/python paper/e2v_extract_embeddings.py; [ $? -ne 3 ] && break; done
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

from clean_extract_features import manifest

GROUPS = ["iemocap_all", "meld_train", "meld_dev", "meld_test"]
FEATS = "paper/clean_feats"
PARTS = f"{FEATS}/e2v_parts"
PART = 500                       # small parts keep the funasr leak far from the RAM ceiling
MAX_SECONDS = 30.0               # must match clean_extract_features.MAX_SECONDS


def load_model():
    from funasr import AutoModel
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return AutoModel(model="iic/emotion2vec_plus_large", device="cuda",
                         disable_update=True)


def embed(model, wav_path: str) -> np.ndarray:
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        sys.exit(f"! {wav_path} is {sr} Hz; expected 16 kHz")
    x = x[: int(MAX_SECONDS * 16000)]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        res = model.generate(x, granularity="utterance", extract_embedding=True,
                             disable_pbar=True)
    return np.asarray(res[0]["feats"], dtype=np.float32).astype(np.float16)


def repair_long(groups: dict) -> None:
    """Recompute, cropped, rows of merged groups whose source audio exceeds 30 s."""
    model = None
    for g in GROUPS:
        final = f"{FEATS}/{g}.e2v.npz"
        if not os.path.exists(final):
            continue
        z = np.load(final)
        emb, clips = z["e2v"].copy(), z["clip"]
        long_idx = [i for i, r in enumerate(groups[g])
                    if sf.info(r["wav"]).duration > MAX_SECONDS]
        if not long_idx:
            continue
        model = model or load_model()
        for i in long_idx:
            emb[i] = embed(model, groups[g][i]["wav"])
        np.savez(final, e2v=emb, clip=clips)
        print(f"{g}: re-embedded {len(long_idx)} clip(s) over {MAX_SECONDS:.0f} s, cropped",
              flush=True)


def main() -> None:
    os.makedirs(PARTS, exist_ok=True)
    groups = manifest()
    if "--repair-long" in sys.argv:
        repair_long(groups)
        return
    model = None
    for g in GROUPS:
        final = f"{FEATS}/{g}.e2v.npz"
        if os.path.exists(final):
            continue
        rows = groups[g]
        nparts = -(-len(rows) // PART)
        missing = [k for k in range(nparts) if not os.path.exists(f"{PARTS}/{g}.{k}.npy")]
        if not missing:
            emb = np.concatenate([np.load(f"{PARTS}/{g}.{k}.npy") for k in range(nparts)])
            np.savez(final, e2v=emb, clip=np.array([r["clip"] for r in rows]))
            for k in range(nparts):
                os.remove(f"{PARTS}/{g}.{k}.npy")
            print(f"{g}: merged {len(rows)} -> {final}", flush=True)
            continue
        if model is None:
            model = load_model()
        k = missing[0]
        chunk = rows[k * PART:(k + 1) * PART]
        out = np.zeros((len(chunk), 1024), dtype=np.float16)
        for i, r in enumerate(chunk):
            out[i] = embed(model, r["wav"])
        np.save(f"{PARTS}/{g}.{k}.npy", out)
        print(f"{g}: part {k + 1}/{nparts} ({len(chunk)} utterances)", flush=True)
        sys.exit(3)                     # a fresh process merges or takes the next part
    print("all groups done", flush=True)


if __name__ == "__main__":
    main()
