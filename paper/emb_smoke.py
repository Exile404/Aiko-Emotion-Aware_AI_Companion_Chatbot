#!/usr/bin/env python3
"""Confirm the emotion2vec utterance embedding is retrievable and its dim, before full extraction.

    .venv/bin/python paper/emb_smoke.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiko import config

import torch
from funasr import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel(model=config.VOICE_EMOTION_MODEL, device=device)

wavs = sorted(glob.glob("paper/meld_wavs/*.wav"))[:2]
print(f"testing {len(wavs)} cached wavs\n")
for w in wavs:
    res = model.generate(w, granularity="utterance", extract_embedding=True)
    r = res[0]
    print(os.path.basename(w), "-> keys:", list(r.keys()))
    for k, v in r.items():
        try:
            arr = np.asarray(v)
            print(f"    {k}: shape={arr.shape} dtype={arr.dtype}")
        except Exception:
            print(f"    {k}: {type(v).__name__} = {str(v)[:60]}")
    print()