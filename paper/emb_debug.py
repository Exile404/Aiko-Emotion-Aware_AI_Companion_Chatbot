#!/usr/bin/env python3
"""Find the clip that kills embedding extraction: print each clip BEFORE processing it."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiko import config

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel

rows = [json.loads(l) for l in open("paper/meld_test_scores.jsonl", encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: (r["dia"], r["utt"]))

print("loading model...", flush=True)
m = AutoModel(model=config.VOICE_EMOTION_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
print("model loaded", flush=True)

for i in range(348, 370):
    r = rows[i]
    f = f"paper/meld_wavs/dia{r['dia']}_utt{r['utt']}.wav"
    dur = sf.info(f).duration if os.path.exists(f) else -1
    print(f"[{i}] dia{r['dia']}_utt{r['utt']} dur={dur:.1f}s words={r['n_words']}", flush=True)
    res = m.generate(f, granularity="utterance", extract_embedding=True)
    print(f"    ok feats={np.asarray(res[0]['feats']).shape}", flush=True)

print("all done", flush=True)