#!/usr/bin/env python3
"""Features for the Expresso read-vs-improvised test (see expresso_fetch.py for why).

Per clip, three voice representations, matching the IEMOCAP/MELD runs exactly:
  dep    the deployed emotion2vec+ seven-class scores, folded exactly as
         aiko/emotion.py::voice_scores folds them (verified identical to calling it)
  e2v    emotion2vec+ 1024-d utterance embedding, from the SAME funasr call
  wavlm  WavLM-large mean-pooled layers 3..24 step 3 plus the all-layer mean, exactly as
         clean_extract_features.py stores them

Design: each (condition, speaker, style) cell is capped at CAP clips, drawn with a fixed
seed, so read and improvised are balanced within every speaker and style and the
compute is bounded. Clips under MIN_SECONDS are dropped (machine segmentation of the
dialogues leaves backchannel fragments); audio is cropped to 30 s.

funasr leaks RAM, so each call computes one part and exits 3 while parts remain:

    while :; do .venv/bin/python paper/expresso_extract.py; [ $? -ne 3 ] && break; done
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

SRC = "data/expresso"
FEATS = "paper/clean_feats"
PARTS = f"{FEATS}/expresso_parts"
CAP = 400
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
PART = 500
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]


def selection() -> list[dict]:
    rows = [json.loads(l) for l in open(f"{SRC}/manifest.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["duration"] >= MIN_SECONDS]
    cells = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        cells[(r["condition"], r["speaker"], r["gold"])].append(r)
    rng = np.random.default_rng(0)
    out = []
    for key in sorted(cells):
        c = cells[key]
        out.extend(c[i] for i in sorted(rng.permutation(len(c))[:CAP]))
    return out


def main() -> None:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    from aiko import config

    os.makedirs(PARTS, exist_ok=True)
    final = f"{FEATS}/expresso.npz"
    if os.path.exists(final):
        print("already merged")
        return
    rows = selection()
    nparts = -(-len(rows) // PART)
    missing = [k for k in range(nparts) if not os.path.exists(f"{PARTS}/{k}.npz")]
    if not missing:
        z = [np.load(f"{PARTS}/{k}.npz") for k in range(nparts)]
        np.savez(final, **{key: np.concatenate([p[key] for p in z])
                           for key in ("wavlm", "e2v", "dep")})
        with open(f"{FEATS}/expresso.meta.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")
        for k in range(nparts):
            os.remove(f"{PARTS}/{k}.npz")
        print(f"merged {len(rows)} -> {final}", flush=True)
        return

    from funasr import AutoModel
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        e2v = AutoModel(model=config.VOICE_EMOTION_MODEL, device="cuda", disable_update=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval()
    labels7 = config.UNIFIED_LABELS

    k = missing[0]
    chunk = rows[k * PART:(k + 1) * PART]
    W = np.zeros((len(chunk), len(LAYERS) + 1, 1024), dtype=np.float16)
    E = np.zeros((len(chunk), 1024), dtype=np.float16)
    D = np.zeros((len(chunk), len(labels7)), dtype=np.float32)
    for i, r in enumerate(chunk):
        x, sr = sf.read(f"{SRC}/wav/{r['id']}.wav", dtype="float32")
        x = x[: int(MAX_SECONDS * 16000)]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            res = e2v.generate(x, granularity="utterance", extract_embedding=True,
                               disable_pbar=True)[0]
        folded: dict[str, float] = {}
        for j, lab in enumerate(config.VOICE_LABELS):
            key = "neutral" if lab in ("other", "unknown") else lab
            folded[key] = folded.get(key, 0.0) + float(res["scores"][j])
        tot = sum(folded.values()) or 1.0
        D[i] = [folded.get(lab, 0.0) / tot for lab in labels7]
        E[i] = np.asarray(res["feats"], dtype=np.float32).astype(np.float16)
        inp = fe(x, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hs = wavlm(inp, output_hidden_states=True).hidden_states
        pooled = torch.stack([h[0].float().mean(0) for h in hs])
        W[i] = torch.cat([pooled[LAYERS], pooled.mean(0, keepdim=True)]).cpu().numpy()
    np.savez(f"{PARTS}/{k}.npz", wavlm=W, e2v=E, dep=D)
    print(f"part {k + 1}/{nparts} ({len(chunk)} clips)", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
