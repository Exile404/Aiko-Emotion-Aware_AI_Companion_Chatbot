#!/usr/bin/env python3
"""Voice features for EARS, windowed, for the gap-1 tests (see ears_fetch.py for why EARS).

Each EARS recording holds several sentences (read, ~10 s) or a free monologue (~15 s), one
recording per speaker x emotion x condition. Recordings are cut into non-overlapping
WINDOW-second windows, so read and freeform are length-matched by construction and
utterance duration cannot confound the read-versus-freeform contrast. Windows that are
mostly silence (fewer than half their 25 ms frames within 35 dB of the recording's
loudest frame) are dropped, and each recording contributes at most MAX_WINDOWS windows so
read and freeform carry equal weight per speaker and emotion.

Per window, exactly the representations used everywhere else in paper/:
  dep    deployed emotion2vec+ seven-class scores, folded as aiko/emotion.py folds them
  e2v    emotion2vec+ 1024-d embedding (same funasr call)
  wavlm  WavLM-large layers 3..24 step 3 plus all-layer mean

funasr leaks RAM, so each call computes one part and exits 3 while parts remain:

    while :; do .venv/bin/python paper/ears_extract.py; [ $? -ne 3 ] && break; done
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

SRC = "data/ears"
FEATS = "paper/clean_feats"
PARTS = f"{FEATS}/ears_parts"
WINDOW = 3.0
MAX_WINDOWS = 3
PART = 600
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]


def windows(x: np.ndarray, sr: int = 16000) -> list[tuple[int, np.ndarray]]:
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[np.clip(idx, 0, len(x) - 1)] ** 2).mean(axis=1) + 1e-12)
    active = rms >= rms.max() * 10 ** (-35 / 20)
    w = int(WINDOW * sr)
    out = []
    for k in range(len(x) // w):
        a, b = k * w, (k + 1) * w
        fa, fb = a // hop, min(n, b // hop)
        if fb > fa and active[fa:fb].mean() >= 0.5:
            out.append((k, x[a:b]))
        if len(out) == MAX_WINDOWS:
            break
    return out


def selection() -> list[dict]:
    recs = [json.loads(l) for l in open(f"{SRC}/manifest.jsonl", encoding="utf-8")]
    recs.sort(key=lambda r: r["id"])
    out = []
    for r in recs:
        x, _ = sf.read(f"{SRC}/wav/{r['id']}.wav", dtype="float32")
        for k, _w in windows(x):
            out.append({"id": f"{r['id']}_w{k}", "recording": r["id"], "window": k,
                        "speaker": r["speaker"], "emotion": r["emotion"],
                        "condition": r["condition"], "duration": WINDOW})
    return out


def main() -> None:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    from aiko import config

    os.makedirs(PARTS, exist_ok=True)
    final = f"{FEATS}/ears.npz"
    if os.path.exists(final):
        print("already merged")
        return
    sel_path = f"{PARTS}/selection.jsonl"
    if not os.path.exists(sel_path):        # freeze the window list once, for alignment
        with open(sel_path, "w", encoding="utf-8") as f:
            for r in selection():
                f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(sel_path, encoding="utf-8")]
    nparts = -(-len(rows) // PART)
    missing = [k for k in range(nparts) if not os.path.exists(f"{PARTS}/{k}.npz")]
    if not missing:
        z = [np.load(f"{PARTS}/{k}.npz") for k in range(nparts)]
        np.savez(final, **{key: np.concatenate([p[key] for p in z])
                           for key in ("wavlm", "e2v", "dep")})
        with open(f"{FEATS}/ears.meta.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        for k in range(nparts):
            os.remove(f"{PARTS}/{k}.npz")
        os.remove(sel_path)
        print(f"merged {len(rows)} windows -> {final}", flush=True)
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
    cache: dict[str, np.ndarray] = {}
    for i, r in enumerate(chunk):
        if r["recording"] not in cache:
            cache.clear()
            cache[r["recording"]] = sf.read(f"{SRC}/wav/{r['recording']}.wav",
                                            dtype="float32")[0]
        s = int(r["window"] * WINDOW * 16000)
        x = cache[r["recording"]][s:s + int(WINDOW * 16000)]
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
    print(f"part {k + 1}/{nparts} ({len(chunk)} windows)", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
