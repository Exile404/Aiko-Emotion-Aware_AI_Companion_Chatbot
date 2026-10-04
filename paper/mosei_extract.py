#!/usr/bin/env python3
"""Features for every CMU-MOSEI segment (gap 2; see mosei_prepare.py for why MOSEI).

Per segment, the representations used everywhere else in paper/:
  dep    deployed emotion2vec+ seven-class scores, folded as aiko/emotion.py folds them
  e2v    emotion2vec+ 1024-d embedding (same funasr call)
  wavlm  WavLM-large layers 3..24 step 3 plus all-layer mean
  minilm MiniLM-L6 embedding of the segment's human transcript (added at merge)

Segments are cut from MOSEI's per-video audio by their start/end times and cropped to the
first 30 s, as everywhere else. All 23,259 segments are processed, so that the label
mapping (intensities -> classes) can be varied at evaluation time without re-extraction.

funasr leaks RAM, so each call computes one part and exits 3 while parts remain:

    while :; do .venv/bin/python paper/mosei_extract.py; [ $? -ne 3 ] && break; done
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

FEATS = "paper/clean_feats"
PARTS = f"{FEATS}/mosei_parts"
PART = 1000
MAX_SECONDS = 30.0
MIN_SAMPLES = 1600
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]


def main() -> None:
    rows = [json.loads(l) for l in open("data/mosei/manifest.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: (r["file"], r["start"]))
    os.makedirs(PARTS, exist_ok=True)
    final = f"{FEATS}/mosei.npz"
    if os.path.exists(final):
        print("already merged")
        return
    nparts = -(-len(rows) // PART)
    missing = [k for k in range(nparts) if not os.path.exists(f"{PARTS}/{k}.npz")]
    if not missing:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        text = st.encode([r["text"] for r in rows], batch_size=256, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
        z = [np.load(f"{PARTS}/{k}.npz") for k in range(nparts)]
        np.savez(final, minilm=text, **{key: np.concatenate([p[key] for p in z])
                                        for key in ("wavlm", "e2v", "dep")})
        with open(f"{FEATS}/mosei.meta.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("text", "file")}
                                   | {"n_words": len(r["text"].split())}) + "\n")
        for k in range(nparts):
            os.remove(f"{PARTS}/{k}.npz")
        print(f"merged {len(rows)} segments -> {final}", flush=True)
        return

    import torch
    from funasr import AutoModel
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    from aiko import config

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
    cur_file, audio, sr = None, None, 16000
    for i, r in enumerate(chunk):
        if r["file"] != cur_file:
            audio, sr = sf.read(r["file"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                sys.exit(f"! {r['file']}: {sr} Hz, expected 16 kHz")
            cur_file = r["file"]
        x = audio[int(r["start"] * sr):int(r["end"] * sr)][: int(MAX_SECONDS * sr)]
        if len(x) < MIN_SAMPLES:
            x = np.pad(x, (0, MIN_SAMPLES - len(x)))
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
    print(f"part {k + 1}/{nparts} ({len(chunk)} segments)", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
