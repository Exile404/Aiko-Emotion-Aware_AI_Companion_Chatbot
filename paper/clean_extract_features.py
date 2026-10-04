#!/usr/bin/env python3
"""Contamination-free features for the C1 boundary condition: WavLM (voice), MiniLM (text).

Why this exists. Both deployed encoders have seen the corpora they are evaluated on.
emotion2vec+ descends from a seed fine-tuned on EmoBox, which contains IEMOCAP, MELD and
CREMA-D, and the DistilRoBERTa text model lists MELD among its training sets. Zero-shot
accuracy on those corpora therefore cannot be read as encoder competence, and the
voice/text inversion could be partly an artifact of WHICH corpus each encoder memorised.

The replacement encoders were chosen because neither has seen emotion labels:
  - microsoft/wavlm-large, self-supervised on Libri-Light, GigaSpeech and VoxPopuli
    (audiobooks, podcasts and video, parliament speech). No emotion corpus, no labels.
  - sentence-transformers/all-MiniLM-L6-v2, contrastive sentence embeddings trained
    without emotion labels.
Both stay FROZEN. All emotion knowledge comes from small probes trained only on each
corpus's training partition (clean_probe_eval.py), so nothing about a test utterance
reaches the model that scores it.

Stored per utterance: WavLM hidden states mean-pooled over time for layers 3, 6, ..., 24
plus the mean over all 25 hidden states (the layer is chosen downstream on training data
only), and the L2-normalised MiniLM embedding of the transcript.

Transcripts come from paper/.text_backup/, which is local only and never committed.
Outputs go to paper/clean_feats/, which is gitignored (binary, regenerable).

    .venv/bin/python paper/clean_extract_features.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

IEMOCAP_ROOT = "IEMOCAP_full_release/IEMOCAP_full_release"
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]          # stored singly; index 8 = all-layer mean
MAX_SECONDS = 30.0                              # crop pathological long clips
MIN_SAMPLES = 1600                              # WavLM's conv stack needs a few frames


def load_backup(name: str) -> list[dict]:
    path = f"paper/.text_backup/{name}.jsonl"
    if not os.path.exists(path):
        sys.exit(f"! missing {path}: full-text caches are needed for the text probes")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def manifest() -> dict[str, list[dict]]:
    """Every utterance whose gold label is in the deployed scheme, grouped by output file."""
    groups: dict[str, list[dict]] = {}

    from crema_extract_scores import SENTENCES
    rows = []
    for r in (json.loads(l) for l in open("paper/crema_scores.jsonl", encoding="utf-8")):
        rows.append({
            "corpus": "crema", "split": "all", "clip": r["clip"], "gold": r["gold"],
            "group": r["actor"], "speaker": r["actor"], "kind": "acted",
            "wav": f"data/CREMA-D/AudioWAV/{r['clip']}.wav",
            "text": SENTENCES.get(r["sentence"], ""),
        })
    groups["crema_all"] = rows

    rows = []
    for r in load_backup("iemocap_scores"):
        if not r.get("gold"):
            continue                                    # fru/oth: no deployed class
        rows.append({
            "corpus": "iemocap", "split": "all", "clip": r["clip"], "gold": r["gold"],
            "group": r["dialog"], "speaker": r["speaker"], "session": r["session"],
            "kind": r["kind"], "duration": r.get("duration"),
            "wav": os.path.join(IEMOCAP_ROOT, f"Session{r['session']}", "sentences", "wav",
                                r["dialog"], f"{r['clip']}.wav"),
            "text": r.get("text") or "",
        })
    groups["iemocap_all"] = rows

    for split, wavdir in (("train", "paper/meld_wavs_train"), ("dev", "paper/meld_wavs_dev"),
                          ("test", "paper/meld_wavs")):
        rows = []
        for r in load_backup(f"meld_{split}_scores"):
            rows.append({
                "corpus": "meld", "split": split, "clip": f"{split}_dia{r['dia']}_utt{r['utt']}",
                "gold": r["gold"], "group": f"{split}_{r['dia']}", "speaker": None,
                "kind": "acted_tv", "dia": r["dia"], "utt": r["utt"],
                "wav": f"{wavdir}/dia{r['dia']}_utt{r['utt']}.wav",
                "text": r.get("text") or "",
            })
        groups[f"meld_{split}"] = rows
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/clean_feats")
    ap.add_argument("--only", default="", help="comma list of groups, e.g. meld_test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval()
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)

    groups = manifest()
    wanted = set(args.only.split(",")) if args.only else set(groups)
    for name, rows in groups.items():
        if name not in wanted:
            continue
        npz = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(npz):
            print(f"{name}: exists, skipping")
            continue
        t0 = time.time()
        feats = np.zeros((len(rows), len(LAYERS) + 1, 1024), dtype=np.float16)
        missing = 0
        for i, r in enumerate(rows):
            try:
                wav, sr = sf.read(r["wav"], dtype="float32")
            except Exception:
                missing += 1
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                sys.exit(f"! {r['wav']} is {sr} Hz; expected 16 kHz")
            wav = wav[: int(MAX_SECONDS * 16000)]
            if len(wav) < MIN_SAMPLES:
                wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)))
            x = fe(wav, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16,
                                                        enabled=dev == "cuda"):
                hs = wavlm(x, output_hidden_states=True).hidden_states
            pooled = torch.stack([h[0].float().mean(0) for h in hs])         # (25, 1024)
            sel = torch.cat([pooled[LAYERS], pooled.mean(0, keepdim=True)])    # (9, 1024)
            feats[i] = sel.cpu().numpy().astype(np.float16)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {name}: {i + 1}/{len(rows)}  {rate:.0f} utt/s", flush=True)

        texts = [r["text"] for r in rows]
        temb = minilm.encode(texts, batch_size=256, normalize_embeddings=True,
                             show_progress_bar=False).astype(np.float32)
        np.savez(npz, wavlm=feats, minilm=temb)
        with open(os.path.join(args.out, f"{name}.meta.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("text", "wav")}
                                   | {"n_words": len(r["text"].split())}) + "\n")
        print(f"{name}: {len(rows)} utterances, {missing} unreadable, "
              f"{time.time() - t0:.0f}s -> {npz}", flush=True)


if __name__ == "__main__":
    main()
