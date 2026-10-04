#!/usr/bin/env python3
"""Blind per-utterance SNR proxy: recording condition as a continuous variable.

The ladder in iemocap_results.md treats recording condition as a corpus label (clean
IEMOCAP, degraded MELD). That is coarse, and it confounds the audio with everything else
that differs between corpora. A per-utterance quality estimate does two jobs:

  1. it lets a deployable fusion rule condition on audio quality (fusion_transfer.py),
     since the estimate is computable from the microphone signal alone at runtime;
  2. it places MELD on the same axis as the controlled degradations in
     noise_degradation.py, by finding the added-noise level at which degraded IEMOCAP
     reaches MELD's estimated quality.

Estimator: frame energies over 25 ms windows with 10 ms hop; noise power is the mean of
the quietest 15 % of frames, signal power the mean of the loudest 50 % minus the noise
power. This is deliberately simple and is NOT an absolute SNR measurement. Its use here
only requires that it is monotone in added noise, which noise_degradation.py verifies
on the controlled mixtures, and that the SAME estimator is applied everywhere.

    .venv/bin/python paper/snr_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

FRAME, HOP, SR = 400, 160, 16000            # 25 ms / 10 ms at 16 kHz
NOISE_Q, SIGNAL_Q = 0.15, 0.50
CLIP_DB = (-20.0, 80.0)


def snr_proxy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    e = np.sort((x[idx] ** 2).mean(axis=1) + 1e-12)
    p_noise = e[: max(1, int(NOISE_Q * n))].mean()
    p_sig = e[-max(1, int(SIGNAL_Q * n)):].mean()
    snr = 10 * np.log10(max(p_sig - p_noise, 1e-12) / p_noise)
    return float(np.clip(snr, *CLIP_DB))


def main() -> None:
    import soundfile as sf
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean_extract_features import manifest

    out = "paper/snr_proxy.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for name, rows in manifest().items():
            vals = []
            for r in rows:
                try:
                    wav, _ = sf.read(r["wav"], dtype="float32")
                except Exception:
                    continue
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                s = snr_proxy(wav)
                vals.append(s)
                f.write(json.dumps({"corpus": r["corpus"], "split": r["split"],
                                    "clip": r["clip"], "snr_db": round(s, 3)}) + "\n")
                n += 1
            q = np.percentile(vals, [25, 50, 75])
            print(f"{name:12s} n={len(vals):5d}  SNR proxy median {q[1]:5.1f} dB "
                  f"(IQR {q[0]:5.1f} to {q[2]:5.1f})", flush=True)
    print(f"\n{n} utterances -> {out}")


if __name__ == "__main__":
    main()
