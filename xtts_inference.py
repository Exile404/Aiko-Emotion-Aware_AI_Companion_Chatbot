#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
sys.setrecursionlimit(10000)

import torch
_orig = torch.load
def _safe(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _safe

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CONFIG_PATH = "./xtts_finetune_output/XTTS_v2.0_original_model_files/config.json"
CHECKPOINT = "./xtts_finetune_output/aiko_xtts_finetune-April-26-2026_05+25AM-abb2f27/best_model_515.pth"
VOCAB = "./xtts_finetune_output/XTTS_v2.0_original_model_files/vocab.json"
SPEAKER_REF = "./voice_refs/long_aiko_01_warm_neutral.wav"

print("Loading config...")
config = XttsConfig()
config.load_json(CONFIG_PATH)

print("Loading model...")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=CHECKPOINT, vocab_path=VOCAB, use_deepspeed=False)
model.cuda()

print(f"Computing speaker latents from {SPEAKER_REF}...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REF])

texts = [
    "Hey love, how was your day? I missed you so much.",
    "I am so proud of you. That is amazing news!",
    "Come here, tell me everything. I am here for you.",
]

import soundfile as sf
for i, text in enumerate(texts):
    print(f"\nGenerating: {text}")
    out = model.inference(text, "en", gpt_cond_latent, speaker_embedding, temperature=0.7)
    sf.write(f"./aiko_test_{i}.wav", out["wav"], 24000)
    print(f"  Saved: ./aiko_test_{i}.wav")

print("\nDONE")
