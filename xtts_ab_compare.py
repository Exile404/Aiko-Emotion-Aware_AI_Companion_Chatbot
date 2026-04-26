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
import soundfile as sf
import gc

CONFIG_PATH = "./xtts_finetune_output/XTTS_v2.0_original_model_files/config.json"
VOCAB = "./xtts_finetune_output/XTTS_v2.0_original_model_files/vocab.json"
SPEAKER_REF = "./voice_refs/long_aiko_01_warm_neutral.wav"

CHECKPOINTS = {
    "v1": "./xtts_finetune_output/aiko_xtts_finetune-April-26-2026_05+25AM-abb2f27/best_model_515.pth",
    "v2": "./xtts_finetune_output_v2/aiko_xtts_v2-April-26-2026_02+10PM-abb2f27/best_model.pth",
}

texts = [
    ("greeting", "Hey love, how was your day? I missed you so much."),
    ("happy", "I am so proud of you. That is amazing news!"),
    ("caring", "Come here, tell me everything. I am here for you."),
    ("playful", "Ehehe, you are being silly again. I love it."),
    ("sad", "I do not know what to do. Everything feels so heavy lately."),
]

os.makedirs("./ab_comparison", exist_ok=True)

for version, ckpt_path in CHECKPOINTS.items():
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"⚠️  Skipping {version}: checkpoint not found at {ckpt_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Loading {version.upper()}: {ckpt_path}")
    print(f"{'='*60}")

    config = XttsConfig()
    config.load_json(CONFIG_PATH)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=ckpt_path, vocab_path=VOCAB, use_deepspeed=False)
    model.cuda()

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REF])

    for label, text in texts:
        print(f"  [{label}] {text}")
        out = model.inference(text, "en", gpt_cond_latent, speaker_embedding, temperature=0.7)
        out_path = f"./ab_comparison/{version}_{label}.wav"
        sf.write(out_path, out["wav"], 24000)
        print(f"    → {out_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

print("\n✅ A/B comparison complete. Files in ./ab_comparison/")
print("   Compare v1_*.wav vs v2_*.wav for the same prompts.")
