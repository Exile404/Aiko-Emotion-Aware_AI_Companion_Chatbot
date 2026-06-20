#!/usr/bin/env python3
import os, sys
os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
from TTS.api import TTS
ref_wavs = sys.argv[1].split(",")
text = sys.argv[2]
out_path = sys.argv[3]
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.tts_to_file(text=text, speaker_wav=ref_wavs, language="en", file_path=out_path)
print(f"DONE:{out_path}")
