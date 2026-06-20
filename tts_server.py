#!/usr/bin/env python3
import os, sys, socket, json
os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"

import torch
_orig = torch.load
def _safe(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _safe

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import re
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF]+")
def clean_for_tts(t):
    t = _EMOJI.sub("", t).replace("~", " ")
    return re.sub(r"\s+", " ", t).strip() or "Mhm."

HOST, PORT = "127.0.0.1", 5123

# --- v2 fine-tune (the one preferred in the A/B compare) ---
# --- v3 fine-tune (best_model = epoch ~4, picked by ear) ---
RUN    = "./xtts_finetune_output_v3/aiko_xtts_v3-June-19-2026_05+12PM-10ef125"
CKPT   = f"{RUN}/best_model.pth"
BASE   = "./xtts_finetune_output_v3/XTTS_v2.0_original_model_files"
CONFIG = f"{BASE}/config.json"
VOCAB  = f"{BASE}/vocab.json"
SPEAKER_REFS = ["./voice_refs/long_aiko_01_warm_neutral.wav"]
print("Loading XTTS v2 fine-tune (this takes a bit)...")
config = XttsConfig(); config.load_json(CONFIG)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=CKPT, vocab_path=VOCAB, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFS)
print(f"✅ Model ready: {CKPT}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT)); server.listen(1)
print(f"TTS v2 server on {HOST}:{PORT} — waiting for requests...")

while True:
    try:
        conn, _ = server.accept()
        data = conn.recv(8192).decode("utf-8")
        if not data:
            conn.close(); continue
        req = json.loads(data)
        text = req.get("text", "")
        out_path = req.get("output", "/tmp/aiko_speech.wav")
        if text == "SHUTDOWN":
            conn.send(b"OK"); conn.close(); break
        out = model.inference(text, "en", gpt_cond_latent, speaker_embedding, temperature=0.7)
        sf.write(out_path, out["wav"], 24000)
        conn.send(b"OK"); conn.close()
        print(f"Generated: {text[:50]}")
    except Exception as e:
        print(f"Error: {e}")
        try:
            conn.send(f"ERROR: {e}".encode()); conn.close()
        except:
            pass
server.close()
