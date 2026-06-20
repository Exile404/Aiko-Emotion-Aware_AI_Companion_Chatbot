#!/usr/bin/env python3
"""Fine-tune Aiko's voice (XTTS v3) from raw recordings.

Three stages, launched from the main environment:

  1. transcribe + segment the recordings in ``voice_processed/`` into clips + metadata
     (Whisper, runs here);
  2. normalize the metadata CSVs to bare clip ids (the ljspeech formatter expects them);
  3. launch GPT fine-tuning in ``tts_venv`` (Coqui TTS pins an older transformers, so the
     training itself runs as a subprocess in that environment).

Requires the XTTS v2 base at ``xtts_finetune_output/XTTS_v2.0_original_model_files/``
(download ``coqui/XTTS-v2`` if missing).

    python scripts/train_xtts_voice.py [--epochs 10] [--skip-transcribe]
"""
from __future__ import annotations

import argparse
import os
import subprocess

VOICE_DIR = "./voice_processed"
TRAINING_DIR = "./xtts_training_data"
WAVS_DIR = f"{TRAINING_DIR}/wavs"
OUTPUT_DIR = "./xtts_finetune_output_v3"
SHARED_BASE = "./xtts_finetune_output/XTTS_v2.0_original_model_files"
TTS_VENV_PYTHON = "./tts_venv/bin/python"
TARGET_SR = 22050

RECORDINGS = [
    "aiko_01_warm_neutral.wav", "aiko_02_happy_excited.wav", "aiko_03_soft_caring.wav",
    "aiko_04_sad_vulnerable.wav", "aiko_05_worried_anxious.wav", "aiko_06_playful_flirty.wav",
    "aiko_07_angry_frustrated.wav", "aiko_08_sleepy_gentle.wav", "aiko_09_surprised_shocked.wav",
    "aiko_10_deep_thoughtful.wav",
]

# Training runs in tts_venv; written out and executed as a subprocess (plain string, not f-string).
TRAIN_SCRIPT = '''#!/usr/bin/env python3
import os, sys
os.environ["MPLBACKEND"] = "Agg"
sys.setrecursionlimit(10000)

import torch
_orig = torch.load
def _safe(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _safe

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig as _XAC, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig as _BDC
    torch.serialization.add_safe_globals([XttsConfig, _XAC, XttsArgs, _BDC])
except Exception as e:
    print(f"safe_globals warning: {e}")

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

TRAINING_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
EPOCHS = int(sys.argv[3])

CHECKPOINTS = os.path.join(OUTPUT_DIR, "XTTS_v2.0_original_model_files")
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS, "dvae.pth")
MEL_NORM_FILE = os.path.join(CHECKPOINTS, "mel_stats.pth")
TOKENIZER_FILE = os.path.join(CHECKPOINTS, "vocab.json")
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS, "model.pth")

config_dataset = BaseDatasetConfig(
    formatter="ljspeech", dataset_name="aiko", path=TRAINING_DIR,
    meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv", language="en",
)

model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150,
    debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
    mel_norm_file=MEL_NORM_FILE, dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT, tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

config = GPTTrainerConfig(
    output_path=OUTPUT_DIR, model_args=model_args,
    run_name="aiko_xtts_v3", project_name="aiko",
    run_description="V3: lr 5e-6, frequent checkpoints",
    dashboard_logger="tensorboard", audio=audio_config,
    batch_size=2, batch_group_size=32, eval_batch_size=2,
    num_loader_workers=2, eval_split_max_size=256,
    print_step=25, plot_step=100, log_model_step=500,
    save_step=100, save_n_checkpoints=6, save_checkpoints=True,
    print_eval=False, run_eval_steps=100,
    optimizer="AdamW", optimizer_wd_only_on_weights=True,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-6,
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [150, 300, 450], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[
        {"text": "Hey love, how was your day?", "speaker_wav": "./voice_refs/long_aiko_01_warm_neutral.wav", "language": "en"},
        {"text": "I am so proud of you. That is amazing news.", "speaker_wav": "./voice_refs/long_aiko_03_soft_caring.wav", "language": "en"},
    ],
    epochs=EPOCHS,
)

train_samples, eval_samples = load_tts_samples(
    config_dataset, eval_split=True,
    eval_split_max_size=config.eval_split_max_size, eval_split_size=0.1,
)

model = GPTTrainer.init_from_config(config)

trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=2),
    config, output_path=OUTPUT_DIR, model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
trainer.fit()
print("DONE")
'''


def transcribe_and_segment() -> None:
    """Whisper-transcribe each recording, slice into 1-15s clips, write metadata CSVs."""
    import gc

    import librosa
    import soundfile as sf
    import torch
    import whisper

    os.makedirs(WAVS_DIR, exist_ok=True)
    print("Loading Whisper (medium)...")
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")

    metadata: list[tuple[str, str, str]] = []
    for filename in RECORDINGS:
        path = os.path.join(VOICE_DIR, filename)
        if not os.path.exists(path):
            print(f"  missing: {filename}")
            continue
        print(f"  {filename}")
        result = model.transcribe(path, language="en", word_timestamps=True)
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        base = filename.replace(".wav", "")
        for i, seg in enumerate(result["segments"]):
            text = seg["text"].strip()
            if not (1.0 <= seg["end"] - seg["start"] <= 15.0) or len(text) < 5:
                continue
            clip = audio[int(seg["start"] * TARGET_SR): int(seg["end"] * TARGET_SR)]
            name = f"{base}_{i:03d}.wav"
            sf.write(os.path.join(WAVS_DIR, name), clip, TARGET_SR)
            metadata.append((f"wavs/{name}", text, "aiko"))

    del model
    torch.cuda.empty_cache()
    gc.collect()
    _write_csvs(metadata)
    print(f"{len(metadata)} clips written to {TRAINING_DIR}/")


def _write_csvs(metadata: list[tuple[str, str, str]]) -> None:
    eval_split = max(5, len(metadata) // 10)

    def dump(path: str, rows: list[tuple[str, str, str]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for audio_file, text, speaker in rows:
                f.write(f"{audio_file}|{text}|{speaker}\n")

    dump(f"{TRAINING_DIR}/metadata.csv", metadata)
    dump(f"{TRAINING_DIR}/metadata_train.csv", metadata[eval_split:])
    dump(f"{TRAINING_DIR}/metadata_eval.csv", metadata[:eval_split])


def normalize_csvs() -> None:
    """Rewrite the CSVs with bare clip ids (no path/extension) for the ljspeech formatter."""
    for name in ("metadata_train.csv", "metadata_eval.csv", "metadata.csv"):
        path = os.path.join(TRAINING_DIR, name)
        if not os.path.exists(path):
            continue
        fixed: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                audio_id = os.path.basename(parts[0]).replace(".wav", "")
                speaker = parts[2] if len(parts) > 2 else "aiko"
                fixed.append(f"{audio_id}|{parts[1]}|{speaker}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(fixed)
        print(f"  normalized {name}: {len(fixed)} rows")


def train(epochs: int) -> None:
    """Symlink the base model and launch GPT fine-tuning inside tts_venv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(SHARED_BASE, "model.pth")):
        raise SystemExit(f"XTTS v2 base missing at {SHARED_BASE} — download coqui/XTTS-v2 first.")
    v3_base = os.path.join(OUTPUT_DIR, "XTTS_v2.0_original_model_files")
    if not os.path.exists(v3_base):
        os.symlink(os.path.abspath(SHARED_BASE), v3_base)

    with open("./xtts_train_v3.py", "w") as f:
        f.write(TRAIN_SCRIPT)

    env = {**os.environ, "MPLBACKEND": "Agg"}
    print(f"Training XTTS via {TTS_VENV_PYTHON} (epochs={epochs})...")
    proc = subprocess.run(
        [TTS_VENV_PYTHON, "./xtts_train_v3.py", TRAINING_DIR, OUTPUT_DIR, str(epochs)],
        env=env,
    )
    if proc.returncode:
        raise SystemExit(proc.returncode)
    print(f"Done -> {OUTPUT_DIR}/ (pick the best-sounding checkpoint by ear, not by loss)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Aiko's XTTS voice.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="reuse the existing xtts_training_data/ instead of re-transcribing")
    args = parser.parse_args()

    if not args.skip_transcribe:
        transcribe_and_segment()
        normalize_csvs()
    train(args.epochs)


if __name__ == "__main__":
    main()
