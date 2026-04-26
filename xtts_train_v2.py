#!/usr/bin/env python3
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
    formatter="ljspeech",
    dataset_name="aiko",
    path=TRAINING_DIR,
    meta_file_train="metadata_train.csv",
    meta_file_val="metadata_eval.csv",
    language="en",
)

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    debug_loading_failures=False,
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

config = GPTTrainerConfig(
    output_path=OUTPUT_DIR,
    model_args=model_args,
    run_name="aiko_xtts_v2",
    project_name="aiko",
    run_description="V2: lower LR, fewer epochs, eval references fixed",
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=2,
    batch_group_size=32,
    eval_batch_size=2,
    num_loader_workers=2,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=500,
    save_step=250,
    save_n_checkpoints=4,
    save_checkpoints=True,
    print_eval=False,
    run_eval_steps=250,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=True,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=1e-6,
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [500, 1000, 1500], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[
        {"text": "Hey love, how was your day?", "speaker_wav": "./voice_refs/long_aiko_01_warm_neutral.wav", "language": "en"},
        {"text": "I am so proud of you. That is amazing news.", "speaker_wav": "./voice_refs/long_aiko_03_soft_caring.wav", "language": "en"},
    ],
    epochs=EPOCHS,
)

train_samples, eval_samples = load_tts_samples(
    config_dataset,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1,
)

model = GPTTrainer.init_from_config(config)

trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=2),
    config,
    output_path=OUTPUT_DIR,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

print("DONE")
