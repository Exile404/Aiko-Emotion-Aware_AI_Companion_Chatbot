#!/usr/bin/env python3
"""Fine-tune Aiko's chat model: Qwen2.5-7B-Instruct + LoRA, response-only QLoRA.

Run in the main environment (with unsloth). Reads the curated TOON dataset, trains a LoRA
adapter on Aiko's replies only, and saves a merged 16-bit model ready for GGUF conversion.

    python scripts/train_chat_lora.py

Afterwards: convert to GGUF (llama.cpp `convert_hf_to_gguf.py`) and
`ollama create aiko-v4 --quantize q4_K_M -f Modelfile`.
"""
from __future__ import annotations

import os
import re
import sys

# Make the `aiko` package importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from aiko.config import AIKO_SYSTEM

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
DATASET_PATH = "./aiko_dataset_v4_clean.toon"
OUTPUT_DIR = "./aiko_training_output"
MERGED_DIR = "aiko_model/aiko_v4_merged_16bit"


def parse_toon(toon_text: str, system_prompt: str) -> list[dict]:
    """Parse the TOON dataset into chat conversations.

    Entry boundaries are detected by field lines (not ``---``) so a missing separator
    can't silently drop or merge examples. ``{AIKO_SYSTEM}`` is substituted with the prompt.
    """
    rows: list[dict] = []
    cur: dict[str, str] = {}
    field: str | None = None

    def flush() -> None:
        if cur.get("user") and cur.get("assistant"):
            sys_text = cur.get("system", "{AIKO_SYSTEM}").strip()
            if sys_text == "{AIKO_SYSTEM}":
                sys_text = system_prompt
            rows.append({"conversations": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": cur["user"].strip()},
                {"role": "assistant", "content": cur["assistant"].strip()},
            ]})

    for line in toon_text.splitlines():
        stripped = re.sub(r"^\s*#+\s*", "", line)  # drop leading markdown hashes
        match = re.match(r"(system|user|assistant):\s?(.*)$", stripped)
        if match:
            fld, rest = match.group(1), match.group(2)
            if (fld == "user" and cur.get("user")) or (fld == "system" and (cur.get("user") or cur.get("assistant"))):
                flush()
                cur = {}
            field, cur[fld] = fld, rest
        elif stripped.strip() == "---":
            continue
        elif field and line.strip():
            cur[field] += "\n" + line
    flush()
    return rows


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = parse_toon(f.read(), AIKO_SYSTEM)
    print(f"Parsed {len(rows):,} examples from {DATASET_PATH}")
    dataset = Dataset.from_list(rows)

    def format_conversations(examples: dict) -> dict:
        return {"text": [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in examples["conversations"]
        ]}

    dataset = dataset.map(format_conversations, batched=True, batch_size=100, desc="Formatting")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,          # effective batch = 16
        num_train_epochs=3,
        warmup_steps=50,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=False,                           # required for clean response masking
        args=args,
    )
    # Train only on Aiko's replies — masks system+user from the loss so she never learns
    # to emit the [voice_emotion] tag and the gradient focuses on the reply.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print(f"Training {len(dataset):,} examples | eff.batch=16 | epochs=3 | lr=2e-4 | response-only")
    stats = trainer.train()
    print(f"Done. loss={stats.training_loss:.4f}, {stats.metrics['train_runtime'] / 60:.1f} min")

    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"Saved merged 16-bit model -> {MERGED_DIR}")


if __name__ == "__main__":
    main()
