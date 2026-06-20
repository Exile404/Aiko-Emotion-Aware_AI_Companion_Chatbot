# Aiko — Emotion-Aware Conversational Companion

A fully-local, real-time AI companion. It hears *how* you feel, replies in a fine-tuned
persona and a cloned voice, remembers your conversations, and animates a Live2D avatar in
sync with its speech. No cloud APIs — everything runs on a single 16 GB GPU.

```
mic → VAD → Whisper STT → hybrid emotion (voice + text)
        → LangChain [ ChromaDB recall → conditional 3B reasoning → Qwen2.5-7B reply ]
        → XTTS v3 voice (streamed) → Live2D avatar
```

End-to-end latency ≈ **2 s** (reply→voice), fully offline.

---

## Components

| Stage | Model / Tool | Notes |
|-------|--------------|-------|
| Speech-to-text | OpenAI Whisper (`small`) | GPU |
| Emotion (voice) | `emotion2vec_plus_large` | GPU; 7-class |
| Emotion (text) | `j-hartmann/emotion-english-distilroberta-base` | CPU |
| Emotion fusion | late fusion + per-emotion reliability weights | short-utterance→text bias, neutral-confidence floor |
| Chat LLM | **Qwen2.5-7B-Instruct + LoRA** (`aiko_lora_v4`) | response-only QLoRA; served **q4_K_M via Ollama** |
| Reasoning | `qwen2.5:3b` "private strategist" | conditional — runs only on emotional turns |
| Memory | ChromaDB + `all-MiniLM-L6-v2` | persistent, stores the user side |
| Voice | **XTTS v3** (fine-tuned) | served by `tts_server.py`; sentence-chunk streaming |
| Avatar | Live2D (pixi-live2d-display) | WebSocket-driven, RMS lip-sync, emotion→expression |

Orchestration is LangChain: memory recall → (conditional) 3B reasoning → Qwen2.5-7B reply.
Only the final reply is sent to TTS and the avatar.

---

## Installation

**Prerequisites:** Python 3.11, NVIDIA GPU (16 GB recommended), CUDA 12+, [Ollama](https://ollama.com), Node/modern browser, Linux.

```bash
git clone <repo> && cd Virtual_GF_LLM

# main environment
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers datasets accelerate trl peft bitsandbytes
pip install langchain langchain-ollama langchain-chroma langchain-huggingface chromadb sentence-transformers
pip install openai-whisper funasr modelscope sounddevice soundfile librosa websockets
sudo apt-get install ffmpeg portaudio19-dev
```

**XTTS environment** (separate venv — Coqui TTS pins an older `transformers`):

```bash
python -m venv tts_venv && tts_venv/bin/pip install TTS==0.21.3 transformers==4.40.0 soundfile librosa
```

**Ollama models** (chat + reasoning):

```bash
ollama create aiko-v4 --quantize q4_K_M -f Modelfile   # imports the fine-tuned GGUF, ~4.7 GB
ollama pull qwen2.5:3b                                  # reasoning model
```

---

## Running the live companion

Three background services + the notebook:

```bash
# 1. avatar WebSocket relay
.venv/bin/python avatar_ws.py                              # :8765

# 2. serve the avatar page, then open http://localhost:8000 and click "Enable Aiko"
.venv/bin/python -m http.server 8000 --directory avatar    # :8000

# 3. the TTS server is launched automatically by the notebook (start_tts_server, :5123)
```

Then open `aiko_notebook_clean.ipynb` and run the inference path (skip §1–4 training):
**§5 Load model + memory → §6 Emotion → §7 Whisper → §8 Voice/Avatar → §9 Conversation.**
Use §9.1 for text/push-to-talk or **§9.2 for hands-free** (energy-VAD — just talk).

---

## Run as a package

The runtime is also packaged as `aiko/` for use outside the notebook (start the same three
services first):

```bash
python -m aiko               # hands-free voice (default)
python -m aiko --mode text   # typed console
python -m aiko --no-tts      # text replies only (skip the voice server)
```

```python
from aiko import Companion
aiko = Companion(); aiko.start(); aiko.converse()
```

| module | responsibility |
|--------|----------------|
| `config.py` | constants, paths, models, prompts |
| `memory.py` | ChromaDB store / recall |
| `emotion.py` | hybrid voice + text emotion |
| `stt.py` | Whisper transcription |
| `llm.py` | Ollama reply + conditional reasoning |
| `tts.py` | XTTS server client, chunking, pause-trim |
| `avatar.py` | WebSocket + expression mapping |
| `audio.py` | mic capture + VAD |
| `companion.py` | orchestrator — `converse()`, `chat_loop()` |

---

## Training

Reproducible from the notebook (§1–4, §10) or the standalone scripts:

```bash
python scripts/train_chat_lora.py      # Qwen2.5-7B LoRA -> aiko_model/aiko_v4_merged_16bit
python scripts/train_xtts_voice.py     # XTTS v3 voice (transcribe -> fine-tune in tts_venv)
```

### Chat model (LoRA)

| Parameter | Value |
|-----------|-------|
| Base | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Method | LoRA (r=16, α=16), **response-only** (`train_on_responses_only`) |
| Data | `aiko_dataset_v4_clean.toon` — ~2,996 curated examples, 7 emotions (TOON format) |
| Epochs / LR | 3 / 2e-4 cosine, warmup 50 |
| Batch | 1 × grad-accum 16 |
| Final loss | ~0.6 (plateaued, not overfit) |
| Deployment | merge → f16 GGUF (`convert_hf_to_gguf.py`) → **q4_K_M** in Ollama |

> The decisive factors for naturalness were **dataset quality** and **response-only training** —
> not model size or LoRA rank. A unified system prompt (`AIKO_SYSTEM`) is used for both training and inference.

### Voice (XTTS v3)

| Parameter | Value |
|-----------|-------|
| Base | Coqui XTTS v2 |
| Data | ~227 clips (~20 min) from 10 source recordings, segmented via Whisper |
| LR / Epochs | 5e-6 / ~10 (best checkpoint ≈ epoch 4 — loss bottoms early) |
| Serving | `tts_server.py` keeps the model resident on :5123; `enable_text_splitting` + emoji strip |

The XTTS appendix (§10) in the notebook covers transcription → fine-tune → listen-test.

---

## Memory

ChromaDB (`./aiko_memory_db`) with `all-MiniLM-L6-v2` embeddings. Each turn stores the user's
message; `recall_memories(query, k=3)` injects the most relevant past context into the prompt.
Storing only the user side (not Aiko's own replies) avoids a self-echo loop that otherwise made
replies repetitive. `clear_memories()` resets the store.

---

## Project structure

```
Virtual_GF_LLM/
├── aiko/                         # runtime package (config, memory, emotion, stt, llm,
│                                 #   tts, avatar, audio, companion, __main__)
├── scripts/                      # standalone training: train_chat_lora.py, train_xtts_voice.py
├── aiko_notebook_clean.ipynb     # training + experimentation notebook
├── aiko_dataset_v4_clean.toon    # curated chat dataset (~2,996 examples)
├── Modelfile                     # Ollama recipe for aiko-v4 (qwen2.5 template + AIKO_SYSTEM)
├── tts_server.py                 # persistent XTTS v3 voice server (:5123)
├── avatar_ws.py                  # WebSocket relay to the browser avatar (:8765)
├── avatar/                       # Live2D viewer (index.html, app.js)
├── aiko_model/                   # LoRA adapter, merged model, GGUF
├── xtts_finetune_output_v3/      # XTTS v3 fine-tune checkpoints
├── xtts_training_data/           # processed voice clips + metadata
├── voice_processed/ · voice_refs/# source recordings + speaker references
├── aiko_memory_db/               # ChromaDB store
├── .venv/ · tts_venv/            # main + TTS environments
└── readme.md
```

---

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Storage | 40 GB | 60 GB |
| Python | 3.11 | 3.11 |

The 16 GB budget is shared at runtime by Ollama (aiko-v4 q4 ≈ 4.7 GB + qwen2.5:3b ≈ 2 GB),
the XTTS server (≈ 5.6 GB), Whisper, and the emotion models. q4 quantization of the chat model
is what makes this fit on one card (f16 spills to CPU and is ~20× slower).

---

## Roadmap

**Done:** fine-tuned emotional persona · hybrid voice+text emotion · ChromaDB memory ·
Whisper STT · custom XTTS v3 voice · TTS streaming · q4 Ollama deployment ·
**Live2D avatar with lip-sync + emotion expressions** · hands-free voice conversation.

**Next:** phoneme-accurate visemes (Rhubarb) · web UI · memory fact-extraction · proactive messaging.

---

## Disclaimer

For personal, educational, and research purposes. Aiko is an AI character, not a substitute for
human relationships. Voice cloning should only use voices you have the rights to.

## License

MIT.

## Acknowledgments

[Unsloth](https://github.com/unslothai/unsloth) · [Qwen](https://github.com/QwenLM/Qwen2.5) ·
[Ollama](https://ollama.com) · [LangChain](https://langchain.com) ·
[OpenAI Whisper](https://github.com/openai/whisper) · [Coqui TTS](https://github.com/coqui-ai/TTS) ·
[emotion2vec](https://github.com/ddlBoJack/emotion2vec) · [pixi-live2d-display](https://github.com/guansss/pixi-live2d-display).

---

> **N.B.** Built with the assistance of Claude AI, drawing on prior Data Science project experience.
