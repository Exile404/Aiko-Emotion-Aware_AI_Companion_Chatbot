# C2 - Efficiency of the fully-local Aiko pipeline

GPU NVIDIA GeForce RTX 5060 Ti, 16311 MB. torch 2.9.1+cu128. 20 measured turns, mean input 2.53s, 3 warmup discarded.

## Table C2-1 - Cold start (load time and VRAM per component)

| Component          | Load s | torch MB | GPU dMB  | GPU MB   |
|--------------------|--------|----------|----------|----------|
| Memory (ChromaDB)  |   5.74 |        0 |      101 |    10614 |
| EmotionDetector    |  11.89 |      628 |      621 |    11235 |
| Whisper-small      |   1.74 |      925 |     1510 |    12745 |
| AikoChat (handles) |   0.04 |        0 |        0 |    12745 |
| VoiceClient        |   0.00 |        0 |        0 |    12745 |

Baseline GPU before loading: 10523 MB of 16311 MB. GPU: NVIDIA GeForce RTX 5060 Ti.


## Table C2-2 - Warm per-stage latency (ms)

| Stage                | n  | mean      | sd      | p50      | p95      |
|----------------------|----|-----------|---------|----------|----------|
| stt_whisper          | 20 |     121.5 |    27.9 |    127.1 |    169.7 |
| emotion_voice        | 20 |      44.3 |     3.7 |     45.0 |     50.3 |
| emotion_text         | 20 |      10.2 |     1.7 |     10.0 |     13.0 |
| emotion_detect_total | 20 |      59.3 |     7.8 |     59.9 |     70.6 |
| memory_recall        | 20 |       6.7 |     1.3 |      6.3 |      9.0 |
| think_3b             | 20 |      94.9 |   169.4 |      0.0 |    405.8 |
| chat_7b_q4           | 20 |     329.0 |    51.2 |    326.1 |    410.0 |
| tts_first_chunk      | 20 |     929.2 |   231.6 |    930.5 |   1282.6 |

**End-to-end (mic-ready audio -> first spoken audio): 1.54 s**; to reply text: 0.61 s. Excludes VAD hang (0.5s) and memory.store (post-reply write).


## Table C2-3 - Conditional reasoning (reply path only, ms)

| Condition                 | n | think 3B | chat 7B   | total     |
|---------------------------|---|----------|-----------|-----------|
| emotional (thinker runs)  | 8 |    350.5 |     323.7 |     680.4 |
| neutral (thinker skipped) | 8 |      0.0 |     303.7 |     309.8 |

## Table C2-4 - GPU memory by process (MB)

| PID   | Process                  | Owner | VRAM MB |
|-------|--------------------------|-------|---------|
| 3204  | cosmic-workspaces        | other | 68      |
| 3194  | cosmic-app-library       | other | 34      |
| 4259  | xdg-desktop-portal-cosmi | other | 32      |
| 6680  | claude-desktop           | other | 110     |
| 13366 | cosmic-term              | other | 40      |
| 14523 | code                     | other | 117     |
| 18264 | brave                    | other | 82      |
| 38968 | llama-server             | aiko  | 4692    |
| 39667 | llama-server             | aiko  | 2214    |
| 41505 | python                   | aiko  | 2368    |
| 47863 | python                   | aiko  | 1830    |

**Aiko's own footprint: 11104 MB of 16311 MB (68 % of the card).** Desktop/IDE overhead present during measurement but not part of the system: 483 MB. Total in use: 12310 MB (75 %).


## Table C2-5 - Ollama generation internals (ms, mean)

| Model   | n | total     | load     | prompt eval | generate  | other   | tokens | tok/s   |
|---------|---|-----------|----------|-------------|-----------|---------|--------|---------|
| aiko-v4 | 5 |       306 |       97 |         12  |       191 |       6 |     17 |    89.2 |
