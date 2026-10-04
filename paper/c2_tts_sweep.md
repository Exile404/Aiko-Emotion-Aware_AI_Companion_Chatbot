# C2 - XTTS first-chunk length sweep

XTTS-v3 fine-tune via tts_server.py on :5123. 5 reps per length, 2 warmup discarded. Text is fixed and LLM-independent, unlike Table C2-2's tts_first_chunk.

## Table C2-6 - XTTS first-chunk length sweep

| target | chars | n | synth ms | sd      | p50      | max      | trim ms | audio s | ms/char | ms/char post | RTF    |
|--------|-------|---|----------|---------|----------|----------|---------|---------|---------|--------------|--------|
| 20     | 30    | 5 |    684.3 |    85.8 |    709.2 |    765.0 |     8.4 |    3.33 |   109.5 |     80.4     |  0.206 |
| 30     | 47    | 5 |    769.7 |   278.1 |    787.6 |   1062.4 |     8.7 |    3.89 |    83.1 |     72.4     |  0.198 |
| 45     | 70    | 5 |    963.2 |    98.9 |    930.1 |   1135.2 |     9.9 |    4.84 |    68.8 |     59.9     |  0.199 |
| 60     | 76    | 5 |   1091.5 |   178.8 |   1058.3 |   1325.8 |    10.7 |    5.60 |    73.4 |     65.2     |  0.195 |
| 80     | 107   | 5 |   1664.8 |   185.5 |   1597.2 |   1886.0 |    15.8 |    8.70 |    81.6 |     64.5     |  0.191 |
| 120    | 147   | 5 |   2173.4 |   318.5 |   2095.5 |   2709.5 |    19.9 |   11.19 |    76.3 |     57.9     |  0.194 |
| 160    | 185   | 5 |   2513.6 |   469.0 |   2501.0 |   3121.8 |    21.4 |   12.90 |    69.6 |     51.7     |  0.195 |
| 200    | 221   | 5 |   2755.8 |   711.5 |   2962.1 |   3505.4 |    23.0 |   14.18 |    64.1 |     48.9     |  0.194 |

**Reading.** Deployed default is `TTS_CHUNK_MIN_CHARS = 60`. Audio-per-character plateau (targets >= 80 chars) = 72.9 ms/char pre-trim, 55.7 post-trim. Inflation above the plateau is EITHER XTTS speaking words the text does not contain (the rambling the floor was introduced to prevent) OR leading/trailing silence. The post-trim column separates them: if inflation survives trimming it is invented speech; if it collapses to the plateau it was silence and costs nothing.

-   20 chars:   684.3 ms to first audio, 109.5 ->  80.4 ms/char  <-- INVENTED SPEECH (survives trim)
-   30 chars:   769.7 ms to first audio,  83.1 ->  72.4 ms/char
-   45 chars:   963.2 ms to first audio,  68.8 ->  59.9 ms/char
-   60 chars:  1091.5 ms to first audio,  73.4 ->  65.2 ms/char
-   80 chars:  1664.8 ms to first audio,  81.6 ->  64.5 ms/char
-  120 chars:  2173.4 ms to first audio,  76.3 ->  57.9 ms/char
-  160 chars:  2513.6 ms to first audio,  69.6 ->  51.7 ms/char
-  200 chars:  2755.8 ms to first audio,  64.1 ->  48.9 ms/char

**Shortest faithful chunk (post-trim): 45 chars at 963 ms** to first audio.
