# Controlled degradation of clean speech (IEMOCAP session 5)

Held-out session 5, common four classes, the full deployed path on every degraded clip. Clean probes trained on sessions 1-4 with settings frozen on MELD dev. `deployed heuristic` is Aiko's fusion rule fed the degraded voice scores and the text scores of the ASR transcript; its seven-way output counts as wrong when it names a class outside the four.

## Table N-1 - babble noise, IEMOCAP session 5 (n = 1241, balanced accuracy)

| SNR    | SNR proxy dB | ASR WER % | deployed voice | deployed text (ASR) | deployed heuristic | clean voice | clean text (ASR) |
|--------|--------------|-----------|----------------|---------------------|--------------------|-------------|------------------|
| clean  | 22.5         | 18.1      | 86.96          | 41.21               | 76.59              | 72.99       | 55.40            |
| +20 dB | 20.3         | 15.4      | 85.56          | 40.31               | 74.99              | 69.84       | 54.94            |
| +10 dB | 15.5         | 17.7      | 82.03          | 39.64               | 71.66              | 54.99       | 53.82            |
| +5 dB  | 12.4         | 30.2      | 77.36          | 39.56               | 66.64              | 47.87       | 51.54            |
| +0 dB  | 9.9          | 65.4      | 64.36          | 35.14               | 49.89              | 36.85       | 42.53            |
| -5 dB  | 8.6          | 112.6     | 33.22          | 28.45               | 28.20              | 30.47       | 31.31            |
## Table N-2 - laughter noise, IEMOCAP session 5 (n = 1241, balanced accuracy)

| SNR    | SNR proxy dB | ASR WER % | deployed voice | deployed text (ASR) | deployed heuristic | clean voice | clean text (ASR) |
|--------|--------------|-----------|----------------|---------------------|--------------------|-------------|------------------|
| clean  | 22.5         | 18.1      | 86.96          | 41.21               | 76.59              | 72.99       | 55.40            |
| +20 dB | 21.2         | 18.1      | 86.35          | 40.65               | 75.86              | 70.25       | 54.66            |
| +10 dB | 19.4         | 21.3      | 85.75          | 40.45               | 73.44              | 63.37       | 54.87            |
| +5 dB  | 18.9         | 24.7      | 82.68          | 40.08               | 71.43              | 57.83       | 54.22            |
| +0 dB  | 19.2         | 30.5      | 60.70          | 38.31               | 51.37              | 51.47       | 51.65            |
| -5 dB  | 21.0         | 42.6      | 28.46          | 38.39               | 29.24              | 43.80       | 50.23            |
## Table N-3 - steady noise, IEMOCAP session 5 (n = 1241, balanced accuracy)

| SNR    | SNR proxy dB | ASR WER % | deployed voice | deployed text (ASR) | deployed heuristic | clean voice | clean text (ASR) |
|--------|--------------|-----------|----------------|---------------------|--------------------|-------------|------------------|
| clean  | 22.5         | 18.1      | 86.96          | 41.21               | 76.59              | 72.99       | 55.40            |
| +20 dB | 20.1         | 15.4      | 76.94          | 40.38               | 67.06              | 65.65       | 55.50            |
| +10 dB | 14.2         | 16.9      | 69.24          | 40.01               | 59.74              | 50.43       | 54.41            |
| +5 dB  | 10.2         | 19.3      | 60.07          | 39.83               | 51.10              | 39.72       | 53.59            |
| +0 dB  | 6.6          | 30.6      | 35.34          | 39.53               | 33.43              | 28.66       | 50.58            |
| -5 dB  | 4.1          | 60.8      | 28.15          | 35.65               | 25.80              | 25.78       | 45.74            |
## Table N-4 - Where voice falls below text (ASR words)

| Noise    | deployed encoders | clean encoders |
|----------|-------------------|----------------|
| babble   | never above -5 dB | +8.8 dB        |
| laughter | -3.5 dB           | +0.2 dB        |
| steady   | +0.9 dB           | +12.8 dB       |

Interpolated SNR at which voice balanced accuracy first drops below text balanced accuracy, where the text model reads Whisper's transcript of the same degraded audio.

## Table N-5 - How much noise makes IEMOCAP look like MELD acoustically?

| Noise    | SNR matching MELD's ASR WER | deployed voice there | clean voice there | SNR matching MELD's SNR proxy | deployed voice there |
|----------|-----------------------------|----------------------|-------------------|-------------------------------|----------------------|
| babble   | +4.8 dB                     | 76.78                | 47.38             | milder than +20 dB            | -                    |
| laughter | -0.5 dB                     | 57.39                | 50.68             | proxy not monotone: invalid   | -                    |
| steady   | -0.2 dB                     | 35.06                | 28.55             | milder than +20 dB            | -                    |

MELD test (common four, clean): Whisper WER 31.8 %, median SNR proxy 21.4 dB, deployed voice balanced accuracy 46.25. IEMOCAP session 5 clean: WER 18.1 %, proxy 22.5 dB. Each anchor finds the added-noise level at which degraded IEMOCAP matches MELD on that measure, then reads the voice accuracy IEMOCAP still has at that level. If that accuracy stays far above MELD's, noise of a severity matching MELD's acoustic difficulty does NOT reproduce MELD's voice collapse, and the IEMOCAP-to-MELD step must be attributed to something other than the audio (for example how MELD was labelled).

