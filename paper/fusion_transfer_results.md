# Fusion transfer across recording regimes (deployed encoders)

emotion2vec+ voice scores and DistilRoBERTa text scores as cached by the extraction scripts. Both encoders have seen these corpora in training; the contamination-free replication is in clean_probe_results.md.

## Table F-1 - Deployed heuristic on both corpora

| Predictor                   | MELD P4 bal acc | MELD P4 macro F1 | MELD P7 WF1 | IEMOCAP P4 bal acc | IEMOCAP P4 macro F1 | IEMOCAP P7 WF1 |
|-----------------------------|-----------------|------------------|-------------|--------------------|---------------------|----------------|
| Voice only (7-way argmax)   | 41.70           | 43.02            | 45.64       | 83.70              | 83.10               | 79.35          |
| Text only (7-way argmax)    | 38.18           | 45.05            | 51.55       | 30.33              | 33.01               | 33.60          |
| Naive fusion (equal weight) | 44.02           | 47.37            | 51.57       | 83.57              | 83.92               | 80.91          |
| Heuristic fusion (deployed) | 41.67           | 45.63            | 51.58       | 74.76              | 77.69               | 75.16          |

Paired dialogue-clustered bootstrap, P4 balanced accuracy, heuristic minus each single modality:
- MELD test: heuristic minus voice only = -0.03 points, 95 % CI [-1.95, +1.92]
- MELD test: heuristic minus text only = +3.49 points, 95 % CI [+1.50, +5.98]
- IEMOCAP: heuristic minus voice only = -8.94 points, 95 % CI [-10.12, -7.61]
- IEMOCAP: heuristic minus text only = +44.43 points, 95 % CI [+42.59, +46.38]

## Table F-2 - Removing each heuristic component, both corpora

| Ablation                                 | MELD d bal acc (P4) | MELD d WF1 (P7) | IEMOCAP d bal acc (P4) | IEMOCAP d WF1 (P7) | sign  |
|------------------------------------------|---------------------|-----------------|------------------------|--------------------|-------|
| remove per-emotion reliability           | +0.64               | +0.54           | +0.44                  | +0.09              |       |
| remove short-utterance voice down-weight | -0.70               | -2.06           | +7.94                  | +4.72              | FLIPS |
| remove neutral-confidence floor          | +1.82               | +0.49           | +0.33                  | +0.39              |       |
| remove agreement boost                   | -0.04               | +0.07           | +0.00                  | +0.00              |       |
| remove heuristic corrections             | +0.20               | +0.27           | +1.17                  | +1.07              |       |
| remove recency smoothing                 | -0.19               | -0.18           | -1.31                  | -0.95              |       |

Positive = the component was HURTING on that corpus. A component that helps on one corpus and hurts on the other (marked FLIPS, threshold 0.25 points each way) is calibrated to one recording regime.

## Table F-3 - Learned fusion across corpora (P4, deployed scores)

| Fusion                               | MELD test bal acc | MELD test macro F1 | IEMOCAP bal acc | IEMOCAP macro F1 |
|--------------------------------------|-------------------|--------------------|-----------------|------------------|
| voice argmax                         | 46.25             | 44.44              | 83.84           | 83.18            |
| text argmax                          | 51.85             | 51.60              | 38.70           | 36.83            |
| head trained on MELD                 | 51.08             | 53.04              | 82.52           | 83.38            |
| head trained on IEMOCAP              | 40.34             | 41.63              | 86.16           | 86.27            |
| head trained on both                 | 50.95             | 52.67              | 86.52           | 86.52            |
| head trained on both + audio quality | 50.79             | 52.49              | 86.47           | 86.51            |

HistGradientBoosting on the 14 deployed scores (seven voice, seven text), default settings as selected on MELD dev in fusion_final.md. IEMOCAP in-corpus predictions are leave-one-session-out, so no head is scored on a session it was trained on. MELD heads train on MELD train only.

Paired dialogue-clustered bootstrap, balanced accuracy:
- MELD test: head trained on IEMOCAP minus head trained on MELD = -10.74, 95 % CI [-12.99, -8.59]
- IEMOCAP: head trained on MELD minus head trained on IEMOCAP = -3.64, 95 % CI [-4.62, -2.72]
- MELD test: head trained on both minus head trained on MELD = -0.13, 95 % CI [-1.47, +1.11]
- IEMOCAP: head trained on both minus head trained on IEMOCAP = +0.36, 95 % CI [-0.30, +0.93]
- MELD test: head trained on both + audio quality minus head trained on both = -0.15, 95 % CI [-1.25, +1.02]
- IEMOCAP: head trained on both + audio quality minus head trained on both = -0.05, 95 % CI [-0.41, +0.29]

## Table F-5 - Accuracy by audio-quality quartile, within each corpus

| Corpus    | SNR-proxy quartile   | n    | voice bal acc | text bal acc | voice - text |
|-----------|----------------------|------|---------------|--------------|--------------|
| MELD test | Q1 (< 16.9 dB)       | 553  | 42.42         | 54.01        | -11.60       |
| MELD test | Q2 (16.9 to 21.4 dB) | 552  | 44.78         | 52.18        | -7.40        |
| MELD test | Q3 (21.4 to 25.6 dB) | 553  | 51.44         | 52.66        | -1.22        |
| MELD test | Q4 (>= 25.6 dB)      | 553  | 45.95         | 47.55        | -1.60        |
| IEMOCAP   | Q1 (< 18.2 dB)       | 1376 | 72.32         | 39.03        | +33.30       |
| IEMOCAP   | Q2 (18.2 to 22.5 dB) | 1375 | 83.26         | 38.28        | +44.98       |
| IEMOCAP   | Q3 (22.5 to 26.7 dB) | 1375 | 85.83         | 38.93        | +46.90       |
| IEMOCAP   | Q4 (>= 26.7 dB)      | 1376 | 82.47         | 35.89        | +46.58       |

Quartiles are computed within each corpus, so corpus identity is held fixed and only estimated audio quality varies. If degraded audio drives the voice collapse, voice accuracy should rise from Q1 to Q4 inside each corpus. Text accuracy is a control: it does not hear the audio.

