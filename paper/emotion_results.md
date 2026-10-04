# Emotion fusion on MELD test (2610 utterances, 715 short <= 3 words)

## Table 1 - Baselines vs proposed (all utterances)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy |
|------------------------------------|-------------|----------|----------|
| Voice only (emotion2vec)           |      45.64 |   28.56 |   45.67 |
| Text only (DistilRoBERTa)          |      51.55 |   35.57 |   49.85 |
| Naive fusion (equal weight)        |      51.57 |   35.07 |   51.69 |
| Proposed (reliability-weighted)    |      51.58 |   35.29 |   53.75 |

## Table 2 - Ablation (remove one component from proposed)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy | ΔWF1  |
|------------------------------------|-------------|----------|----------|-------|
| Proposed (full)                    |      51.58 |   35.29 |   53.75 |     — |
|   - per-emotion reliability        |      52.12 |   35.64 |   54.21 | +0.54 |
|   - short-utterance down-weight    |      49.52 |   32.35 |   51.72 | -2.06 |
|   - neutral-confidence floor       |      52.07 |   36.23 |   51.99 | +0.49 |
|   - agreement boost                |      51.65 |   35.28 |   53.87 | +0.07 |
|   - heuristic corrections          |      51.85 |   35.30 |   54.21 | +0.27 |
|   - recency smoothing              |      51.40 |   35.17 |   53.64 | -0.18 |

## Table 3 - Short-utterance subset (715 utterances, <= 3 words)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy | ΔWF1  |
|------------------------------------|-------------|----------|----------|-------|
| Voice only (emotion2vec)           |      45.38 |   26.60 |   44.34 | -12.93 |
| Text only (DistilRoBERTa)          |      59.14 |   43.03 |   57.62 | +0.83 |
| Proposed (full)                    |      58.31 |   43.29 |   60.28 |     — |
| Proposed - short down-weight       |      51.06 |   31.81 |   52.87 | -7.25 |

## Per-class - Proposed (all utterances)

| Emotion     | Precision | Recall | F1     | Support |
|-------------|-----------|--------|--------|---------|
| angry       |    53.42 |  22.61 |  31.77 |     345 |
| disgusted   |    23.08 |   8.82 |  12.77 |      68 |
| fearful     |    14.63 |  12.00 |  13.19 |      50 |
| happy       |    44.55 |  47.76 |  46.10 |     402 |
| neutral     |    61.61 |  73.73 |  67.13 |    1256 |
| sad         |    38.84 |  22.60 |  28.57 |     208 |
| surprised   |    43.27 |  52.67 |  47.51 |     281 |

