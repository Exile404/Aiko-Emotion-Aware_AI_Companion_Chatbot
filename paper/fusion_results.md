# Learned fusion: train on MELD dev (1108), test on MELD test (2610)

## Table 1 - Baselines vs heuristic vs learned fusion (test set)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy |
|------------------------------------|-------------|----------|----------|
| Voice only (emotion2vec)           |      45.64 |   28.56 |   45.67 |
| Text only (DistilRoBERTa)          |      51.55 |   35.57 |   49.85 |
| Naive fusion (equal weight)        |      51.57 |   35.07 |   51.69 |
| Heuristic fusion (deployed)        |      51.58 |   35.29 |   53.75 |
| Learned fusion (LogReg)            |      56.25 |   35.04 |   58.89 |
| Learned fusion (HistGBM)           |      54.41 |   34.14 |   56.97 |

(oracle ceiling, either modality right: Acc = 67.78)

## Table 2 - Learned-fusion feature ablation (HistGBM, test)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy | ΔWF1  |
|------------------------------------|-------------|----------|----------|-------|
| voice + text (14)                  |      54.41 |   34.14 |   56.97 |     — |
| voice features only (7)            |      43.42 |   24.38 |   46.67 | -10.99 |
| text features only (7)             |      53.07 |   32.23 |   55.75 | -1.34 |

## Per-class - Learned fusion (LogReg) (test set)

| Emotion     | Precision | Recall | F1     | Support |
|-------------|-----------|--------|--------|---------|
| angry       |    43.70 |  34.20 |  38.37 |     345 |
| disgusted   |     0.00 |   0.00 |   0.00 |      68 |
| fearful     |     0.00 |   0.00 |   0.00 |      50 |
| happy       |    56.96 |  44.78 |  50.14 |     402 |
| neutral     |    67.24 |  81.05 |  73.50 |    1256 |
| sad         |    38.62 |  26.92 |  31.73 |     208 |
| surprised   |    45.96 |  58.72 |  51.56 |     281 |
