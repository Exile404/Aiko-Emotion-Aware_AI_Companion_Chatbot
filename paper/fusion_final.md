# C1 final: train MELD train (9988), select on dev, test on test (2610)

## Model selection on dev

| Method                             | Weighted-F1 | Macro-F1 | Accuracy |
|------------------------------------|-------------|----------|----------|
| logreg                             |      54.25 |   39.46 |   56.59 |
| histgbm                            |      56.32 |   41.24 |   58.48 |
| logreg (balanced)                  |      52.89 |   40.33 |   50.99 |
| histgbm (balanced)                 |      54.94 |   40.42 |   54.60 |

  -> standard = histgbm, balanced = histgbm (balanced)

## Table 1 - Headline (TEST)

| Method                             | Weighted-F1 | Macro-F1 | Accuracy |
|------------------------------------|-------------|----------|----------|
| Voice only (emotion2vec)           |      45.64 |   28.56 |   45.67 |
| Text only (DistilRoBERTa)          |      51.55 |   35.57 |   49.85 |
| Naive fusion (equal weight)        |      51.57 |   35.07 |   51.69 |
| Heuristic fusion (deployed)        |      51.58 |   35.29 |   53.75 |
| Learned fusion, standard (histgbm) |      56.72 |   37.57 |   58.93 |
| Learned fusion, balanced (histgbm) |      54.31 |   36.63 |   53.75 |

(oracle ceiling, either modality right: Acc = 67.78)

## Per-class - standard (TEST)

| Emotion     | Precision | Recall | F1     | Support |
|-------------|-----------|--------|--------|---------|
| angry       |    42.21 |  35.36 |  38.49 |     345 |
| disgusted   |    24.14 |  10.29 |  14.43 |      68 |
| fearful     |    14.29 |   6.00 |   8.45 |      50 |
| happy       |    54.57 |  53.48 |  54.02 |     402 |
| neutral     |    67.91 |  80.89 |  73.84 |    1256 |
| sad         |    36.79 |  18.75 |  24.84 |     208 |
| surprised   |    49.45 |  48.40 |  48.92 |     281 |

## Per-class - balanced (TEST)

| Emotion     | Precision | Recall | F1     | Support |
|-------------|-----------|--------|--------|---------|
| angry       |    39.42 |  39.42 |  39.42 |     345 |
| disgusted   |    14.49 |  14.71 |  14.60 |      68 |
| fearful     |     4.84 |   6.00 |   5.36 |      50 |
| happy       |    48.16 |  55.22 |  51.45 |     402 |
| neutral     |    73.09 |  65.53 |  69.10 |    1256 |
| sad         |    29.15 |  27.88 |  28.50 |     208 |
| surprised   |    43.39 |  53.74 |  48.01 |     281 |
