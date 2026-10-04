# Contamination check: emotion2vec+ embeddings under the clean-probe protocol

Voice systems compared on identical utterances and splits (IEMOCAP leave-one-session-out, MELD train -> test, common four classes). `deployed head` and `e2v probe` share the encoder and differ in read-out; `e2v probe` and `WavLM probe` share the read-out and differ in encoder.

## Table E-1 - e2v probe regularisation, selected on MELD dev

| Setting    | MELD dev bal acc |          |
|------------|------------------|----------|
| C = 0.0001 | 53.49            |          |
| C = 0.001  | 53.77            |          |
| C = 0.01   | 53.92            | selected |
| C = 0.1    | 53.72            |          |
## Table E-2 - Voice balanced accuracy by condition

| Voice system  | IEMOCAP scripted | IEMOCAP improvised | IEMOCAP all | MELD test |
|---------------|------------------|--------------------|-------------|-----------|
| deployed head | 82.23            | 83.63              | 83.76       | 46.25     |
| e2v probe     | 85.52            | 87.83              | 87.41       | 48.86     |
| WavLM probe   | 64.04            | 76.22              | 72.01       | 54.02     |
## Table E-3 - Scripted minus improvised, voice (95 % CI)

| Voice system  | all utterances         | duration x emotion matched |
|---------------|------------------------|----------------------------|
| deployed head | -1.40 [-5.21, +2.89]   | -1.21 [-5.07, +3.37]       |
| e2v probe     | -2.31 [-5.94, +1.77]   | -2.66 [-6.63, +1.81]       |
| WavLM probe   | -12.18 [-16.71, -7.22] | -13.77 [-18.29, -9.23]     |

Dialogue-clustered bootstrap, 1000 resamples; scripted and improvised dialogues resampled independently. Negative = improvised is easier.

## Table E-4 - Voice minus text (MiniLM probe), by condition

| Voice system | IEMOCAP scripted        | IEMOCAP improvised      | MELD test            |
|--------------|-------------------------|-------------------------|----------------------|
| e2v probe    | +21.88 [+18.27, +25.23] | +37.36 [+33.99, +40.65] | -5.14 [-8.34, -1.96] |
| WavLM probe  | +0.40 [-4.11, +4.36]    | +25.76 [+22.02, +29.32] | +0.03 [-3.19, +2.83] |

Paired dialogue-clustered bootstrap. The WavLM row reproduces Table S-4. If the improvised-only voice advantage reappears with emotion2vec+ embeddings, it is not a WavLM artifact either.

