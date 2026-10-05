# Text-encoder strength check: XLM-RoBERTa-large against MiniLM-L6

Voice is WavLM-large (~316M parameters) throughout. Text is MiniLM-L6 (~22M, as in every earlier table) or xlm-roberta-large (~560M, larger than the voice encoder). Both text encoders and the voice encoder are frozen, selected once on MELD dev, and probed with the same linear probes and splits.

## Table H-1 - Selection on MELD dev (balanced accuracy)

| Encoder                                  | selected             | MELD dev bal acc |
|------------------------------------------|----------------------|------------------|
| MiniLM-L6 (~22M)                         | C = 0.1              | 53.74            |
| XLM-RoBERTa-large (~560M)                | layer L16, C = 0.001 | 58.73            |
| WavLM-large voice (~316M), for reference | layer L18, C = 0.001 | 57.36            |
## Table H-2 - Voice minus text with each text encoder (balanced accuracy; 95 % CI)

| Condition                      | n    | voice | text MiniLM | text XLM-R | voice - MiniLM          | voice - XLM-R           |
|--------------------------------|------|-------|-------------|------------|-------------------------|-------------------------|
| IEMOCAP improvised             | 2943 | 76.22 | 50.46       | 56.22      | +25.76 [+22.02, +29.32] | +20.00 [+16.06, +23.55] |
| IEMOCAP improvised, unanimous  | 1519 | 84.04 | 54.62       | 61.95      | +29.42 [+25.35, +33.48] | +22.10 [+16.62, +26.47] |
| MOSEI natural, rule A          | 2207 | 56.16 | 43.23       | 49.44      | +12.93 [+9.03, +16.67]  | +6.72 [+2.94, +10.19]   |
| MOSEI natural, rule B          | 3039 | 50.00 | 41.45       | 45.21      | +8.54 [+5.25, +11.82]   | +4.79 [+1.66, +7.82]    |
| MOSEI natural, rule C          | 2838 | 43.55 | 37.38       | 39.16      | +6.17 [+2.67, +9.98]    | +4.39 [+1.08, +7.75]    |
| IEMOCAP scripted               | 2588 | 64.04 | 63.64       | 62.68      | +0.40 [-4.11, +4.36]    | +1.36 [-3.05, +5.33]    |
| IEMOCAP scripted, unique lines | 1698 | 64.29 | 61.90       | 60.57      | +2.38 [-1.76, +6.46]    | +3.72 [-0.77, +7.89]    |
| IEMOCAP scripted, unanimous    | 1020 | 66.94 | 67.31       | 63.01      | -0.37 [-6.73, +5.21]    | +3.93 [-2.30, +10.08]   |
| MELD test (scripted TV)        | 2211 | 54.02 | 53.99       | 58.92      | +0.03 [-3.19, +2.83]    | -4.90 [-7.97, -2.02]    |
## Table H-3 - Improvised margin minus scripted margin (IEMOCAP)

| Text encoder | difference [95 % CI]    |
|--------------|-------------------------|
| MiniLM       | +25.36 [+19.56, +30.37] |
| XLM-R        | +18.64 [+12.98, +23.96] |
## Table H-4 - Verdict per claim with the larger text encoder

| Condition                                    | claim            | with XLM-RoBERTa-large |
|----------------------------------------------|------------------|------------------------|
| IEMOCAP improvised                           | voice beats text | HOLDS                  |
| IEMOCAP improvised, unanimous                | voice beats text | HOLDS                  |
| MOSEI natural, rule A                        | voice beats text | HOLDS                  |
| MOSEI natural, rule B                        | voice beats text | HOLDS                  |
| MOSEI natural, rule C                        | voice beats text | HOLDS                  |
| IEMOCAP scripted                             | voice ties text  | HOLDS                  |
| IEMOCAP scripted, unique lines               | voice ties text  | HOLDS                  |
| IEMOCAP scripted, unanimous                  | voice ties text  | HOLDS                  |
| MELD test (scripted TV)                      | voice ties text  | text wins              |
| IEMOCAP, improvised margin - scripted margin | positive         | HOLDS                  |

9 of 10 claims hold with a text encoder larger than the voice encoder. Any claim that does not is reported as such and must be narrowed in the paper accordingly.

