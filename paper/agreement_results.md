# Gap 3: label reliability and the shared IEMOCAP-to-MELD drop

Every IEMOCAP utterance has exactly three external evaluators; the released label is their majority, so a scored utterance is unanimous (3 of 3) or a split decision (2 of 3). Contamination-free probes (WavLM / MiniLM), leave-one-session-out, settings from MELD dev. CIs resample dialogues, paired.

## Table G-0 - Unanimous and split-decision utterances

| Subset     | n    | unanimous | split | split share |
|------------|------|-----------|-------|-------------|
| all        | 5531 | 2539      | 2992  | 54.1 %      |
| scripted   | 2588 | 1020      | 1568  | 60.6 %      |
| improvised | 2943 | 1519      | 1424  | 48.4 %      |
## Table G-1 - Accuracy on unanimous versus split-decision utterances

| Subset     | modality | unanimous bal acc | split bal acc | split - unanimous [95 % CI] |
|------------|----------|-------------------|---------------|-----------------------------|
| all        | voice    | 80.76             | 63.22         | -17.53 [-21.12, -14.17]     |
| all        | text     | 64.04             | 54.97         | -9.08 [-12.40, -5.78]       |
| scripted   | voice    | 66.94             | 58.16         | -8.77 [-14.27, -3.83]       |
| scripted   | text     | 67.31             | 59.60         | -7.71 [-13.19, -3.34]       |
| improvised | voice    | 84.04             | 66.94         | -17.10 [-21.96, -12.00]     |
| improvised | text     | 54.62             | 45.88         | -8.74 [-13.49, -3.94]       |
## Table G-2 - Do voice and text fall together?

| Contrast                       | voice drop | text drop | voice drop - text drop |
|--------------------------------|------------|-----------|------------------------|
| all                            | -17.53     | -9.08     | -8.45 [-12.74, -4.36]  |
| scripted                       | -8.77      | -7.71     | -1.06 [-7.83, +5.67]   |
| improvised                     | -17.10     | -8.74     | -8.36 [-15.71, -1.35]  |
| IEMOCAP scripted -> MELD (S-5) | -10.02     | -9.65     | -0.37                  |

A difference near zero means the two modalities lose the same amount, the pattern a shared cause such as label reliability produces and the pattern seen from IEMOCAP scripted to MELD. The last row is the observed cross-corpus drop for comparison.

## Table G-3 - Is MELD's labelling less reliable? Fleiss' kappa, three annotators

| Corpus                                              | Fleiss' kappa |
|-----------------------------------------------------|---------------|
| IEMOCAP, all utterances (computed here)             | 0.276         |
| IEMOCAP, scripted                                   | 0.203         |
| IEMOCAP, improvised                                 | 0.349         |
| IEMOCAP, all, excited -> happiness (9 categories)   | 0.352         |
| IEMOCAP, all, + frustration -> anger (8, MELD-like) | 0.435         |
| IEMOCAP, all, + frustration -> neutral instead (8)  | 0.393         |
| MELD (Poria et al., 2019, as reported)              | 0.43          |

**Verdict.** MECHANISM: within scripted speech, split-decision labels cost voice -8.77 and text -7.71, falling together as they do from IEMOCAP scripted to MELD (-10.02, -9.65), so unreliable labels can produce that drop at that size. LEVEL: the kappa comparison shows COMPARABLE reliability once the category sets are aligned (0.435 against 0.43), so MELD's labels are not measurably less reliable than IEMOCAP's. Label reliability is therefore a sufficient mechanism but is not shown to be the cause; the shared, non-acoustic IEMOCAP-to-MELD drop is attributed to cross-corpus differences that affect both modalities equally (label construct and context, domain, multi-party structure), which this test cannot separate. Kappa from different category inventories remains an indicative comparison.


IEMOCAP's kappa is computed over 10039 utterances with three external evaluators, ten categories, first-listed label per evaluator. MELD's is over seven categories. Kappa depends on the category inventory and prevalence, so this comparison is indicative rather than exact.

## Table G-4 - The headline on unanimous labels only (voice - text)

| Condition  | agreement | n    | voice - text [95 % CI]  |
|------------|-----------|------|-------------------------|
| scripted   | unanimous | 1020 | -0.37 [-6.42, +5.58]    |
| scripted   | split     | 1568 | -1.43 [-6.23, +3.34]    |
| improvised | unanimous | 1519 | +29.42 [+25.18, +33.56] |
| improvised | split     | 1424 | +21.07 [+15.13, +26.65] |

Table S-4's pattern (voice beats text in improvised speech, ties it in scripted) is compared here within each agreement level. If it survives on unanimous utterances it is not an artifact of IEMOCAP's scripted labels being less reliable.

## Reference - deployed emotion2vec+ (contaminated) by agreement

| System              | unanimous | split | split - unanimous |
|---------------------|-----------|-------|-------------------|
| deployed voice head | 88.31     | 80.75 | -7.56             |

Shown for completeness only. This encoder trained on IEMOCAP's released labels, so on split decisions it may reproduce the majority label from memory rather than from the audio; its drop is not comparable to the probes'.

