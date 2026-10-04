# Gap 2: voice against text on natural speech (CMU-MOSEI)

Real YouTube opinion monologues, crowd-rated. Contamination-free encoders, probes trained on the official train split, scored once on test, settings from MELD dev. CIs resample videos.

## Table M-0 - Segments per labelling rule (test counts per class)

| Rule | train | test | angry | happy | neutral | sad |
|------|-------|------|-------|-------|---------|-----|
| A    | 7762  | 2207 | 277   | 1039  | 702     | 189 |
| B    | 10494 | 3039 | 435   | 1496  | 702     | 406 |
| C    | 9752  | 2838 | 217   | 1609  | 702     | 310 |
## Table M-1 - Voice minus text on natural speech (clean encoders, test split)

| Rule | test segments | classes | voice bal acc | text bal acc | voice - text [95 % CI] |
|------|---------------|---------|---------------|--------------|------------------------|
| A    | 2207          | 4       | 56.16         | 43.23        | +12.93 [+9.03, +16.67] |
| B    | 3039          | 4       | 50.00         | 41.45        | +8.54 [+5.25, +11.82]  |
| C    | 2838          | 4       | 43.55         | 37.38        | +6.17 [+2.67, +9.98]   |

Reference, clean_spontaneity_results.md Table S-4: IEMOCAP improvised +25.76, IEMOCAP scripted +0.40, MELD (scripted television) +0.03.

**Verdict: the voice advantage extends to natural speech.** Under every labelling rule the interval excludes zero, so the conclusion does not depend on how MOSEI's intensities are mapped to classes. Two features of MOSEI favour TEXT, which makes the result conservative: its transcripts are long (see the word counts below) and its raters saw the words. The size of the advantage sits between scripted speech (about zero) and actors' improvisation (about +26), giving the ordering scripted < natural < actor-improvised. Cross-corpus magnitudes compare different label spaces and protocols, so the ordering is indicative; the sign and its interval are the robust part.

## Table M-2 - Aiko's own detectors on natural speech (rule A, test)

| Detector                               | bal acc |
|----------------------------------------|---------|
| emotion2vec+ (voice, deployed head)    | 35.90   |
| DistilRoBERTa (text, human transcript) | 34.73   |
| voice - text                           | +1.16   |

Chance is 25.0 for 4 classes. Neither deployed detector was trained on MOSEI's labels; emotion2vec+'s self-supervised pretraining may have included MOSEI audio.

**Reading.** On natural speech both of Aiko's detectors sit close to chance (35.9 and 34.7 against 25.0), while a linear probe on a generic encoder trained on natural speech reaches 56.2 (Table M-1, rule A). The emotional signal is present in natural voices; the off-the-shelf detectors, trained largely on acted speech, do not transfer to it. For a companion whose users speak naturally, the deployed emotion pipeline needs adaptation to natural speech before its outputs can be trusted.

Test segments average 19.7 words (median 17), against IEMOCAP's 11.5 and MELD's 8.3.

