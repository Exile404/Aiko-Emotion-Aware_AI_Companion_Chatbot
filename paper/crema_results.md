# C1 boundary condition - voice encoder on clean versus in-the-wild speech

emotion2vec_plus_large, voice modality only, argmax restricted to the six categories CREMA-D and MELD share.

## Table C1-7 - Voice encoder competence, shared six classes

| Corpus                  | n    | acc   | bal acc | WF1   | macro F1 | majority | acc over majority |
|-------------------------|------|-------|---------|-------|----------|----------|-------------------|
| CREMA-D (clean, acted)  | 7442 | 86.41 | 86.50   | 86.43 | 86.53    | 17.08    | +69.34            |
| MELD test (in the wild) | 2329 | 49.68 | 30.83   | 49.63 | 29.48    | 53.93    | -4.25             |

281 MELD utterances labelled `surprised` were dropped so both corpora span the same six categories. Argmax is restricted to those six on both sides.

**Primary comparison (balanced accuracy, immune to the two corpora's very different class priors): CREMA-D 86.50 % against MELD 30.83 %, a gap of +55.67 points.** Macro F1 tells the same story: 86.53 against 29.48. Raw accuracy is reported for completeness but is NOT comparable across the two corpora, because MELD's neutral-dominated prior lifts it independently of encoder quality; the `acc over majority` column is the honest accuracy-based reading.

**Text-only control on CREMA-D: acc 14.61 %, balanced acc 16.26 %, macro F1 7.40** against a majority baseline of 17.08 %. CREMA-D's twelve carrier sentences are each recorded in all six emotions, so the transcript carries no emotion information by design. This confirms empirically that the CREMA-D result isolates the voice modality and cannot be explained by lexical cues.

## Per-class F1 (voice only, shared six)

```
CREMA-D
              precision    recall  f1-score   support

       angry      0.962     0.888     0.924      1271
   disgusted      0.839     0.880     0.859      1271
     fearful      0.869     0.750     0.805      1271
       happy      0.878     0.922     0.899      1271
     neutral      0.904     0.900     0.902      1087
         sad      0.760     0.850     0.802      1271

    accuracy                          0.864      7442
   macro avg      0.869     0.865     0.865      7442
weighted avg      0.868     0.864     0.864      7442

```

```
MELD test
              precision    recall  f1-score   support

       angry      0.454     0.270     0.338       345
   disgusted      0.026     0.015     0.019        68
     fearful      0.053     0.020     0.029        50
       happy      0.340     0.604     0.435       402
     neutral      0.678     0.595     0.634      1256
         sad      0.287     0.346     0.314       208

    accuracy                          0.497      2329
   macro avg      0.306     0.308     0.295      2329
weighted avg      0.519     0.497     0.496      2329

```

## Table C1-8 - CREMA-D accuracy by acted intensity (neutral excluded)

| Intensity | n   | acc   | bal acc | macro F1 |
|-----------|-----|-------|---------|----------|
| LO        | 455 | 79.56 | 79.56   | 80.69    |
| MD        | 455 | 81.98 | 81.98   | 81.98    |
| HI        | 455 | 82.86 | 82.86   | 83.13    |

Macro F1 is averaged over the five classes present in these subsets. These rows treat the levels as independent samples, which understates the design; the paired analysis below is the one to cite.

## Table C1-9 - Intensity as a paired within-actor contrast

| Level | n triples | acc   | mean p(true class) |
|-------|-----------|-------|--------------------|
| LO    | 455       | 79.56 | 0.7715             |
| MD    | 455       | 81.98 | 0.8045             |
| HI    | 455       | 82.86 | 0.8140             |

455 matched triples: the same actor speaking the same sentence with the same intended emotion at all three intensities. Actor, lexical content and emotion are therefore held constant, and only portrayal strength varies.

**Correctness.** Cochran's Q across the three levels = 2.13, p = 0.3448. LO vs HI by exact McNemar on 117 discordant triples (66 gained at HI, 51 lost): p = 0.1953. Paired bootstrap (1000 resamples) on HI minus LO accuracy: +3.30 points, 95 % CI [-1.32, +7.91].

**Confidence in the true class.** Mean p(true) rises 0.7715 -> 0.8045 -> 0.8140. Wilcoxon signed-rank HI vs LO: W = 43884, p = 4.43e-03. Paired bootstrap on HI minus LO: +0.0425, 95 % CI [+0.0039, +0.0826]. Strictly increasing across all three levels in 90/455 triples (19.8 %), against the 16.7 % expected if orderings were random (p = 7.81e-02).

**Reading.** The graded measure moves with intensity while argmax correctness does not separate reliably. The encoder's CONFIDENCE tracks portrayal strength even where its decision is already correct at low intensity, which is the expected signature when accuracy is near ceiling.

## Sentence balance check

| Sentence | n    | most common gold | % of that sentence |
|----------|------|------------------|--------------------|
| DFA      | 546  | angry            | 16.7               |
| IEO      | 1456 | angry            | 18.8               |
| IOM      | 546  | angry            | 16.7               |
| ITH      | 540  | angry            | 16.7               |
| ITS      | 545  | angry            | 16.7               |
| IWL      | 546  | angry            | 16.7               |
| IWW      | 546  | angry            | 16.7               |
| MTI      | 540  | angry            | 16.7               |
| TAI      | 546  | angry            | 16.7               |
| TIE      | 545  | angry            | 16.7               |
| TSI      | 546  | angry            | 16.7               |
| WSI      | 540  | angry            | 16.7               |

Each carrier sentence should be near uniform across the six emotions. Large skew on any row would mean the text-only control is weaker than it appears.

