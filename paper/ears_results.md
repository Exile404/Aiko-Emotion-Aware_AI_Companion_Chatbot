# Gap 1 on EARS: emotion2vec+ on a corpus released after it

EARS (arXiv 2406.06185, 10 June 2024) post-dates emotion2vec+ (May 2024), so none of it can be in emotion2vec+'s training data. 3 s windows, length-matched across conditions; 107 speakers. Probes grouped 5-fold by speaker, regularisation fixed from MELD dev; nothing tuned on EARS.

## Table Z-0 - Composition (three classes)

| Condition | windows | angry | neutral | sad |
|-----------|---------|-------|---------|-----|
| read      | 930     | 316   | 294     | 320 |
| freeform  | 940     | 310   | 316     | 314 |
## Table Z-1 - Voice balanced accuracy, angry / neutral / sad

| Voice system  | read  | freeform | all   |
|---------------|-------|----------|-------|
| deployed head | 61.16 | 55.93    | 58.50 |
| e2v probe     | 84.99 | 72.23    | 78.47 |
| WavLM probe   | 98.28 | 90.34    | 94.28 |
## Table Z-2 - Read minus freeform, voice (95 % CI, speakers resampled)

| Voice system  | read - freeform        |
|---------------|------------------------|
| deployed head | +5.23 [+1.60, +8.84]   |
| e2v probe     | +12.76 [+9.59, +16.23] |
| WavLM probe   | +7.94 [+5.79, +10.19]  |
## Table Z-3 - Do the encoders disagree about delivery?

| Corpus                     | contrast              | e2v effect - WavLM effect | source        |
|----------------------------|-----------------------|---------------------------|---------------|
| IEMOCAP (labels seen)      | scripted - improvised | +9.87                     | e2v_probe E-3 |
| Expresso (labels not seen) | read - improvised     | +5.63 [-6.69, +14.59]     | expresso X-3  |
| EARS (never seen)          | read - freeform       | +4.82 [+0.91, +8.86]      | this file     |

**Verdict: BOTH: the encoders still disagree on a corpus emotion2vec+ never saw, so part of the IEMOCAP disagreement is a general property of the encoder, but by less than on IEMOCAP, which leaves room for contamination to account for the rest.** In both corpora the disagreement has the same sign: relative to WavLM, emotion2vec+ does worse on the spontaneous condition than on the read or scripted one. That is the signature of an encoder biased toward read, acted speech, which its training mix (mostly acted corpora) would produce. Caveats: the contrasts are analogous, not identical (IEMOCAP scripted against improvised dialogue; EARS read sentences against freeform monologue), and the IEMOCAP value is a point estimate carrying its own uncertainty, so 'less than on IEMOCAP' is indicative, not a formal test.

Note the direction of the delivery effect itself differs between corpora: on IEMOCAP the contamination-free probe found improvised speech EASIER for voice than scripted, on EARS every system finds read speech easier than freeform. 'Improvised speech is easier for voice' is therefore an IEMOCAP finding and must not be generalised.

## Table Z-4 - Same task, seen versus never seen (angry / neutral / sad, read acted speech)

| Voice system  | CREMA-D (in training) | EARS read (never seen) | change |
|---------------|-----------------------|------------------------|--------|
| deployed head | 95.02                 | 61.16                  | -33.87 |
| e2v probe     | 95.48                 | 86.55                  | -8.93  |
| WavLM probe   | 90.81                 | 97.85                  | +7.04  |
## Table Z-5 - emotion2vec+'s edge over WavLM, seen versus never seen

| Corpus                        | e2v probe - WavLM probe [95 % CI] |
|-------------------------------|-----------------------------------|
| CREMA-D (in training)         | +4.67 [+3.43, +6.11]              |
| EARS read (never seen)        | -11.30 [-13.42, -9.17]            |
| change, never seen minus seen | -15.97 [-18.43, -13.57]           |

Both corpora are read, acted, studio-quality speech with intended labels, scored on the same three classes under identical probes, so the task is held fixed and only emotion2vec+'s prior exposure changes. A genuinely better encoder keeps its edge on the unseen corpus; a memorising one loses it. The deployed zero-shot head is in Table Z-4 for the same contrast.

**Reading.** The contamination-free WavLM probe finds EARS read speech EASIER than CREMA-D, so the unseen corpus is not a harder version of the task. Against that, the deployed emotion2vec+ head falls by 33.9 points and emotion2vec+'s representational edge over WavLM reverses sign. Its CREMA-D performance therefore overstates what it does on speakers and recordings it has not trained on, which is the deployment case: every real user of a companion system is an unseen speaker. Residual caveats: the corpora differ in actors, sentences and recording chain (EARS is anechoic, 48 kHz), and a zero-shot head always pays some domain shift; the probe rows remove the label-mapping part of that shift and the gap persists.

## Table Z-6 - Four-class EARS, each happy proxy (read + freeform)

| Mapping              | e2v probe | WavLM probe | e2v advantage |
|----------------------|-----------|-------------|---------------|
| happy := amusement   | 71.28     | 92.31       | -21.03        |
| happy := contentment | 68.76     | 89.88       | -21.12        |
| happy := extasy      | 70.40     | 91.85       | -21.45        |

For comparison with the four-class rows of expresso_results.md X-4: CREMA-D +6.46, IEMOCAP +15.40 (labels seen); MELD test -5.16, Expresso -14.15 (not seen). EARS has no plain happiness category, so every proxy is reported rather than one chosen.

