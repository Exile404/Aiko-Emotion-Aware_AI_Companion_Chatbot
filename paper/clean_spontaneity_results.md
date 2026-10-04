# Stress test of the clean-encoder spontaneity effect (IEMOCAP, LOSO)

WavLM L18 / MiniLM probes, settings frozen on MELD dev, identical to clean_probe_eval.py.

## Table S-1 - Paired within-speaker, within-emotion cells

| Modality | cells | mean script - impro | script higher | Wilcoxon W | p        |
|----------|-------|---------------------|---------------|------------|----------|
| voice    | 40    | -12.78              | 12/40         | 169        | 0.000844 |
| text     | 40    | +13.41              | 32/40         | 140        | 0.00015  |
## Table S-2 - Matched on duration quartile and emotion

| Modality | n per side | script bal acc | impro bal acc | script - impro [95 % CI] |
|----------|------------|----------------|---------------|--------------------------|
| voice    | 1974       | 63.35          | 77.13         | -13.77 [-18.29, -9.23]   |
| text     | 1974       | 63.42          | 51.39         | +12.03 [+8.10, +15.58]   |
## Table S-3a - Utterances whose exact transcript occurs in another session

| kind   | n    | transcript seen in another session |
|--------|------|------------------------------------|
| script | 2588 | 34.4 %                             |
| impro  | 2943 | 11.1 %                             |
## Table S-3b - Accuracy by whether the line was seen in training

| subset                      | n    | text bal acc | voice bal acc |
|-----------------------------|------|--------------|---------------|
| script, line seen elsewhere | 890  | 65.71        | 62.83         |
| script, line unique         | 1698 | 61.90        | 64.29         |
| impro (all)                 | 2943 | 50.46        | 76.22         |

Under leave-one-session-out, a 'seen elsewhere' scripted line appeared verbatim in the training sessions. If the text probe's scripted advantage is concentrated there, it is memorised script content, not emotion read from language, and the text side of Table K-4 must not be reported as a property of scripted speech. Voice is reported alongside: a voice probe cannot memorise a line it hears in a different speaker's voice nearly as easily, so it acts as a control.

## Table S-4 - Voice minus text, by condition (clean encoders)

| Condition                                | n    | voice bal acc | text bal acc | voice - text [95 % CI]  |
|------------------------------------------|------|---------------|--------------|-------------------------|
| IEMOCAP scripted (clean audio)           | 2588 | 64.04         | 63.64        | +0.40 [-4.11, +4.36]    |
| IEMOCAP scripted, unique lines only      | 1698 | 64.29         | 61.90        | +2.38 [-1.76, +6.46]    |
| IEMOCAP improvised (clean audio)         | 2943 | 76.22         | 50.46        | +25.76 [+22.02, +29.32] |
| MELD test (scripted TV, broadcast audio) | 2211 | 54.02         | 53.99        | +0.03 [-3.19, +2.83]    |

Improvised margin minus scripted margin, within IEMOCAP (same actors, same recording chain): +25.36 points, 95 % CI [+19.65, +30.59].

## Table S-5 - IEMOCAP scripted to MELD, each modality

| Modality | IEMOCAP scripted | MELD test | MELD - IEMOCAP scripted [95 % CI] |
|----------|------------------|-----------|-----------------------------------|
| voice    | 64.04            | 54.02     | -10.02 [-13.94, -5.97]            |
| text     | 63.64            | 53.99     | -9.65 [-12.97, -6.25]             |

Both conditions are scripted performance; they differ in recording conditions, multi-party context, corpus and annotation. The text probe never hears the audio, so any drop it shares with the voice probe cannot be acoustic. A voice drop that the text probe matches points to something both modalities share (label reliability, domain), not to degraded audio.

