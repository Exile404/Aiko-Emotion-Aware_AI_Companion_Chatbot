# Gap 1: emotion2vec+ where it has not seen the labels (Expresso)

Same four actors reading and improvising in happy, sad and default (neutral) styles. Labels are intended styles. Probes leave-one-speaker-out, regularisation fixed from MELD dev. Nothing is tuned on Expresso.

## Table X-0 - Composition

| Condition  | n    | happy | neutral | sad  | median dur s | clusters |
|------------|------|-------|---------|------|--------------|----------|
| read       | 4558 | 1520  | 1519    | 1519 | 2.58         | 12       |
| improvised | 4267 | 1489  | 1600    | 1178 | 2.29         | 83       |
## Table X-1 - Voice balanced accuracy (three classes)

| Voice system  | read  | improvised | all   |
|---------------|-------|------------|-------|
| deployed head | 64.01 | 54.83      | 60.30 |
| e2v probe     | 70.20 | 64.31      | 67.84 |
| WavLM probe   | 81.86 | 81.59      | 81.99 |
## Table X-2 - Read minus improvised, voice (95 % CI)

| Voice system  | all clips              | duration x style matched |
|---------------|------------------------|--------------------------|
| deployed head | +9.18 [-3.78, +20.29]  | +9.02 [-4.31, +20.11]    |
| e2v probe     | +5.90 [-7.58, +17.23]  | +6.06 [-6.90, +17.54]    |
| WavLM probe   | +0.27 [-10.04, +10.25] | -0.37 [-10.78, +9.76]    |

Cluster bootstrap over recording groups (dialogue takes for improvised, speaker x style for read; 95 clusters), 1000 resamples. The duration-matched subsample has 6494 clips.

## Table X-3 - Do the two encoders disagree about delivery?

| Corpus                                | contrast              | e2v effect - WavLM effect | source                   |
|---------------------------------------|-----------------------|---------------------------|--------------------------|
| IEMOCAP (labels seen by emotion2vec+) | scripted - improvised | +9.87                     | e2v_probe_results.md E-3 |
| Expresso (labels not seen)            | read - improvised     | +5.63 [-6.69, +14.59]     | this file                |

Near zero on Expresso, the encoders agree once nothing is memorised, which favours contamination as the cause of their IEMOCAP disagreement. A disagreement of similar size and sign on Expresso favours specialisation. Note the contrasts are analogous, not identical: IEMOCAP compares scripted with improvised dialogue, Expresso read with improvised.

**Verdict: INCONCLUSIVE: the interval contains both zero and the IEMOCAP value, so this test cannot separate contamination from specialisation.** Expresso's read speech comes in only 12 recording blocks (speaker x style), and an honest bootstrap resamples blocks, not clips, so this interval is wide. MSP-IMPROV, with many sentence x emotion x session blocks and perceived labels, is the test with the power to decide.

## Table X-4 - emotion2vec+ minus WavLM under identical probes

| Corpus                | emotion2vec+ exposure                  | e2v probe | WavLM probe | advantage |
|-----------------------|----------------------------------------|-----------|-------------|-----------|
| CREMA-D (4 classes)   | in EmoBox                              | 93.40     | 86.94       | +6.46     |
| IEMOCAP (4 classes)   | in EmoBox                              | 87.41     | 72.01       | +15.40    |
| MELD test (4 classes) | in EmoBox (test split may be held out) | 48.86     | 54.02       | -5.16     |
| Expresso (3 classes)  | labels not in training                 | 67.84     | 81.99       | -14.15    |

IEMOCAP and MELD rows are from e2v_probe_results.md (leave-one-session-out; train -> test). CREMA-D is computed here, 5-fold grouped by actor, both probes with the MELD-dev regularisation. Label spaces differ between rows, so compare the SIGN and rough size of the advantage, not absolute accuracies.

**Reading.** emotion2vec+ was fine-tuned on tens of thousands of hours of emotional speech, so if its advantage reflected representation quality it should lead a generic speech encoder on every corpus. It leads only on the corpora whose labels it trained on, and trails where it did not: the advantage tracks training exposure, which is the signature contamination predicts. Two caveats bound this. Expresso's labels are intended speaking styles rather than perceived emotions, a task emotion2vec+ was not built for, which could depress it there independently of contamination; MELD, whose labels ARE perceived emotions, shows the same sign, but its test split may have been held out of emotion2vec+'s training rather than truly unseen. The pattern is strong circumstantial evidence, not proof.

