# C1 boundary condition, part 2: the voice encoder across corpora, with IEMOCAP

> **Read first.** Every number below uses the DEPLOYED encoders (emotion2vec+ for voice, DistilRoBERTa for text), and both saw these corpora in training: emotion2vec+ via EmoBox (IEMOCAP, MELD, CREMA-D), DistilRoBERTa via MELD. The numbers are correct as measurements of the deployed system, but two interpretations an earlier version of this file drew from them did not survive contamination-free replication: the voice/text ranking does not invert with recording conditions, and spontaneity is not null. See clean_probe_results.md, clean_spontaneity_results.md, noise_results.md and fusion_transfer_results.md for what holds.

emotion2vec_plus_large unless stated. IEMOCAP is studio-recorded and conversational and contains both scripted and actor-improvised dialogue from the same ten actors.

## Table C1-10 - Voice encoder across the cleanliness/spontaneity ladder (common four classes)

| Condition                                         | n    | acc   | bal acc | WF1   | macro F1 | majority | acc over majority |
|---------------------------------------------------|------|-------|---------|-------|----------|----------|-------------------|
| CREMA-D (acted, isolated utterance, clean)        | 4900 | 93.39 | 93.28   | 93.40 | 93.33    | 25.94    | +67.45            |
| IEMOCAP script (acted, conversational, clean)     | 2588 | 82.15 | 82.23   | 81.95 | 81.09    | 31.45    | +50.70            |
| IEMOCAP impro (improvised, conversational, clean) | 2943 | 83.18 | 83.63   | 82.91 | 83.52    | 37.34    | +45.84            |
| IEMOCAP all                                       | 5531 | 82.70 | 83.76   | 82.46 | 83.12    | 30.88    | +51.82            |
| MELD test (acted TV, conversational, degraded)    | 2211 | 52.92 | 46.25   | 53.48 | 44.44    | 56.81    | -3.89             |

All rows: emotion2vec_plus_large, voice modality only, argmax restricted to angry/happy/neutral/sad. These four are the only categories with usable support in all three corpora (IEMOCAP has two `disgusted` utterances and 40 `fearful`). Four-class figures are NOT comparable to Table C1-7's six-class figures.

## Table C1-11 - Gap decomposition (balanced accuracy)

| Step                            | from  | to    | delta  | property varied            |
|---------------------------------|-------|-------|--------|----------------------------|
| CREMA-D -> IEMOCAP script       | 93.28 | 82.23 | -11.05 | isolated -> conversational |
| IEMOCAP script -> MELD test     | 82.23 | 46.25 | -35.98 | clean -> degraded audio    |
| IEMOCAP script -> IEMOCAP impro | 82.23 | 83.63 | +1.40  | acted -> improvised        |
| TOTAL CREMA-D -> MELD test      | 93.28 | 46.25 | -47.02 | sum of the first two       |

The first two steps share IEMOCAP script as their anchor, which is scripted and conversational as MELD is, so their deltas sum to the total gap. Step 1 adds dialogue context and open vocabulary. Step 2 changes recording conditions, but also corpus, annotation procedure and multi-party structure, and must NOT be read as an acoustic effect: realistic noise at MELD-matched speech-recognition difficulty does not reproduce it (noise_results.md, Table N-5), and under contamination-free encoders text falls as far as voice across this step (clean_spontaneity_results.md, Table S-5), which audio cannot explain. Step 3 is the only within-corpus contrast; under this deployed encoder it is null, but see the caveat under Table C1-13. Steps 1 and 2 are cross-corpus and absorb everything else that differs between corpora.

## Table C1-12 - Spontaneity as a within-corpus contrast

| Condition | n    | acc   | bal acc | macro F1 | majority |
|-----------|------|-------|---------|----------|----------|
| script    | 2588 | 82.15 | 82.23   | 81.09    | 31.45    |
| impro     | 2943 | 83.18 | 83.63   | 83.52    | 37.34    |

**Dialogue-clustered bootstrap** (1000 resamples, the dialogue is the resampled unit because utterances within a dialogue share speaker, topic and take): script minus impro balanced accuracy = -1.25 points, 95 % CI [-5.21, +2.89].

**Paired within-speaker, within-emotion test** on 40 (speaker x emotion) cells present in both conditions with at least 5 utterances each: mean script minus impro accuracy = -0.10 points, Wilcoxon signed-rank W = 352, p = 0.4437. Script higher in 14/40 cells.

## Table C1-13 - Same contrast, matched on duration quartile and emotion

| Condition        | n    | acc   | bal acc | macro F1 |
|------------------|------|-------|---------|----------|
| script (matched) | 1974 | 81.05 | 82.32   | 81.67    |
| impro (matched)  | 1974 | 83.79 | 83.53   | 84.01    |

duration quartile edges (s): 2.32, 3.58, 5.77. Cells are matched to equal counts in every (quartile x emotion) combination, so neither utterance length nor emotion prior differs between the conditions. Script minus impro balanced accuracy = -1.04 points, 95 % CI [-5.07, +3.37] under the same dialogue-clustered bootstrap.

**Caveat: this null is a property of the deployed encoder, not of speech.** emotion2vec+ saw IEMOCAP labels in training (via EmoBox). Under a contamination-free encoder (frozen WavLM with a probe trained leave-one-session-out) improvised speech is 12 to 14 points EASIER for voice than scripted speech, surviving the same paired-cell and duration-and-emotion-matched controls used here (clean_probe_results.md K-4; clean_spontaneity_results.md S-1, S-2). Probing emotion2vec+'s own embeddings with that same protocol is still null (e2v_probe_results.md, E-3), so the flattening lives in the encoder's representation, not its classification head. Do not report the null above as a finding about spontaneity.

## Table C1-14 - Voice against text on IEMOCAP (common four)

| Method                                      | n    | acc   | bal acc | WF1   | macro F1 |
|---------------------------------------------|------|-------|---------|-------|----------|
| Voice only (emotion2vec)                    | 5502 | 82.75 | 83.84   | 82.52 | 83.18    |
| Text only (DistilRoBERTa, gold transcripts) | 5502 | 41.20 | 38.70   | 37.71 | 36.83    |

Oracle ceiling (either modality correct): 91.35 % accuracy on 5502 utterances. Transcripts are the corpus's own, so the text pathway is evaluated under conditions more favourable than deployment. Note also that the DistilRoBERTa text model was trained on MELD but not on IEMOCAP, so its weakness here and its strength on MELD partly reflect which corpus it has seen.

## Table C1-15 - Voice against text across corpora, deployed encoders (does not survive contamination-free replication)

| Corpus                          | n    | voice bal acc | text bal acc | voice - text | voice macro F1 | text macro F1 | oracle acc |
|---------------------------------|------|---------------|--------------|--------------|----------------|---------------|------------|
| IEMOCAP (clean, conversational) | 5502 | 83.84         | 38.70        | +45.14       | 83.18          | 36.83         | 91.35      |
| MELD test (degraded TV audio)   | 2211 | 46.25         | 51.85        | -5.59        | 44.44          | 51.60         | 78.20      |

Identical label space, identical argmax restriction, identical deployed encoders. Under these encoders the sign of the voice-minus-text margin reverses between the corpora, a swing of 50.73 points. **This is not a finding about the modalities.** Both deployed encoders have seen these corpora in training: emotion2vec+ saw IEMOCAP, MELD and CREMA-D through EmoBox, and the DistilRoBERTa text model was trained on MELD but not IEMOCAP, which favours voice on IEMOCAP and text on MELD exactly as the table shows. With contamination-free encoders the margins are +13.05 on IEMOCAP and +0.03 on MELD (clean_probe_results.md, K-3): voice's advantage vanishes on MELD but text does not overtake it, and the swing shrinks by about three quarters. The defensible pattern is in clean_spontaneity_results.md, Table S-4: voice beats text only in improvised speech, and the two tie for scripted speech in both the studio (IEMOCAP scripted) and on television (MELD). That the deployed text-leading heuristic is miscalibrated is measured directly in fusion_transfer_results.md (F-1, F-2).

## Table C1-16 - Brevity control on step 2

| Subset              | n    | bal acc | macro F1 |
|---------------------|------|---------|----------|
| IEMOCAP, <= 5 words | 1826 | 80.85   | 79.20    |
| IEMOCAP, > 5 words  | 3705 | 85.37   | 85.09    |

MELD's utterances are shorter than IEMOCAP's (mean 8.34 against 11.49 words; 26.2 % against 19.5 % at three words or fewer), and short utterances are already known to hurt the voice modality. Brevity is therefore an alternative explanation for step 2 and is tested directly: restricting IEMOCAP to its short utterances costs 4.52 balanced-accuracy points, which is a small fraction of step 2. For the deployed encoder, step 2 is therefore not an artifact of utterance length. This does not make it acoustic: see the note under Table C1-11.

## Per-class F1 - IEMOCAP, voice only, common four

```
              precision    recall  f1-score   support

       angry      0.957     0.831     0.890      1103
       happy      0.755     0.938     0.837      1617
     neutral      0.921     0.657     0.767      1704
         sad      0.757     0.927     0.833      1078

    accuracy                          0.828      5502
   macro avg      0.848     0.838     0.832      5502
weighted avg      0.847     0.828     0.825      5502

```

## Frustration has no home in the deployed label scheme

| voice argmax | n   | %    |
|--------------|-----|------|
| sad          | 514 | 27.8 |
| happy        | 463 | 25.0 |
| angry        | 441 | 23.9 |
| neutral      | 423 | 22.9 |
| surprised    | 3   | 0.2  |
| disgusted    | 3   | 0.2  |
| fearful      | 2   | 0.1  |

1849 utterances labelled `fru` in IEMOCAP, the corpus's second largest category, have no counterpart among the deployed seven classes and are excluded from every accuracy table above. This row shows where the encoder puts them instead: almost evenly across four classes, so the encoder has no representation of frustration. If frustrated speech is harder than the retained classes, as that spread suggests, excluding it flatters IEMOCAP. That makes the CREMA-D-to-IEMOCAP step in Table C1-11 conservative but the IEMOCAP-to-MELD step an OVERestimate, not a lower bound as an earlier version of this file stated.

## Sensitivity - the excited-into-happy merge

| Scheme                       | n    | bal acc | macro F1 |
|------------------------------|------|---------|----------|
| happy = hap + exc (reported) | 5531 | 83.76   | 83.12    |
| happy = hap only             | 4490 | 82.64   | 78.90    |

Merging `exc` into happy is conventional on IEMOCAP and is forced by the deployed scheme, which has no excited category. It moves 1,041 utterances, so it is reported both ways rather than left implicit.

## Corpus composition

| kind   | n cached | n common-four | mean dur s | median dur s | mean words |
|--------|----------|---------------|------------|--------------|------------|
| impro  | 3984     | 2943          | 4.41       | 3.63         | 11.9       |
| script | 3548     | 2588          | 4.73       | 3.58         | 11.8       |
