# Gap 2, part 1: the voice-text pattern on EARS (read versus freeform)

640 recordings, 107 speakers, angry / neutral / sad. Voice: WavLM probe on the mean of each recording's 3 s windows. Text: MiniLM probe on the Whisper-small transcript of the whole recording. Probes grouped 5-fold by speaker, settings from MELD dev.

## Table T-1 - Voice minus text, by condition

| Condition | recordings | voice bal acc | text bal acc | voice - text [95 % CI] |
|-----------|------------|---------------|--------------|------------------------|
| read      | 321        | 100.00        | 100.00       | +0.00 [+0.00, +0.00]   |
| freeform  | 319        | 95.31         | 100.00       | -4.69 [-6.90, -2.51]   |

Freeform margin minus read margin: -4.69 points, 95 % CI [-6.90, -2.51] (speakers resampled; every speaker contributes to both conditions, so the comparison is paired).

For reference, clean_spontaneity_results.md Table S-4 (IEMOCAP, gold transcripts): scripted +0.40, improvised +25.76, improvised minus scripted +25.36 [+19.65, +30.59]; MELD (scripted television) +0.03.

Mean transcript length: read 31.0 words, freeform 38.0 words. Of 112 distinct read transcripts, 0 occur under more than one emotion. If that share were high, the read sentences would be carrier sentences (as in CREMA-D) and text would be uninformative by design; it is reported so the read condition's text accuracy can be interpreted.

**Verdict: UNINFORMATIVE, text is at ceiling in both conditions.** The read sentences are emotion-specific and shared across all speakers, so a text probe evaluated on held-out speakers still sees held-in SENTENCES and memorises sentence -> emotion; and freeform monologues, recorded on the instruction to speak in an emotion, name or describe it. Neither condition can show whether voice or text carries more emotion, so EARS does not test the Table S-4 pattern. The margins above are reported for completeness and must not be cited as evidence for or against it.

